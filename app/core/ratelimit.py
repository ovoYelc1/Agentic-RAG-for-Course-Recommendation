# AI 端点限流
#
# 两道闸门：
#   1) 频率闸门：滑动窗口近似（固定窗口计数），限制每分钟请求数。
#   2) 并发闸门：同一用户同时仅允许 N 个进行中的 AI 请求（默认 1），
#      防单用户占满推理队列。
#
# 失败取舍（与黑名单相反，务必区分）：
#   黑名单查询失败 → fail-closed（拒绝），因为那是**安全**判断；
#   限流查询失败   → fail-open（放行），因为那是**容量**保护，
#   若 Redis 抖动就让全站 AI 不可用，等于把限流做成了自伤开关。
import os
import time
from typing import Optional

from fastapi import HTTPException

_RATE_PREFIX = "airate:"
_CONC_PREFIX = "aiconc:"


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


async def check_rate_limit(user_id: str) -> None:
    """
    频率闸门：固定窗口计数。超限抛 429。

    窗口边界处最多放过 2 倍额度（固定窗口的固有缺陷），
    对本场景（防滥用而非精确计费）足够，换取实现简单与单次 Redis 往返。
    """
    limit = _int_env("AI_RATE_LIMIT_PER_MIN", 20)
    if limit <= 0:
        return

    window = 60
    key = f"{_RATE_PREFIX}{user_id}:{int(time.time()) // window}"
    try:
        from app.services.cache import get_async_redis

        r = await get_async_redis()
        pipe = r.pipeline()
        pipe.incr(key)
        pipe.expire(key, window)
        count, _ = await pipe.execute()
    except Exception as exc:  # noqa: BLE001
        print(f"[RateLimit] Redis 不可用，跳过频率限流（fail-open）: {exc}")
        return

    if count and int(count) > limit:
        raise HTTPException(
            status_code=429,
            detail=f"提问太频繁了，请稍后再试（每分钟上限 {limit} 次）",
            headers={"Retry-After": "60"},
        )


class ConcurrencyGuard:
    """
    并发闸门：以 Redis 计数器约束单用户同时进行中的请求数。

    两种用法：
      1) 普通接口 —— async with（异常时自动释放）：
             async with ConcurrencyGuard(user_id):
                 ...
      2) SSE 流式接口 —— 必须显式 acquire/release：
             guard = ConcurrencyGuard(user_id)
             await guard.acquire()          # 在返回 StreamingResponse 之前，才能正确返回 429
             ... 生成器 finally 中 await guard.release()
         原因：一旦 StreamingResponse 开始输出，HTTP 200 与响应头已发出，
         此时再抛 HTTPException 无法改写状态码，客户端只会收到一个畸形流。

    释放采用「先减后判」而非 DEL，避免多请求间互相误删对方的槽位。
    计数器带 TTL 兜底：进程被 kill 导致 finally 未执行时，槽位不会永久泄漏。
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.key = f"{_CONC_PREFIX}{user_id}"
        self.limit = _int_env("AI_MAX_CONCURRENCY_PER_USER", 1)
        self.ttl = _int_env("AI_CONCURRENCY_TTL", 300)
        self._redis = None
        self._acquired = False

    async def acquire(self) -> "ConcurrencyGuard":
        """占用一个并发槽位；超限抛 429。Redis 不可用时 fail-open 放行。"""
        if self.limit <= 0:
            return self
        try:
            from app.services.cache import get_async_redis

            self._redis = await get_async_redis()
            pipe = self._redis.pipeline()
            pipe.incr(self.key)
            pipe.expire(self.key, self.ttl)
            current, _ = await pipe.execute()
        except Exception as exc:  # noqa: BLE001
            print(f"[RateLimit] Redis 不可用，跳过并发限制（fail-open）: {exc}")
            self._redis = None
            return self

        self._acquired = True
        if current and int(current) > self.limit:
            # 超限：立刻归还自己刚占用的槽位，再拒绝
            await self.release()
            raise HTTPException(
                status_code=429,
                detail="您有一个提问正在处理中，请等它回答完再发下一条",
            )
        return self

    async def __aenter__(self):
        return await self.acquire()

    async def __aexit__(self, exc_type, exc, tb):
        await self.release()
        return False

    async def release(self) -> None:
        if not self._acquired or self._redis is None:
            return
        self._acquired = False
        try:
            remaining = await self._redis.decr(self.key)
            if remaining is not None and int(remaining) <= 0:
                await self._redis.delete(self.key)
        except Exception as exc:  # noqa: BLE001
            print(f"[RateLimit] 并发槽位释放失败（TTL {self.ttl}s 后自动回收）: {exc}")


def resolve_session_id(user_id: str, client_session_id: Optional[str]) -> str:
    """
    生成**绑定到用户**的会话 ID，阻断跨用户读取对话历史。

    漏洞背景：会话历史存于 Redis `aichat:history:{session_id}`，而原 session_id
    完全由客户端在请求体自报 → 猜测/枚举他人 session_id 即可读取并污染其聊天记录（IDOR）。

    修复：无论客户端传什么，都强制加上服务端可信的 user_id 前缀，
    使不同用户的命名空间天然隔离；客户端仍可用后缀区分自己的多个会话。
    """
    safe_suffix = ""
    if client_session_id:
        # 只保留安全字符并截断，防止注入冒号伪造出别人的命名空间
        cleaned = "".join(c for c in str(client_session_id) if c.isalnum() or c in "-_")
        safe_suffix = cleaned[:64]
    return f"u{user_id}:{safe_suffix}" if safe_suffix else f"u{user_id}:default"
