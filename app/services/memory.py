# 会话记忆服务 - Redis 优先，内存兜底
#
# 产品形态：单一持续聊天页。所有消息复用同一个 conversation_id，
# 每条消息独立保存 24 小时（expires_at = created_at + 86400）。
# 24 小时后消息不再展示，并由 Redis 过期机制清理。
#
# 实现：保留 Redis JSON 数组，key = aichat:history:{conversation_id}。
#   - 每条消息带 message_id / role / content / created_at / expires_at；
#   - 整 key 留一个宽松 TTL 作整体兜底清理（30 天），逐条过期判断才是主机制；
#   - 读取/写入前过滤过期消息，按 created_at 排序返回；
#   - 不再因固定 40 条上限提前删除尚未过期的历史消息；
#   - message_id 幂等：重复写入同 message_id 直接跳过，避免重试产生重复消息。
#
# 内存兜底与"以 Redis 当前值为基准"的既有行为一并保留。
import os
import json
import time
import uuid
from typing import Dict, List, Optional

import redis.asyncio as aioredis

# 单条消息有效期（秒）：created_at + 86400
MESSAGE_TTL = 86400
# 整 key 兜底 TTL：逐条过期判断主导清理，此处仅防进程长期不续期造成滞留
HISTORY_KEY_TTL = 60 * 60 * 24 * 30

# 内存兜底存储：conversation_id -> [{'message_id','role','content','created_at','expires_at'}]
_MEMORY: Dict[str, List[Dict]] = {}

_redis = None


async def _get_redis():
    """会话记忆专用异步连接（懒加载单例；DB 取 REDIS_DB，与问答缓存同库 DB1）。"""
    global _redis
    if _redis is None:
        _redis = aioredis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            password=os.getenv("REDIS_PASSWORD") or None,
            decode_responses=True,
            socket_timeout=3,
            socket_connect_timeout=3,
        )
    return _redis


def _new_message_id(role: str, created_at: int) -> str:
    return f"msg_{role[:1]}_{created_at}_{uuid.uuid4().hex[:8]}"


def _prune_expired(msgs: List[Dict], now: int) -> List[Dict]:
    """过滤并用『最近 24h 内 created_at 大于窗口起点』作为展示边界（逐条）。"""
    out = []
    cutoff = now - MESSAGE_TTL
    for m in msgs:
        created = int(m.get("created_at") or now)
        expires = m.get("expires_at")
        if expires is not None and int(expires) <= now:
            continue
        out.append(m)
    # 上面逻辑：不过期指的是未到 expires_at；created_expired 统一再按 created_at 窗口收尾
    # （保守：把过老的旧格式消息也排除）
    result = [m for m in out if int(m.get("created_at") or now) >= cutoff]
    return result


def _to_compact(msg: Dict) -> Dict:
    """返回给 LLM/UI 的最小消息结构（role/content），保留完整字段在存储层。"""
    return {"role": msg.get("role", "user"), "content": msg.get("content", "")}


async def get_history(conversation_id: str, *, include_full: bool = False,
                      now: Optional[int] = None) -> List[Dict]:
    """
    返回对话历史（未过期、按 created_at 排序）。

    include_full=False（默认）：仅 role/content，供 LLM _trim_history 消费。
    include_full=True：含 message_id/created_at/expires_at 完整结构，供 UI 翻阅。
    无 conversation_id 时返回空列表。
    """
    if not conversation_id:
        return []
    now_unix = now if now is not None else int(time.time())
    try:
        r = await _get_redis()
        raw = await r.get(f"aichat:history:{conversation_id}")
        if raw:
            msgs = _prune_expired(json.loads(raw), now_unix)
            msgs.sort(key=lambda m: int(m.get("created_at") or 0))
            if not include_full:
                return [_to_compact(m) for m in msgs]
            return msgs
    except Exception as e:
        print(f"[Memory] Redis 读取失败，使用内存: {e}")
    mem = _prune_expired(_MEMORY.get(conversation_id, []), now_unix)
    mem.sort(key=lambda m: int(m.get("created_at") or 0))
    if not include_full:
        return [_to_compact(m) for m in mem]
    return mem


async def append_history(conversation_id: str, role: str, content: str,
                         message_id: Optional[str] = None,
                         created_at: Optional[int] = None,
                         now: Optional[int] = None) -> str:
    """追加一条消息（逐条 24h 过期）。返回 message_id（可用于幂等）。

    - 写入前清掉过期消息；
    - message_id 已存在则跳过（幂等，重试/缓存回放不重复）；
    - 不因固定上限提前删除未过期消息。
    """
    if not conversation_id:
        return ""
    now_unix = now if now is not None else int(time.time())
    created = created_at or now_unix
    mid = message_id or _new_message_id(role, created)
    if mid in _msg_ids(_MEMORY.get(conversation_id, [])):
        return mid  # 内存内已存在即幂等

    key = f"aichat:history:{conversation_id}"
    entry = {
        "message_id": mid,
        "role": role,
        "content": content,
        "created_at": created,
        "expires_at": created + MESSAGE_TTL,
    }
    try:
        r = await _get_redis()
        raw = await r.get(key)
        msgs = json.loads(raw) if raw else list(_MEMORY.get(conversation_id, []))
        if mid in _msg_ids(msgs):
            return mid  # Redis 已存在 → 幂等
        msgs = _prune_expired(msgs, now_unix)
        msgs.append(entry)
        msgs.sort(key=lambda m: int(m.get("created_at") or 0))
        _MEMORY[conversation_id] = msgs  # 内存兜底同步
        await r.set(key, json.dumps(msgs, ensure_ascii=False), ex=HISTORY_KEY_TTL)
        return mid
    except Exception as e:
        # Redis 不可用：退回纯内存追加
        if mid in _msg_ids(_MEMORY.get(conversation_id, [])):
            return mid
        msgs = list(_MEMORY.get(conversation_id, []))
        msgs = _prune_expired(msgs, now_unix)
        msgs.append(entry)
        msgs.sort(key=lambda m: int(m.get("created_at") or 0))
        _MEMORY[conversation_id] = msgs
        print(f"[Memory] Redis 写入失败，使用内存: {e}")
        return mid


def _msg_ids(msgs: List[Dict]) -> set:
    return {m.get("message_id") for m in (msgs or []) if m.get("message_id")}


async def close_memory_redis():
    """关闭记忆连接（服务关闭时调用，释放连接池）。"""
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None
