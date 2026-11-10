# AI 服务用户级鉴权
#
# 背景（修复「未登录访客可白嫖 LLM + 伪造他人会话」漏洞）：
#   原实现只在 main.py 校验 Nginx 注入的 X-API-Token，那个令牌回答的是
#   「这个请求是否经由我们的网关」，而不是「发起者是哪个登录用户」。
#   由于 Nginx 对 /api/ai/ 无条件 proxy_set_header 盖章，任何匿名请求
#   到达 AI 服务时都自带合法令牌 → 直接放行 → 免登录调用 LLM。
#
#   本模块补上缺失的第二层：验证调用方持有后端签发的有效用户 JWT。
#   两层是不同的信任边界，必须同时成立：
#     1) X-API-Token  → 证明「来自内网网关」（防外部直连绕过）
#     2) Authorization → 证明「来自某个已登录用户」（防匿名滥用与越权）
#
# 与 Spring Boot 后端的契约（务必保持同步，任一侧改动都会导致全量 401）：
#   - 密钥：application.yml `jwt.secret` ← 环境变量 JWT_SECRET，且 Java 侧用
#           Keys.hmacShaKeyFor(Decoders.BASE64.decode(secret))，即字符串是
#           **Base64 编码**的，须先解码再作 HMAC 原始密钥。
#   - 载荷：sub=userId、username、role、jti、iat、exp（见 JwtUtil.generateToken）
#   - 登出黑名单：JwtBlacklistUtil 优先以 "jwt:black:" + **jti** 为 key（旧条目为
#           "jwt:black:" + 完整 token 原文），写在 Redis DB 0
#           （application.yml spring.data.redis.database: 0）。本模块同时查两种 key。
import base64
import binascii
import hmac
import os
from typing import Optional

import jwt
from fastapi import Header, HTTPException

BLACKLIST_PREFIX = "jwt:black:"

# 对称算法白名单。JJWT 的 signWith(key) 按密钥长度自动选 HS256/384/512，
# 这里全部接受；因均为 HMAC，不存在「alg 混淆降级到非对称」的风险。
_ALLOWED_ALGS = ["HS256", "HS384", "HS512"]


class Identity:
    """经校验的调用方身份。字段来源均为 JWT 签名载荷，不接受客户端自报。"""

    __slots__ = ("user_id", "username", "role", "token")

    def __init__(self, user_id: str, username: str, role: str, token: str):
        self.user_id = user_id
        self.username = username
        self.role = role
        self.token = token


def _hmac_key() -> bytes:
    """
    还原后端使用的 HMAC 原始密钥。

    必须与 Java 侧 Decoders.BASE64.decode(secret) 完全一致，否则验签必然失败。
    JWT_SECRET 缺失时直接抛错（fail-fast），避免「配置漏了却静默放行」。
    """
    secret = os.getenv("JWT_SECRET", "")
    if not secret:
        raise RuntimeError(
            "JWT_SECRET 未配置：AI 服务无法校验用户身份。"
            "请在 docker-compose 中为 ai-service 注入与后端相同的 JWT_SECRET。"
        )
    try:
        return base64.b64decode(secret, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(
            "JWT_SECRET 不是合法 Base64：后端以 Base64 解码后作为 HMAC 密钥，两侧须一致。"
        ) from exc


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    """从 Authorization 头取出 Bearer token；格式不符返回 None。"""
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


_blacklist_redis = None


async def _get_blacklist_redis():
    """
    连接**后端所用的** Redis 逻辑库，用于读取 JWT 登出黑名单。

    为什么不复用 app.services.cache 的连接（这是个必须踩明白的坑）：
      docker-compose 给 ai-service 设的是 REDIS_DB=1（对话记忆库，与后端隔离），
      而后端把黑名单写在 DB 0（application.yml `spring.data.redis.database: 0`）。
      若复用 DB1 的连接查 `jwt:black:*`，永远查不到 key → exists=0 → 判定「不在黑名单」
      → **已登出的 Token 在 AI 服务依旧畅通**，而且是静默放行（不会报错），极难察觉。
      故此处独立建连，库号由 BACKEND_REDIS_DB 指定（默认 0），与后端保持一致。
    """
    global _blacklist_redis
    if _blacklist_redis is None:
        import redis.asyncio as aioredis

        _blacklist_redis = aioredis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("BACKEND_REDIS_DB", 0)),
            password=os.getenv("REDIS_PASSWORD") or None,
            decode_responses=True,
            socket_timeout=3,
            socket_connect_timeout=3,
        )
    return _blacklist_redis


async def _is_blacklisted(token: str, jti: Optional[str] = None) -> bool:
    """
    查询后端登出黑名单。

    与后端 JwtBlacklistUtil 保持契约：优先查 "jwt:black:" + jti，
    同时查 "jwt:black:" + 完整 token（兼容 jti 改造前的旧黑名单条目）。

    安全取舍与后端 JwtBlacklistUtil 对齐：Redis 不可用时 **fail-closed**（返回 True 拒绝），
    否则 Redis 故障窗口内已登出的 Token 可被重放。
    注意这与限流模块的 fail-open 相反——安全判断宁可拒绝，容量保护宁可放行。
    """
    try:
        client = await _get_blacklist_redis()
        if jti:
            return bool(await client.exists(BLACKLIST_PREFIX + jti, BLACKLIST_PREFIX + token))
        return bool(await client.exists(BLACKLIST_PREFIX + token))
    except Exception as exc:  # noqa: BLE001 - 任何异常都按拒绝处理
        print(f"[Auth] 黑名单查询失败，按 fail-closed 拒绝: {exc}")
        return True


async def close_blacklist_redis() -> None:
    """服务关闭时释放黑名单连接池。"""
    global _blacklist_redis
    if _blacklist_redis is not None:
        await _blacklist_redis.aclose()
        _blacklist_redis = None


def _gateway_token_ok(x_api_token: Optional[str], authorization: Optional[str]) -> bool:
    """
    校验服务间令牌（第一层：证明来自内网网关）。

    使用 hmac.compare_digest 做定长比较，消除原 `==` 短路比较的时序侧信道。
    """
    expected = os.getenv("AI_SERVICE_TOKEN", "")
    if not expected:
        return False
    if x_api_token and hmac.compare_digest(x_api_token, expected):
        return True
    # 兼容旧调用方式：Authorization: Bearer <服务令牌>（非用户 JWT）
    bearer = _extract_bearer(authorization)
    if bearer and hmac.compare_digest(bearer, expected):
        return True
    return False


async def require_identity(
    x_api_token: Optional[str] = Header(None, alias="X-API-Token"),
    authorization: Optional[str] = Header(None),
) -> Identity:
    """
    FastAPI 依赖：同时校验「网关令牌」与「用户 JWT」，返回可信身份。

    AI_REQUIRE_AUTH != "true" 时（本地联调）跳过校验，返回匿名身份，
    使 uvicorn 直跑、curl 调试仍然可用；生产 compose 已设为 true。
    """
    if os.getenv("AI_REQUIRE_AUTH", "false").lower() != "true":
        return Identity(user_id="local-dev", username="local-dev", role="student", token="")

    # ---- 第一层：必须来自内网网关 ----
    if not _gateway_token_ok(x_api_token, authorization):
        raise HTTPException(status_code=401, detail="缺少或错误的 AI 服务令牌")

    # ---- 第二层：必须是已登录用户 ----
    token = _extract_bearer(authorization)
    # 网关令牌可能也放在 Authorization 里（旧式服务间调用），此时不构成用户身份
    if token and hmac.compare_digest(token, os.getenv("AI_SERVICE_TOKEN", "") or "\0"):
        token = None
    if not token:
        raise HTTPException(
            status_code=401,
            detail="请先登录后再使用 AI 客服",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(
            token,
            _hmac_key(),
            algorithms=_ALLOWED_ALGS,
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    except jwt.InvalidTokenError:
        # 统一话术，不回显具体验签失败原因，避免为爆破提供反馈
        raise HTTPException(status_code=401, detail="登录凭证无效，请重新登录")
    except RuntimeError as exc:
        # JWT_SECRET 配置问题属服务端错误，不能当成「凭证无效」误导用户
        print(f"[Auth] 配置错误: {exc}")
        raise HTTPException(status_code=500, detail="AI 服务鉴权配置异常")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="登录凭证无效，请重新登录")

    if await _is_blacklisted(token, payload.get("jti")):
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")

    return Identity(
        user_id=str(user_id),
        username=payload.get("username") or "",
        role=payload.get("role") or "",
        token=token,
    )
