# Redis 缓存服务
#
# 修复记录：原实现用同步 `redis.Redis` 却放在 `async def` 里调用（`r.get(...)` 未 await，
# 也无法 await），导致每次缓存读写都在事件循环线程上做阻塞网络 I/O：
# 一条慢查询会卡住整个 uvicorn worker 的所有并发请求（含 SSE 流）。
# 现改用 `redis.asyncio`（redis-py 5.x 内置，无需新增依赖），全链路真异步。
import json
import hashlib
import os
import re

import redis.asyncio as aioredis


_redis = None

# ====================== 答案缓存 key 构建（v7，归一化 + canonical identity + filters）======================
# v4 的问题：
#   1) 槽位指纹仅『精化轮』（无 category 方向信号）才拼入 → 首轮/精化轮语义依赖
#      隐含分支，且『最终生效的过滤条件』未必与槽位一致（如 DB2 ai_field 补充）；
#   2) 没有把 DB2 intent_match / 路由纳入，曾在『相同 query 不同难度』误命中。
# v7：key 纳入所有实际影响答案的条件：
#    - query 归一化（normalize_query，v4 保留）；
#    - canonical(cache_intent)；
#    - canonical(search_filters)（排序后 join 的规范化指纹：category/level/max_price/
#      tech_keywords/ai_field/ai_level，保证『相同结构化约束才命中』）；
#    - chat:index_ver（索引版本，课程变更即全量失效）。
# 不使用『会话全部历史』作 key；追问严重依赖历史指代时由调用方置
# QueryContext.cache_skip_reason 跳过缓存读。


_TONE_WORDS = ("呢", "吗", "吧", "啊", "呀", "哦", "嘛", "哈", "啦")
_CN_DIGITS = {"零": "0", "一": "1", "二": "2", "两": "2", "三": "3", "四": "4",
              "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}


def normalize_query(q: str) -> str:
    """问题归一化：全角→半角、去语气词、中文数字→阿拉伯、压空白、小写。
    只做确定性变换（不碰语义），保证归一化幂等、key 稳定。"""
    if not q:
        return q
    # 全角字母/数字/标点 → 半角（全角空格单独处理）
    out = []
    for ch in q:
        code = ord(ch)
        if code == 0x3000:
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    s = "".join(out)
    s = "".join(ch for ch in s if ch not in _TONE_WORDS)
    # 独立中文数字转阿拉伯（"三百"→"300" 简化为逐字映射 + 十位处理不值得，只转个位与"两百"类常见价格）
    for cn, ar in _CN_DIGITS.items():
        s = re.sub(rf"(?<![0-9]){cn}(?![0-9])", ar, s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


# ====================== 上下文依赖追问判定（P0#3）======================
# 设计原则：答案缓存 key 只纳入『真正影响答案』的字段（query/cache_intent/filters），
# **不把整段历史拼进 key**。对明显依赖前文的追问，直接跳过缓存读（cache_skip_reason）。
# 命中规则要保守：拿不准是否依赖前文 → 跳过缓存，避免错误共享。
#
# 1) 显式指代 / 相对比较 / 对已推荐内容的追问（无条件跳过）
_FOLLOWUP_DEICTIC = [
    "还有其他的", "还有别的", "再来几个", "换一批", "换些", "多来几个",
    "其他的呢", "别的呢", "还有其他", "别的推荐", "另外",
    "这个怎么样", "这个课程怎么样", "这个课怎么样", "这个呢", "那门",
    "刚才那门", "第二个", "第三个", "上面那个", "我上面", "前面那个",
    "前一个", "上一条", "和上面", "和刚才", "和之前", "和那个",
    "比哪个好", "哪个更好",
]
# 2) 相对价格/难度精化（依赖当前推荐集合的新的一轮）
_FOLLOWUP_REFINE = [
    "再便宜", "便宜点", "便宜些", "便宜一些", "便宜一点", "更便宜",
    "太贵了", "贵了", "超预算", "实惠一点", "再低价", "报价更低",
    "再优惠", "降价", "低一些", "降低预算", "再便宜些",
    "再难一点", "难一些", "太难", "太难了", "进阶一点", "更高级", "再深", "更难",
    "简单一点", "更简单",
]
# 3) 纯指代短句（无独立业务语义）
_FOLLOWUP_SHORT = [
    "还有吗", "还有么", "别的呢", "然后呢", "然后", "继续", "继续吧",
    "下一个", "再一个", "再来", "换一个",
]


def follows_context(query: str, has_independent_terms: bool = True) -> bool:
    """判定该 query 是否强依赖前文（若是 → 应跳过答案缓存）。

    has_independent_terms：调用方判断当前 query 是否含独立业务语义（明确领域/
    技术词/价格/难度）。含独立语义的短句（"还有没有 Python 课"）优先认为可缓存，
    仅当命中强指代表达才跳过。
    """
    if not query:
        return True
    q = query.lower()
    if any(p in q for p in _FOLLOWUP_DEICTIC):
        return True
    if any(p in q for p in _FOLLOWUP_REFINE):
        return True
    if not has_independent_terms:
        if any(p in q for p in _FOLLOWUP_SHORT):
            return True
    return False


_CANON_KEY_ORDER = ("category", "categories", "level", "max_price",
                    "tech_keywords", "excluded_tech_keywords",
                    "ai_field", "ai_level")


def canonical_search_filters(search_filters: dict) -> str:
    """把 search_filters 转成稳定、有序的规范化指纹（参与缓存 key）。
    只纳入『真实影响答案』的结构化约束；空约束为空串。
    keywords 保留给检索和工具，但不进入答案缓存指纹，避免泛化口语词
    （例如“太贵了”“便宜些”“能退吗”）造成无意义的 key 分裂。
    categories 为有序列表排序后 join。"""
    if not search_filters:
        return ""
    parts = []
    for k in _CANON_KEY_ORDER:
        v = search_filters.get(k)
        if v is None or v == "" or v == []:
            continue
        if k == "tech_keywords":
            norm = ",".join(sorted(str(x).lower() for x in v if x))
            if norm:
                parts.append(f"{k}={norm}")
        elif k == "categories":
            norm = ",".join(sorted(str(x).lower() for x in v if x))
            if norm:
                parts.append(f"{k}={norm}")
        elif isinstance(v, (list, tuple)):
            norm = ",".join(sorted(str(x).lower() for x in v if x))
            if norm:
                parts.append(f"{k}={norm}")
        elif isinstance(v, float):
            parts.append(f"{k}={v:g}")
        else:
            parts.append(f"{k}={v}".lower())
    return "&".join(parts)


async def build_answer_cache_key(message: str,
                                 *, intent: str = "",
                                 cache_intent: str = "",
                                 search_filters: dict | None = None) -> str:
    """
    构建答案缓存 key（v7）。所有调用方（routes 非流式/流式、agent 后台润色）
    必须传 intent + search_filters，保证写读一致、约束不同不共享答案。
    """
    norm = normalize_query(message)
    sig = canonical_search_filters(search_filters or {})
    identity = cache_intent or intent
    locale = f"intent={identity}" if identity else ""
    key_body = ":".join([x for x in (locale, sig) if x])
    try:
        r = await get_async_redis()
        index_ver = await r.get("chat:index_ver") or "0"
    except Exception:
        index_ver = "0"
    digest = hashlib.sha256(f"{norm}:{key_body}".encode("utf-8")).hexdigest()
    return f"chat:v7:{index_ver}:{digest}"


async def get_async_redis():
    """
    获取异步 Redis 连接（懒加载单例，内部自带连接池）。

    连接参数：
    - db 取 REDIS_DB（compose 设为 1），这是 **AI 服务自己的**逻辑库：
      对话记忆、问答缓存、限流计数都在这里，与后端缓存(DB0)、意图缓存(DB2)隔离。
    - 因此**不能**用这个连接去读后端写的 JWT 黑名单（那在 DB0）；
      黑名单查询见 app/core/security.py 的独立连接，切勿图省事复用此处。
    - 设置超时，避免 Redis 无响应时请求无限挂起。
    """
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


async def get_cached(key):
    """
    从 Redis 获取缓存

    Args:
        key: 缓存键

    Returns:
        缓存数据，未命中返回 None
    """
    try:
        r = await get_async_redis()
        value = await r.get(key)
        if value:
            return json.loads(value)
    except Exception as e:
        print(f"[Cache] Redis GET 失败: {e}")
    return None


async def set_cache(key, value, ttl=3600):
    """
    设置 Redis 缓存

    Args:
        key: 缓存键
        value: 缓存值
        ttl: 过期时间（秒），默认 1 小时
    """
    try:
        r = await get_async_redis()
        await r.set(key, json.dumps(value, ensure_ascii=False, default=str), ex=ttl)
    except Exception as e:
        print(f"[Cache] Redis SET 失败: {e}")


async def close_redis():
    """关闭 Redis 连接（服务关闭时调用，释放连接池）"""
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None
