# 轻量运行指标：Redis INCR 计数器（DB1，复用缓存连接池）。
#
# 设计取舍：不上 S3/ES 采集管道——当前规模（单副本、30 门课）下，
# 用几个计数器回答"热路径覆盖率多少 / 兜底触发多频繁 / 追问预算拦截几次"
# 已足够，且零外部依赖。埋点失败静默忽略，绝不影响主流程。
# 读取端：GET /api/ai/metrics（走双层鉴权）。
from app.services.cache import get_async_redis


# 读写共用同一份清单，避免新增埋点后忘记在 /api/ai/metrics 暴露。
METRIC_NAMES = (
    "cache_hit", "cache_miss", "cache_skip",
    "route_faq", "route_structured", "route_agent", "route_general",
    "route_unknown",
    "intent_hot", "general_llm", "tool_loop",
    "forced_course_rescue", "rescue", "clarify_used", "clarify_blocked",
)


async def incr(name: str):
    """计数器 +1（key: ai:metrics:{name}）。失败静默。"""
    try:
        r = await get_async_redis()
        await r.incr(f"ai:metrics:{name}")
    except Exception:
        pass
