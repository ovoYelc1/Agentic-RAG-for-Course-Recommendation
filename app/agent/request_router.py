# Request Router —— 决定当前请求交给哪条处理路径（纯 Python 规则，零额外 LLM）
#
# 职责边界（与 DB2 意图缓存的严格区分）：
#   - DB2 意图缓存（lookup_intent_filter）继续负责“课程领域/别名/技术词识别”，
#     只是给 QueryContext.search_filters 补充字段，**不决定路由**。
#   - 本模块负责“这条请求整体走 FAQ / 结构化快查 / Agent / 通用回答”。
#     两者不互通：意图命中某课程别名 ≠ 一定走结构化快查。
#
# 规则拿不准时**默认走 AGENT_COURSE**（委派给原 Agent / 通用回答），绝不强行进
# 结构化快查热路径，避免“语义不清却套模板”。
import re
from dataclasses import dataclass
from typing import List

from app.services.context_manager import QueryContext
from app.services.query_signals import has_course_refinement_signal, is_faq_trial_query
from app.services.cache import follows_context

# 路由枚举（与任务建议一致）
FAQ = "FAQ"
STRUCTURED = "STRUCTURED_COURSE_SEARCH"
AGENT = "AGENT_COURSE"
GENERAL = "GENERAL_LLM"

# AGENT 专属信号 1：多技术领域比较（"java 和 python 哪个好""前端还是后端"）
_MULTI_TECH_RE = re.compile(
    r"和|与|还是|对比|比较|哪个好|哪个更适合|区别|选哪个|哪个简单|vs",
    re.IGNORECASE,
)

# AGENT 专属信号 2：学习路径/路线规划（跨多方向、需要展开长文）
_PATH_RE = re.compile(
    r"学习路线|路线|怎么学|如何学|学习路径|怎么入门|规划|职业发展|转行.*路线|入门.*规划"
)

# AGENT 专属信号 3：复杂咨询/追问（需结合会话语境）
_COMPLEX_RE = re.compile(
    r"帮我|给个方案|建议学|适合.*[学选]|该不该|能不能.*学|值得.*学|大概学多久|前景|薪资"
)
_RELATIVE_LEVEL_RE = re.compile(
    r"太难|难了|简单一点|更简单|容易一点|难一点|更难|进阶一点|高级一点"
)

@dataclass
class RouteDecision:
    route: str
    reason: str
    confidence: float  # 规则优先级标记（非模型概率）


def _tech_words(ctx: QueryContext) -> List[str]:
    """收集本轮 + 槽位里的技术词（本体判断是否多技术比较）。"""
    words = list(ctx.extracted.get("tech_keywords") or [])
    for t in ctx.slot_snapshot.get("tech_keywords") or []:
        if t not in words:
            words.append(t)
    return words


def _resembles_comparison(query: str, tech_words: List[str]) -> bool:
    q = (query or "").lower()
    if not _MULTI_TECH_RE.search(q):
        return False
    # 比较连词 + 至少两个不同技术词，或连词 + 明显的“二选一/对比”结构
    if len(set(tech_words)) >= 2:
        return True
    return len(tech_words) >= 1 and _PATH_RE.search(q) is not None


def route(ctx: QueryContext) -> RouteDecision:
    """基于 QueryContext 产出路由决策。"""
    q = ctx.query or ""
    hint = (ctx.intent or "generic").lower()
    extracted = ctx.extracted or {}
    tech_words = _tech_words(ctx)
    reason = ""

    # 1) 明确平台政策 / 会员规则 → FAQ（政策事实已内联 system prompt，禁工具）
    if hint == "faq" and not _course_refinement_signal(ctx):
        return RouteDecision(FAQ, "explicit_faq_intent", 0.97)

    # 2) 多技术领域比较 → Agent（跨领域、需综合权衡）
    if _resembles_comparison(q, tech_words):
        reason = "multi_technology_comparison"
        return RouteDecision(AGENT, reason, 0.9)

    # 3) 学习路径 / 路线规划 → Agent（跨多方向，需展开详答）
    if _PATH_RE.search(q):
        reason = "learning_path_planning"
        return RouteDecision(AGENT, reason, 0.88)

    # 4) 复杂咨询 / 追问 / 结合语境 → Agent
    if _COMPLEX_RE.search(q):
        reason = "complex_course_consult"
        return RouteDecision(AGENT, reason, 0.8)

    # 继承槽位上的短追问和相对精化要交给 Agent 结合上下文处理。唯一保留的
    # 结构化例外是产品已冻结的"再便宜一些"：有既有预算时按旧上限重查，不生成新价。
    explicit_delta = bool(
        extracted.get("category") or extracted.get("categories")
        or extracted.get("tech_keywords") or extracted.get("excluded")
        or extracted.get("excluded_tech_keywords") or extracted.get("level")
        or extracted.get("max_price") is not None
    )
    slot_snapshot = ctx.slot_snapshot or {}
    has_prior_course_context = bool(
        slot_snapshot.get("category") or slot_snapshot.get("categories")
        or slot_snapshot.get("level") or slot_snapshot.get("max_price") is not None
        or slot_snapshot.get("tech_keywords")
        or slot_snapshot.get("excluded_tech_keywords")
    )
    relative_price_recheck = (
        "再便宜" in q.lower()
        and slot_snapshot.get("max_price") is not None
    )
    if (has_prior_course_context and not explicit_delta
            and follows_context(q, has_independent_terms=False)
            and not relative_price_recheck):
        return RouteDecision(AGENT, "context_dependent_followup", 0.82)
    if has_prior_course_context and _RELATIVE_LEVEL_RE.search(q):
        return RouteDecision(AGENT, "relative_level_followup", 0.82)

    # Explicit exclusions cannot be satisfied by the structured fast path:
    # it only has positive category/level/price predicates. Keep the request
    # on the Agent path so the tool/validator can enforce the exclusion.
    if extracted.get("excluded") or extracted.get("excluded_tech_keywords"):
        return RouteDecision(AGENT, "excluded_technology_constraint", 0.8)

    # 5) 通用知识 → GENERAL_LLM（工具无法提供有效信息时直接自身知识答）
    if hint == "generic" and not _has_course_signal(ctx) and not ctx.intent_match:
        reason = "general_knowledge"
        return RouteDecision(GENERAL, reason, 0.85)

    # 6) 明确单领域 / 简单属性检索 → 结构化快查热路径
    if _simple_structured(ctx):
        reason = "single_domain_simple_retrieval"
        return RouteDecision(STRUCTURED, reason, 0.86)

    # 7) 课程类但规则拿不准 → Agent（不强行进热路径）
    if ctx.intent == "course_recommend" or ctx.intent_match:
        reason = "course_but_ambiguous"
        return RouteDecision(AGENT, reason, 0.6)

    # 8) 兜底：默认 Agent / 通用回答
    reason = "default_agent"
    return RouteDecision(AGENT, reason, 0.5)


def _has_course_signal(ctx: QueryContext) -> bool:
    extracted = ctx.extracted or {}
    slots = ctx.slot_snapshot or {}
    filters = ctx.search_filters or {}
    return bool(
        extracted.get("category")
        or extracted.get("categories")
        or extracted.get("tech_keywords")
        or extracted.get("excluded")
        or extracted.get("excluded_tech_keywords")
        or extracted.get("level")
        or extracted.get("max_price") is not None
        or has_course_refinement_signal(ctx.query)
        or slots.get("category")
        or slots.get("categories")
        or slots.get("tech_keywords")
        or slots.get("excluded_tech_keywords")
        or slots.get("level")
        or slots.get("max_price") is not None
        or filters.get("category")
        or filters.get("categories")
        or filters.get("tech_keywords")
        or filters.get("excluded_tech_keywords")
        or filters.get("level")
        or filters.get("max_price") is not None
    )


def _course_refinement_signal(ctx: QueryContext) -> bool:
    """判断 FAQ 词是否其实是课程筛选/精化条件。"""
    extracted = ctx.extracted or {}
    if (str(ctx.intent or "").lower() == "faq"
            and is_faq_trial_query(ctx.query)
            and not (extracted.get("category") or extracted.get("categories")
            or extracted.get("tech_keywords") or extracted.get("excluded")
            or extracted.get("excluded_tech_keywords") or extracted.get("level"))):
        return False
    if (
        extracted.get("category")
        or extracted.get("tech_keywords")
        or extracted.get("excluded")
        or extracted.get("excluded_tech_keywords")
        or extracted.get("level")
        or extracted.get("max_price") is not None
    ):
        return True
    return bool(has_course_refinement_signal(ctx.query))


def prepare_cache_identity(ctx: QueryContext) -> QueryContext:
    """生成缓存前的 canonical identity，并整理有效过滤条件。

    这不是执行分支；它只解决“同一句话在 FAQ/课程精化之间被错误复用”
    的问题，因此可以安全地发生在缓存查询之前。
    """
    refinement = _course_refinement_signal(ctx)
    has_course = (
        refinement
        or ctx.intent == "course_recommend"
        or bool(ctx.intent_match)
        or bool(
            (ctx.search_filters or {}).get("category")
            or (ctx.search_filters or {}).get("categories")
            or (ctx.search_filters or {}).get("level")
            or (ctx.search_filters or {}).get("max_price") is not None
            or (ctx.search_filters or {}).get("tech_keywords")
            or (ctx.search_filters or {}).get("excluded_tech_keywords")
        )
    )
    if ctx.intent == "faq" and not refinement:
        identity = "faq"
    elif has_course:
        identity = "course_refinement" if refinement else "course"
    elif ctx.intent == "generic" and not ctx.intent_match:
        identity = "general"
    else:
        identity = "agent"

    ctx.cache_intent = identity
    if identity not in ("course", "course_refinement"):
        ctx.search_filters = {}
        ctx.hint_filters = {}
        ctx.categories = []
        ctx.category_mode = "none"
    return ctx


def _simple_structured(ctx: QueryContext) -> bool:
    """是否适合直接结构化快查：明确方向 / 简单属性且无矛盾比较嫌。

    需至少一项可结构化约束（category / tech_keyword / price / level），且
    不是多技术比较、不是路线规划、不是复杂咨询。交由真正的过滤执行。
    """
    sf = ctx.search_filters or {}
    # 相对价格抱怨（“太贵了/贵了”）依赖上一轮结果，不能直接按当前槽位
    # 模板检索；交给 Agent 结合上下文和工具处理。
    extracted = ctx.extracted or {}
    if (
        has_course_refinement_signal(ctx.query)
        and not (
            extracted.get("category")
            or extracted.get("tech_keywords")
            or extracted.get("level")
            or extracted.get("max_price") is not None
        )
        and not (
            "再便宜" in (ctx.query or "").lower()
            and sf.get("max_price") is not None
        )
    ):
        return False
    has_struct = bool(
        sf.get("category") or sf.get("level")
        or sf.get("ai_field") or sf.get("ai_level")
        or sf.get("max_price") is not None
    )
    if not has_struct:
        return False
    # 明确单方向且带属性 → 足够清晰，走快查
    cat = sf.get("category") or sf.get("ai_field")
    if cat and _has_course_signal(ctx):
        return True
    # 仅有属性（价格/难度）且已有会话方向 → 精化轮，也走快查
    if sf.get("max_price") is not None or sf.get("level"):
        return True
    return False
