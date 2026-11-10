# 统一 QueryContext —— 一次请求生命周期内的上下文数据对象
#
# 目标：把此前散落在 run_agent / run_agent_stream / routes 各局部变量里的
#   query / intent / slots / intent_match / search_filters / recent_history
# 收敛成一个请求级对象，供检索、工具循环、Rescue、Validator、答案缓存共享。
#
# 原则：
#   - QueryContext 是纯 Python 数据对象（dataclass），不是全局变量，不跨请求共享；
#   - 每次用户消息构建一份，请求结束即释放；
#   - 不把整个对象写入 System Prompt，由 build_context_prompt 择要转文本；
#   - 通过显式函数参数传给下游模块。
#
# search_filters 只包含课程库 metadata **真实支持**的过滤键：
#   category / level / ai_field / ai_level  → structured where
#   max_price                               → 内存后置过滤（price 是字符串）
#   tech_keywords / keywords                → 课名收窄 / 排序
# `learning_style` 在课程 metadata 中不存在，不得作为硬过滤字段。
import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


from app.services.query_signals import has_course_refinement_signal, is_faq_trial_query


PLAN_TOTEM = "query-context-v1"


@dataclass
class QueryContext:
    """一次请求的上下文。

    slot_snapshot: 当前对话『有效（未过期）槽位合并结果』的纯值字典
                   （{category, level, max_price, tech_keywords, keywords}），不
                   含 clarify_used（那是 Agent 对话控制状态，不进业务检索）。
    intent_match : DB2 离线课程别名匹配结果（lookup_intent_filter 返回值，
                   如 {"ai_field": "后端"} 或 None）。它只做领域/别名补充，
                   **不**决定路由。
    search_filters: 可真正交给课程检索执行的过滤条件，是其唯一权威来源。
    recent_history: 供 LLM 使用的近期历史（已裁剪）；不等于全部 24h 历史。
    cache_skip_reason: 当需要跳过答案缓存时置位（如追问严重依赖历史指代）。
    """

    conversation_id: str
    user_id: str
    query: str

    intent: str = "generic"
    route: str = "AGENT_COURSE"          # 由 Request Router 回填
    route_reason: str = ""               # Router 规则原因；用于区分默认兜底 AGENT
    route_confidence: float = 0.0

    # —— 本轮规则解析结果 ——
    extracted: dict = field(default_factory=dict)
    intent_match: Optional[dict] = None
    slot_snapshot: dict = field(default_factory=dict)

    # —— 派生 ——
    search_filters: dict = field(default_factory=dict)
    recent_history: list = field(default_factory=list)

    # —— 多领域（P0#1）——
    # categories: 本轮明确涉及的全部领域（有序、去重）。单领域时为空。
    # category_mode: "single" | "multi" | "none"
    categories: list = field(default_factory=list)
    category_mode: str = "single"
    # hint_filters: DB2 离线推断（ai_field/ai_level）——只作方向补充，
    # 不作为删除其它领域的硬约束。与 search_filters(hard) 分离（P0#2）。
    hint_filters: dict = field(default_factory=dict)

    # —— 控制 ——
    cache_skip_reason: str = ""
    # 缓存身份由 Context Manager 之后的轻量归一化阶段回填。
    cache_intent: str = "agent"

    def is_course_route(self) -> bool:
        return self.route in ("STRUCTURED_COURSE_SEARCH", "AGENT_COURSE")


class SlotState:
    """会话槽位（含来源 / 有效期）的构建与合并。

    槽位值结构：{field: {"value": v, "source_message_id": ..., "updated_at": ...,
                          "expires_at": ...}}
    这里只负责「合并 & 去过期 & 摊平成纯值」，底层读写交给 services.slots（同文件
    也可被服务层复用）。history/slots 的 get 由调用方注入（便于测试与异步聚合）。
    """

    SLOT_FIELDS = ("category", "categories", "level", "max_price", "keywords",
                   "tech_keywords", "excluded_tech_keywords")

    def __init__(self, raw: Optional[dict] = None, now: Optional[int] = None):
        self.raw = dict(raw or {})     # 原始（含来源/有效期元数据，或旧裸值）
        self._now = now if now is not None else int(time.time())

    # ---------- 过期判断 ----------
    @staticmethod
    def _is_expired(entry: dict, now: int) -> bool:
        exp = entry.get("expires_at")
        if exp is None:
            return False
        return int(exp) <= now

    # ---------- 读取：仅返回未过期有效纯值 ----------
    def effective_values(self) -> Dict:
        out: Dict = {}
        for k in self.SLOT_FIELDS:
            entry = self.raw.get(k)
            if entry is None:
                continue
            if isinstance(entry, dict):
                if self._is_expired(entry, self._now):
                    continue
                v = entry.get("value")
            else:
                # 兼容旧裸值（无来源字段）
                v = entry
            if v is not None and v != "" and v != []:
                out[k] = v
        return out

    # ---------- 更新：显式迭代所提字段 ----------
    def set_fields(self, updates: dict, source_message_id: Optional[str] = None,
                   ttl: int = 86400) -> None:
        """本轮『明确提及』的字段覆盖；未提及/空值字段不做任何操作（继承旧值）。

        显式语义：None / 空列表 / 空字符串 一律视为『未提及』，不覆盖不删除——
        避免 extract_recommend_params 每轮都返回空列表把继承值清掉。
        """
        for k, v in updates.items():
            if v is None:
                continue
            if hasattr(v, "__len__") and len(v) == 0:
                continue
            self._set_one(k, v, source_message_id, ttl)

    def clear_fields(self, fields) -> None:
        """用户『明确取消』某条件时删除；未提及 ≠ 取消。"""
        for k in fields:
            self.raw.pop(k, None)

    def _set_one(self, k, v, source_message_id, ttl) -> None:
        now = self._now
        self.raw[k] = {
            "value": v,
            "source_message_id": source_message_id or "",
            "updated_at": now,
            "expires_at": now + ttl,
        }


def _is_topic_switch(intent_hint: str) -> bool:
    return intent_hint in ("faq", "order_query", "account_help")


def _has_course_signal(extracted: dict) -> bool:
    """本轮是否有明确课程属性增量信号。

    普通中文 keywords 只是抽取器的文本残片，不足以证明用户仍在课程话题。
    """
    return bool(
        extracted.get("category")
        or extracted.get("categories")
        or extracted.get("tech_keywords")
        or extracted.get("excluded")
        or extracted.get("excluded_tech_keywords")
        or extracted.get("level")
        or extracted.get("max_price") is not None
    )


def _relative_level(query: str, current: str):
    """按用户的相对难度表达调整已有难度槽位，不凭空创建难度。"""
    q = (query or "").lower()
    if not current:
        return None
    if any(x in q for x in ("太难", "难了", "简单一点", "更简单", "容易一点")):
        return {"advanced": "intermediate", "intermediate": "beginner"}.get(current)
    if any(x in q for x in ("太简单", "难一点", "更难", "进阶一点", "高级一点")):
        return {"beginner": "intermediate", "intermediate": "advanced"}.get(current)
    return None


def _preserve_ai_followup(query: str, current_category: str,
                          extracted_category: str | None) -> bool:
    """AI 话题中的代词式“深度学习”追问沿用已确认的 ai 方向。"""
    q = (query or "").lower()
    return (
        current_category == "ai"
        and extracted_category == "data"
        and "深度学习" in q
        and any(marker in q for marker in ("呢", "那", "的"))
    )


def _extract_clearance(extracted: dict) -> list:
    """识别『明确取消某条件』的口语（如'不限预算''取消价格限制'）。"""
    cleared = []
    msg = (extracted.get("_raw_query") or "").lower()
    if "不限" in msg and ("价格" in msg or "预算" in msg):
        cleared.append("max_price")
    return cleared


async def build_query_context(
    conversation_id: str,
    user_id: str,
    query: str,
    *,
    intent_hint: str,
    extracted: dict,
    intent_match: Optional[dict],
    get_slots_history,          # async () -> (raw_slots, history)
    now: Optional[int] = None,
) -> QueryContext:
    """按依赖顺序构建 QueryContext。

    顺序：
      1) 本轮规则解析（hint_intent / extract_recommend_params —— 已在调用方完成，传入）；
      2) 判断是否课程相关，决定是否取 DB2 别名补充（intent_match 由调用方决定是否查询）；
      3) 读取当前会话历史 + 槽位（无依赖，可并行——调用方已用 asyncio.gather 聚合）；
      4) 合并槽位与冲突处理（集中在本函数）；
      5) 摊平出 search_filters 并装配 QueryContext。
    """
    # 兼容两种注入方式：调用方可能已用 asyncio.gather 聚合好并传同步 lambda
    # （routes._build_context_for_request），也可能直接传 async 函数。
    # 之前无条件下 await，遇到同步 lambda 会 TypeError: object tuple can't be used
    # in 'await' expression —— 该异常发生在构建上下文阶段，会让 /api/ai/chat 与
    # /chat/stream 全部 500（单元测试传的是同步函数，故未暴露）。
    loaded = get_slots_history()
    if inspect.isawaitable(loaded):
        loaded = await loaded
    raw_slots, history = loaded

    state = SlotState(raw_slots, now=now)
    now_unix = now if now is not None else int(time.time())
    previous = state.effective_values()

    # FAQ hint 只有在本轮带课程精化语义时才保留课程槽位；
    # 普通退款/价格政策仍按话题切换清空旧课程上下文。
    faq_course_refinement = intent_hint == "faq" and (
        has_course_refinement_signal(query)
        or extracted.get("category")
        or extracted.get("categories")
        or extracted.get("tech_keywords")
        or extracted.get("excluded")
        or extracted.get("excluded_tech_keywords")
        or extracted.get("level")
        or extracted.get("max_price") is not None
    )
    if (intent_hint == "faq" and is_faq_trial_query(query)
            and not (extracted.get("category") or extracted.get("categories")
                     or extracted.get("tech_keywords")
                     or extracted.get("excluded")
                     or extracted.get("excluded_tech_keywords")
                     or extracted.get("level"))):
        faq_course_refinement = False
    if _is_topic_switch(intent_hint) and not _has_course_signal(extracted) and not faq_course_refinement:
        state = SlotState(None, now=now_unix)

    # ---- 用户明确取消某条件 ----
    for f in _extract_clearance(extracted):
        state.clear_fields([f])

    # 相对难度在已有难度上调整，“太难了”不会脱离会话单独猜一个等级。
    relative_level = _relative_level(query, previous.get("level"))
    if relative_level:
        extracted = dict(extracted)
        extracted["level"] = relative_level

    # AI 话题中“那深度学习的呢”是上下文延续，不应把人工智能方向覆成 data。
    if _preserve_ai_followup(query, previous.get("category"),
                             extracted.get("category")):
        extracted = dict(extracted)
        extracted["category"] = previous.get("category")
        extracted["categories"] = []

    current_categories = list(extracted.get("categories") or [])
    if extracted.get("category") and len(current_categories) < 2:
        state.clear_fields(["categories"])

    # 明确换方向时，旧技术词不能继续约束新方向。
    new_category = extracted.get("category")
    if new_category and previous.get("category") and new_category != previous["category"]:
        state.clear_fields(["tech_keywords", "excluded_tech_keywords"])

    # ---- 合并本轮字段（提及即覆盖，未提及继承）----
    # category 非空 → 覆盖，并清理旧领域技术词残留（旧领域关键词不残留到新领域）
    source = extracted.get("_message_id")
    state.set_fields(
        {
            "category": extracted.get("category"),
            "categories": current_categories if len(current_categories) >= 2 else None,
            "level": extracted.get("level"),
            "max_price": extracted.get("max_price"),
            "keywords": extracted.get("keywords"),
            "tech_keywords": extracted.get("tech_keywords"),
            "excluded_tech_keywords": (
                extracted.get("excluded_tech_keywords")
                or extracted.get("excluded")
            ),
        },
        source_message_id=source,
    )

    # 领域切换清理：本轮提到新 category 时，清掉旧方向的技术词（如
    # "前端"语境沉淀的 vue 词，切到"后端"后不应再参与后端课程的名称收窄）。
    # ---- 摊平成最终有效槽位纯值快照（不含 clarify_used）----
    slot_snapshot = state.effective_values()

    # ---- 装配 QueryContext ----
    ctx = QueryContext(
        conversation_id=conversation_id,
        user_id=user_id,
        query=query,
        intent=intent_hint,
        extracted=extracted,
        intent_match=intent_match,
        slot_snapshot=slot_snapshot,
        recent_history=history,
    )
    _finalize_filters(ctx, extracted)
    return ctx


def _finalize_filters(ctx: QueryContext, extracted: dict) -> None:
    """从『合并后的有效槽位』+ 'intent_match' 摊平出 search_filters。

    先摊平有效槽位；FAQ/通用问题是否清空由缓存身份归一化阶段决定，
    避免在 raw hint 阶段误伤“便宜些/太贵了”等课程精化请求。

    多领域（categories）与来源性质（hard / hint）：
    - category=categories 本轮明确表达 → hard，放入 search_filters；
    - intent_match（DB2 ai_field/ai_level）→ 只作 hint_filters 方向补充，
      **不作为删除其它领域的硬约束**（P0#2）。
    """
    if (
        ctx.intent == "faq"
        and not (
            has_course_refinement_signal(ctx.query)
            or extracted.get("category")
            or extracted.get("categories")
            or extracted.get("tech_keywords")
            or extracted.get("excluded")
            or extracted.get("excluded_tech_keywords")
            or extracted.get("level")
            or extracted.get("max_price") is not None
        )
    ):
        if (is_faq_trial_query(ctx.query)
                and not (extracted.get("category") or extracted.get("categories")
                         or extracted.get("tech_keywords")
                         or extracted.get("excluded")
                         or extracted.get("excluded_tech_keywords")
                         or extracted.get("level"))):
            ctx.search_filters = {}
            ctx.hint_filters = {}
            ctx.categories = []
            ctx.category_mode = "none"
            return
        ctx.search_filters = {}
        ctx.hint_filters = {}
        ctx.categories = []
        ctx.category_mode = "none"
        return

    cat = ctx.slot_snapshot.get("category")
    lvl = ctx.slot_snapshot.get("level")
    price = ctx.slot_snapshot.get("max_price")
    techs = ctx.slot_snapshot.get("tech_keywords") or []
    excluded_techs = ctx.slot_snapshot.get("excluded_tech_keywords") or []
    kws = ctx.slot_snapshot.get("keywords")

    # ---- 多领域判定（P0#1）：本轮明确提到的多个领域走 categories ----
    # 注：否定领域已在 router.extract_recommend_params 剔除，这里不再重复依赖
    #     CATEGORY_MAP（避免 context_manager→router 交叉导入）。
    categories = list(extracted.get("categories") or [])
    if len(categories) < 2 and not extracted.get("category"):
        categories = list(ctx.slot_snapshot.get("categories") or [])
    # 单领域时：若 categories 去重后只有一个，退化为单值 category
    if len(categories) >= 2:
        ctx.categories = categories
        ctx.category_mode = "multi"
        cat = None  # 交给多领域检索，不再用单值
    else:
        ctx.categories = []
        ctx.category_mode = "single" if cat else "none"

    filters: dict = {}
    if cat:
        filters["category"] = cat
    if lvl:
        filters["level"] = lvl
    if price is not None:
        filters["max_price"] = price
    if techs:
        filters["tech_keywords"] = list(techs)
    if excluded_techs:
        filters["excluded_tech_keywords"] = list(excluded_techs)
    # keywords 还会承载“便宜些/太贵了/能退吗”等中文口语片段。
    # 这些词不是课程实体，喂给 keyword_narrow 会把真实候选无意义地收窄；
    # 原始槽位仍保留，只有交给检索的 search_filters 做语义清洗。
    from app.services.query_signals import filter_search_keywords
    kws = filter_search_keywords(kws)
    excluded_norm = {str(value).lower() for value in excluded_techs or []}
    kws = [word for word in kws if str(word).lower() not in excluded_norm]
    if kws:
        filters["keywords"] = kws

    # DB2 离线别名补充 → **hint 层**（来源可异于 user 明确表达，不作为硬约束）
    hint: dict = {}
    if ctx.intent_match:
        if ctx.intent_match.get("ai_field"):
            hint["ai_field"] = ctx.intent_match["ai_field"]
        if ctx.intent_match.get("ai_level"):
            hint["ai_level"] = ctx.intent_match["ai_level"]
    ctx.hint_filters = hint

    ctx.search_filters = filters
