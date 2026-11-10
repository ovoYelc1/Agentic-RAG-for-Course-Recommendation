# 对话槽位服务 - 多轮推荐约束的结构化状态
#
# 设计动机：热路径的意图缓存是"用户原话精确匹配"（消费者写入的课程同义词），
# 多轮精化（"便宜点""换个进阶的"）永远 miss，只能走完整 LLM 工具循环。
# 槽位把 category/level/max_price 从对话中结构化沉淀下来：
#   - 本轮提及则覆盖、未提及则保留 —— 第二轮起也能命中结构化快查（<100ms）；
#   - 话题转移（明确问 FAQ/订单/账号且本轮无任何推荐信号）时清空重建，
#     防止"聊完退款回来问课"被旧槽位带偏。
#
# 同时承载追问预算（clarify_used）：阻断式澄清追问每会话最多 MAX_CLARIFY 次，
# 耗尽后 agent 层注入"禁止反问"强指令（机制兜底而非 prompt 祈祷）；
# 邀请式补充（答完附一句"告诉我预算可以更精准"）不受预算限制。
#
# 存储：Redis DB1（与对话记忆同库），key = aichat:slots:{conversation_id}。
# 槽位值带来源 / 有效期：
#   {field: {"value": v, "source_message_id": ..., "updated_at": ..., "expires_at": ...}}
# 过期槽位不参与检索；整体 key 保留一个 TTL 兜底。
#
# 说明：槽位的『合并语义』（提及即覆盖 / 未提及继承 / 迁移清理）已上移到
# app/services/context_manager.SlotState。本模块继续负责 Redis 读写、旧格式兼容、
# 提问预算计数，并提供 `merge_slots`（内部转调 SlotState 后持久化），保持与
# 旧调用点签名兼容。
import json
import logging
from typing import Dict, Optional


from app.services import context_manager
from app.services.cache import get_async_redis


logger = logging.getLogger("slots")

# 槽位整体兜底 TTL：逐槽位 expires_at 主导（24h），这里给整 key 一个宽松兜底，
# 防止进程长期不续期导致 key 永久滞留 Redis。
SLOT_KEY_TTL = 60 * 60 * 24 * 30   # 30 天兜底
SLOT_FIELD_TTL = 60 * 60 * 24      # 单槽位 24h（跟随来源消息有效期）

# 兼容旧调用：删掉旧的整 key 30 分钟 TTL 常量，避免日志误导
MAX_CLARIFY = 1      # 阻断式追问预算（邀请式补充不计数）


_EMPTY: Dict = {
    "category": None,
    "categories": [],
    "level": None,
    "max_price": None,
    "keywords": [],
    "tech_keywords": [],
    "excluded_tech_keywords": [],
    "clarify_used": 0,
}

# 话题转移判定用：意图明确属于这三类时，视为用户已离开课程推荐话题
_TOPIC_SWITCH_INTENTS = {"faq", "order_query", "account_help"}


def _key(conversation_id: str) -> str:
    return f"aichat:slots:{conversation_id}"


def _normalize(raw) -> Dict:
    """把 Redis 里的 JSON 规整成完整槽位（兼容旧裸值 + 新带来源结构）。"""
    slots = dict(_EMPTY)
    if isinstance(raw, dict):
        for k in _EMPTY:
            if raw.get(k) is not None:
                slots[k] = raw[k]
        slots["clarify_used"] = int(raw.get("clarify_used", 0) or 0)
    return slots


async def get_slots(conversation_id: Optional[str]) -> Dict:
    """读取槽位；返回『已去过期』的纯值快照 + 保留 clarify_used。

    兼容旧调用点（agent/routes 读 slots.get('category') 等）：返回纯值字典，
    clarify_used 一并带上（Agent 控制状态）。过期槽位不会出现在返回值里。
    """
    if not conversation_id:
        return dict(_EMPTY)
    try:
        r = await get_async_redis()
        raw = await r.get(_key(conversation_id))
        if raw:
            parsed = _normalize(json.loads(raw))
            pure = context_manager.SlotState(parsed).effective_values()
            pure["clarify_used"] = int(parsed.get("clarify_used", 0) or 0)
            return pure
    except Exception as e:
        logger.warning("[Slots] 读取失败（按空槽位处理）: %s", e)
    return dict(_EMPTY)


async def save_slots(conversation_id: Optional[str], slots: Dict, *, with_meta:
                     Optional[Dict] = None):
    """持久化槽位。

    slots: 纯值（旧调用形态），或已带来源/有效期的完整结构（with_meta 时）。
    兼容：既有调用点传纯值 slots，这里自动补齐来源/有效期元数据。

    动态 TTL（P1）：Key TTL = 最长有效字段的剩余有效期（clamp 到 [SLOT_FIELD_TTL,
    SLOT_KEY_TTL]），避免整 key 30 天滞留而内部字段早已过期。全部字段已过期时
    删除整 key（不写入无意义残骸）。
    """
    if not conversation_id:
        return
    try:
        r = await get_async_redis()
        import time as _t
        value = _ensure_meta(slots, with_meta)
        ttl = _key_ttl_from_slots(value, now=int(_t.time()))
        key = _key(conversation_id)
        if ttl <= 0:
            await r.delete(key)  # 全字段已过期 → 清整个 key
            return
        await r.set(
            key,
            json.dumps(value, ensure_ascii=False, default=str),
            ex=ttl,
        )
    except Exception as e:
        logger.warning("[Slots] 写入失败（忽略）: %s", e)


def _key_ttl_from_slots(value: Dict, now: int, *, allow_empty_clear: bool = True) -> int:
    """计算槽位 Key 应设的 TTL（秒）。

    取所有字段 expires_at 的最大剩余值，clamp 到 [0, SLOT_KEY_TTL]。
    - 所有字段都已过期 → 返回 ≤0（调用方据此删除整 key）；
    - 明确更新会刷新某字段 expires_at → 反映到剩余 TTL（继承旧值不延长）。
    """
    import time as _t
    now = int(now if now is not None else _t.time())
    expires = [int(e.get("expires_at")) for e in value.values()
               if isinstance(e, dict) and e.get("expires_at")]
    if not expires:
        # 无任何来源字段 → 用默认 24h（防 key 永久滞留）
        return SLOT_FIELD_TTL
    remaining = max(expires) - now
    if remaining <= 0:
        return 0
    # 上限 30 天兜底；下限给 60s 防止极小剩余导致每次写入都竞争删除
    return min(max(remaining, 60), SLOT_KEY_TTL)


def _ensure_meta(slots: Dict, with_meta: Optional[Dict] = None) -> Dict:
    """把纯值或已带来源的结构统一成『带来源/有效期』的形式。

    优先保留已存在的来源/有效期（with_meta / 本身已带），否则为纯值当场补齐
    有效期（当前时间 +24h），以保证新格式下过期判断有据可依。
    """
    import time
    now = int(time.time())
    out = {}
    meta = with_meta or {}
    for k, v in (slots or {}).items():
        if k == "clarify_used":
            out[k] = v
            continue
        if isinstance(v, dict) and "value" in v:
            entry = dict(v)
            if "expires_at" not in entry:
                entry["expires_at"] = now + SLOT_FIELD_TTL
            out[k] = entry
            continue
        src = (meta.get(k) or {}).get("source_message_id") if isinstance(meta.get(k), dict) else None
        out[k] = {
            "value": v,
            "source_message_id": src or "",
            "updated_at": now,
            "expires_at": now + SLOT_FIELD_TTL,
        }
    return out


async def merge_slots(conversation_id: Optional[str], extracted: Dict, intent) -> Dict:
    """把本轮抽取的推荐参数合并进既有槽位，返回合并后的纯值快照。

    规则（语义见 context_manager.SlotState）：
    - category / level / max_price / tech_keywords：本轮提及（非 None）才覆盖，
      未提及保留旧值；
    - keywords：本轮有非空提取则整体覆盖（防多轮累积污染）；
    - 话题转移（意图明确是 FAQ/订单/账号，且本轮抽不到任何推荐信号）→ 清空重建；
    - 领域切换：新 category 覆盖旧 category，并清理旧领域技术词残留，
      **不主动清空**仍有效的 level / max_price；
    - clarify_used 追问预算计数跨轮保留。
    """
    if not conversation_id:
        return dict(_EMPTY)

    raw = {}
    try:
        r = await get_async_redis()
        cached = await r.get(_key(conversation_id))
        if cached:
            raw = _normalize(json.loads(cached))
    except Exception as e:
        logger.warning("[Slots] merge_slots 读取失败（按空）: %s", e)

    state = context_manager.SlotState(raw)
    intent_val = getattr(intent, "value", str(intent)) if intent is not None else ""

    # 话题切换：清空课程业务槽位（保留 clarify_used）
    has_signal = bool(
        extracted.get("category") or extracted.get("tech_keywords")
        or extracted.get("excluded") or extracted.get("excluded_tech_keywords")
        or extracted.get("categories")
        or extracted.get("level") or extracted.get("max_price") is not None
    )
    faq_course_refinement = (
        intent_val == "faq"
        and context_manager.has_course_refinement_signal(
            str(extracted.get("_raw_query") or "")
        )
    )
    if intent_val in _TOPIC_SWITCH_INTENTS and not has_signal and not faq_course_refinement:
        raw = {"clarify_used": raw.get("clarify_used", 0)}
        state = context_manager.SlotState(raw)

    # 明确取消：'不限预算' 等口语清除 max_price
    msg = str(extracted.get("_raw_query") or (extracted.get("_message_id") or "")).lower()
    if "不限" in msg and ("预算" in msg or "价格" in msg):
        state.clear_fields(["max_price"])

    previous = state.effective_values()
    relative_level = context_manager._relative_level(
        msg, previous.get("level"))
    if relative_level:
        extracted = dict(extracted)
        extracted["level"] = relative_level
    if context_manager._preserve_ai_followup(
            msg, previous.get("category"), extracted.get("category")):
        extracted = dict(extracted)
        extracted["category"] = previous.get("category")
        extracted["categories"] = []
    current_categories = list(extracted.get("categories") or [])
    if extracted.get("category") and len(current_categories) < 2:
        state.clear_fields(["categories"])
    new_category = extracted.get("category")
    if new_category and previous.get("category") and new_category != previous["category"]:
        state.clear_fields(["tech_keywords", "excluded_tech_keywords"])

    # 合并本轮字段
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
        source_message_id=extracted.get("_message_id"),
    )

    # 领域切换清理旧技术词（由 set_fields 已覆盖 tech_keywords，这里无需额外）
    effective = state.effective_values()

    # 持久化 with 元数据（state.raw 已含来源/有效期 + clarify_used），返回纯值快照
    persist = dict(state.raw)
    persist["clarify_used"] = int(raw.get("clarify_used", 0) or 0)
    await save_slots(conversation_id, persist)

    result = {**dict(_EMPTY), **effective}
    result["clarify_used"] = int(persist.get("clarify_used", 0) or 0)
    return result


async def merge_tool_args(conversation_id: Optional[str], args: Dict):
    """工具循环里 LLM 调 search_courses 时的参数同样是高质量槽位信号，顺手回填。"""
    if not conversation_id or not args:
        return
    picked = {k: args.get(k) for k in ("category", "level", "max_price")}
    if not any(v is not None for v in picked.values()):
        return
    slots = await get_slots_raw(conversation_id)
    state = context_manager.SlotState(slots)
    state.set_fields(picked)
    state.raw["clarify_used"] = int(slots.get("clarify_used", 0) or 0)
    await save_slots(conversation_id, state.raw)


async def get_slots_raw(conversation_id: Optional[str]) -> Dict:
    """读取带来源/有效期的完整槽位（内部用；语义保留 clarify_used）。"""
    if not conversation_id:
        return {"clarify_used": 0}
    try:
        r = await get_async_redis()
        raw = await r.get(_key(conversation_id))
        if raw:
            return _normalize(json.loads(raw))
    except Exception as e:
        logger.warning("[Slots] get_slots_raw 失败: %s", e)
    return {"clarify_used": 0}


async def bump_clarify(conversation_id: Optional[str], slots: Dict) -> int:
    """阻断式追问计数 +1（agent 检测到追问形态的回答时调用）。返回最新计数。"""
    slots["clarify_used"] = int(slots.get("clarify_used", 0) or 0) + 1
    await save_slots(conversation_id, slots)
    try:
        from app.services.metrics import incr

        await incr("clarify_used")
    except Exception:
        pass
    logger.info(
        "[clarify] 阻断式追问已计数: %d/%d（预算耗尽后将强制直答）",
        slots["clarify_used"], MAX_CLARIFY,
    )
    return slots["clarify_used"]


def budget_exhausted(slots: Dict) -> bool:
    """追问预算是否已耗尽（agent 据此注入'禁止反问'强指令）。"""
    return int(slots.get("clarify_used", 0) or 0) >= MAX_CLARIFY


__all__ = [
    "SLOT_FIELD_TTL",
    "MAX_CLARIFY",
    "get_slots", "save_slots", "merge_slots", "merge_tool_args",
    "get_slots_raw", "bump_clarify", "budget_exhausted",
]
