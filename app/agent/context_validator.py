# Context Validator —— 确定性 Python 规则校验检索结果 + 一次定向修正
#
# 不做 Reflection Agent，不调用 LLM 批评自己的答案。以课程库真实 metadata 为准：
#   course_name / category / level / price(str) / teacher_name / description /
#   avg_rating / ai_field / ai_level ...（见 app/mq/consumer._upsert）。

from typing import Dict, List


class ValidationResult:
    __slots__ = ("ok", "violations", "courses")

    def __init__(self, ok: bool, violations: List[str], courses: List[Dict]):
        self.ok = ok
        self.violations = violations
        self.courses = courses


def _price_of(c: dict):
    try:
        return float(c.get("price") or 0)
    except (TypeError, ValueError):
        return None


def _techs(ctx) -> List[str]:
    """本轮 + 槽位里的技术关键词（校验 tech 是否命中课程名）。"""
    if ctx is None:
        return []
    sf = ctx.search_filters or {}
    return list(sf.get("tech_keywords") or [])


def validate_course_result(courses: List[Dict], ctx) -> ValidationResult:
    """校验课程卡片是否符合 QueryContext 已确认的硬约束。

    仅核对课程库真实支持的字段；不存在字段（如 learning_style）不校验。
    返回 both ok 判定 + 违规列表（便于定向修正时知道『修什么』）。
    """
    violations: List[str] = []
    if ctx is None:
        return ValidationResult(True, violations, courses or [])

    sf = ctx.search_filters or {}

    # 1) 课程问题（route 课程相关）必须有真实卡片输出
    route = getattr(ctx, "route", "")
    is_course_route = route in ("STRUCTURED_COURSE_SEARCH", "AGENT_COURSE")
    input_courses = courses or []
    if is_course_route and not input_courses:
        violations.append("course_result_empty")

    cat = sf.get("category")
    lvl = sf.get("level")
    ai_field = sf.get("ai_field")
    ai_level = sf.get("ai_level")
    max_price = sf.get("max_price")
    techs = _techs(ctx)
    excluded = [
        str(value).lower() for value in
        (sf.get("excluded_tech_keywords") or sf.get("excluded") or [])
        if str(value).strip()
    ]
    # 多领域（P0#1）：任一领域命中即合法，不做单值 category 独判
    ctx_multi = (getattr(ctx, "categories", None) or []).copy() if ctx else []
    ctx_cat_mode = getattr(ctx, "category_mode", "single") if ctx else "single"

    # 2) category / ai_field 一致性
    if ctx_cat_mode == "multi" and len(ctx_multi) >= 2:
        # 多领域：每门课的 category 必须 ∈ 本轮涉及领域（避免混入库中其它无关领域）
        allowed = set(ctx_multi)
        wrong = [c for c in input_courses
                 if str(c.get("category") or "") and str(c.get("category")) not in allowed]
        for c in wrong:
            violations.append(
                f"category_mismatch:course={c.get('course_id')}"
                f"(got {c.get('category')}, want any of {sorted(allowed)})")
    elif cat:
        wrong = [c for c in input_courses
                 if str(c.get("category") or "") not in ("", str(cat))]
        for c in wrong:
            violations.append(
                f"category_mismatch:course={c.get('course_id')}"
                f"(got {c.get('category')}, want {cat})")
    elif ai_field:
        # ai_field 是中文标签（如"后端"），核对 course_name 方向或 ai_field 本身
        wrong = [c for c in input_courses
                 if c.get("ai_field") and c.get("ai_field") != ai_field
                 and not _name_has(c, ai_field)]
        for c in wrong:
            violations.append(
                f"ai_field_mismatch:course={c.get('course_id')}")

    # 3) level / ai_level 一致性
    if lvl:
        wrong = [c for c in input_courses
                 if str(c.get("level") or "") not in ("", str(lvl))]
        for c in wrong:
            violations.append(
                f"level_mismatch:course={c.get('course_id')}(got {c.get('level')}, want {lvl})")
    elif ai_level:
        wrong = [c for c in input_courses
                 if c.get("ai_level") and c.get("ai_level") != ai_level]
        for c in wrong:
            violations.append(f"ai_level_mismatch:course={c.get('course_id')}")

    # 4) 价格不超过 max_price
    if max_price is not None:
        for c in input_courses:
            p = _price_of(c)
            if p is not None and p > float(max_price):
                violations.append(
                    f"price_over_budget:course={c.get('course_id')}({p} > {max_price})")

    # 5) 技术关键词：存在 tech_keywords 时，至少一门课命中其一（用于收窄校验）。
    #    匹配范围（P1）：course_name 强 / ai_tech 强 / tags 可靠 / course_code 辅助 /
    #    description 仅辅助证据（不单独判精确命中）。
    if techs:
        if not any(_course_hits_tech(c, techs) for c in input_courses):
            violations.append("tech_keyword_no_name_match")

    if excluded:
        for course in input_courses:
            name = str(course.get("course_name") or "").lower()
            hit = [token for token in excluded if token in name]
            if hit:
                violations.append(
                    f"excluded_tech_keyword:course={course.get('course_id')}({hit})"
                )

    # 7) FAQ 混淆课程卡片
    if route == "FAQ" and input_courses:
        violations.append("faq_should_not_have_course_cards")

    return ValidationResult(not violations, violations, input_courses)


def _name_has(c: dict, token: str) -> bool:
    token = (token or "").strip()
    if not token:
        return False
    return any(tok in (c.get("course_name") or "") for tok in _split(token))


def _course_hits_tech(c: dict, techs: List[str]) -> bool:
    """技术词是否命中课程 metadata（P1）：课名/ai_tech/tags 强；code/描述辅助。

    只要任一技术词命中任一『强匹配』字段即视为命中（防误删正确课程）。
    description 仅作辅助证据——普通正文提及不单独判精确命中（避免宽泛误放行）。
    """
    kws = [str(k).lower() for k in (techs or []) if str(k).strip()] or []
    if not kws:
        return False
    name = str(c.get("course_name") or "").lower()
    ai_tech = str(c.get("ai_tech") or "").lower()
    tags = str(c.get("tags") or "").lower()
    code = str(c.get("course_code") or "").lower()
    # 强匹配字段
    strong = f"{name}\n{ai_tech}\n{tags}\n{code}"
    for k in kws:
        if k and k in strong:
            return True
    return False


def _split(token: str):
    # ai_field 可能是"后端开发"，按字面在课名里找；拆掉常见后缀放宽匹配
    for suf in ("开发", "方向", "领域"):
        if token.endswith(suf):
            return [token, token[: -len(suf)]]
    return [token]


async def correct_and_validate(courses: List[Dict], ctx, limit: int = 5,
                               *, search_fn=None) -> ValidationResult:
    """最多一次定向修正：用 QueryContext.search_filters 重新检索并再次校验。

    - 通过 → 直接返回；
    - 不通过 → 用同一个统一结构化检索入口（search_fn）重新检索（一次），
      再校验；仍失败 → 返回【仍失败的校验结果】（调用方据 violations 说明
      '未找到满足条件的结果'）。
    - 硬约束不放宽（不为了凑数量删约束）。
    """
    first = validate_course_result(courses, ctx)
    if first.ok:
        return first

    # 已无课程，无需修正
    if not courses and not (ctx.search_filters or {}):
        return first

    if search_fn is None or ctx is None:
        return first

    # 定向修正：以 ctx.search_filters 为唯一权威重新结构化检索
    try:
        corrected = await search_fn(ctx, limit)
    except Exception:
        corrected = []
    second = validate_course_result(corrected, ctx)
    return second


def explain_violations(violations: List[str]) -> str:
    """把违规项转成对用户的中文说明（'未找到满足条件的结果' 的依据）。"""
    if not violations:
        return ""
    return "未找到同时满足您全部条件的结果（" + "，".join(violations) + "）。"
