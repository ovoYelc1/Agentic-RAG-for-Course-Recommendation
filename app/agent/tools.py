# 工具定义与执行器
#
# 这里定义 AI Agent 可调用的两个工具，并负责实际执行：
#   1. search_courses - 从课程数据库检索真实课程
#   2. web_search     - 联网搜索实时/外部信息
#
# 平台政策类问题（原 lookup_faq 工具）已改为 SYSTEM_PROMPT 内联事实：
# 7 条政策约几百 token，模型每轮直接看到，语义理解覆盖各种问法，
# 省一次工具往返；字面匹配版 lookup_faq 仅保留在 _rescue_without_llm 作无 LLM 兜底。
#
# 所有工具都做了异常保护：执行出错时返回错误信息文本，而不是抛异常让上层崩，
# 避免触发"抱歉我不确定"这类反射式推脱。
import logging
from typing import Dict

logger = logging.getLogger("agent_tools")

# 步3：课程检索改走 Chroma 向量库（语义检索），不再直连 MySQL。
from app.rag.course_search import search_courses as search_courses_rag, keyword_narrow
from app.services.web_search import web_search
from app.agent.router import extract_tech_keywords
from app.agent.course_recommender import _generate_match_reason


# ====================== 工具 Schema（OpenAI 工具调用格式） ======================

SEARCH_COURSES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_courses",
        "description": (
            "根据关键词、分类、难度、价格等条件，从课程数据库中检索已发布的课程。"
            "当用户询问课程、想学某项技术、寻找推荐课程、找低价/免费课程时使用。务必优先调用以获取真实课程数据。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "搜索关键词，如 'Java'、'Python 数据分析'、'前端'、'Docker'",
                },
                "category": {
                    "type": "string",
                    "enum": ["backend", "frontend", "data", "devops"],
                    "description": "课程分类，无法确定可留空",
                },
                "level": {
                    "type": "string",
                    "enum": ["beginner", "intermediate", "advanced"],
                    "description": "难度等级（入门/进阶/高级），无法确定可留空",
                },
                "max_price": {
                    "type": "number",
                    "description": "价格上限（元），不限制可留空",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回课程数量，默认 5",
                },
            },
            "required": [],
        },
    },
}

WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "联网搜索实时信息、外部知识、技术文档、行业新闻等。"
            "当问题需要最新信息、课程数据库和知识库都没有答案、或涉及平台以外的内容时使用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词或问题，例如 '2025 年最流行的后端框架'",
                },
                "num_results": {
                    "type": "integer",
                    "description": "返回结果数量，默认 5",
                },
            },
            "required": ["query"],
        },
    },
}

TOOL_SCHEMAS = [SEARCH_COURSES_SCHEMA, WEB_SEARCH_SCHEMA]


# ====================== 工具执行 ======================

def _course_to_text(c: Dict) -> str:
    # avg_rating 在 metadata 中存为数值（见 consumer._upsert）。
    # 0 / None / "" 均表示「暂无评分」，避免向用户展示「评分:0」或「评分:None」。
    rating = c.get("avg_rating")
    if rating in (None, "", 0, 0.0):
        rating_text = "暂无评分"
    else:
        # 去掉多余的 .0（4.0 -> 4.0 也保留一位即可，4.7 原样）
        rating_text = f"{rating}分"
    code = c.get("course_code") or ""
    code_text = f" 编号:{code}" if code else ""
    return (
        f"- 《{c.get('course_name')}》 分类:{c.get('category')} 难度:{c.get('level')} "
        f"价格:¥{c.get('price')} 评分:{rating_text} 讲师:{c.get('teacher_name')} "
        f"ID:{c.get('id')}{code_text}\n  简介:{c.get('description')}"
    )


async def _search_multi_categories(keyword: str, categories: list, level, max_price,
                                   limit: int) -> list:
    """多领域合并检索（P0#1）：每领域各自结构化过滤，合并后按 course_id 去重。

    不放宽 level/max_price 等其它硬约束；每领域分摊 limit，合并后取前 limit，
    避免任一领域独占结果。
    """
    per = max(2, limit // max(len(categories), 1)) if len(categories) > 1 else limit
    seen, merged = set(), []
    try:
        for cat in categories:
            batch = await search_courses_rag(keyword, cat, level, max_price, per)
            for c in batch or []:
                cid = str(c.get("course_id") or c.get("id") or c.get("course_name") or "")
                if cid and cid not in seen:
                    seen.add(cid)
                    merged.append(c)
    except Exception as e:
        logger.error(
            "[Tools] 工具执行失败 tool=search_courses operation=multi_category_search error_type=%s",
            type(e).__name__,
        )
    return merged[:limit]


async def execute_tool(name: str, arguments: dict,
                       query_context=None) -> dict:
    """
    执行指定工具，返回统一结构：
        {"content": str, "courses": list|None, "intent_tag": str}

    第二个可选参数 query_context 是『后端内部参数』，不暴露在工具 Schema。
    对 search_courses：
      - LLM 可提供进一步收窄的关键词 / limit；
      - 但 QueryContext.search_filters 里的『硬约束』（category/level/max_price/
        tech_keywords/ai_field/ai_level）必须叠加保留——LLM 不得通过工具参数
        删除或放宽它们。多轮工具调用共用同一份 QueryContext。
    """
    arguments = arguments or {}
    try:
        if name == "search_courses":
            keyword = arguments.get("keyword") or ""
            # 后端已确认的硬约束（来自 QueryContext.search_filters），比 LLM 参数优先
            sf = (query_context.search_filters or {}) if query_context else {}
            category = sf.get("category")
            level = sf.get("level")
            max_price = sf.get("max_price")
            # 多领域（P0#1）：ctx.categories 合并检索每领域，再合并且按 tech 收窄
            multi_cats = []
            if query_context is not None and getattr(query_context, "categories", None) \
                    and getattr(query_context, "category_mode", "") == "multi":
                multi_cats = [c for c in query_context.categories if c]
            if category:
                pass  # 单领域
            elif multi_cats:
                category = None  # 交由下方多领域合并，不走单 category
            else:
                category = arguments.get("category")

            if max_price is None:
                max_price = arguments.get("max_price")
            else:
                # 硬约束存在时，LLM 给的更宽价格不能放宽
                llm_price = arguments.get("max_price")
                if llm_price is not None and float(llm_price) < float(max_price):
                    max_price = float(llm_price)  # 更严的也接受（只收不放）
            # LLM 提供的 level 只作为单领域下的补充；多领域时 level 硬约束优先
            if level is None and not multi_cats:
                level = arguments.get("level")
            limit = int(arguments.get("limit", 5))

            if multi_cats:
                # 多领域合并检索：每领域各自结构化过滤 + 去重合并
                courses = await _search_multi_categories(
                    keyword, multi_cats, level, max_price, limit)
            else:
                # 语义检索 + 结构化过滤
                courses = await search_courses_rag(
                    keyword, category, level, max_price, limit)
            # 向量噪音收窄：优先 QueryContext 技术关键词，其次工具 keyword
            excluded = [
                str(value).lower() for value in
                (sf.get("excluded_tech_keywords") or sf.get("excluded") or [])
                if str(value).strip()
            ]
            excluded_set = set(excluded)
            techs = [
                value for value in (
                    list(sf.get("tech_keywords") or [])
                    or extract_tech_keywords(keyword)
                )
                if str(value).lower() not in excluded_set
            ]
            courses = keyword_narrow(courses, techs)
            if excluded:
                courses = [
                    course for course in courses
                    if not any(
                        token in str(course.get("course_name") or "").lower()
                        for token in excluded
                    )
                ]
            if not courses:
                return {
                    "content": "（课程库中暂无匹配的课程，可尝试更换关键词或放宽条件）",
                    "courses": [],
                    "intent_tag": "course_recommend",
                }
            # 对齐前端卡片字段（metadata 用 course_id，这里补 id）；
            # keyword_narrow 已写"匹配关键词：X"的卡片不覆盖，其余才用通用理由
            for c in courses:
                c["id"] = c.get("course_id")
                if not c.get("match_reason"):
                    c["match_reason"] = _generate_match_reason(c, {})
            text = "\n".join(_course_to_text(c) for c in courses)
            return {
                "content": f"检索到 {len(courses)} 门课程：\n{text}",
                "courses": courses,
                "intent_tag": "course_recommend",
            }

        elif name == "web_search":
            query = arguments.get("query", "")
            num = int(arguments.get("num_results", 5))
            results = await web_search(query, num)
            if not results:
                return {
                    "content": "（联网搜索未返回有效结果，可能是当前网络环境限制或未配置搜索 API Key）",
                    "courses": None,
                    "intent_tag": "web_search",
                }
            text = "\n".join(
                f"- {r['title']} | {r['url']} | {r['content']}" for r in results
            )
            return {
                "content": f"联网搜索结果：\n{text}",
                "courses": None,
                "intent_tag": "web_search",
            }

        else:
            return {
                "content": f"未知工具: {name}",
                "courses": None,
                "intent_tag": "generic",
            }
    except Exception as e:
        logger.error(
            "[Tools] 工具执行失败 tool=%s error_type=%s",
            name,
            type(e).__name__,
        )
        return {
            "content": "工具执行出错: 工具执行失败，当前无法完成该操作，请尝试其他方式或稍后重试。",
            "courses": None,
            "intent_tag": "generic",
        }
