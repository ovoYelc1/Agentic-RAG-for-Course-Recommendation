# Agent 编排器 - 工具调用循环 + 兜底检索 + 会话记忆
#
# 这是 AI 客服的"大脑"。相比旧版的"关键词路由 -> 命中不了就甩锅"，
# 这里改为让 LLM 自主调用工具（查课程库 / 联网搜索 / 查知识库），
# 并且：
#   - 禁止 AI 条件反射式地说"抱歉我不太确定"
#   - 当 LLM 没有调用工具、或回答里出现"无法回答/不太确定"时，
#     按意图兜底直接检索数据库/知识库，确保"能查到的就一定回答"
#   - 通过 session_id 维护多轮记忆
import asyncio
import json
import os
import re
import time
from typing import Dict, List, Optional

from openai import AsyncOpenAI

from app.models.chat import Intent
from app.agent.router import hint_intent, extract_recommend_params, extract_tech_keywords
from app.agent.tools import TOOL_SCHEMAS, execute_tool
from app.agent.context_validator import correct_and_validate
from app.services.memory import get_history, append_history
from app.services.slots import (
    merge_slots, merge_tool_args, bump_clarify, budget_exhausted,
)
from app.services.faq_service import faq_facts_text
from app.services.metrics import incr as metrics_incr
# 阶段2/3：混合检索热路径（意图缓存命中 → 结构化快查 + 单次润色，跳过工具循环）
from app.rag.course_search import (
    structured_search, structured_search_with_context,
    slots_where, keyword_narrow,
)

# 后台润色任务需要强引用，避免请求返回后任务被垃圾回收。
_BG_TASKS = set()


def _spawn_background(coro):
    task = asyncio.create_task(coro)
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)
    return task


SYSTEM_PROMPT = """你是 ITCloud 云课堂的智能 AI 客服助手，专业、友好、乐于助人。


你可以使用以下工具来回答用户问题：
1. search_courses —— 从课程数据库检索真实课程（推荐课程、查某技术课、找低价/免费课等）
2. web_search —— 联网搜索实时/外部信息（最新技术、文档、平台外内容、行业资讯）


工作原则（非常重要）：
- **工具并行**：如需调用多个工具，必须在同一轮一次性并行声明所有工具调用（如同时查 search_courses 和 web_search），严禁逐轮串行调用——每多一轮决策往返用户就多等数秒。
- 课程相关问题（推荐课程、查价格、找课），务必先调用 search_courses 从数据库找真实课程再回答，不要凭空编造课程名或价格。
- 平台政策类问题（退款、发票、证书、支付、试看、学习周期等），依据下方"平台政策"直接回答，不需要调用任何工具。
- 需要最新外部信息时，调用 web_search。
- **通用知识问题**（学习路线、技术解释、编程入门指导、职业建议等），如果工具无法提供有效信息，应直接用你自身的知识给出详细、有条理的回答，而不是反问用户或说"无法回答"。
- 回答要结构化、使用中文。推荐课程时用列表呈现：课程名、价格、适合人群、讲师。
- 用户问宽泛问题（如"初学者学什么"）且检索结果跨多个方向时，按方向分组推荐并给一句选择建议，结尾自然附一句"告诉我方向或预算可以推荐得更精准"之类的邀请。
- 篇幅按场景控制：课程推荐类回答保持简洁（200 字内，信息密度优先）；**仅当用户问复杂知识问题**（学习路线、技术原理、职业规划）时才展开详尽回答。
- 严禁主动说"抱歉，这个问题我不太确定"这类推脱话术。


输出格式要求（聊天界面渲染，务必严格遵守）：
- 回答的第一个字符必须是正文，严禁以空行、空格或"\n\n"开头。
- 严禁使用 Markdown 表格（竖线 |）和"---"分隔线；多个信息用"- "列表逐行呈现。
- 严禁使用 # 标题；需要突出关键词时用 **加粗** 即可。
- 推荐学习路线时按由浅入深排序（先基础入门课，后框架进阶课，最后高级专项课）。


澄清追问规则：
- 只有在完全没有任何可依据的信号（无分类/难度/价格/关键词线索）时，才可以向用户提出一个澄清问题并停下等待回复。
- 其余情况一律先给出最佳推荐/答案；可以在回答末尾自然附一句"如果告诉我预算或学习目标，可以推荐得更精准"之类的邀请（这不算追问，随时都可以）。
- 严禁连续反问，严禁用追问代替回答。


平台政策（回答政策类问题时只能依据以下事实，禁止编造或套用其他平台的规则）：
""" + faq_facts_text()

# 追问预算耗尽后追加的强指令：机制兜底而非 prompt 祈祷
_BUDGET_EXHAUSTED_SUFFIX = (
    "\n\n注意：本会话的澄清追问次数已用完。禁止再向用户反问任何问题，"
    "必须基于现有信息直接给出最佳答案与推荐。"
)


def _system_prompt(slots: Dict, query_context=None) -> str:
    """system prompt + 追问预算状态（耗尽时注入强指令，由机制保证而非模型自觉）。"""
    base = SYSTEM_PROMPT
    if query_context is not None:
        base = SYSTEM_PROMPT + "\n\n" + build_context_prompt(query_context)
    if budget_exhausted(slots):
        return base + _BUDGET_EXHAUSTED_SUFFIX
    return base


def build_context_prompt(query_context) -> str:
    """把 QueryContext 的必要信息转成简洁文本注入 system prompt。

    绝不把整个 Python 对象直接塞入 prompt；只取对检索/回答真正必要的
    用户当前请求、当前有效硬约束与回答要求。
    """
    if query_context is None:
        return ""
    q = query_context.query or ""
    sf = query_context.search_filters or {}
    lines = ["以下是根据当前对话为你整理的用户当前请求与有效约束："]

    lines.append(f"用户当前请求：{q}")

    if sf:
        lines.append("当前有效条件：")
        if sf.get("category"):
            lines.append(f"- category：{sf['category']}")
        if sf.get("level") or sf.get("ai_level"):
            lines.append(f"- level：{sf.get('level') or sf.get('ai_level')}")
        if sf.get("max_price") is not None:
            lines.append(f"- max_price：{sf['max_price']}")
        techs = sf.get("tech_keywords")
        if techs:
            lines.append(f"- technology：{'、'.join(techs)}")
        if sf.get("ai_field"):
            lines.append(f"- 领域：{sf['ai_field']}")
    else:
        lines.append("当前有效条件：无（未确认硬性过滤条件）")

    lines.extend([
        "回答要求：",
        "1. 检索和回答都要考虑上述约束。",
        "2. 不得忽略明确写出的硬约束（category / level / max_price / technology）。",
        "3. 没有符合条件的结果时如实说明，不要为了推荐而放宽约束或编造课程。",
        "4. 不得编造课程名称、价格和链接。",
    ])
    return "\n".join(lines)


async def _ttft_guard(stream, timeout: float = 5.0):
    """流式首 token 守卫（TTFT gate）：`timeout` 秒内拿不到首个事件即抛
    asyncio.TimeoutError（由调用方统一走兜底降级）。

    场景：云端 LLM "假死"——连接建立但迟迟不吐首 token。此时用户看到的是
    空白界面而非逐字输出（SSE 下总时长不致命，**卡住不动**才致命）。
    5s ≈ 2.5× 实测 TTFT p99（1.9s），防假死且不误杀慢而正常的请求。
    """
    it = stream.__aiter__()
    try:
        yield await asyncio.wait_for(it.__anext__(), timeout)
    except StopAsyncIteration:
        return
    async for ev in it:
        yield ev


def _sanitize_answer(text: str) -> str:
    """出口统一清理：去首尾空白、压缩连续空行、markdown 标题降级为加粗。

    LLM 对"禁止 # 标题"的 prompt 约束屡教不改，聊天界面也没有标题样式，
    在出口机械降级为 **加粗**（前端 formatMessage 只认 **bold**），
    保证缓存/记忆/非流式响应里不残留 "##"。
    """
    if not text:
        return text
    t = text.strip()
    t = re.sub(r"^#{1,6}\s*(.+)$", r"**\1**", t, flags=re.M)   # md 标题 → 加粗
    t = re.sub(r"^\s*-{3,}\s*$", "", t, flags=re.M)            # --- 分隔线
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t


def _looks_like_clarify(answer: str, courses) -> bool:
    """
    启发式判定"阻断式追问"：以问句结尾、没有推荐结果、且内容很短。
    邀请式（先给答案再附一句精化邀请）因为"有内容/有课程"不会被误伤；
    误判的代价只是预算多扣或少扣一次，可接受。
    """
    if courses:
        return False
    a = (answer or "").strip()
    return 0 < len(a) < 120 and a.endswith(("？", "?"))


# 工具调用最大轮数（防止死循环）
MAX_ITER = 4

# 工具循环总预算（秒）：v2 由 100s 收紧至 60s——GLM-4.5-Air 实测冷路径 p99 21s，
# 60s 覆盖 3 倍 p99；剩余场景由兜底接管，避免长尾拖到网关 300s。
#
# 2026-09-17：换成 CPU 版 bge-reranker-v2-m3 后，单次 search_courses 需要对
# 混合召回后的最多 50 个候选做交叉编码，实测 15~26s（首次含模型加载 26s），
# 冷路径整体被拉长，
# 生产默认 90s，与 docker-compose/.env.example 保持一致，覆盖 v2-m3 CPU
# 重排器单次 search_courses 15~26s 的长尾并给后续 LLM 回答轮留出余量。
TOOL_LOOP_BUDGET = int(os.getenv("AI_TOOL_LOOP_BUDGET", "90"))

# 单个工具执行超时（秒）：生产默认 40s，覆盖本地 v2-m3 CPU 重排耗时；
# 外部搜索工具仍由自身网络超时控制。
#
# ⚠ 2026-09-17 该前提被 RERANKER_MODEL 切换打破：
#   search_courses → retrieve_hybrid(top_k=15) → RRF Top50 候选过交叉编码器，
#   CPU 上 bge-reranker-base 约 1.6s，而 bge-reranker-v2-m3 实测 **15~26s**。
#   若仍固定 5s，search_courses **每次都会超时**，Agent 静默退化为「零卡片兜底」，
#   语义检索在生产上等于关闭（而本地检索耗时又打不出任何日志，极难发现）。
#   故改为可由 AI_TOOL_TIMEOUT 覆盖，compose 置 40（< 循环预算 90s，给 LLM 续答留余量）。
#   回滚 RERANKER_MODEL 回 bge-reranker-base 时仍可显式调小该值。
TOOL_TIMEOUT = int(os.getenv("AI_TOOL_TIMEOUT", "40"))

# 流式生成总预算（秒）：自首个内容 token 起计时，超时截断流并追加提示。
# 与 TTFT 守卫互补：TTFT 保证"开始说话"（防假死），本预算保证"说完话"的体验
# 下限。8s 由实测校准：正常 300 字级答案生成 ≤8s，超时即真长尾，截断后 P99
# 自首 token 起收敛到 8s 以内。
STREAM_ANSWER_BUDGET = 8
_TRUNCATE_NOTICE = "\n\n（回答较长，已为您截断。完整信息可查看上方课程卡片，或换个更具体的问法～）"

# 兜底判定用的"无帮助"关键词（run_agent / run_agent_stream 共用）
_UNHELPFUL_KW = (
    "无法回答", "不太确定", "抱歉", "误译", "不清楚",
    "具体说明", "能否请您", "请告诉我更多", "请详细描述",
    "请重新表述", "无法理解", "没有理解",
)


def _trim_history(history: List[Dict], max_turns: int = 6) -> List[Dict]:
    """LLM 上下文瘦身：只带最近 max_turns 轮；除最近 2 条外单条截断到 120 字。

    冷路径长尾由云端 TTFT 主导（实测单样本 9.5s），prefill 越长首 token 越慢；
    历史只需保留"话题与槽位语境"，完整答案文本对决策轮没有增量价值。
    """
    msgs = history[-max_turns * 2:]
    out = []
    for i, m in enumerate(msgs):
        c = m.get("content") or ""
        if i < len(msgs) - 2 and len(c) > 120:
            c = c[:120] + "…"
        out.append({"role": m.get("role", "user"), "content": c})
    return out


# ---- 冷路径工具遵从修复（2026-08-27）----
# 实测（benchmark_full 12 题）：GLM-4.5-Air 对"想学Java从哪门课开始"类问询有
# 概率不调 search_courses 直接凭记忆作答——文字里课程全是编造，卡片为空，
# E2E 召回率被拉到 58.3%。两道防线：
#   ① prompt 层：课程类问询在 user 消息尾部注入"必须先调工具"强指令（降概率）
#   ② 机制层：工具循环结束仍零卡片 → 强制本地混合检索补卡片（兜住漏网）
_COURSE_DIRECTIVE = (
    "\n\n（系统注：本问题与课程推荐相关。回答前必须先调用 search_courses 工具"
    "检索真实课程库，基于检索结果推荐；严禁不调用工具、凭记忆编造课程名或价格。）"
)


def _course_related(extracted: Dict, intent_hint, query_context=None) -> bool:
    """课程类问询判定（工具遵从修复用）。

    只以明确课程意图、课程属性或课程精化表达触发课程指令和补卡。
    AGENT 路由也承载通用咨询，不能单凭路由标签推断用户要课程。
    """
    from app.services.query_signals import has_course_refinement_signal

    route = getattr(query_context, "route", "") if query_context is not None else ""
    if route == "STRUCTURED_COURSE_SEARCH":
        return True
    if route in ("FAQ", "GENERAL_LLM"):
        return False
    if intent_hint == Intent.COURSE_RECOMMEND:
        return True
    if intent_hint == Intent.FAQ:
        return False

    has_course_attributes = bool(
        extracted.get("category")
        or extracted.get("tech_keywords")
        or extracted.get("level")
        or extracted.get("max_price") is not None
    )
    query = getattr(query_context, "query", "") if query_context is not None else ""
    if has_course_attributes or has_course_refinement_signal(query):
        return True

    # 带有明确学习路径意图的 AGENT 请求仍属于课程场景，即使参数提取器
    # 没有抽出可直接用于检索的方向或技术词。
    return (
        route == "AGENT_COURSE"
        and getattr(query_context, "route_reason", "") == "learning_path_planning"
    )


async def _force_course_rescue(message: str, extracted: Dict, limit: int,
                               query_context=None) -> List[Dict]:
    """零卡片机制兜底：课程类问询 LLM 未调工具/检索空时，强制本地检索补卡片。

    定位是"瞬时补卡片"，必须快——不走重量级语义检索（查询嵌入冷态 2s+ bge
    精排，实测 8s 超时全空手而归），改用结构化快查（纯 metadata 过滤，<100ms）：
    category/level 槽位过滤 → 技术词收窄 → 无槽位时退到全库前 N。
    候选池取 limit×5（Chroma get 无排序，池太小会让技术词收窄漏掉目标课，
    实测 backend 池 8 张只含 1 门 Java 课）。
    """
    from app.rag.course_search import structured_search
    if query_context is not None:
        # 统一走「带上下文的统一结构化搜索入口」，与主路径/Validator 同一套逻辑
        courses = await structured_search_with_context(query_context, limit)
        return courses or []
    category = extracted.get("category")
    level = extracted.get("level")
    pool = max(limit * 5, 25)
    try:
        if category or level:
            filt = slots_where({"category": category, "level": level}) or {}
            courses = await asyncio.to_thread(structured_search, filt, pool)
        else:
            courses = await asyncio.to_thread(structured_search, {}, pool)
        courses = keyword_narrow(courses, extracted.get("tech_keywords")) or courses
        for c in courses:
            c["id"] = c.get("course_id")
        return courses[:limit]
    except Exception as e:
        print(f"[Agent] 零卡片强制检索失败 error_type={type(e).__name__} kind=course_search")
        return []


def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=__import__("os").getenv("OPENAI_API_KEY"),
        base_url=__import__("os").getenv("OPENAI_BASE_URL"),
        # 关键：默认 600s 超时太离谱，外部 LLM 卡住会把整个请求拖死、
        # 前端先超时报"网络超时"。这里限定 60s 快速失败，失败即走兜底检索。
        # （云端 API 常态 <10s；60s 是异常场景的硬上限，各调用点不再单独放宽）
        timeout=60.0,
        max_retries=1,
    )


async def _ensure_validated(courses: List[Dict], query_context, limit: int) -> List[Dict]:
    """课程结果过 Validator + 至多一次定向修正。返回最终可输出的课程列表。

    - 通过 → 原样返回；
    - 不通过 → 用 QueryContext.search_filters 走统一结构化搜索修正一次，
      再校验；仍失败 → 返回空（调用方据此说明"未找到满足条件的结果"）。
    - 硬约束不放宽。
    """
    if query_context is None or not courses:
        return courses or []
    result = await correct_and_validate(
        courses, query_context, limit,
        search_fn=structured_search_with_context,
    )
    return result.courses if result.ok else []


def _to_float(v):
    """把 Decimal / 字符串数字安全转 float；None 原样返回。

    数据库里的 price、avg_rating 是 Decimal 类型，json.dumps 不认识 Decimal，
    会在流式接口 / Redis 缓存处抛 'Object of type Decimal is not JSON serializable'
    —— 这正是前端'一输入不到五秒就报错'的根因。这里统一转成 float 根治。
    """
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _normalize_courses(courses: List[Dict]) -> List[Dict]:
    """
    将数据库课程字段转为前端期望的 camelCase 结构。
    （旧版直接返回 snake_case，导致前端课程卡片一直是空的）
    """
    out = []
    for c in courses:
        out.append({
            "id": c.get("id"),
            "courseName": c.get("course_name"),
            "courseCode": c.get("course_code"),
            "category": c.get("category"),
            "level": c.get("level"),
            "price": _to_float(c.get("price")),
            "description": c.get("description"),
            "coverImage": c.get("cover_image"),
            "teacherName": c.get("teacher_name"),
            "studentCount": c.get("student_count"),
            "avgRating": _to_float(c.get("avg_rating")),
            "matchReason": c.get("match_reason") or "综合匹配度较高",
        })
    return out


def _build_polish_prompt(message: str, courses: List[Dict]) -> str:
    """热路径润色提示词（流式/非流式共用）"""
    course_lines = "\n".join(
        f"- 《{c.get('course_name')}》 分类:{c.get('category')} 难度:{c.get('level')} "
        f"价格:¥{c.get('price')} 简介:{c.get('description')}"
        for c in courses
    )
    return (
        "你是 ITCloud 云课堂 AI 客服。用户问：" + message + "\n"
        "以下是系统已为你检索到的相关课程（来自结构化检索）：\n" + course_lines + "\n"
        "请用友好、简洁的中文回复，**全文控制在 200 字以内**，要求：\n"
        "1. 若课程集中在同一方向，直接列表推荐最相关的 3 门左右，说明适合人群与亮点；\n"
        "2. 若课程跨多个方向（如用户问'初学者学什么'这类宽泛问题），按方向分组呈现，"
        "每组 1-2 门，并给一句'怎么选'的建议（如'零基础建议从某方向入手'）；\n"
        "3. 结尾自然附一句邀请，如'告诉我你想学的方向或预算，我可以推荐得更精准'；\n"
        "4. 不要编造课程名或价格，严格基于上面的列表。"
    )


def _template_answer(courses: List[Dict]) -> str:
    """热路径直出模板：无 LLM、微秒级。结构化呈现检索结果，
    后台润色任务完成后写入答案缓存，重复问自然升级为润色版。"""
    lines = [
        f"**{i}. {c.get('course_name') or '未命名课程'}** "
        f"（{c.get('category') or '综合'}/{c.get('level') or '不限'}）"
        f" ¥{c.get('price') or 0} - {c.get('teacher_name') or '资深讲师'}"
        for i, c in enumerate(courses, 1)
    ]
    return "为您找到以下相关课程：\n" + "\n".join(lines) + "\n\n如需了解某门课的详细内容或想按预算筛选，告诉我即可。"


async def _background_polish(message: str, courses: List[Dict],
                             user_id: Optional[str], query_context=None):
    """后台润色任务：热路径直出模板后 fire-and-forget，润色完成写答案缓存。
    下次相同问题（归一化后）命中润色版——首次 <100ms，重复问也是 <100ms 且文案自然。

    v7：用 QueryContext 的 canonical cache_intent + search_filters 构建与主路径一致的缓存 key，
    保证写入的润色版能被后续同约束请求命中。
    """
    try:
        answer = await _polish_reply(message, courses)
        if not answer or len(answer) < 20:
            return
        from app.services.cache import build_answer_cache_key, set_cache
        kw = {}
        if query_context is not None:
            kw = {"intent": query_context.intent,
                  "cache_intent": getattr(query_context, "cache_intent", ""),
                  "search_filters": query_context.search_filters or {}}
        key = await build_answer_cache_key(message, **kw)
        await set_cache(key, {
            "intent": "course_recommend",
            "answer": answer,
            "courses": _normalize_courses(courses),
            "confidence": 0.95,
        }, ttl=3600)
    except Exception as e:
        print(f"[PolishBg] 后台润色失败 error_type={type(e).__name__} kind=cache_write")


async def _polish_reply(message: str, courses: List[Dict]) -> str:
    """
    热路径专用：拿到结构化检索出的课程后，用 LLM 生成一次友好推荐文案（不再调工具、不再 embedding）。
    失败则退回纯文本拼接，保证至少有结果返回。
    """
    client = _get_client()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _build_polish_prompt(message, courses)}],
            temperature=0.3,
            max_tokens=450,  # 200 字级推荐文案足够；过大是端到端延迟的主要成分
            timeout=30,  # 覆盖实测 p99（21s）即止；超时放弃润色（模板版已在缓存）
        )
        # 出口清理：去前导空白 + markdown 标题降级（前端渲染适配）
        return _sanitize_answer(resp.choices[0].message.content or "")
    except Exception as e:
        print(f"[Polish] 润色失败，退回拼接 error_type={type(e).__name__} kind=llm_polish")
        return "为您找到相关课程：\n" + "\n".join(f"- 《{c.get('course_name')}》" for c in courses)


async def run_agent(
    message: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 5,
    query_context=None,
) -> Dict:
    """
    运行 Agent，返回与 ChatResponse 兼容的字典：
        {"intent": Intent, "answer": str, "courses": list|None, "confidence": float}

    query_context：可选。传入后——
      - system prompt 注入 build_context_prompt 的约束；
      - search_courses 工具叠加 QueryContext 硬约束（execute_tool）；
      - 结构化快查统一走 structured_search_with_context（不再维护两套热路径）；
      - 课程结果过 Validator + 一次定向修正。
    """
    intent_hint = hint_intent(message)
    client = _get_client()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # -------- 槽位状态（多轮精化的前提）--------
    # 本轮抽取的推荐参数先合并进会话槽位（提及覆盖/未提及保留/话题转移清空）。
    # 无论后续走哪条路径，槽位始终反映最新约束；追问预算计数也随槽位跨轮保留。
    extracted = extract_recommend_params(message)
    slots = await merge_slots(session_id, extracted, intent_hint)

    # -------- 结构化快查（由 Request Router 决定是否走；统一入口）--------
    # v3：意图缓存热路径 + 槽位精化快查两条独立分支合并为一个统一入口
    #   structured_search_with_context。是否走快查由 Request Router 决定
    #   （query_context.route == STRUCTURED_COURSE_SEARCH），Agent 不再重复维护
    #   两套「意图缓存 / 槽位精化」的检索代码。
    if query_context is not None and getattr(query_context, "route", "") == "STRUCTURED_COURSE_SEARCH":
        hot_courses = await structured_search_with_context(query_context, limit)
        if hot_courses:
            hot_courses = await _ensure_validated(hot_courses, query_context, limit)
            if hot_courses:
                answer = _template_answer(hot_courses)
                _spawn_background(_background_polish(
                    message, hot_courses, user_id, query_context))
                await metrics_incr("intent_hot")
                # 非流式接口没有外层 SSE 收尾器，因此热路径在这里保存历史。
                await append_history(session_id, "user", message)
                await append_history(session_id, "assistant", answer)
                return {
                    "intent": Intent.COURSE_RECOMMEND,
                    "answer": answer,
                    "courses": _normalize_courses(hot_courses),
                    "confidence": 0.95,
                }
    # （未传入 query_context / 非 STRUCTURED 路由 / 快查无结果 → 落到完整工具循环，保持原行为）

    # 拼接会话历史（多轮记忆）；system prompt 带追问预算状态
    if budget_exhausted(slots):
        await metrics_incr("clarify_blocked")  # 预算拦截生效（"禁止反问"强指令注入）
    # Context Manager 已经读取过本轮历史；复用它，避免重复访问 Redis。
    history_source = (
        getattr(query_context, "recent_history", None)
        if query_context is not None else None
    )
    history = _trim_history(
        history_source if history_source is not None else await get_history(session_id)
    )

    # GENERAL_LLM 是独立的无工具路径。通用问题不应进入课程工具循环，
    # 也不应触发课程卡片 rescue。
    if query_context is not None and getattr(query_context, "route", "") == "GENERAL_LLM":
        direct_messages = [
            {"role": "system", "content": (
                "你是 ITCloud 云课堂的技术助手。直接回答用户问题，不调用工具，"
                "不反问；回答准确、清晰，必要时分点说明。"
            )},
            *history,
            {"role": "user", "content": message},
        ]
        answer = ""
        try:
            resp = await client.chat.completions.create(
                model=model, messages=direct_messages, temperature=0.5,
                max_tokens=768, timeout=30,
            )
            answer = _sanitize_answer(resp.choices[0].message.content or "")
        except Exception as e:
            print(f"[Agent] GENERAL_LLM 直答失败 error_type={type(e).__name__} kind=general_llm")
        if not answer:
            answer = "抱歉，当前服务暂时无法处理您的问题，请稍后再试。"
        await metrics_incr("general_llm")
        await append_history(session_id, "user", message)
        await append_history(session_id, "assistant", answer)
        if _looks_like_clarify(answer, None):
            await bump_clarify(session_id, slots)
        return {
            "intent": Intent.GENERIC,
            "answer": answer,
            "courses": None,
            "confidence": 0.85 if answer else 0.2,
        }

    messages = [{"role": "system", "content": _system_prompt(slots, query_context)}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})
    # 工具遵从防线①：课程类问询注入"必须先调 search_courses"强指令
    _is_course_q = _course_related(extracted, intent_hint, query_context)
    if _is_course_q:
        messages[-1]["content"] += _COURSE_DIRECTIVE

    collected_courses: List[Dict] = []
    # 跨工具调用累积的技术关键词（含用户消息里提取的），用于对最终卡片收窄
    tool_tech_kws: List[str] = list(extracted.get("tech_keywords") or [])
    final_intent = intent_hint
    answer = ""

    # -------- 工具调用循环 --------
    # 总预算门闩：LLM 若反复重试畸形参数，逐轮超时会累积到网关超时（300s）；
    # 超预算立即跳出，走下方 rescue/直答兜底，保证请求有始有终。
    _loop_t0 = time.monotonic()
    try:
        for _ in range(MAX_ITER):
            if time.monotonic() - _loop_t0 > TOOL_LOOP_BUDGET:
                print(f"[Agent] 工具循环总预算 {TOOL_LOOP_BUDGET}s 耗尽，跳出走兜底")
                break
            is_faq_route = (
                getattr(query_context, "route", "") == "FAQ"
                or (query_context is None and intent_hint == Intent.FAQ)
            )
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                # FAQ 政策类：平台事实已内联 system prompt，禁用工具（防无关课程卡片）。
                # 最终策略使用 canonical route，避免 raw hint 与 Router 分裂。
                tools=None if is_faq_route else TOOL_SCHEMAS,
                tool_choice="none" if is_faq_route else "auto",
                temperature=0.3,
                max_tokens=512,
                timeout=30,  # 覆盖实测 p99（21s）即止；超时走兜底
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                # 把 assistant 的工具调用消息原样加入上下文（供模型继续推理）
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                })
                _t_tools = time.monotonic()
                # 同轮多个工具调用并行执行（v2：原逐个 await 串行累加延迟）
                async def _run_one(tc):
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}
                    try:
                        result = await asyncio.wait_for(
                            execute_tool(tc.function.name, args, query_context),
                            timeout=TOOL_TIMEOUT,
                        )
                    except Exception as e:
                        print(
                            f"[Agent] 工具执行失败 tool={tc.function.name} "
                            f"error_type={type(e).__name__} kind=timeout_or_failure"
                        )
                        result = {
                            "content": (
                                f"工具 {tc.function.name} 执行超时或失败："
                                "当前无法完成该操作，请稍后重试或尝试其他方式。"
                            ),
                            "courses": None,
                            "intent_tag": None,
                        }
                    return tc, args, result

                for tc, args, result in await asyncio.gather(
                        *[_run_one(tc) for tc in msg.tool_calls]):
                    if result.get("courses"):
                        # 跨多次工具调用累积去重：原来"后一次覆盖前一次"，
                        # 多轮探索式搜索后早期命中的课程会从卡片上丢失
                        for c in result["courses"]:
                            if not any(
                                str(x.get("course_id")) == str(c.get("course_id"))
                                for x in collected_courses
                            ):
                                collected_courses.append(c)
                        # 累积检索关键词并对累积课程收窄：LLM 探索式搜索可能
                        # 连带查回无关课程（问 java 时顺带查了前端课），
                        # 收窄保证前端卡片与回答内容一致
                        for kw in extract_tech_keywords(str(args.get("keyword") or "")):
                            if kw not in tool_tech_kws:
                                tool_tech_kws.append(kw)
                        collected_courses = keyword_narrow(collected_courses, tool_tech_kws)
                        # LLM 调 search_courses 的参数是高质量槽位信号，顺手回填
                        # （下一轮精化即可走结构化快查）
                        await merge_tool_args(session_id, args)
                    if result.get("intent_tag"):
                        # 同步更新意图标签（如 web_search / course_recommend / faq）
                        try:
                            final_intent = Intent(result["intent_tag"])
                        except Exception:
                            pass
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result["content"],
                    })
                print(f"[Agent] 本轮 {len(msg.tool_calls)} 个工具并行执行 "
                      f"{time.monotonic() - _t_tools:.2f}s")
                continue
            else:
                answer = msg.content or ""
                break
    except Exception as e:
        # LLM 调用异常（网络/密钥/模型不可用）不再直接返回"抱歉"，
        # 而是走下方兜底检索，尽量用数据库/知识库给出答案。
        print(f"[Agent] LLM 调用异常 error_type={type(e).__name__} kind=llm_call")
        answer = ""

    # -------- 工具遵从防线②：课程类问询零卡片 → 强制本地检索补卡片 --------
    # LLM 未调工具直接文字作答（或调用失败）时，文字答案保留、卡片补真实课程；
    # 放在 rescue 之前，避免"短答案"条件又被模板覆盖。
    if _is_course_q and not collected_courses:
        forced = await _force_course_rescue(message, extracted, limit, query_context)
        if forced:
            forced = await _ensure_validated(forced, query_context, limit)
        if forced:
            await metrics_incr("forced_course_rescue")
            collected_courses = forced
            final_intent = Intent.COURSE_RECOMMEND
            print(f"[Agent] 零卡片强制检索补 {len(forced)} 张卡片")

    # -------- 兜底救援 --------
    # v2 触发条件收紧：v1 的 `len(answer)<200` 把大量**合法短答案**（带课程卡片的
    # 简洁推荐、FAQ 直答）误判为"无帮助"，再用模板答案覆盖模型的好答案
    # （实测 rescue 12.5% 大半由此而来）。现在：空/推脱词/既短又无课程卡片才兜底。
    if (
        not answer
        or any(kw in answer for kw in _UNHELPFUL_KW)
        or (len(answer) < 120 and not collected_courses)
    ):
        try:
            # 兜底定位是"瞬时返回"（本地 Chroma + FAQ 常态 <100ms），
            # 20s 只会放任联网搜索拖长尾——8s 封顶，超时放弃联网部分
            rescue = await asyncio.wait_for(
                _rescue_without_llm(message, limit, query_context), timeout=8
            )
        except Exception as e:
            rescue = None
            print(f"[Agent] 兜底检索超时/失败 error_type={type(e).__name__} kind=rescue_search")
        if rescue:
            await metrics_incr("rescue")
            answer = rescue["answer"]
            if rescue.get("courses"):
                collected_courses = rescue["courses"]
            final_intent = rescue["intent"]

    # -------- 纯 LLM 直答兜底 --------
    # 若数据库/知识库也没给出有效答案，做一次无工具的纯 LLM 调用，
    # 让模型用自身知识回答通用问题（学习路线、技术解释、编程指导等）。
    if not answer or len(answer) < 120:
        try:
            direct_messages = [
                {"role": "system", "content": "你是技术导师。直接回答用户问题，不要反问或要求澄清。回答要分点、详细，至少300字。"},
                *history,
                {"role": "user", "content": message},
            ]
            direct_resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=direct_messages,
                    temperature=0.7,
                    max_tokens=768,
                    timeout=30,
                ),
                timeout=35,
            )
            direct_answer = direct_resp.choices[0].message.content or ""
            if len(direct_answer) > len(answer):
                answer = direct_answer
                print(f"[Agent] 纯 LLM 直答成功，长度: {len(answer)}")
            else:
                print(f"[Agent] 纯 LLM 直答仍较短: {len(direct_answer)}")
        except Exception as e:
            print(f"[Agent] 纯 LLM 直答失败 error_type={type(e).__name__} kind=direct_llm")

    # -------- 记忆写入 + 追问计数 --------
    await metrics_incr("tool_loop")  # 走完整工具循环的请求（与 intent_hot/general_llm 对照看覆盖率）
    if answer:
        await append_history(session_id, "user", message)
        await append_history(session_id, "assistant", answer)
    # 检测阻断式追问（问句结尾+无课程+短答），消耗本会话的一次性预算
    if _looks_like_clarify(answer, collected_courses):
        await bump_clarify(session_id, slots)

    final_answer = (
        _sanitize_answer(answer or "")
        or "抱歉，当前服务暂时无法处理您的问题，请稍后再试或联系人工客服 support@itcloud.com。"
    )

    # 最终卡片集合统一过一遍校验（P0#4）：无论来自工具循环、hot 直出还是
    # 强制补卡，返回/缓存/推送的都是校验后的合法集合。
    if collected_courses:
        validated = await _ensure_validated(collected_courses, query_context, limit)
        if validated:
            collected_courses = validated
        else:
            collected_courses = []

    return {
        "intent": final_intent,
        "answer": final_answer,
        "courses": _normalize_courses(collected_courses) if collected_courses else None,
        "confidence": 0.9 if (collected_courses or answer) else 0.2,
    }


async def _rescue_without_llm(message: str, limit: int, query_context=None) -> Optional[Dict]:
    """
    LLM 不可用（token 失效 / 网络错误 / 模型不支持工具）时的综合兜底：
    依次尝试 课程库 -> 知识库 -> 联网搜索，尽量直接给出有用答案。
    """
    from app.rag.course_search import search_courses as search_courses_rag
    from app.agent.course_recommender import _generate_match_reason
    from app.services.faq_service import lookup_faq
    from app.services.web_search import web_search

    # 1) 课程库语义检索（即使没有 LLM 也能答课程问题，走 Chroma 不再连 MySQL）
    params = extract_recommend_params(message)
    if (
        params.get("category")
        or params.get("keywords")
        or params.get("level")
        or params.get("max_price")
    ):
        keyword = " ".join(params.get("keywords", []))
        try:
            courses = await asyncio.wait_for(
                search_courses_rag(
                    keyword,
                    params.get("category"),
                    params.get("level"),
                    params.get("max_price"),
                    limit,
                ),
                timeout=10,
            )
        except Exception as e:
            courses = None
            print(f"[Rescue] 课程检索超时/失败 error_type={type(e).__name__} kind=course_search")
        if courses:
            # 技术关键词收窄（与主路径一致）：RAG 向量噪音会把 Go 混进"java"的兜底答案
            courses = keyword_narrow(courses, params.get("tech_keywords"))
            for c in courses:
                c["id"] = c.get("course_id")
                if not c.get("match_reason"):
                    c["match_reason"] = _generate_match_reason(c, {})
            if query_context is not None:
                courses = await _ensure_validated(courses, query_context, limit)
            # Chroma 旧索引/异构 metadata 可能缺 category 等字段，用 .get 兜底避免 KeyError
            lines = [
                f"**{i}. {c.get('course_name') or '未命名课程'}** "
                f"({c.get('category') or '综合'}/{c.get('level') or '不限'}) ¥{c.get('price') or 0}"
                for i, c in enumerate(courses, 1)
            ]
            return {
                "answer": "为您找到以下课程：\n" + "\n".join(lines),
                "courses": courses,
                "intent": Intent.COURSE_RECOMMEND,
            }

    # 2) 知识库（平台政策类问题；与 SYSTEM_PROMPT 内联事实同一份 FAQ_KB，
    #    这里是 LLM 不可用时的字面匹配兜底）
    ans = lookup_faq(message)
    if ans:
        return {"answer": ans, "courses": None, "intent": Intent.FAQ}

    # 3) 联网搜索（尽力而为，不依赖 LLM）
    try:
        results = await web_search(message, 5)
        if results:
            text = "\n".join(
                f"- {r['title']} | {r['url']} | {r['content']}" for r in results
            )
            return {
                "answer": f"我暂时无法调用 AI 模型，但为您联网检索到以下信息：\n{text}",
                "courses": None,
                "intent": Intent.GENERIC,
            }
    except Exception:
        pass

    return None


async def run_agent_stream(message: str, user_id: Optional[str] = None,
                           session_id: Optional[str] = None, limit: int = 5,
                           query_context=None):
    """
    run_agent 的流式版本，供 /api/ai/chat/stream 消费。

    真 token 级流式：LLM 生成侧即用 stream=True，首字延迟 ≈ 模型首 token，
    替代旧版「先 await 全量结果再按句切分」的伪流式。

    事件协议（与旧版保持一致，前端无需改动）：
    - type: intent  （在任何 chunk 之前发出）
    - type: courses （有推荐课程时，在首个 chunk 之前发出）
    - type: chunk   （token 级文本片段）
    - type: done

    一致性约束：所有已推送给客户端的 chunk 拼接后即为写入会话记忆的最终答案，
    保证多轮上下文与用户所见一致（流式下无法"撤回"已发送文本）。

    query_context：可选。传入后与 run_agent 相同的上下文注入（system prompt 约束、
    工具硬约束、统一结构化入口、Validator）。
    """
    intent_hint = hint_intent(message)
    client = _get_client()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    header_sent = [False]

    def _header_events(intent, courses):
        """首个 chunk 前发送 intent/courses 头事件（只发一次）"""
        if header_sent[0]:
            return []
        header_sent[0] = True
        intent_val = intent.value if hasattr(intent, "value") else str(intent)
        events = [{"type": "intent", "data": intent_val}]
        if courses:
            events.append({"type": "courses", "data": _normalize_courses(courses)})
        return events

    async def _send_header(intent, courses):
        for ev in _header_events(intent, courses):
            yield ev

    # -------- 槽位状态（多轮精化的前提，与非流式版对称）--------
    extracted = extract_recommend_params(message)
    slots = await merge_slots(session_id, extracted, intent_hint)

    # -------- 1) 结构化快查（由 Request Router 决定是否走；统一入口）--------
    # 与非流式版对称：把意图热路径 + 槽位精化两条分支合并为统一入口
    # structured_search_with_context，是否走快查由 query_context.route 决定。
    if query_context is not None and getattr(query_context, "route", "") == "STRUCTURED_COURSE_SEARCH":
        hot_courses = await structured_search_with_context(query_context, limit)
        if hot_courses:
            hot_courses = await _ensure_validated(hot_courses, query_context, limit)
            if hot_courses:
                async for ev in _send_header(Intent.COURSE_RECOMMEND, hot_courses):
                    yield ev
                answer = _template_answer(hot_courses)
                _spawn_background(_background_polish(
                    message, hot_courses, user_id, query_context))
                await metrics_incr("intent_hot")
                # 模板直出按行切块推送（保持流式观感）
                for i in range(0, len(answer), 20):
                    yield {"type": "chunk", "data": answer[i:i + 20]}
                yield {"type": "done", "data": ""}
                return

    # -------- 2) 常规路径：工具调用循环 --------
    # 工具调用轮：结构化输出本无 token 可流，按非流式聚合处理；
    # 最终回答轮（无 tool_calls）：逐 delta 边收边推。
    if budget_exhausted(slots):
        await metrics_incr("clarify_blocked")  # 预算拦截生效（"禁止反问"强指令注入）
    # 与非流式路径一致：优先复用 QueryContext 中的历史快照。
    history_source = (
        getattr(query_context, "recent_history", None)
        if query_context is not None else None
    )
    history = _trim_history(
        history_source if history_source is not None else await get_history(session_id)
    )

    # GENERAL_LLM 流式直答：不声明工具、不进入工具循环或课程 rescue。
    if query_context is not None and getattr(query_context, "route", "") == "GENERAL_LLM":
        general_messages = [
            {"role": "system", "content": (
                "你是 ITCloud 云课堂的技术助手。直接回答用户问题，不调用工具，"
                "不反问；回答准确、清晰，必要时分点说明。"
            )},
            *history,
            {"role": "user", "content": message},
        ]
        parts: List[str] = []
        # 先发 intent，保证 LLM 建连或首包失败时 SSE 事件协议仍完整。
        yield {"type": "intent", "data": Intent.GENERIC.value}
        try:
            stream = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model, messages=general_messages, temperature=0.5,
                    max_tokens=768, timeout=30, stream=True,
                ), timeout=15,
            )
            async for event in _ttft_guard(stream):
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                if delta and delta.content:
                    parts.append(delta.content)
                    yield {"type": "chunk", "data": delta.content}
        except Exception as e:
            print(f"[AgentStream] GENERAL_LLM 直答失败 error_type={type(e).__name__} kind=general_llm")
        answer = _sanitize_answer("".join(parts))
        if not answer:
            answer = "抱歉，当前服务暂时无法处理您的问题，请稍后再试。"
            yield {"type": "chunk", "data": answer}
        await metrics_incr("general_llm")
        if _looks_like_clarify(answer, None):
            await bump_clarify(session_id, slots)
        yield {"type": "done", "data": ""}
        return

    messages = [{"role": "system", "content": _system_prompt(slots, query_context)}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})
    # 工具遵从防线①（与非流式版对称）：课程类问询注入"必须先调工具"强指令
    _is_course_q = _course_related(extracted, intent_hint, query_context)
    if _is_course_q:
        messages[-1]["content"] += _COURSE_DIRECTIVE

    collected_courses: List[Dict] = []
    # 跨工具调用累积的技术关键词（含用户消息里提取的），用于对最终卡片收窄
    tool_tech_kws: List[str] = list(extracted.get("tech_keywords") or [])
    final_intent = intent_hint
    answer = ""

    try:
        # 总预算门闩（与非流式版对称）：防止逐轮超时累积突破网关 300s
        _loop_t0 = time.monotonic()
        # 生成总预算起点：跨轮持久——自用户看到的首个内容 token 起算整条流，
        # 而非每轮各算 8s（修复：过渡语轮 + 工具轮 + 答案轮串联可叠到 17s+）
        _gen_t0 = None
        _truncated = False
        for _ in range(MAX_ITER):
            if time.monotonic() - _loop_t0 > TOOL_LOOP_BUDGET:
                print(f"[AgentStream] 工具循环总预算 {TOOL_LOOP_BUDGET}s 耗尽，跳出走兜底")
                break
            is_faq_route = (
                getattr(query_context, "route", "") == "FAQ"
                or (query_context is None and intent_hint == Intent.FAQ)
            )
            stream = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    # FAQ 政策类：平台事实已内联 system prompt，禁用工具。
                    tools=None if is_faq_route else TOOL_SCHEMAS,
                    tool_choice="none" if is_faq_route else "auto",
                    temperature=0.3,
                    max_tokens=512,
                    stream=True,
                ),
                # create 级守卫（含云端首包停滞 + SDK 内部重试）：响应头 15s 内
                # 不到即放弃走兜底。实测离群样本：首包停滞 60s 读超时 + 重试
                # 成功 = 用户白等 63s——TTFT 守卫只管 headers 之后，管不到这里。
                timeout=15,
            )
            content_parts: List[str] = []
            tool_calls_acc: Dict[int, Dict] = {}  # index -> {id, name, arguments}
            async for event in _ttft_guard(stream):
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                if delta is None:
                    continue
                if delta.content:
                    if _gen_t0 is None:
                        _gen_t0 = time.monotonic()
                    elif time.monotonic() - _gen_t0 > STREAM_ANSWER_BUDGET:
                        # 生成总预算耗尽（跨轮累计）：截断流 + 追加提示，跳出整个循环
                        content_parts.append(_TRUNCATE_NOTICE)
                        async for ev in _send_header(final_intent, collected_courses):
                            yield ev
                        yield {"type": "chunk", "data": _TRUNCATE_NOTICE}
                        print(f"[AgentStream] 生成总预算 {STREAM_ANSWER_BUDGET}s 截断"
                              f"（已生成 {sum(len(p) for p in content_parts)} 字）")
                        _truncated = True
                        break
                    content_parts.append(delta.content)
                    # 内容一到达即推送；若本轮随后还带工具调用，已推送文本作为
                    # 模型的过渡说明保留（OpenAI 惯例 tool_calls 轮 content 为空，极少触发）
                    async for ev in _send_header(final_intent, collected_courses):
                        yield ev
                    yield {"type": "chunk", "data": delta.content}
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        slot = tool_calls_acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                slot["name"] = tc.function.name
                            if tc.function.arguments:
                                slot["arguments"] += tc.function.arguments

            if _truncated:
                # 截断即终局：不再执行本轮已积累的工具调用，直接以现有内容收尾
                answer = "".join(content_parts)
                break
            if tool_calls_acc:
                # 过渡语轮（content 与 tool_calls 并存）：模型的 narration 不算
                # "正式开答"，重置预算起点——否则工具轮耗时会把最终答案轮
                # 挤成"只剩截断提示"（实测 40 字纯提示的坏案例）
                if content_parts:
                    _gen_t0 = None
                messages.append({
                    "role": "assistant",
                    "content": "".join(content_parts) or "",
                    "tool_calls": [
                        {
                            "id": slot["id"],
                            "type": "function",
                            "function": {"name": slot["name"], "arguments": slot["arguments"]},
                        }
                        for slot in tool_calls_acc.values()
                    ],
                })
                for slot in tool_calls_acc.values():
                    try:
                        args = json.loads(slot["arguments"] or "{}")
                    except Exception:
                        args = {}
                    slot["_args"] = args

                async def _run_one(slot):
                    # 同轮多工具并行执行（v2，与非流式版对称）
                    try:
                        result = await asyncio.wait_for(
                            execute_tool(slot["name"], slot["_args"], query_context),
                            timeout=TOOL_TIMEOUT,
                        )
                    except Exception as e:
                        print(
                            f"[AgentStream] 工具执行失败 tool={slot['name']} "
                            f"error_type={type(e).__name__} kind=timeout_or_failure"
                        )
                        result = {
                            "content": (
                                f"工具 {slot['name']} 执行超时或失败："
                                "当前无法完成该操作，请稍后重试或尝试其他方式。"
                            ),
                            "courses": None,
                            "intent_tag": None,
                        }
                    return slot, result

                for slot, result in await asyncio.gather(
                        *[_run_one(s) for s in tool_calls_acc.values()]):
                    args = slot["_args"]
                    if result.get("courses"):
                        # 跨多次工具调用累积去重 + 按累积技术关键词收窄
                        # （与非流式版对称：卡片与回答内容保持一致）
                        for c in result["courses"]:
                            if not any(
                                str(x.get("course_id")) == str(c.get("course_id"))
                                for x in collected_courses
                            ):
                                collected_courses.append(c)
                        for kw in extract_tech_keywords(str(args.get("keyword") or "")):
                            if kw not in tool_tech_kws:
                                tool_tech_kws.append(kw)
                        collected_courses = keyword_narrow(collected_courses, tool_tech_kws)
                        # LLM 调 search_courses 的参数是高质量槽位信号，顺手回填
                        # （下一轮精化即可走结构化快查）
                        await merge_tool_args(session_id, args)
                    if result.get("intent_tag"):
                        try:
                            final_intent = Intent(result["intent_tag"])
                        except Exception:
                            pass
                    messages.append({
                        "role": "tool",
                        "tool_call_id": slot["id"],
                        "content": result["content"],
                    })
                # header 已发（过渡语轮以空卡片发出）而工具刚带回课程 → 补发
                # courses 事件。否则卡片被"只发一次"守卫吞掉：LLM 常先吐一句
                # "我帮您查一下"再调工具，header 恰在彼时空手发出（benchmark
                # 实测冷路径 cards=0 的根因），前端/缓存从此拿不到卡片。
                if header_sent[0] and collected_courses:
                    # 工具循环返回的卡片在正式输出前过一遍校验（P0#4），
                    # 保证 courses 事件携带的是已校验卡片（文本为逐 token 流式，
                    # 无法撤回，只对"卡片集合"做硬约束）。
                    validated = await _ensure_validated(
                        collected_courses, query_context, limit)
                    if validated:
                        collected_courses = validated
                    yield {"type": "courses",
                            "data": _normalize_courses(collected_courses)}
                continue
            else:
                answer = "".join(content_parts)
                break
    except Exception as e:
        print(f"[AgentStream] LLM 调用异常 error_type={type(e).__name__} kind=llm_call")
        answer = ""

    streamed = answer  # 已推送给客户端的文本（拼接即最终答案）

    # -------- 工具遵从防线②（与非流式版对称）：课程类问询零卡片 → 强制补卡片 --------
    # LLM 未调工具直接文字作答时补齐真实课程卡片：header 未发则随 header 带出，
    # 已发则补发独立 courses 事件（前端按最新事件渲染）
    if _is_course_q and not collected_courses:
        forced = await _force_course_rescue(message, extracted, limit, query_context)
        if forced:
            forced = await _ensure_validated(forced, query_context, limit)
        if forced:
            await metrics_incr("forced_course_rescue")
            collected_courses = forced
            final_intent = Intent.COURSE_RECOMMEND
            # header 可能已随首批 chunk 发出（彼时零卡片，_header_events 只发
            # 一次不会再带）——补发独立 courses 事件，前端按最新事件渲染
            if header_sent[0]:
                yield {"type": "courses", "data": _normalize_courses(forced)}
            print(f"[AgentStream] 零卡片强制检索补 {len(forced)} 张卡片")

    # -------- 3) 兜底救援（非 LLM：课程库/知识库/联网，瞬时返回） --------
    # v2 条件收紧（与非流式版对称）：空/推脱词/既短又无课程卡片才兜底；
    # 截断（_truncated）不算"空/推脱"——截断即终局，不再叠加任何后续段
    if (
        not _truncated
        and (
            not answer
            or any(kw in answer for kw in _UNHELPFUL_KW)
            or (len(answer) < 120 and not collected_courses)
        )
    ):
        try:
            # 兜底定位是"瞬时返回"（本地 Chroma + FAQ 常态 <100ms），
            # 20s 只会放任联网搜索拖长尾——8s 封顶，超时放弃联网部分
            rescue = await asyncio.wait_for(
                _rescue_without_llm(message, limit, query_context), timeout=8
            )
        except Exception as e:
            rescue = None
            print(f"[AgentStream] 兜底检索超时/失败 error_type={type(e).__name__} kind=rescue_search")
        if rescue:
            rescue_answer = rescue["answer"]
            await metrics_incr("rescue")
            if rescue.get("courses"):
                collected_courses = rescue["courses"]
            final_intent = rescue["intent"]
            # 流式下无法撤回已发送文本：已有内容时以换行衔接追加，保持所见即所存
            if streamed:
                sep = "\n\n"
                async for ev in _send_header(final_intent, collected_courses):
                    yield ev
                yield {"type": "chunk", "data": sep + rescue_answer}
                streamed += sep + rescue_answer
            else:
                async for ev in _send_header(final_intent, collected_courses):
                    yield ev
                yield {"type": "chunk", "data": rescue_answer}
                streamed = rescue_answer

    # -------- 4) 纯 LLM 直答兜底（token 级流式） --------
    # 截断后不再追加直答：总预算的意义就是封顶用户等待，叠加只会破 cap
    if (not streamed or len(streamed) < 120) and not _truncated:
        try:
            direct_messages = [
                {"role": "system", "content": "你是技术导师。直接回答用户问题，不要反问或要求澄清。回答要分点、详细，至少300字。"},
                *history,
                {"role": "user", "content": message},
            ]
            direct_stream = await client.chat.completions.create(
                model=model,
                messages=direct_messages,
                temperature=0.7,
                max_tokens=768,
                timeout=30,
                stream=True,
            )
            parts: List[str] = []
            _gen_t0 = None
            async for event in _ttft_guard(direct_stream):
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                if delta and delta.content:
                    if _gen_t0 is None:
                        _gen_t0 = time.monotonic()
                    elif time.monotonic() - _gen_t0 > STREAM_ANSWER_BUDGET:
                        parts.append(_TRUNCATE_NOTICE)
                        async for ev in _send_header(final_intent, collected_courses):
                            yield ev
                        yield {"type": "chunk", "data": _TRUNCATE_NOTICE}
                        print(f"[AgentStream] 直答生成总预算 {STREAM_ANSWER_BUDGET}s 截断")
                        break
                    parts.append(delta.content)
                    async for ev in _send_header(final_intent, collected_courses):
                        yield ev
                    yield {"type": "chunk", "data": delta.content}
            direct_answer = "".join(parts)
            if direct_answer:
                if streamed:
                    streamed += "\n\n" + direct_answer
                else:
                    streamed = direct_answer
                print(f"[AgentStream] 纯 LLM 直答完成，累计长度: {len(streamed)}")
        except Exception as e:
            print(f"[AgentStream] 纯 LLM 直答失败 error_type={type(e).__name__} kind=direct_llm")

    # -------- 5) 记忆写入 + 追问计数 + 收尾 --------
    await metrics_incr("tool_loop")  # 走完整工具循环的请求（与 intent_hot/general_llm 对照看覆盖率）
    # 出口清理（已发出的 chunk 由前端兜底清洗，这里保证入记忆的干净）
    streamed = _sanitize_answer(streamed or "")
    # 历史写入由 /api/ai/chat/stream 的外层路由统一负责，避免同一条 SSE
    # 请求产生两组 user/assistant 消息。
    # 检测阻断式追问（问句结尾+无课程+短答），消耗本会话的一次性预算
    if _looks_like_clarify(streamed, collected_courses):
        await bump_clarify(session_id, slots)

    if not streamed:
        streamed = "抱歉，当前服务暂时无法处理您的问题，请稍后再试或联系人工客服 support@itcloud.com。"
        async for ev in _send_header(final_intent, collected_courses):
            yield ev
        yield {"type": "chunk", "data": streamed}

    # 若至今未发出头事件（极端情况：无任何内容产出），补发以免前端悬挂
    async for ev in _send_header(final_intent, collected_courses):
        yield ev

    yield {"type": "done", "data": ""}
