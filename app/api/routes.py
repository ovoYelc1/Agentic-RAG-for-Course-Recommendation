# API 路由
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
import asyncio
import json
import logging
from datetime import datetime

# 鉴权与限流：身份一律取自 JWT，禁止沿用请求体中客户端自报的 user_id/session_id
from app.core.security import Identity, require_identity
from app.core.ratelimit import ConcurrencyGuard, check_rate_limit, resolve_session_id

from app.models.chat import (
    ChatRequest,
    ChatResponse,
    HistoryResponse,
)
from app.agent.agent import run_agent, run_agent_stream
from app.agent.router import extract_recommend_params, hint_intent
from app.agent.course_recommender import _generate_match_reason
from app.agent import request_router
from app.services.slots import merge_slots
from app.services.memory import append_history, get_history
from app.services.context_manager import build_query_context, QueryContext
# 步骤3：课程推荐改走 Chroma 语义检索，不再直连 MySQL
from app.rag.course_search import search_courses
from app.services.cache import get_cached, set_cache, get_async_redis, build_answer_cache_key
from app.services.metrics import METRIC_NAMES, incr as metrics_incr

router = APIRouter()
logger = logging.getLogger(__name__)


def _intent_val(v) -> str:
    return v.value if hasattr(v, "value") else str(v)


def _cache_filters(ctx) -> dict:
    """答案缓存 key 使用的过滤指纹：search_filters + 多领域 categories。

    不走槽位纯值（那些是会话派生，非本轮约束）；只纳入『真正影响答案』的
    本轮约束。categories 参与 key 使多领域比较不与单领域共享答案（P0#1）。
    """
    filters = dict((ctx.search_filters or {}))
    if getattr(ctx, "category_mode", "") == "multi" and getattr(ctx, "categories", None):
        filters["categories"] = list(ctx.categories)
    return filters


def _cache_skip_reason_if_followup(ctx, message: str) -> str:
    """上下文依赖追问 → 返回跳过缓存的理由；可独立缓存 → 空串（P0#3）。

    判定：follows_context(query, has_independent_terms)。has_independent_terms 取
    自本轮是否含独立业务语义（明确领域/技术词/价格/难度）。
    """
    from app.services.cache import follows_context
    expr = ctx.extracted or {}
    has_ind = bool(
        expr.get("category") or expr.get("categories")
        or expr.get("tech_keywords") or expr.get("level")
        or expr.get("max_price") is not None
        or ctx.intent == "course_recommend"
    )
    if follows_context(message, has_independent_terms=has_ind):
        return "context_dependent_followup"
    return ""


async def _build_context_for_request(message: str, user_id: str, session_id: str,
                                     message_id: str = "") -> QueryContext:
    """按依赖顺序构建 QueryContext（Context Manager）。

    1) 本轮规则解析（hint_intent / extract_recommend_params，纯本地正则）；
    2) 判断是否课程相关，决定是否查 DB2 离线别名（intent_match）；
    3) 并行读取历史 + 槽位（无依赖，asyncio.gather）；
    4) 合并槽位与冲突（build_query_context 内部完成，摊平出 search_filters）。
    """
    intent_hint = _intent_val(hint_intent(message))
    extracted = extract_recommend_params(message)
    extracted["_message_id"] = message_id or ""
    extracted["_raw_query"] = message

    is_course_related = (
        intent_hint == "course_recommend"
        or extracted.get("category")
        or extracted.get("tech_keywords")
        or extracted.get("max_price") is not None
        or extracted.get("keywords")
    )

    from app.services.slots import get_slots_raw
    from app.rag.course_search import lookup_intent_filter

    async def _load_slots_history():
        try:
            slots_raw = await get_slots_raw(session_id) if session_id else {}
        except Exception:
            slots_raw = {}
        history = await get_history(session_id, include_full=False) if session_id else []
        return slots_raw, history

    async def _load_intent():
        if not is_course_related:
            return None
        try:
            return await lookup_intent_filter(message)
        except Exception:
            return None

    # 历史 + 槽位（get_slots_history 会并行取）与 DB2 别名无依赖，可并行
    (slots_raw, history), intent_match = await asyncio.gather(
        _load_slots_history(), _load_intent(),
    )

    return await build_query_context(
        conversation_id=session_id,
        user_id=user_id,
        query=message,
        intent_hint=intent_hint,
        extracted=extracted,
        intent_match=intent_match,
        get_slots_history=lambda: (slots_raw, history),
    )


async def _save_ai_reply_history(session_id: str, message: str, result: dict,
                                 message_id: str = "") -> None:
    """缓存命中 / Agent 完成后统一：保存本次 user + assistant 历史（幂等）。"""
    await append_history(session_id, "user", message, message_id=message_id or None)
    await append_history(
        session_id, "assistant", (result.get("answer") or ""),
        message_id=_msg("a", message_id),
    )


def _msg(role_tag: str, base: str = "") -> str:
    """为幂等生成带 role 标记的 message_id 前缀。"""
    import uuid
    suffix = base or uuid.uuid4().hex[:8]
    return f"msg_{role_tag}_{suffix}"


@router.post("/api/ai/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, identity: Identity = Depends(require_identity)):
    """
    统一聊天接口（Agent 驱动）
    - 由 Agent 自主调用工具：查课程库 / 联网搜索 / 查知识库
    - 返回结构化答案 + 推荐课程（如有）

    身份处理：user_id / session_id **一律以 JWT 为准**，忽略请求体中的自报值，
    杜绝「伪造 user_id 蹭他人画像」与「猜 session_id 读他人聊天记录」。
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    user_id = identity.user_id
    session_id = resolve_session_id(user_id, request.session_id)
    await check_rate_limit(user_id)

    # 0. Context Manager 构建 QueryContext（保存用户消息前完成解析与槽位合并）
    ctx = await _build_context_for_request(request.message, user_id, session_id)
    # 缓存先于执行分支，但必须先完成 canonical cache identity。
    request_router.prepare_cache_identity(ctx)

    # 1. 答案缓存（canonical cache_intent + effective search_filters；v7）
    #    缓存命中不需要完整 Request Router，命中后直接返回/回放。
    #     上下文依赖追问（指代/相对精化/无独立语义）→ cache_skip_reason 跳过读+写
    if not ctx.cache_skip_reason:
        from app.services.cache import follows_context
        ctx.cache_skip_reason = _cache_skip_reason_if_followup(ctx, request.message)
    cache_key = await build_answer_cache_key(
        request.message, intent=ctx.intent, cache_intent=ctx.cache_intent,
        search_filters=_cache_filters(ctx),
    )
    cached = None if ctx.cache_skip_reason else await get_cached(cache_key)
    if cached:
        await metrics_incr("cache_hit")
        # 缓存命中也要沉淀槽位（多轮精化依赖每一轮的抽取结果）
        await merge_slots(
            session_id, extract_recommend_params(request.message),
            hint_intent(request.message),
        )
        # 保存本次交互历史（user + assistant 各一条，幂等）
        await _save_ai_reply_history(session_id, request.message, cached)
        return ChatResponse(**cached)
    await metrics_incr("cache_skip" if ctx.cache_skip_reason else "cache_miss")

    # 2. 缓存未命中后才执行完整 Request Router。
    decision = request_router.route(ctx)
    ctx.route = decision.route
    ctx.route_reason = decision.reason
    ctx.route_confidence = decision.confidence
    await metrics_incr({
        request_router.FAQ: "route_faq",
        request_router.STRUCTURED: "route_structured",
        request_router.AGENT: "route_agent",
        request_router.GENERAL: "route_general",
    }.get(decision.route, "route_unknown"))

    # 3. 交给 Agent 处理（并发闸门防单用户占满推理算力）
    async with ConcurrencyGuard(user_id):
        result = await run_agent(
            message=request.message,
            user_id=user_id,
            session_id=session_id,
            query_context=ctx,
        )

    # 4. 缓存结果（热路径直出由 agent 后台润色任务覆盖为润色版）
    if not ctx.cache_skip_reason:
        await set_cache(cache_key, result, ttl=3600)

    return ChatResponse(**result)


@router.post("/api/ai/chat/stream")
async def chat_stream(request: ChatRequest, identity: Identity = Depends(require_identity)):
    """
    流式聊天接口 - SSE 推送
    #16 修复：改由 run_agent_stream 异步生成器逐片段产出（intent/courses/chunk/done），
    边生成边推送，避免原「先 await 全量结果再逐字 yield」的首字延迟与事件循环阻塞。
    事件:
    - type: intent, data: 意图标识
    - type: courses, data: 推荐课程列表（如有）
    - type: chunk, data: 文本片段
    - type: done, data: ""
    - type: error, data: 错误信息

    鉴权与配额：由 require_identity 保证「来自网关 + 已登录用户」；
    user_id / session_id 以 JWT 为准，忽略请求体自报值。
    限流在返回 StreamingResponse **之前**判定，否则 200 已发出、无法再返回 429。
    """
    if not request.message.strip():
        async def error_stream():
            yield json.dumps({"type": "error", "data": "消息不能为空"}, ensure_ascii=False, default=str) + "\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    user_id = identity.user_id
    session_id = resolve_session_id(user_id, request.session_id)

    # 两道闸门都在流开始前判定，超限时以标准 429 响应（而非流内 error 事件）
    await check_rate_limit(user_id)

    # 0. Context Manager（课程相关才查 DB2 并装配过滤条件）
    ctx = await _build_context_for_request(request.message, user_id, session_id)
    request_router.prepare_cache_identity(ctx)

    # 1. 答案缓存（v7：canonical cache_intent + search_filters 指纹；SSE 也查也写）。
    #    命中时直接回放，完整 Request Router 只在 miss 后执行。
    if not ctx.cache_skip_reason:
        from app.services.cache import follows_context
        ctx.cache_skip_reason = _cache_skip_reason_if_followup(ctx, request.message)
    cache_key = await build_answer_cache_key(
        request.message, intent=ctx.intent, cache_intent=ctx.cache_intent,
        search_filters=_cache_filters(ctx),
    )
    cached = None if ctx.cache_skip_reason else await get_cached(cache_key)

    async def _cached_stream(result):
        """缓存命中：按 ~32 字切块模拟流式吐出（纯内存操作，延迟 <50ms）。

        修复：命中时补写 user + assistant 历史（旧版漏写，导致缓存命中后的
        多轮上下文缺本轮一问一答）。
        """
        await metrics_incr("cache_hit")
        await merge_slots(
            session_id,
            extract_recommend_params(request.message),
            hint_intent(request.message),
        )
        await _save_ai_reply_history(session_id, request.message, result)
        yield json.dumps({"type": "intent", "data": _intent_val(result.get("intent")),
                          "timestamp": datetime.now().isoformat()},
                         ensure_ascii=False, default=str) + "\n\n"
        if result.get("courses"):
            yield json.dumps({"type": "courses", "data": result["courses"],
                              "timestamp": datetime.now().isoformat()},
                             ensure_ascii=False, default=str) + "\n\n"
        answer = result.get("answer") or ""
        # 回放节奏自适应：总节奏预算 ~80ms
        chunks = [answer[i:i + 32] for i in range(0, len(answer), 32)]
        pace = min(0.012, 0.08 / max(len(chunks), 1))
        for piece in chunks:
            yield json.dumps({"type": "chunk", "data": piece,
                              "timestamp": datetime.now().isoformat()},
                             ensure_ascii=False, default=str) + "\n\n"
            await asyncio.sleep(pace)
        yield json.dumps({"type": "done", "data": "",
                          "timestamp": datetime.now().isoformat()},
                         ensure_ascii=False, default=str) + "\n\n"

    if cached:
        return StreamingResponse(_cached_stream(cached), media_type="text/event-stream")
    await metrics_incr("cache_skip" if ctx.cache_skip_reason else "cache_miss")

    decision = request_router.route(ctx)
    ctx.route = decision.route
    ctx.route_reason = decision.reason
    ctx.route_confidence = decision.confidence
    await metrics_incr({
        request_router.FAQ: "route_faq",
        request_router.STRUCTURED: "route_structured",
        request_router.AGENT: "route_agent",
        request_router.GENERAL: "route_general",
    }.get(decision.route, "route_unknown"))

    guard = await ConcurrencyGuard(user_id).acquire()

    async def event_stream():
        collected = {"answer": [], "courses": None, "intent": None}
        try:
            async for piece in run_agent_stream(
                message=request.message,
                user_id=user_id,
                session_id=session_id,
                query_context=ctx,
            ):
                ptype = piece.get("type")
                if ptype == "chunk" and piece.get("data"):
                    collected["answer"].append(piece["data"])
                elif ptype == "courses" and collected["courses"] is None:
                    collected["courses"] = piece.get("data")
                elif ptype == "intent" and collected["intent"] is None:
                    collected["intent"] = piece.get("data")
                piece["timestamp"] = datetime.now().isoformat()
                yield json.dumps(piece, ensure_ascii=False, default=str) + "\n\n"
            # 流正常结束 → 写答案缓存 + 保存本轮历史（缓存命中后的历史在 _cached_stream 写）
            if collected["answer"]:
                _ans = "".join(collected["answer"])
                result = {
                    "intent": collected["intent"] or "course_recommend",
                    "answer": _ans,
                    "courses": collected["courses"],
                    "confidence": 0.9,
                }
                if "已为您截断" not in _ans:
                    if not ctx.cache_skip_reason:
                        await set_cache(cache_key, result, ttl=3600)
                await _save_ai_reply_history(session_id, request.message, result)
        except Exception as e:
            logger.error(
                "api.chat.stream failed error_type=%s",
                type(e).__name__,
            )
            yield json.dumps({
                "type": "error",
                "data": "服务暂时不可用，请稍后重试。",
                "timestamp": datetime.now().isoformat(),
            }, ensure_ascii=False, default=str) + "\n\n"
        finally:
            # 客户端中途断开时，Starlette 会关闭生成器（抛 GeneratorExit），
            # finally 仍会执行 → 并发槽位不会泄漏。
            await guard.release()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/api/ai/recommend")
async def recommend(
    query: str = Query(..., min_length=1, description="课程查询描述"),
    limit: int = Query(5, ge=1, le=20, description="返回数量"),
    category: str = Query("", description="可选分类约束（backend/frontend/data/devops），详情页相关推荐传入当前课程分类"),
    identity: Identity = Depends(require_identity),
):
    """
    课程推荐接口（结构化返回，适合前端卡片展示）

    示例: GET /api/ai/recommend?query=我想学Java入门&limit=3

    整合 course_recommender：先用 extract_recommend_params 解析查询中的
    结构化维度（分类/难度/预算），优先做「语义检索 + 结构化过滤」；
    过滤后为空再回退纯语义检索，避免过拟合关键词导致零结果。
    每门课带 match_reason 匹配理由，供前端卡片展示推荐依据。

    category 显式参数优先于从 query 提取的分类（详情页"相关推荐"场景：
    当前课程分类是已知最强信号，纯语义检索易跨领域误推）。

    注意：已移除 user_id 查询参数——它同样是客户端自报、不可信；
    个性化维度统一取自 JWT，避免通过改 URL 冒用他人身份。
    """
    await check_rate_limit(identity.user_id)

    params = extract_recommend_params(query)
    # 显式传入的分类优先（调用方已知当前课程所属领域）
    effective_category = category or params.get("category")
    courses = await search_courses(
        query,
        category=effective_category,
        level=params.get("level"),
        max_price=params.get("max_price"),
        limit=limit,
    )
    if not courses and (effective_category or params.get("level") or params.get("max_price")):
        # 结构化过滤太严格导致空结果 → 回退纯语义检索
        courses = await search_courses(query, limit=limit)

    for c in courses:
        c["id"] = c.get("course_id")
        c["match_reason"] = _generate_match_reason(c, params)

    return {
        "query": query,
        "total": len(courses),
        "courses": courses,
    }


@router.get("/api/ai/history", response_model=HistoryResponse)
async def chat_history(session_id: str = "",
                       identity: Identity = Depends(require_identity)):
    """
    读取会话历史（前端恢复同一聊天页用）。鉴权走 JWT。

    会话归属：session_id 由服务端 `resolve_session_id` 绑定 user 前缀，无法跨用户
    读取/污染他人历史（IDOR 防护，见 ratelimit.resolve_session_id）。前端可传
    sessionStorage 里存的原始 conversation_id 后缀，服务端会重新绑定到当前 JWT 用户。
    返回 24h 内未过期消息，按创建时间升序。
    """
    cid = resolve_session_id(identity.user_id, session_id)
    messages = await get_history(cid, include_full=True)
    return HistoryResponse(
        conversation_id=session_id or "new",
        messages=messages,
    )


@router.get("/api/ai/health")
async def health(identity: Identity = Depends(require_identity)):
    """
    业务健康检查（需登录）。

    注意分工：
    - 容器/运维**存活**探测请用 main.py 的 `GET /health`（无鉴权，只回 status）；
    - 本端点属 /api/ai/* 业务域，受双层鉴权保护，供已登录前端确认 AI 可用性。

    已移除 llm 型号回显：那属于对外指纹信息（便于攻击者针对特定模型构造提示注入），
    对终端用户也无意义。需要排查模型配置时看服务启动日志。
    """
    return {"status": "healthy", "auth": "on"}


@router.get("/api/ai/metrics")
async def ai_metrics(identity: Identity = Depends(require_identity)):
    """
    AI 服务运行指标（Redis INCR 计数器，走双层鉴权）。

    用途：确认热路径覆盖率（intent_hot + cache_hit vs tool_loop）、
    兜底触发频率（rescue）、追问预算拦截情况（clarify_used / clarify_blocked）。
    计数器自服务启动起累加（Redis 持久，重启不清零）。
    """
    try:
        r = await get_async_redis()
        data = {}
        for n in METRIC_NAMES:
            v = await r.get(f"ai:metrics:{n}")
            data[n] = int(v) if v else 0
        return {"success": True, "data": data}
    except Exception as e:
        logger.error(
            "api.ai.metrics failed error_type=%s",
            type(e).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="指标读取失败，请稍后重试。",
        )
