# 课程检索封装：在 retriever 的混合检索之上施加分类/难度/价格过滤，
# 返回可直接映射成前端课程卡片的字典列表。完全不依赖 MySQL。
import asyncio
import os
import re
import time as _time

import redis.asyncio as aioredis

from app.rag.retriever import retrieve, retrieve_hybrid
from app.rag import chroma_store


# ====================== 阶段2/3：意图缓存 + 结构化快查（混合检索热路径）======================
_intent_redis = None
_intent_keys_cache = None   # 意图键集合进程内缓存（最长包含匹配用）
_intent_keys_ts = 0.0

# ====================== Query 改写：离线同义词/扩展词库（与意图缓存同库 DB2）======================
# 离线阶段（app/mq/consumer.py 的 _write_intent）把 LLM 抽取的课程「别称/缩写/口语叫法」
# 以 qrewrite:{短语} -> 规范课名 写入同一 Redis 库；在线阶段用户口语/缩略问法（"SB课多少钱"）
# 经最长包含匹配归一化为规范课名（"Spring Boot实战开发"），再走原检索链路。
# 进程内缓存键集合 5min TTL，避免每请求 SCAN（与意图缓存一致）。
_qrewrite_keys_cache = None
_qrewrite_keys_ts = 0.0

# 全量规范课名词表缓存（防御：原话已含规范课名则跳过改写，避免短别名子串泄漏）
_canon_names_cache = None
_canon_names_ts = 0.0


def _norm_ws(s: str) -> str:
    """去空格归一：用户常把 "K8s 编排" 打成带空格，而字典键 "K8s编排" 无空格，
    原始子串匹配对空格敏感会漏命中；归一后两者都能匹配（详见内存日志 2026-09-01）。"""
    return re.sub(r"\s+", "", s or "")


async def _redis_intent():
    """意图缓存专用异步连接（独立 DB2，懒加载单例，与对话记忆隔离）。

    修复：原来每次调用都新建同步连接——热路径每请求一次 TCP 建连，
    且同步客户端在事件循环线程上阻塞（同 memory.py 修复前的病灶）。
    """
    global _intent_redis
    if _intent_redis is None:
        _intent_redis = aioredis.Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_INTENT_DB", 2)),
            # Redis 已开启 requirepass（见 docker-compose redis 服务），必须携带认证
            password=os.getenv("REDIS_PASSWORD") or None,
            decode_responses=True,
            socket_timeout=3,
            socket_connect_timeout=3,
        )
    return _intent_redis


async def lookup_intent_filter(query: str) -> dict | None:
    """
    查 Redis 意图缓存（DB2）。命中返回 {'ai_field': X} 或 {'ai_level': X}，否则 None。

    匹配策略：**最长包含匹配**（v2）。键是课名/同义词（如 intent:Spring Boot企业级开发），
    v1 只做原话精确匹配——用户说"有没有Spring Boot企业级开发这门课？"永远 miss
    （intent_hot 长期为 0 的根因）。v2 改为：用户原话**包含**某键即命中，
    多键命中取最长（区分力最强，"Java入门"优先于"Java"）。
    键集合进程内缓存 5 分钟，避免每请求 SCAN。
    """
    global _intent_keys_cache, _intent_keys_ts
    try:
        r = await _redis_intent()
        now = _time.monotonic()
        if _intent_keys_cache is None or now - _intent_keys_ts > 300:
            keys = await r.keys("intent:*")
            # 按长度降序：首中即最长匹配；过滤太短的键（"java"2字以下噪音大）
            _intent_keys_cache = sorted(
                (k[len("intent:"):] for k in keys if len(k) > len("intent:") + 1),
                key=len, reverse=True,
            )
            _intent_keys_ts = now
        for k in _intent_keys_cache:
            if k in query:
                val = await r.get(f"intent:{k}")
                if not val:
                    continue
                if val.startswith("ai_field:"):
                    return {"ai_field": val[len("ai_field:"):]}
                if val.startswith("ai_level:"):
                    return {"ai_level": val[len("ai_level:"):]}
    except Exception as e:
        print(f"[Intent] 缓存查询失败: {e}")
    return None


async def _load_canonical_names():
    """懒加载全量规范课名（改写防御：原话已含规范课名则跳过改写）。缓存 5min。"""
    global _canon_names_cache, _canon_names_ts
    now = _time.monotonic()
    if _canon_names_cache is None or now - _canon_names_ts > 300:
        s = set()
        try:
            col = chroma_store.get_collection()
            data = col.get(include=["metadatas"])
            for m in data.get("metadatas") or []:
                n = (m.get("course_name") or "").strip()
                if n:
                    s.add(n)
        except Exception as e:
            print(f"[QueryRewrite] 加载规范课名词表失败: {e}")
            return _canon_names_cache or set()
        _canon_names_cache = s
        _canon_names_ts = now
    return _canon_names_cache


async def rewrite_query(raw: str) -> str:
    """
    离线同义词/扩展词库 Query 改写（与意图缓存同库 DB2）。

    离线阶段（app/mq/consumer.py 的 _write_intent）把 LLM 抽取的课程
    「别称/缩写/口语叫法」以 qrewrite:{短语} -> 规范课名 写入同一 Redis 库；
    在线阶段这里做**最长包含匹配**：用户口语/缩略问法（"SB课多少钱"）被归一化为
    规范课名（"Spring Boot实战开发"），再交给原检索链路（retriever.retrieve）。

    设计要点：
    - 无命中则原样返回——对规范问法完全是 no-op，不影响现有 83% 基线；
    - 匹配键集合进程内缓存 5min（与意图缓存一致），热路径额外开销 <1ms；
    - 任何异常都降级为原 query，绝不阻断检索。
    """
    # 总开关（RAG_QREWRITE，默认 0 = 关闭）。
    # 2026-09-16 在 128 门语料 / 180 条独立评测集上实测：改写口径 R@1 50.0%，
    # 比不改写（55.0%）低 5 个点；把 430 条别名值里的难度词剥离后仍只有 52.8%。
    # 根因是句子级替换会抹掉原句中的判别词（"调优"→"原理"），把精通档翻成入门档。
    # 词库不删除（离线抽取能力仍是项目亮点），只在线上检索链路摘除。
    if os.getenv("RAG_QREWRITE", "0") != "1":
        return raw
    global _qrewrite_keys_cache, _qrewrite_keys_ts
    if not raw:
        return raw
    try:
        r = await _redis_intent()
        now = _time.monotonic()
        if _qrewrite_keys_cache is None or now - _qrewrite_keys_ts > 300:
            keys = await r.keys("qrewrite:*")
            _qrewrite_keys_cache = sorted(
                (k[len("qrewrite:"):] for k in keys if len(k) > len("qrewrite:")),
                key=len, reverse=True,  # 首中即最长匹配
            )
            _qrewrite_keys_ts = now
        q = _norm_ws(raw)
        # 防御：若用户原话已包含某规范课名，说明问法已足够规范，直接原样返回，
        # 避免短别名子串泄漏把本已正确的检索改写到错误兄弟课。比较走空格归一，
        # 空格不再成为漏判因素（归一化会同时唤醒"指错课"的潜在坏别名，
        # 已通过清理 qrewrite 字典中映射错误的条目解决，详见 memory/2026-09-01.md）。
        cnames = await _load_canonical_names()
        if any(cn and _norm_ws(cn) in q for cn in cnames):
            return raw
        for p in _qrewrite_keys_cache:
            pn = _norm_ws(p)
            if pn and pn in q:
                canonical = await r.get(f"qrewrite:{p}")
                # 守卫：规范名已出现在归一化 query 中时跳过，避免 "SpringSpringBoot" 双插；
                # 同时若改写会把原 query 中已正确的词覆盖成兄弟课（别名漂移），
                # 这一步仅做最小防护，根因仍在离线别名的"唯一指向"质量上。
                if canonical and _norm_ws(canonical) not in q:
                    q = q.replace(pn, _norm_ws(canonical))
        return q
    except Exception as e:
        print(f"[QueryRewrite] 改写失败降级原样: {e}")
        return raw


def structured_search(filter: dict, limit: int = 5) -> list:
    """
    结构化快查：按 metadata 字段过滤 Chroma 集合，**不走向量检索（无 embedding，<100ms）**。
    支持两类过滤键：ai_field / ai_level（意图缓存热路径，中文标签），
    category / level（槽位热路径，英文枚举）。多条件用显式 $and 形式。
    返回 metadata 字典列表（与 search_courses 同 schema，含 course_id / course_name 等）。
    """
    try:
        res = chroma_store.get_collection().get(where=filter, limit=limit)
        out = [m for m in (res.get("metadatas") or [])]
        for m in out:
            m["id"] = m.get("course_id")
        return out
    except Exception as e:
        print(f"[Structured] 查询失败: {e}")
        return []


def structural_where_from_filters(filters: dict, ctx=None) -> dict:
    """把 QueryContext.search_filters 里『可作结构化 where』的键转成 Chroma where。

    仅 category / level / ai_field / ai_level 参与 where（真实 metadata 过滤键）；
    max_price（price 是字符串，无法 where 比较）由 price_filter 内存后置处理；
    tech_keywords / keywords 由 keyword_narrow 后置处理。

    多领域（ctx.categories ∩2，P0#1）：category 用 $in 一次放行多个领域，
    而不是把其它领域过滤死。level 等其余硬约束仍 $eq 生效。
    """
    conds = []
    if ctx is not None and getattr(ctx, "categories", None) and getattr(ctx, "category_mode", "") == "multi":
        cats = [c for c in ctx.categories if c]
        if len(cats) >= 2:
            conds.append({"category": {"$in": cats}})
            # category 已由 $in 覆盖，跳过下面单值 category 处理
            rest = {k: v for k, v in filters.items() if k != "category"}
        else:
            rest = filters
    else:
        rest = filters
    for k in ("category", "level", "ai_field", "ai_level"):
        v = rest.get(k)
        if v:
            conds.append({"category": {"$in": [v]}} if k == "category" else {k: {"$eq": v}})
    if not conds:
        return {}
    return conds[0] if len(conds) == 1 else {"$and": conds}


async def structured_search_with_context(ctx, limit: int = 5) -> list:
    """统一结构化课程搜索入口（意图缓存热路径 与 槽位精化快查 合并后共用）。

    由 Request Router 决定是否走本入口；所有结构化课程检索都经它执行，
    Agent 内不再维护两套结构化检索代码。

    流程：
      1) 从 ctx.search_filters 构造 `where`（category/level/ai_field/ai_level；
         多领域 category 用 $in）→ structured_search（纯 metadata 过滤）；
      2) 用 max_price 做内存后置过滤（price 是字符串型 metadata）；
      3) 用 tech_keywords / keywords 对课名收窄/排序（keyword_narrow）；
      4) DB2 hint_filters（ai_field/ai_level）只作方向**排序加分**，不作为
         硬 where —— 避免弱推断把其它领域过滤死（P0#2）；
      5) 守卫：过滤过严导致空结果时，保留足够样本回退给上层（Rescue）。

    返回 metadata 字典列表（含 course_id / course_name 等）。
    """
    filters = (ctx.search_filters or {}) if ctx else {}
    candidate_pool = max(limit * 4, 25)
    where = structural_where_from_filters(filters, ctx)
    try:
        courses = await asyncio.to_thread(structured_search, where, candidate_pool)
    except Exception as e:
        print(f"[StructuredWithCtx] 查询异常降级: {e}")
        courses = []

    excluded = [
        str(value).lower() for value in
        (filters.get("excluded_tech_keywords") or filters.get("excluded") or [])
        if str(value).strip()
    ]
    excluded_set = set(excluded)
    techs = [
        value for value in (
            list(filters.get("tech_keywords") or [])
            or ((ctx.extracted or {}).get("tech_keywords") if ctx else None)
            or []
        )
        if str(value).lower() not in excluded_set
    ]
    kws = list(filters.get("keywords") or []) if filters.get("keywords") else (
        (ctx.extracted or {}).get("keywords") if ctx else None
    ) or []
    kws = [value for value in kws if str(value).lower() not in excluded_set]

    # 价格内存过滤（price 字符串 → float）
    narrowed = price_filter(courses, filters.get("max_price"))
    # 技术关键词课名收窄（只收窄英文具体技术词；宽泛词不收窄）
    if techs:
        narrowed = keyword_narrow(narrowed, list(techs)) or narrowed
    elif kws:
        narrowed = keyword_narrow(narrowed, list(kws)) or narrowed

    if excluded:
        narrowed = [
            course for course in narrowed
            if not any(
                token in str(course.get("course_name") or "").lower()
                for token in excluded
            )
        ]

    # DB2 hint 方向排序加分（P0#2）：同 matching 前缀的课程往前提，不删除其它领域
    hint = (ctx.hint_filters or {}) if ctx else {}
    hint_field = hint.get("ai_field")
    if hint_field and narrowed:
        def _hint_key(c):
            ai = str(c.get("ai_field") or "")
            name = str(c.get("course_name") or "")
            hit = (ai and hint_field in ai) or (hint_field in name)
            return 0 if hit else 1  # 命中的排在前面
        narrowed = sorted(narrowed, key=_hint_key)

    for c in narrowed:
        c["id"] = c.get("course_id")
    return narrowed[:limit]


def slots_where(slots: dict) -> dict | None:
    """
    把会话槽位转成 Chroma where 过滤条件（供多轮精化热路径）。
    只有 category / level 参与过滤（price 是字符串 metadata，无法 where 比较，
    由 price_filter 在内存中后置过滤）。无可用约束时返回 None。
    """
    conds = []
    if slots.get("category"):
        conds.append({"category": {"$eq": slots["category"]}})
    if slots.get("level"):
        conds.append({"level": {"$eq": slots["level"]}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def price_filter(courses: list, max_price) -> list:
    """按价格上限内存过滤（metadata 里 price 是字符串，统一 float 比较；
    解析失败的条目保留——宁多勿缺，交给上层展示）。"""
    if max_price is None:
        return courses
    out = []
    for m in courses:
        try:
            if float(m.get("price") or 0) <= float(max_price):
                out.append(m)
        except (TypeError, ValueError):
            out.append(m)
    return out


def keyword_narrow(courses: list, keywords: list) -> list:
    """
    按技术关键词收窄/排序结构化检索结果——修复"问 java 推荐 Go"：
    意图缓存只给领域级过滤（java→ai_field:后端），Chroma where 过滤不排序，
    捞回的任意后端课里会混入 Go/微服务等无关课程。

    规则：
    - 仅"课程名命中"才收窄（调用方已保证 keywords 是具体技术词，不含宽泛分类词）；
    - 简介命中只参与排序（描述里也提该技术的更相关），不参与收窄；
    - 全部未命中时原样返回（宽泛查询不收窄）。
    命中的课程顺带写入 match_reason，前端卡片不再显示万能的"综合匹配度较高"。
    """
    kws = [str(k).lower() for k in (keywords or []) if len(str(k)) >= 2]
    if not kws or not courses:
        return courses

    def name_hits(c):
        name = (c.get("course_name") or "").lower()
        return [k for k in kws if k in name]

    def desc_hits(c):
        desc = (c.get("description") or "").lower()
        return sum(1 for k in kws if k in desc)

    matched = [c for c in courses if name_hits(c)]
    if not matched:
        return courses
    matched.sort(key=lambda c: (len(name_hits(c)), desc_hits(c)), reverse=True)
    for c in matched:
        c["match_reason"] = f"匹配关键词：{'、'.join(name_hits(c))}"
    return matched


async def search_courses(
    query: str,
    category=None,
    level=None,
    max_price=None,
    limit: int = 5,
) -> list:
    """
    语义检索课程，并施加结构化过滤。

    返回：metadata 字典列表，字段含 course_id / course_name / category /
          level / price / teacher_name / description，与旧 MySQL 行字段对齐，
          方便上层工具直接复用 _normalize_courses 等逻辑。
    """
    # 阶段0：离线同义词/扩展词库 Query 改写（口语/缩略问法 -> 规范课名），
    # 命中才改写，未命中 no-op，不改动原检索链路语义。
    query = await rewrite_query(query or "")
    # 第一阶段候选生成开关：RAG_HYBRID=1（生产默认开启）走
    # 「Vector + BM25 → RRF → IDF → BGE」双路召回；设为 0 可回滚到
    # Vector → IDF → BGE 基线。
    if os.getenv("RAG_HYBRID", "1") == "1":
        hits = await retrieve_hybrid(query, top_k=limit * 3)
    else:
        hits = await retrieve(query, top_k=limit * 3)
    results = []
    for h in hits:
        m = h.get("metadata", {})
        if category and m.get("category") != category:
            continue
        if level and m.get("level") != level:
            continue
        if max_price is not None:
            try:
                if float(m.get("price") or 0) > float(max_price):
                    continue
            except (TypeError, ValueError):
                pass
        results.append(m)
        if len(results) >= limit:
            break
    return results
