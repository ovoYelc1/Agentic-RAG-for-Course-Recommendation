# 课程语义检索：向量召回 + jieba 关键词 IDF 加权（混合检索）+ 交叉编码器重排。
#
# 设计原则（对应步骤2「从一开始不依赖 MySQL」）：
#   - 检索只走 Chroma 向量库，本模块不 import 任何 MySQL 代码；
#   - 课程详情直接来自入库时写入 Chroma 的 metadata，无需再回查业务库。
#
# 融合策略（v2，IDF 加权）：
#   v1 问题：score = rerank + 0.05 * 裸词元命中数。命中数不区分词的区分力，
#   「课程/入门/实战」这类几乎每篇文档都有的高频词同样计分，长描述课程白捡分，
#   0.05×N 的加分足以推翻交叉编码器排序（实测翻错 2 题）。
#   v2 方案：关键词命中分按 IDF 加权后归一化到 [0,1]（高频词 IDF→0 自动失权），
#   再乘固定系数融合——关键词只能"助推"，不能凭噪声词推翻精排结果。
import asyncio
import math
import os
import re
import threading

import jieba


from app.rag import embeddings, reranker, chroma_store
from app.rag import bm25 as bm25_engine

# jieba 日志太吵，关掉
jieba.setLogLevel(20)

# IDF 命中分的融合系数：kw_idf ∈ [0,1]，最高加 α 分——足以在精排分差接近的
# 近邻候选间完成纠偏，又不会压过 reranker 的强信号（mmarco-mMiniLM 分数量级）。
#
# 2026-09-17 起由环境变量 RAG_FUSION_ALPHA 控制（默认 0，匹配当前生产配置）：
#   这个系数是**为旧 reranker（bge-reranker-base）的分数分布标定**的。
#   换成 bge-reranker-v2-m3 后，语义 reranker 对重口语 query 给出的分数更低更集中，
#   固定 α=0.5 的绝对加分相对权重过大，会把"关键词重叠高但语义错"的课翻到 Top1。
#   实测（真实生产 Top60 候选池）：v2-m3 下 α 从 0 单调加到 0.5，整体 R@1 56.7% → 52.8%
#   （-3.9pt），其中 colloquial_heavy 39.4% → 28.2%（-11.2pt）——IDF 加分对 v2-m3 是纯负优化。
#   因此生产随 RERANKER_MODEL 一起切到 α=0（见 docker-compose.yml 的 RAG_FUSION_ALPHA）。
#   复现旧基线（base + α=0.5）时：RAG_FUSION_ALPHA=0.5。
_FUSION_ALPHA = float(os.getenv("RAG_FUSION_ALPHA", "0"))

# 课名完整命中加分：查询词元完全覆盖某课程名称（如用户直接念出课名）时，
# 该课程就是用户所指，给决定性加分。动机：168 门近邻克隆语料实测发现
# mmarco-mMiniLM 对中文近重复文档区分力弱（克隆课 rerank 10.86 > 真课 10.32），
# 关键词 IDF 分对"两门课都全命中"的情形也无从区分，需要唯一性信号兜底。
_NAME_EXACT_BONUS = 3.0

# 课名子串/双向精确匹配加分：用户查询（或其一段）与课名存在"去空格后"的双向子串包含时，
# 该课程极可能就是所指。动机：v3 的 name_exact 要求"课名词元 ⊆ 查询词元"，方向过严——
# 用户只念出课名一部分（如 "微服务架构设计"）时，课名（含"实战"）不被查询词元覆盖，
# 两门近重复课 rerank 分并列(0.997)且无唯一性信号可区分，靠 chroma 返回顺序被任意打破，
# 致期望课（名字更长、含区别词）输给兄弟课。双向子串匹配补上这个信号：
#   - cname_norm ⊆ q_norm（课名是查询的子串）：强信号"用户念的就是这门课"，无条件生效；
#   - q_norm ⊆ cname_norm（查询是课名的子串）：查询为课名片段，需 len≥3 防 "开发" 等泛词误伤。
_NAME_SUBSTR_BONUS = 3.0

# ———————————————— 难度感知融合（治「向入门档塌陷」）———————————————
# 背景：2026-09-16 在 128 门语料上统计 83 条未命中样本的难度流向，发现
#   intermediate → beginner 26 条、advanced → beginner 10 条，
#   实际返回 beginner 占 52%，远高于课程库中入门档 38.3% 的占比——
#   即无论用户要什么难度，检索结果都系统性偏向最基础那门课（"入门档塌陷"）。
#
# 根因之一：ingest._build_doc 写入的是英文难度值（"难度：advanced"），
#   而用户 query 说的是中文（"精通""调优"）。中文 embedding 对 "advanced"
#   这种英文词的表征很弱，难度信号在语义层几乎失联；且入门课 doc 普遍更长、
#   用词更泛，在 IDF 加权下天然占优。
#
# 方案：从 query 中识别**明确**的难度信号词，命中时对同档位课程加分。
#   之所以做成"结构化读取 metadata.level"而不是再改 doc 文本，是因为 metadata
#   里已有 level 字段，无需重建索引即可验证效果。
#
# 设计取舍：
#   1) 只加分不惩罚——惩罚会误伤"名字含原理但其实是入门课"这类样本，
#      先单独验证加分的净收益，不够再加惩罚项。
#   2) 词表刻意宁缺毋滥——"实战/项目/架构/优化"这类跨档歧义词一律不收，
#      宁可漏判也不误判；且词表取自通用中文技术学习用语，不来自评测集。
#   3) 信号冲突（如"从零到精通"）时整体不加权，交由 reranker 自行裁决。
_LEVEL_QUERY_SIGNALS = {
    "beginner": ("入门", "零基础", "从零", "初学", "小白", "新手", "菜鸟", "打下基础"),
    "intermediate": ("进阶", "进一步提升", "提高班"),
    # 注意：不收裸词"深度"——"深度学习"是学科名而非难度；同理不收"实战/项目/架构/优化"
    # 这类跨档歧义词。宁可漏判（降级为不加分）也不误判（把分加给错误档位）。
    "advanced": ("精通", "高阶", "高级", "底层", "底层原理", "原理", "源码",
                 "调优", "剖析", "极致", "骨灰级", "深度剖析", "深入理解"),
}

# 命中同档位的加分。量纲参照：cross-encoder 对近邻候选的分差常在 0.1~0.5，
# 1.2 足以在近邻间完成纠偏，但不足以推翻 reranker 的强信号（避免重演 v1 的失败）。
_LEVEL_MATCH_BONUS = 1.2


def _detect_level_intent(query: str):
    """识别 query 中明确的难度意图；无信号或信号冲突时返回 None（不加权）。"""
    q = query or ""
    hits = [lv for lv, words in _LEVEL_QUERY_SIGNALS.items() if any(w in q for w in words)]
    if len(hits) != 1:  # 0 个=保守不加权；≥2 个=语义冲突，交给重排自行裁决
        return None
    return hits[0]


def _level_aware_enabled() -> bool:
    return os.getenv("RAG_LEVEL_AWARE", "0") == "1"


def _term_ok(t: str) -> bool:
    """词元过滤：单字/数字/标点噪声大，只保留有区分力的语义词元。"""
    t = t.strip()
    if not t or t.isdigit():
        return False
    if t.isascii():  # vue3 / go / es6 / jvm 等英文技术词
        return t.isalnum() and len(t) >= 2
    return len(t) >= 2  # 中文词元 jieba 已切成语义词，单字（的/课/与…）丢弃


def _keywords(text: str) -> set:
    return {t for t in jieba.lcut(text or "") if _term_ok(t)}


# —— IDF 统计（惰性缓存）——
# 语料规模变化（课程增删）时自动重算；对既有课程的原位更新（count 不变）不敏感，
# 但关键词区分力对文档微调并不苛求，TTL 级别的滞后可接受。
_idf_lock = threading.Lock()
_idf_state = {"count": -1, "idf": {}}


def _get_idf(collection) -> dict:
    with _idf_lock:
        n = collection.count()
        if _idf_state["count"] == n:
            return _idf_state["idf"]
        data = collection.get(include=["documents"])
        total = max(len(data["documents"]), 1)
        df = {}
        for doc in data["documents"]:
            for t in _keywords(doc):
                df[t] = df.get(t, 0) + 1
        _idf_state["idf"] = {t: math.log(1 + total / (1 + c)) for t, c in df.items()}
        _idf_state["count"] = n
        return _idf_state["idf"]


def _oov_excluded() -> bool:
    """是否把"语料外词"从 IDF 归一化分母中剔除（RAG_IDF_OOV=1 启用）。"""
    return os.getenv("RAG_IDF_OOV", "0") == "1"


def _kw_idf_score(q_keys: set, doc_keys: set, idf: dict, total_docs: int,
                  exclude_oov: bool = None) -> float:
    """IDF 加权的关键词命中分，归一化到 [0,1]。

    exclude_oov 显式传入 True/False 可绕过环境变量 RAG_IDF_OOV，用于在同一次检索里
    同时算出两种口径（离线扫参时一次推理即可覆盖全部组合，见 RAG_SCORE_DUMP）。

    旧实现：`default = log(1+total_docs)` 给语料外词记最高 IDF（128 篇时≈4.86，
    而常见词"入门"仅 1.27、"实战" 2.25），这些词命中不了任何文档却占据最大分母，
    把有效词的贡献稀释掉。实测问题（2026-09-16）：

      - 改写注入语料外词："Docker 实战课"→"Docker容器化实战" 分母 5.36→8.47，
        同等命中下 kw_idf 被稀释到 63%；"微服务课哪家强"→"微服务实战开发哪家强" 稀释到 61%；
      - 更严重的是**不改写也中毒**：口语 query 的虚词"怎么/哪家/课有"全是语料外词，
        如"ES 搜索课怎么选" denom=13.00，其中"ES"+"怎么"占 9.72（75%），
        真正有效词"搜索"只占 3.28。

    新实现（RAG_IDF_OOV=1）：语料外词不计入分母——它命中不了任何文档，没有区分力，
    保留在分母里只会惩罚所有候选。全部词都在语料外时（纯口语）退化为等权命中率，
    避免关键词信号直接归零。
    """
    if not q_keys:
        return 0.0
    if exclude_oov is None:
        exclude_oov = _oov_excluded()
    if not exclude_oov:
        default = math.log(1 + total_docs)
        q_idf = {t: idf.get(t, default) for t in q_keys}
        denom = sum(q_idf.values())
        if denom <= 0:
            return 0.0
        hit = sum(v for t, v in q_idf.items() if t in doc_keys)
        return hit / denom
    # 只统计语料中出现过的词
    q_idf = {t: idf[t] for t in q_keys if t in idf}
    if q_idf:
        denom = sum(q_idf.values())
        if denom <= 0:
            return 0.0
        hit = sum(v for t, v in q_idf.items() if t in doc_keys)
        return hit / denom
    # 全部为语料外词：退化为等权命中率，保留一丝关键词信号
    return len(q_keys & doc_keys) / len(q_keys)


async def retrieve(query: str, top_k: int = 5, fusion: str = "", q_vec: list = None) -> list:
    """
    返回重排后的课程候选列表，每项结构：
        {"id", "text", "metadata": {course_id, course_name, ...}, "score"}
    metadata 含课程展示所需的全部字段，可直接给前端卡片用。

    fusion:
        "v3"（默认）— IDF 加权命中分 + 课名完整命中决定性加分
        "idf"       — v2：仅 IDF 加权归一化命中分，有界融合
        "add"       — v1：裸词元命中数 ×0.05，保留作 A/B 对照
    q_vec:
        可选，预计算好的 query 向量。批量评测时一次 embed 全部 query 后逐条传入，
        避免 N 次串行 embed（单条 retrieve 默认会自行 embed）。
    """
    fusion = fusion or os.getenv("RAG_FUSION", "v3")
    loop = asyncio.get_event_loop()
    collection = chroma_store.get_collection()
    total = collection.count()
    if total == 0:
        return []

    # 1) 向量召回（多取一些供重排精筛）
    if q_vec is None:
        q_vec = await loop.run_in_executor(None, embeddings.embed, [query])
    n = min(max(top_k * 4, 12), total)
    res = collection.query(query_embeddings=q_vec, n_results=n)
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    # 2) 组装候选 + 计算关键词命中分（v1 命中数 / v2 IDF 归一分 / v3 课名完整命中）
    q_keys = _keywords(query)
    q_norm = re.sub(r"\s+", "", query or "")
    need_idf = fusion in ("idf", "v3")
    idf = _get_idf(collection) if need_idf else {}
    candidates = []
    for vi, (cid, doc, meta) in enumerate(zip(ids, docs, metas), start=1):
        d_keys = _keywords(doc)
        c = {
            "id": cid, "text": doc, "metadata": meta,
            "kw_hit": len(q_keys & d_keys),
            "kw_idf": _kw_idf_score(q_keys, d_keys, idf, total) if need_idf else 0.0,
            # 与 kw_idf 同一次计算出的「剔除语料外词」口径，仅用于离线扫参/归因，
            # 不参与线上打分。need_idf 为假时恒为 0。
            "kw_idf_oov": (_kw_idf_score(q_keys, d_keys, idf, total, exclude_oov=True)
                           if need_idf else 0.0),
            "vec_rank": vi,          # 纯向量召回名次（reranker 之前），供归因分析
            "name_exact": False,
        }
        if fusion == "v3":
            # 课名词元被查询完整覆盖 → 用户念的就是这门课
            n_keys = _keywords(meta.get("course_name", ""))
            c["name_exact"] = bool(n_keys) and n_keys.issubset(q_keys)
            # 课名与查询去空格后双向子串包含 → 补 name_exact 方向过严的盲区
            # （用户只念出课名一段，如 "微服务架构设计" 是 "微服务架构设计实战" 子串）
            cname_norm = re.sub(r"\s+", "", meta.get("course_name", "") or "")
            c["name_substr"] = bool(cname_norm) and (
                cname_norm in q_norm
                or (len(q_norm) >= 3 and q_norm in cname_norm)
            )
        candidates.append(c)

    # 3) 交叉编码器精排
    reranked = await loop.run_in_executor(
        None, reranker.rerank, query, candidates, top_k * 2
    )

    # 4) 融合重排分与关键词信号，输出最终 top_k
    lvl_intent = None   # v3 分支才会赋值；dump 时统一读取
    if fusion == "add":  # v1 旧策略（A/B 对照用）
        for c in reranked:
            c["score"] = c.get("rerank_score", 0.0) + 0.05 * c.get("kw_hit", 0)
    elif fusion == "idf":  # v2：IDF 加权，命中分有界 [0,1]，只助推不翻盘
        for c in reranked:
            c["score"] = c.get("rerank_score", 0.0) + _FUSION_ALPHA * c.get("kw_idf", 0.0)
    else:  # v3：IDF 助推 + 课名完整命中 / 子串命中决定性加分
        # 难度意图无条件识别（纯字符串匹配，零成本），但只在开关打开时才参与打分。
        # 这样 RAG_SCORE_DUMP 一次就能同时拿到「有 / 无难度加权」两种口径的结果。
        lvl_intent = _detect_level_intent(query)
        lvl_apply = lvl_intent if _level_aware_enabled() else None
        for c in reranked:
            score = (c.get("rerank_score", 0.0)
                     + _FUSION_ALPHA * c.get("kw_idf", 0.0)
                     + (_NAME_EXACT_BONUS if c.get("name_exact")
                        else (_NAME_SUBSTR_BONUS if c.get("name_substr") else 0.0)))
            if lvl_apply and (c.get("metadata") or {}).get("level") == lvl_apply:
                score += _LEVEL_MATCH_BONUS
            c["score"] = score
    reranked.sort(key=lambda c: c["score"], reverse=True)

    # ——— 打分分量 dump（RAG_SCORE_DUMP=1，仅离线评测用）———
    # 返回全部候选（不做 top_k 截断），并保留 rerank 原始分、两种 IDF 口径、课名命中
    # 标志、难度档位、向量召回名次等分量。这样 α、RAG_IDF_OOV、难度加权、课名加分等
    # 组合可以在「一次模型推理」之后离线穷举，不必为每种配置重跑一遍 embedding+rerank。
    # 线上默认关闭，行为与原先完全一致。
    if os.getenv("RAG_SCORE_DUMP") == "1":
        for c in reranked:
            c["_lvl_intent"] = lvl_intent
        return reranked
    return reranked[:top_k]


# ————————————————————————————— 第一阶段：Vector + BM25 双路候选生成（RRF 融合）—————————————————————————————
# 背景：原链路只有向量召回，IDF 只对「已进入向量候选」的课程重打分，无法召回被向量
# 漏掉的课程。第一阶段把 BM25 作为独立倒排通道引入「候选生成」层，与向量召回并列，
# 用 RRF 名次融合（不直接相加 vector/BM25 score，二者量纲不同且无物理意义）。
#
# 职责边界（本阶段）：
#   Vector Recall(Top40) + BM25 Recall(Top20) → RRF → Candidate(Top50)
#   → IDF Feature（仅排序特征，不再承担独立召回） → BGE Reranker → TopK
# 原 retrieve() 保留不动，作为 A/B 的 baseline。


_VEC_TOP = 40     # 向量召回窗口（固定，与 top_k 解耦，给 BM25 融合留空间）
_BM25_TOP = 20    # BM25 召回窗口；作为向量召回的补充，避免引入过多词法噪声
_RRF_TOP = 50     # RRF 融合后保留的候选数（Top40~60 建议区间）
_HYBRID_RRF_K = 60  # RRF 平滑常数（沿用默认稳定值）
_HYBRID_WEIGHTS = [1.0, 0.3]  # 更信任语义召回，BM25 仅补充明确技术词/课程名命中


def _rrf_fuse(ranked_lists, k: int = 60, weights: list = None) -> list:
    """RRF 名次融合：score(doc) = Σ_w  w / (k + rank)，rank 从 1 起。

    ranked_lists: 若干路已排序的候选列表（元素需含 'id'）。
    不直接相加各路 score（量纲不同 + 随 query 漂移），只取名次，天然免疫量纲与漂移。
    """
    ws = list(weights or [1.0] * len(ranked_lists))
    fused = {}
    for li, lst in enumerate(ranked_lists):
        w = ws[li] if li < len(ws) else ws[-1]
        for rank, c in enumerate(lst, start=1):
            cid = str(c.get("id"))
            if cid not in fused:
                fused[cid] = dict(c)          # 复制，避免污染上游候选对象
                fused[cid]["_rrf"] = 0.0
            fused[cid]["_rrf"] += w / (k + rank)
    for c in fused.values():
        c["_rrf"] = round(c["_rrf"], 6)
    return sorted(fused.values(), key=lambda c: c["_rrf"], reverse=True)


async def _hybrid_candidates(query: str, q_vec: list = None):
    """候选生成层：Vector Top40 + BM25 Top20 → RRF Top50。

    返回三元组 (vec_cands, bm25_cands, rrf_cands)，每项为含 'id' 的候选字典列表，
    供检索漏斗诊断（Vector/BM25/RRF 各自的 TopN 命中）与 retrieve_hybrid 复用。
    不在此做重排，避免评测与生产重复推理。
    """
    collection = chroma_store.get_collection()
    total = collection.count()
    if total == 0:
        return [], [], []
    loop = asyncio.get_event_loop()

    # 1) 向量召回 Top40
    if q_vec is None:
        q_vec = await loop.run_in_executor(None, embeddings.embed, [query])
    n_vec = min(_VEC_TOP, total)
    res = collection.query(query_embeddings=q_vec, n_results=n_vec)
    vec_cands = [{"id": cid, "text": doc, "metadata": meta}
                 for cid, doc, meta in zip(res["ids"][0], res["documents"][0],
                                           res["metadatas"][0])]

    # 2) BM25 独立倒排召回 Top20
    bm25_cands = bm25_engine.search(query, top_k=min(_BM25_TOP, total))

    # 3) RRF 融合，保留 Top50
    rrf_cands = _rrf_fuse([vec_cands, bm25_cands],
                          k=_HYBRID_RRF_K, weights=_HYBRID_WEIGHTS)[:min(_RRF_TOP, total)]
    return vec_cands, bm25_cands, rrf_cands


async def retrieve_hybrid(query: str, top_k: int = 5, fusion: str = "",
                          q_vec: list = None, candidates: list = None) -> list:
    """第一阶段新检索链路：Vector + BM25 → RRF → IDF Feature → BGE Reranker。

    返回结构与 retrieve 完全一致（{id,text,metadata,score}），生产侧可无缝替换。
    融合口径沿用 v3（IDF 助推 + 课名命中 + 难度加权），但候选集来自 RRF 双路，
    IDF 从「独立召回」降级为「候选阶段的排序 Feature」。

    fusion: 同 retrieve，默认 v3；"idf" 仅 IDF 助推（不做课名/难度加权），A/B 对照用。
    candidates: 可选，预先由 _hybrid_candidates 算好的 RRF 候选（评测漏斗复用，避免重复
                chroma 查询 + bm25 检索）；不传则内部自行计算。
    """
    fusion = fusion or os.getenv("RAG_FUSION", "v3")
    loop = asyncio.get_event_loop()
    collection = chroma_store.get_collection()
    total = collection.count()
    if total == 0:
        return []

    if candidates is None:
        _, _, candidates = await _hybrid_candidates(query, q_vec=q_vec)

    # 2) 计算关键词 IDF 命中分 + 课名命中标志（与 retrieve v3 同口径）
    q_keys = _keywords(query)
    q_norm = re.sub(r"\s+", "", query or "")
    need_idf = fusion in ("idf", "v3")
    idf = _get_idf(collection) if need_idf else {}
    for c in candidates:
        d_keys = _keywords(c["text"])
        c["kw_idf"] = _kw_idf_score(q_keys, d_keys, idf, total) if need_idf else 0.0
        if fusion == "v3":
            n_keys = _keywords((c.get("metadata") or {}).get("course_name", ""))
            c["name_exact"] = bool(n_keys) and n_keys.issubset(q_keys)
            cname_norm = re.sub(r"\s+", "", (c.get("metadata") or {}).get("course_name", "") or "")
            c["name_substr"] = bool(cname_norm) and (
                cname_norm in q_norm or (len(q_norm) >= 3 and q_norm in cname_norm)
            )
        else:
            c["name_exact"] = False
            c["name_substr"] = False

    # 3) BGE 交叉编码器精排（候选集比原链路更大：RRF Top50 vs 向量 Top20）
    reranked = await loop.run_in_executor(
        None, reranker.rerank, query, candidates, top_k * 2
    )

    # 4) 融合重排分与关键词/课名/难度信号（与 retrieve v3 完全一致）
    lvl_intent = _detect_level_intent(query) if fusion == "v3" else None
    lvl_apply = lvl_intent if _level_aware_enabled() else None
    for c in reranked:
        score = (c.get("rerank_score", 0.0)
                 + _FUSION_ALPHA * c.get("kw_idf", 0.0)
                 + (_NAME_EXACT_BONUS if c.get("name_exact")
                    else (_NAME_SUBSTR_BONUS if c.get("name_substr") else 0.0)))
        if lvl_apply and (c.get("metadata") or {}).get("level") == lvl_apply:
            score += _LEVEL_MATCH_BONUS
        c["score"] = score
    reranked.sort(key=lambda c: c["score"], reverse=True)
    return reranked[:top_k]


# ————————————————————————————— Phase 2：Intent Recall + Query Understanding —————————————————————————————
# 背景：Phase 2A 的 beginner penalty 已被否证（把向下塌陷 1:1 置换成向上过难，净负）。
# Phase 2 改为在「候选生成」层引入 Query Understanding，补两类缺口：
#   1) Technical Query → BM25：把原始 query 里的技术实体/关键词提出来做倒排，减少口语虚词
#      对 BM25 的稀释（重口语 query 整句无技术名词 → BM25 结构性召不回）；
#   2) Structured Intent → Intent Recall：把「部署/自动部署/写后台」等意图词经有限 domain 映射
#      到 category → 召回该领域课程，补「无技术词但有意图」的语义缺口。
# 原 original_query 全程保留：Vector Recall 仍用原句；level 只作弱 Feature，不硬过滤。
#
# 职责边界：不改 embedding / BGE reranker / 原 retrieve/retrieve_hybrid；Intent Recall 是新增的
# 第三路召回，与 Vector、BM25 并列进 RRF（等权，第一版不调权）。

# Intent Recall 的候选缓存（与 bm25._get_index 同款：按 Chroma count 失效）
_intent_cache = {"count": -1, "docs": None}
_intent_lock = threading.Lock()


def _intent_docs() -> list:
    global _intent_cache
    with _intent_lock:
        collection = chroma_store.get_collection()
        n = collection.count()
        if _intent_cache["count"] == n and _intent_cache["docs"] is not None:
            return _intent_cache["docs"]
        data = collection.get(include=["documents", "metadatas"])
        docs = [{"id": cid, "text": doc, "metadata": meta}
                for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])]
        _intent_cache["docs"] = docs
        _intent_cache["count"] = n
        return docs


def intent_recall(intent: dict, top_k: int = 40) -> list:
    """基于 Structured Intent 的领域/技术召回。

    打分（可追踪、无复杂公式）：
      - category 命中 intent.domain        → +2.0（领域硬匹配）
      - technology 词命中 course_name/tags → 每个 +1.0（技术软匹配）
    返回 score>0 的候选按 score 降序 top_k。level 不参与打分（不硬过滤难度）。
    """
    domain = set(intent.get("domain", []))
    techs = [t.lower() for t in intent.get("technology", []) if t]
    scored = []
    for c in _intent_docs():
        meta = c["metadata"]
        cat = meta.get("category", "")
        score = 0.0
        if cat in domain:
            score += 2.0
        if techs:
            text = (meta.get("course_name", "") + " " + meta.get("tags", "")).lower()
            for t in techs:
                if t in text:
                    score += 1.0
        if score > 0:
            scored.append({"id": c["id"], "text": c["text"],
                           "metadata": meta, "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


async def retrieve_phase2(query: str, top_k: int = 5, fusion: str = "",
                          q_vec: list = None) -> list:
    """Phase 2 检索链路：Original→Vector + Technical→BM25 + Intent→Intent Recall，三路 RRF。

    与 retrieve_hybrid 返回结构完全一致（{id,text,metadata,score}），生产侧可无缝替换。
    候选生成层换成三路（Vector Top40 + BM25(Technical) Top20 + Intent Top20 → RRF Top50），
    重排层（IDF Feature + BGE Reranker + v3 融合）直接复用 retrieve_hybrid，不改排序逻辑。
    """
    from app.rag import query_understanding as qu

    loop = asyncio.get_event_loop()
    collection = chroma_store.get_collection()
    total = collection.count()
    if total == 0:
        return []

    # 1) Query Understanding（original_query 原样保留，不改写）
    intent = qu.understand(query)

    # 2) Vector Recall（Original query）
    if q_vec is None:
        q_vec = await loop.run_in_executor(None, embeddings.embed, [query])
    n_vec = min(_VEC_TOP, total)
    res = collection.query(query_embeddings=q_vec, n_results=n_vec)
    vec_cands = [{"id": cid, "text": doc, "metadata": meta}
                 for cid, doc, meta in zip(res["ids"][0], res["documents"][0],
                                           res["metadatas"][0])]

    # 3) BM25 Recall（Technical query；无技术词时回退 original）
    tq = intent.get("technical_query", "").strip()
    bm25_cands = bm25_engine.search(tq if tq else query, top_k=min(_BM25_TOP, total))

    # 4) Intent Recall（Structured intent）
    intent_cands = intent_recall(intent, top_k=min(_BM25_TOP, total))

    # 5) RRF 三路融合，保留 Top50
    rrf_cands = _rrf_fuse([vec_cands, bm25_cands, intent_cands],
                          k=_HYBRID_RRF_K,
                          weights=[1.0, 1.0, 1.0])[:min(_RRF_TOP, total)]

    # 6) 复用 retrieve_hybrid 的重排（IDF Feature + BGE + v3 融合），不改排序逻辑
    return await retrieve_hybrid(query, top_k=top_k, fusion=fusion,
                                 q_vec=q_vec, candidates=rrf_cands)
