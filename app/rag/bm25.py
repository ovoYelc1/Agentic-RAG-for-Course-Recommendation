# 独立 BM25 / 倒排检索通道（第一阶段：候选生成）。
#
# 背景：原检索链路只有向量召回（Chroma collection.query），jieba+IDF 只是对
# 已进入向量候选的课程「重打分」，没有独立的关键词召回能力——被向量漏掉的课程
# （尤其近重复簇、口语/技术词命中但语义相似度不高的查询）永远进不了候选集。
# 本模块补一条真正独立的倒排通道：用 BM25 对课程字段做词频-逆文档频率打分，
# 与向量召回在「候选生成」层并列，二者再走 RRF 融合（见 retriever.retrieve_hybrid）。
#
# 设计原则（第一版）：
#   - 正确性与可维护性优先，不引入复杂算法（无 NLP 扩展、无 synonym 表）；
#   - 索引从 Chroma 集合本身读取（与向量召回同源、天然一致），不碰 MySQL；
#   - 惰性构建 + 线程安全缓存，按 Chroma count 失效（语料增删时自动重建）；
#   - 分词口径与 retriever._keywords 保持一致（jieba + 同款词元过滤），避免两套
#     检索通道对同一文本切出不同词、导致融合错位。
import math
import threading

import jieba

from app.rag import chroma_store

# jieba 日志太吵，关掉（与 retriever 一致）
jieba.setLogLevel(20)

# ============================ 字段权重 ============================
# 用户指定：course_name / tags / course_code 高；category 中高；
#           level_cn / description 中；teacher_name 低。
# 量纲取整数倍便于解读；高=3.0 中高=2.0 中=1.5 低=0.5。
FIELD_WEIGHTS = {
    "course_name": 3.0,
    "tags": 3.0,
    "course_code": 3.0,
    "category": 2.0,
    "level_cn": 1.5,
    "description": 1.5,
    "teacher_name": 0.5,
}

# 参与检索的字段顺序（与 FIELD_WEIGHTS 一致，保证确定性）
_FIELDS = ["course_name", "tags", "course_code", "category", "level_cn",
           "description", "teacher_name"]

# ============================ BM25 参数 ============================
K1 = 1.5      # 词频饱和参数（标准默认）
B = 0.75      # 文档长度归一化参数（标准默认）

# level → 中文难度标签（level_cn 字段用；Chroma 只存英文枚举）
LEVEL_CN = {
    "beginner": "入门",
    "intermediate": "进阶",
    "advanced": "精通",
}


def _term_ok(t: str) -> bool:
    """词元过滤：单字/数字/标点噪声大，只保留有区分力的语义词元。

    与 retriever._term_ok 完全同款：ascii 技术词（vue3/go/es6/jvm）要求
    len>=2；中文词元 jieba 已切成语义词，单字（的/课/与…）丢弃。
    """
    t = t.strip()
    if not t or t.isdigit():
        return False
    if t.isascii():
        return t.isalnum() and len(t) >= 2
    return len(t) >= 2


def _tokenize(text: str) -> list:
    return [t for t in jieba.lcut(text or "") if _term_ok(t)]


def _tags_from_doc(doc: str) -> str:
    """兜底：从文档文本的「标签：」行提取标签（元数据缺 tags 时用）。

    文档文本由 ingest._build_doc 固定拼装，格式确定：
      课程名称：…\n分类：… 难度：…\n讲师：…\n价格：…\n标签：…\n简介：…
    """
    for line in (doc or "").splitlines():
        if line.startswith("标签："):
            return line[len("标签："):].strip()
    return ""


def _field_texts(meta: dict, doc: str) -> dict:
    """把一门课拆成 BM25 检索所需的各字段文本（优先元数据，tags 缺省回退文档行）。"""
    tags = meta.get("tags") or _tags_from_doc(doc)
    return {
        "course_name": meta.get("course_name", ""),
        "tags": tags,
        "course_code": meta.get("course_code", ""),
        "category": meta.get("category", ""),
        "level_cn": LEVEL_CN.get(meta.get("level", ""), ""),
        "description": meta.get("description", ""),
        "teacher_name": meta.get("teacher_name", ""),
    }


class _FieldIndex:
    """单字段的倒排索引 + BM25 打分。"""

    def __init__(self, token_lists):
        self.n = len(token_lists)
        self.doc_len = [len(tl) for tl in token_lists]
        self.avg_dl = (sum(self.doc_len) / self.n) if self.n else 1.0
        # term -> list[(doc_idx, tf)]
        self.postings = {}
        for di, tl in enumerate(token_lists):
            tf = {}
            for t in tl:
                tf[t] = tf.get(t, 0) + 1
            for t, f in tf.items():
                self.postings.setdefault(t, []).append((di, f))

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        # BM25 经典 IDF：df 越大区分力越低；df=0 时调用方已跳过
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))


class BM25Index:
    """多字段加权 BM25 索引。"""

    def __init__(self, docs):
        # docs: list[{"id","text","metadata"}]
        self.docs = docs
        self.field_index = {}
        for f in _FIELDS:
            token_lists = [_tokenize(_field_texts(d["metadata"], d["text"])[f])
                           for d in docs]
            self.field_index[f] = _FieldIndex(token_lists)

    def search(self, query: str, top_k: int = 40) -> list:
        """返回按 BM25 分降序的 top_k 候选，结构同 retriever：{id,text,metadata,score}。"""
        qtoks = _tokenize(query)
        if not qtoks:
            return []
        scores = {}
        for f in _FIELDS:
            w = FIELD_WEIGHTS[f]
            fi = self.field_index[f]
            for t in qtoks:
                post = fi.postings.get(t)
                if not post:
                    continue
                idf = fi.idf(t)
                for di, tf in post:
                    dl = fi.doc_len[di]
                    # BM25 词频饱和 + 长度归一
                    tf_norm = tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl / fi.avg_dl))
                    scores[di] = scores.get(di, 0.0) + w * idf * tf_norm
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out = []
        for di, sc in ranked:
            d = self.docs[di]
            out.append({"id": d["id"], "text": d["text"],
                        "metadata": d["metadata"], "score": sc})
        return out

    def score_all(self, query: str) -> dict:
        """返回所有命中 query 词元的文档的 {doc_id: bm25_score}（含未进 Top40 的低分候选）。

        与 search 同口径打分（字段权重 / K1 / B 完全一致），但不截断、不排序成候选结构。
        供排序诊断阶段对任意候选（真值课程 / 误判 Top1）反查其 BM25 分——search 只返回
        Top40，真值若落在 Top40 之外就拿不到分，故需此全量口径。改动面最小：独立实现，
        不动 search 的既有逻辑。
        """
        qtoks = _tokenize(query)
        if not qtoks:
            return {}
        scores = {}
        for f in _FIELDS:
            w = FIELD_WEIGHTS[f]
            fi = self.field_index[f]
            for t in qtoks:
                post = fi.postings.get(t)
                if not post:
                    continue
                idf = fi.idf(t)
                for di, tf in post:
                    dl = fi.doc_len[di]
                    tf_norm = tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl / fi.avg_dl))
                    scores[di] = scores.get(di, 0.0) + w * idf * tf_norm
        return {self.docs[di]["id"]: sc for di, sc in scores.items()}


# ============================ 惰性缓存 ============================
# 与 retriever._get_idf 同款：按 Chroma count 失效；count 变化（课程增删）时重建。
# 对既有课程的原位更新（count 不变）不敏感，TTL 级滞后可接受。
_state = {"count": -1, "index": None}
_lock = threading.Lock()


def _get_index() -> BM25Index:
    global _state
    with _lock:
        collection = chroma_store.get_collection()
        n = collection.count()
        if _state["index"] is not None and _state["count"] == n:
            return _state["index"]
        data = collection.get(include=["documents", "metadatas"])
        docs = [
            {"id": cid, "text": doc, "metadata": meta}
            for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
        ]
        _state["index"] = BM25Index(docs)
        _state["count"] = n
        return _state["index"]


def search(query: str, top_k: int = 40) -> list:
    """对外检索入口：惰性构建索引后返回 top_k 候选。"""
    if not query:
        return []
    return _get_index().search(query, top_k=top_k)


def score_all(query: str) -> dict:
    """对外：返回 {doc_id: bm25_score} 全量命中（供排序诊断反查任意候选的 BM25 分）。"""
    if not query:
        return {}
    return _get_index().score_all(query)
