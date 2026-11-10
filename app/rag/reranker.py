# 中文重排：交叉编码器精排（离线、不联网，模型来自本机 HF 缓存）。
#
# 交叉编码器对 (query, doc) 联合编码，排序质量显著高于纯向量召回，
# 是 RAG「先召后排」链路里的精排环节。
# 模型由 RERANKER_MODEL 环境变量指定（.env）：
#   - BAAI/bge-reranker-base：中文优化（2026-08-26 起，近重复区分力强）
#   - cross-encoder/mmarco-mMiniLMv2-L12-H384-v1：旧模型，英文为主，
#     拥挤语料实测对中文近重复区分力弱（克隆课 rerank 分反超真课）
import os
import functools

_MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


@functools.lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(_MODEL_NAME, local_files_only=True)


def rerank(query: str, docs: list, top_k: int = 5) -> list:
    """
    对候选文档按与 query 的相关性重排。

    docs: list[dict]，每个需含 'text' 字段（用于排序的文本）。
    返回：重排并截断到 top_k 的同一批 dict，附带 'rerank_score' 字段。
    """
    if not docs:
        return []
    pairs = [(query, d.get("text", "")) for d in docs]
    scores = _model().predict(pairs)
    for d, s in zip(docs, scores):
        d["rerank_score"] = float(s)
    docs.sort(key=lambda d: d["rerank_score"], reverse=True)
    return docs[:top_k]
