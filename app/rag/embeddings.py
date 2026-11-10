# 中文嵌入包装：使用本机已下载的 text2vec-base-chinese（离线、不联网）。
#
# 维度 768。与 Chroma 集合的向量维度绑定 —— 一旦入库，不可随意更换模型，
# 否则已有向量与新查询不在同一语义空间，检索失效（需重建集合）。
#
# 线程安全（启动预热 / 全量重建 / 首个请求可能并发触发懒加载）：
# lru_cache 不加锁，两个线程同时构造 SentenceTransformer 会互相踩踏
# （实测报 "Cannot copy out of meta tensor"，一方拿到未初始化完的权重）。
# 改为双重检查锁：只有一个线程真正加载，其余等锁后直接复用。
import os
import threading

# 默认值与 docker-compose 中 EMBEDDING_MODEL 对齐；必须是本机 HF 缓存里存在的仓库名。
_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "shibing624/text2vec-base-chinese")

_MODEL = None
_MODEL_LOCK = threading.Lock()


def _model():
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                from sentence_transformers import SentenceTransformer

                # local_files_only=True：绝不去 HuggingFace 下载，只用本机缓存（满足"不下载"要求）。
                _MODEL = SentenceTransformer(_MODEL_NAME, local_files_only=True)
    return _MODEL


def embed(texts):
    """texts: str | list[str] -> list[list[float]]（已 L2 归一化，便于余弦相似度）。"""
    if isinstance(texts, str):
        texts = [texts]
    vecs = _model().encode(texts, normalize_embeddings=True)
    return vecs.tolist()
