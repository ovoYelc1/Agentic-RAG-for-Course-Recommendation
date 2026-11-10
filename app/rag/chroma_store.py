# Chroma 向量库连接与集合管理（server 模式，连 docker-compose 里的 chromadb 服务）。
import os

_COLLECTION = "itcloud_courses"


def get_client():
    import chromadb

    # 安全说明：chromadb 1.x（Rust 版）已移除内建鉴权，CHROMA_SERVER_AUTHN_* 等旧
    # 配置被服务端静默忽略。向量库的安全边界由 docker-compose 提供：
    # chromadb 服务不映射宿主机端口，仅 docker 内网可达（AI 服务专用）。
    return chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "chromadb"),
        port=int(os.getenv("CHROMA_PORT", 8000)),
    )


def get_collection():
    """获取（不存在则创建）课程集合，使用余弦距离。"""
    client = get_client()
    return client.get_or_create_collection(
        name=_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def count() -> int:
    try:
        return get_collection().count()
    except Exception as e:
        print(f"[Chroma] 连接失败: {e}")
        return 0


def upsert_course(doc_id: str, text: str, metadata: dict, embedding: list):
    """写入/更新一门课程的向量与元数据。

    doc_id 必须与 ingest.py 一致，形如 'course:{id}'，否则无法与已有索引对应。
    embedding 由调用方用 app.rag.embeddings.embed 计算（server 模式不自动嵌入）。
    """
    get_collection().upsert(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[metadata],
    )


def delete_course(doc_id: str):
    """从课程集合中删除一门课（下架/归档时调用）。"""
    try:
        get_collection().delete(ids=[doc_id])
    except Exception as e:
        print(f"[Chroma] 删除失败 {doc_id}: {e}")
