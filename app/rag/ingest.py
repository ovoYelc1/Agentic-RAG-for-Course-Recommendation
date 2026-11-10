# 课程入库：把 course_corpus.json 中的课程切分/嵌入后写入 Chroma。
#
# 语料来源：scripts/build_corpus.py 一次性从业务库导出（仅构建期运行，非服务运行时）。
# 运行（在容器内或宿主机均可）：
#   python -m app.rag.ingest
import json
import os

from app.rag import embeddings, chroma_store

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "course_corpus.json")


def _build_doc(c: dict) -> str:
    parts = [
        f"课程名称：{c.get('course_name', '')}",
        f"分类：{c.get('category', '')} 难度：{c.get('level', '')}",
        f"讲师：{c.get('teacher_name', '')}",
        f"价格：{c.get('price', '')}",
        f"标签：{_tags_text(c.get('tags'))}",
        f"简介：{c.get('description', '')}",
    ]
    return "\n".join(p for p in parts if p)


def _tags_text(tags) -> str:
    """把 tags 统一成纯文本（兼容 JSON 数组字符串 / 列表 / 空值）。"""
    if not tags:
        return ""
    if isinstance(tags, (list, tuple)):
        return " ".join(str(t) for t in tags)
    s = str(tags).strip()
    if not s or s in ("[]", "null"):
        return ""
    if s.startswith("["):
        try:
            import json as _json
            parsed = _json.loads(s)
            if isinstance(parsed, list):
                return " ".join(str(t) for t in parsed)
        except Exception:
            pass
        return s.strip("[]").replace('"', "").replace(",", " ")
    return s


def ingest_from_corpus(path: str = CORPUS_PATH) -> int:
    if not os.path.exists(path):
        print(f"[Ingest] 语料文件不存在: {path}")
        print("          请先运行：python scripts/build_corpus.py 生成课程语料。")
        return 0

    with open(path, "r", encoding="utf-8") as f:
        courses = json.load(f)
    if not courses:
        print("[Ingest] 语料为空，跳过。")
        return 0

    collection = chroma_store.get_collection()

    # 去重：先删除同 id 旧数据，再写入（支持重复入库更新）
    ids = [f"course:{c['id']}" for c in courses]
    try:
        collection.delete(ids=ids)
    except Exception:
        pass

    docs = [_build_doc(c) for c in courses]
    metas = [
        {
            "course_id": c.get("id"),
            "course_name": c.get("course_name", ""),
            "category": c.get("category", ""),
            "level": c.get("level", ""),
            "price": c.get("price", ""),
            "teacher_name": c.get("teacher_name", ""),
            "description": c.get("description", ""),
            # 课程编码与评分（离线语料 course_corpus.json 已含这两个字段），与线上路径对齐。
            # avg_rating 存【数值】（下游 _generate_match_reason 需做 >= 比较）。
            "course_code": c.get("course_code", ""),
            "avg_rating": (float(c["avg_rating"]) if c.get("avg_rating") is not None else 0.0),
            # tags 入库为元数据（供 BM25 独立倒排通道按字段加权检索；此前只进 doc 文本、
            # 不进 metadata，BM25 拿不到结构化标签）。
            "tags": _tags_text(c.get("tags")),
        }
        for c in courses
    ]
    vecs = embeddings.embed(docs)
    collection.add(ids=ids, documents=docs, embeddings=vecs, metadatas=metas)
    print(f"[Ingest] 已写入 {len(courses)} 门课程到 Chroma 集合 itcloud_courses。")
    return len(courses)


if __name__ == "__main__":
    n = ingest_from_corpus()
    print(f"done: {n}")
