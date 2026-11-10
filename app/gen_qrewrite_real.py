#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实 qrewrite 词库补全生成器（独立脚本，不修改项目代码）。
- 课程来源：Chroma 元数据（course_id / course_name / description），避免再打后端。
- 抽取：复用 consumer._extract，命中 tagcache 则秒回（已抽过的 56 门不重复花钱），
        未命中的约 34 门真实调用 glm-4.5-air；连接失败按指数退避重试，规避限流。
- 写入：复用 consumer._write_intent，写 DB2 的 qrewrite:{别名}->规范名 + intent 缓存。
目标：把真实 LLM 词库补到接近全量（90 门），供盲测使用。
"""
import os
import sys
sys.path.insert(0, "/app")  # 容器内 app 包位于 /app/app，需将 /app 加入搜索路径
import time
import asyncio

from app.rag import chroma_store
from app.mq import consumer

REDIS_PW = os.getenv("REDIS_PASSWORD")
MAX_RETRIES = 4
BACKOFF = [3, 6, 12, 20]
INTER_COURSE_DELAY = 1.2


async def extract_with_retry(name, desc):
    for i in range(MAX_RETRIES):
        try:
            tags = await consumer._extract(name, desc)
            if tags and (tags.get("synonyms") or tags.get("aliases")):
                return tags
        except Exception as e:
            print(f"    [retry {i+1}] 抽取异常: {e!r}")
        if i < MAX_RETRIES - 1:
            time.sleep(BACKOFF[i])
    return {}


async def main():
    col = chroma_store.get_collection()
    data = col.get(include=["metadatas"])
    courses = []
    for cid, m in zip(data["ids"], data["metadatas"]):
        courses.append({
            "course_id": m.get("course_id", cid.replace("course:", "")),
            "name": m.get("course_name", ""),
            "desc": m.get("description", "") or "",
        })
    print(f"语料课程数: {len(courses)}")

    done = 0
    wrote_alias_courses = 0
    total_alias_keys = 0
    t0 = time.time()
    for idx, c in enumerate(courses, 1):
        tags = await extract_with_retry(c["name"], c["desc"])
        if not tags:
            print(f"  [{idx}/{len(courses)}] {c['name']}: 抽取失败，跳过")
            continue
        synonyms = tags.get("synonyms") if isinstance(tags.get("synonyms"), list) else []
        aliases = tags.get("aliases") if isinstance(tags.get("aliases"), list) else []
        # _write_intent 内部用 asyncio.to_thread 写 Redis，这里直接调度
        await asyncio.to_thread(
            consumer._write_intent, c["course_id"], tags, synonyms, aliases, c["name"]
        )
        done += 1
        if aliases:
            wrote_alias_courses += 1
            total_alias_keys += len(aliases)
        if idx % 10 == 0:
            print(f"  [{idx}/{len(courses)}] 已处理，累计写别名课程 {wrote_alias_courses} 门 / {total_alias_keys} 键")
        time.sleep(INTER_COURSE_DELAY)

    import redis
    r = redis.Redis(host="redis", port=6379, db=2, password=REDIS_PW, decode_responses=True)
    qk = r.keys("qrewrite:*")
    print(f"\n完成：处理 {done}/{len(courses)} 门，含别名课程 {wrote_alias_courses} 门")
    print(f"qrewrite 总键数: {len(qk)}，耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
