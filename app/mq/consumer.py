# 课程事件消费者（"预制菜"方案的核心离线模块）
#
# 消费 RabbitMQ 的 course.* 事件（由 Java 后端在课程新增/变更/下架时发出）：
#   1. 调本地 qwen2.5:7b 抽取结构化标签 + 用户常见搜索同义词（一次调用拿全）；
#   2. 把课程向量 + 元数据 upsert 进 Chroma（与 ingest.py 同一集合/同一 ID 规则）；
#   3. 把同义词 -> 标签 写入 Redis 意图缓存（独立 DB2），供搜索热路径秒级命中。
#
# 仅对已发布(published)课程建索引；未发布确保不进索引。下架/归档事件则移除。
# 全程本地模型，无 API 成本；失败有日志，不阻塞主流程（索引兜底靠每日全量重建）。
import os
import json
import asyncio
import hashlib
import logging

import aio_pika

from app.rag import chroma_store, embeddings

logger = logging.getLogger("course_mq_consumer")

COURSE_QUEUE = os.getenv("COURSE_MQ_QUEUE", "queue.course.ai")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", 5672))
RABBITMQ_USER = os.getenv("RABBITMQ_USERNAME")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASSWORD")
REDIS_INTENT_DB = int(os.getenv("REDIS_INTENT_DB", 2))

_SYSTEM_PROMPT = """你是一个课程标签与同义词抽取器，服务于在线教育平台的课程搜索引擎。
根据课程名和描述，抽取结构化信息并生成用户常见搜索词。
严格只输出一个 JSON 对象（不要任何解释、不要 markdown 代码块），格式：
{
  "field": "技术领域，如 后端开发 / 前端开发 / 人工智能 / 数据库 / 运维 等",
  "level": "难度，如 入门 / 中级 / 高级",
  "audience": "适用人群，如 学生 / 在职 / 转行",
  "tech": "核心技术栈关键词，逗号分隔，如 Spring Boot, MySQL",
  "synonyms": ["用户可能搜的常见短语1", "如 学Java", "写后台接口", "零基础学编程"],
  "aliases": ["能唯一指代本课的缩写/黑话/口语叫法", "如 K8s", "SB课", "Vue三"]
}
字段说明：
- synonyms：泛化搜索意图（可能命中多门课），写入意图缓存做领域/难度过滤；
- aliases：唯一指代本课的【缩写 / 黑话 / 口语别称】，用于离线 Query 改写词库。严格要求：
  1) 必须是用户真正会敲的【简短缩写或黑话】，例如 Kubernetes→"K8s"、Spring Boot→"SB课/Boot"、Vue3→"Vue三"、TypeScript→"TS"、JavaScript→"JS"、Python→"PY"（仅为格式示例，请按本课实际技术栈推断，不要照抄示例词）；
  2) 严禁只是「课程规范名 + 课/实战/教程」的同义改写（不要写「Vue3实战」「Java高阶课」「ES6+教程」这类）；
  3) 必须能【唯一指向本课】，不要写也会命中兄弟课的泛化词（例如本课是「Redis高性能缓存实战」就不要写模糊的「Redis缓存」，应写「Redis缓存实战」等更具体的词，避免与兄弟课冲突）；
  4) 至少给 3 个，越多越好；若确实无合适缩写可给空数组 []。
若信息不足，synonyms 至少给 5 个，aliases 按上述要求尽力给。"""


def _get_llm_client():
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        timeout=30.0,
        max_retries=1,
    )


def _parse_json_safe(text: str) -> dict:
    """先直接解析；失败则截取第一个 { 到最后一个 } 兜底（qwen2.5 偶发带多余文字）。"""
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return {}
    return {}


def _as_first_text(v) -> str:
    """取「过滤键」的单一取值：LLM 偶发把 field/level 返成列表（如 ['运维','开发']）。
    这里取首个非空元素——注意**不能拼接**：field/level 是 metadata 等值过滤键，
    拼成 "运维、开发" 后 structured_search 永远匹配不到任何课程（热路径静默失效）。"""
    if isinstance(v, (list, tuple)):
        for x in v:
            if isinstance(x, str) and x.strip():
                return x.strip()
        return ""
    if v is None:
        return ""
    return str(v).strip()


def _as_join_text(v) -> str:
    """取「描述性标签」的文本：列表则用「、」连接（audience/tech 只用于展示与检索增强）。"""
    if isinstance(v, (list, tuple)):
        return "、".join(str(x).strip() for x in v if str(x).strip())
    if v is None:
        return ""
    return str(v).strip()


def _normalize_tags(tags: dict) -> dict:
    """把 LLM 抽取结果规整成「标量字段必为 str、列表字段必为 list[str]」。

    背景（2026-09-17 全量重建事故）：_extract 不做类型校验，而 GLM-4.5-Air 偶发把
    audience 返成列表 ['学生', '转行']。该列表一路传到 Chroma upsert，触发
    `ValueError: Expected metadata value to be a str, int, float, bool, or None, got [...]`，
    使整个重建任务在第 23 门课时崩溃退出（MQ 消费路径调用同一个 _upsert，同样会崩）。
    在 _extract 出口统一规整，可同时保护 Chroma metadata、Redis 意图缓存两处写入。
    """
    if not isinstance(tags, dict):
        return {}
    out = dict(tags)
    for k in ("field", "level"):
        if k in out:
            out[k] = _as_first_text(out[k])
    for k in ("audience", "tech"):
        if k in out:
            out[k] = _as_join_text(out[k])
    for k in ("synonyms", "aliases"):
        v = out.get(k)
        if v is None:
            out[k] = []
        elif isinstance(v, str):
            out[k] = [s.strip() for s in v.split(",") if s.strip()]
        elif isinstance(v, (list, tuple)):
            out[k] = [str(s).strip() for s in v if str(s).strip()]
        else:
            out[k] = []
    return out


async def _extract(course_name: str, description: str) -> dict:
    """抽标签（幂等）：结果按 prompt+模型+课程内容 hash 缓存。

    全量重建/每日对账时内容未变的课程直接复用缓存，LLM 调用 ≈0；
    改 prompt 或换模型 → hash 自动变 → 旧缓存天然失效重抽，零人为维护。
    抽取失败（空 dict）不缓存，下个周期自然重试。
    """
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    prompt_hash = hashlib.md5(f"{_SYSTEM_PROMPT}|{model}".encode()).hexdigest()[:8]
    content_hash = hashlib.md5(f"{course_name}\n{description or ''}".encode()).hexdigest()[:12]
    cache_key = f"tagcache:{prompt_hash}:{content_hash}"
    try:
        cached = await asyncio.to_thread(_redis_intent().get, cache_key)
        if cached:
            # 缓存里可能是规整前写入的旧值（含列表），出口再规整一次保证类型契约
            return _normalize_tags(json.loads(cached))
    except Exception:
        pass  # 缓存读失败不挡主流程，退化为直接调 LLM

    client = _get_llm_client()
    user = f"课程名：{course_name}\n描述：{description or ''}"
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        tags = _parse_json_safe(resp.choices[0].message.content or "")
        tags = _normalize_tags(tags) if tags else {}
        if tags:
            try:
                await asyncio.to_thread(
                    _redis_intent().set, cache_key,
                    json.dumps(tags, ensure_ascii=False), ex=7 * 86400,
                )
            except Exception:
                pass
        return tags
    except Exception as e:
        logger.error("标签抽取失败: %s", e)
        return {}


_redis_intent_client = None


def _redis_intent():
    """意图缓存同步连接（模块级单例）。

    调用方（_write_intent/_delete）都经由 asyncio.to_thread 在工作线程执行，
    不占事件循环，同步客户端即可；单例避免每条课程事件都新建 TCP 连接。
    """
    global _redis_intent_client
    if _redis_intent_client is None:
        import redis

        _redis_intent_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=REDIS_INTENT_DB,
            # Redis 已开启 requirepass（见 docker-compose redis 服务），必须携带认证
            password=os.getenv("REDIS_PASSWORD") or None,
            decode_responses=True,
            socket_timeout=5,
        )
    return _redis_intent_client


async def _bump_answer_cache_version():
    """课程索引已变更 → 递增全局版本号，令 DB1 的答案缓存全量失效。

    解决"课程降价/下架后，TTL（1h）内用户仍读到旧价格/旧课程"的缓存一致性：
    routes.py 的缓存 key 拼入该版本号（chat:v3:{ver}:*），版本一变旧 key 全部 miss。
    注意 rebuild_all（每日全量重建）**不**递增：重建通常无变化，
    递增会让缓存每天白失效一次；真正变更走 MQ 事件路径时已在此处递增。
    """
    try:
        from app.services.cache import get_async_redis

        r = await get_async_redis()
        await r.incr("chat:index_ver")
    except Exception as e:
        # 失败仅意味着缓存多活到 TTL 自然过期（最长 1h 旧数据），可接受
        logger.warning("答案缓存版本号递增失败（旧缓存将等待 TTL 过期）: %s", e)


def _doc_text(course: dict) -> str:
    price = course.get("price")
    price_text = f"价格：{price}" if price is not None else ""
    parts = [
        f"课程名称：{course.get('courseName', '')}",
        f"分类：{course.get('category', '')} 难度：{course.get('level', '')}",
        f"讲师：{course.get('teacherName', '')}",
        price_text,
        f"简介：{course.get('description', '')}",
    ]
    return "\n".join(p for p in parts if p)


def _upsert(course: dict, tags: dict):
    course_id = course.get("courseId")
    if not course_id:
        return
    doc_id = f"course:{course_id}"  # 与 ingest.py 一致
    text = _doc_text(course)
    vec = embeddings.embed(text)[0]  # 同步 CPU 密集，外层用 to_thread 调用
    metadata = {
        "course_id": course_id,
        "course_name": course.get("courseName", ""),
        "category": course.get("category", ""),
        "level": course.get("level", ""),
        "price": str(course.get("price")) if course.get("price") is not None else "",
        "teacher_name": course.get("teacherName", ""),
        "description": course.get("description", ""),
        # 课程编码与评分：补齐语义卡片展示字段。
        # course_code 来自后端 Course 实体；avg_rating 来自 PublishedCourseVO（BigDecimal，
        # 经 JSON 反序列化为数值）。注意 avg_rating 必须存【数值】而非字符串——下游
        # course_recommender._generate_match_reason 会做 rating >= 4.5 比较，字符串比较会抛 TypeError。
        "course_code": course.get("courseCode") or "",
        "avg_rating": (float(course["avgRating"]) if course.get("avgRating") is not None else 0.0),
        # 以下为 LLM 抽取的增值标签（预制菜方案），供意图缓存与结构化过滤使用
        "ai_field": tags.get("field", "") or "",
        "ai_level": tags.get("level", "") or "",
        "ai_audience": tags.get("audience", "") or "",
        "ai_tech": tags.get("tech", "") or "",
    }
    chroma_store.upsert_course(doc_id, text, metadata, vec)


# 全量规范课名词表（离线别名子串安全过滤用，懒加载缓存一次）
_ALL_CNAMES = None


def _load_all_cnames():
    """懒加载全量规范课名集合，供 qrewrite 别名的【子串安全过滤】使用。
    仅加载一次并缓存；失败返回 None（降级为不过滤，保持原行为）。"""
    global _ALL_CNAMES
    if _ALL_CNAMES is not None:
        return _ALL_CNAMES
    s = set()
    try:
        col = chroma_store.get_collection()
        data = col.get(include=["metadatas"])
        for m in data.get("metadatas") or []:
            n = (m.get("course_name") or "").strip()
            if n:
                s.add(n)
    except Exception as e:
        print(f"[qrewrite] 加载规范课名词表失败，降级不过滤: {e}")
        return None
    _ALL_CNAMES = s
    return s


def _write_intent(course_id: str, tags: dict, synonyms: list, aliases: list = None, course_name: str = ""):
    r = _redis_intent()
    field = tags.get("field") or ""
    level = tags.get("level") or ""
    course_id = course_id or ""
    # 1) 泛化搜索意图 -> 意图缓存（领域/难度过滤用）
    for p in (synonyms or []):
        if not isinstance(p, str) or not p.strip():
            continue
        p = p.strip()
        if field:
            r.set(f"intent:{p}", f"ai_field:{field}")
        elif level:
            r.set(f"intent:{p}", f"ai_level:{level}")
        r.sadd(f"intent_courses:{course_id}", p)
    r.expire(f"intent_courses:{course_id}", 60 * 60 * 24 * 30)
    # 2) 唯一指代本课的口语/缩写/别称 -> qrewrite 词库（Query 改写的离线词典）
    #    与意图缓存同库(DB2)，键前缀隔离；course_name 即规范课名，线上据此归一化。
    cname = (course_name or "").strip()
    if cname:
        cnames = _load_all_cnames()  # 可能为 None（降级不过滤）
        for a in (aliases or []):
            if not isinstance(a, str) or not a.strip():
                continue
            a = a.strip()
            # 子串安全过滤：别名 a 若是【其他】规范课名的子串，则用户一旦原样敲出该
            # 规范课名（如 "Spring Boot实战开发课"），a 会作为子串命中、被改写到错误的
            # 兄弟课，破坏本已正确的检索（盲测曾致 canonical recall 100%→50%）。
            # 故剔除这类会"泄漏"的短别名，保留唯一指向本课的黑话/缩写（SB课/K8s/TS…）。
            if cnames is not None and any(a and a in cn for cn in cnames if cn != cname):
                continue
            r.set(f"qrewrite:{a}", cname)


def _delete(course_id: str):
    if not course_id:
        return
    doc_id = f"course:{course_id}"
    chroma_store.delete_course(doc_id)
    r = _redis_intent()
    for p in r.smembers(f"intent_courses:{course_id}"):
        r.delete(f"intent:{p}")
    r.delete(f"intent_courses:{course_id}")


async def _handle(body: dict):
    action = body.get("action")
    course_id = body.get("courseId")
    if action in ("new", "update"):
        tags = await _extract(body.get("courseName", ""), body.get("description", ""))
        # 仅对已发布课程建索引；未发布则确保它不出现在检索中（可能曾发布后又下架）
        if body.get("status") != "published":
            await asyncio.to_thread(_delete, course_id)
            await _bump_answer_cache_version()
            logger.info("课程 %s 状态=%s，跳过建索引", course_id, body.get("status"))
            return
        synonyms = tags.get("synonyms") if isinstance(tags.get("synonyms"), list) else []
        aliases = tags.get("aliases") if isinstance(tags.get("aliases"), list) else []
        await asyncio.to_thread(_upsert, body, tags)
        await asyncio.to_thread(
            _write_intent, course_id, tags, synonyms,
            aliases, body.get("courseName", ""),
        )
        await _bump_answer_cache_version()
        logger.info("课程 %s 已建索引 (field=%s, level=%s)", course_id, tags.get("field"), tags.get("level"))
    elif action == "delete":
        await asyncio.to_thread(_delete, course_id)
        await _bump_answer_cache_version()
        logger.info("课程 %s 已从索引移除", course_id)
    else:
        logger.warning("未知课程事件 action=%s", action)


async def start_consumer():
    """启动消费者，带自动重连。作为 FastAPI 启动时的后台任务运行。"""
    while True:
        try:
            missing_credentials = [
                name
                for name, value in (
                    ("RABBITMQ_USERNAME", RABBITMQ_USER),
                    ("RABBITMQ_PASSWORD", RABBITMQ_PASS),
                )
                if not value or not value.strip()
            ]
            if missing_credentials:
                logger.error(
                    "RabbitMQ 配置错误：缺少或为空的环境变量 %s；10s 后重试",
                    ", ".join(missing_credentials),
                )
                await asyncio.sleep(10)
                continue

            connection = await aio_pika.connect_robust(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                login=RABBITMQ_USER,
                password=RABBITMQ_PASS,
            )
        except Exception as e:
            logger.error("RabbitMQ 连接失败，10s 后重试: %s", e)
            await asyncio.sleep(10)
            continue
        try:
            async with connection:
                channel = await connection.channel()
                # 主动声明 topic 交换机并绑定队列（与 Java 端 RabbitMQConfig.courseBinding 等价）。
                # 消费者侧也声明，可消除"先启动消费者、后端尚未建绑定时事件被静默丢弃"的时序依赖。
                exchange = await channel.declare_exchange(
                    "course.exchange", aio_pika.ExchangeType.TOPIC, durable=True
                )
                queue = await channel.declare_queue(COURSE_QUEUE, durable=True)
                for rk in ("course.new", "course.update", "course.delete"):
                    await queue.bind(exchange, routing_key=rk)
                logger.info("课程事件消费者已启动，监听队列 %s", COURSE_QUEUE)
                async for message in queue.iterator():
                    async with message.process():
                        try:
                            body = json.loads(message.body.decode())
                        except Exception:
                            logger.warning("消息体解析失败，已忽略")
                            continue
                        try:
                            await _handle(body)
                        except Exception as e:
                            logger.exception("处理课程事件异常: %s", e)
        except Exception as e:
            logger.error("消费者循环异常，5s 后重连: %s", e)
            await asyncio.sleep(5)


# ====================== 阶段4：每日全量重建兜底 ======================
async def rebuild_all() -> int:
    """
    从 Java 内部接口拉取全量已发布课程，重新抽取标签并写 Chroma + 意图缓存。
    作为事件驱动的兜底：即便某次 MQ 事件丢失，每日也能对齐索引。
    返回成功处理的课程数；拉取失败返回 -1（区别于"成功但 0 门课"），供调用方重试。
    """
    import httpx

    base = os.getenv("BACKEND_BASE_URL", "http://backend:8080")
    token = os.getenv("INTERNAL_API_TOKEN", "")
    headers = {"X-Internal-Token": token} if token else {}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{base}/api/internal/courses", headers=headers)
            resp.raise_for_status()
            courses = resp.json()
    except Exception as e:
        logger.error("全量重建：拉取课程失败: %s", e)
        return -1  # 拉取失败与「成功但 0 门课」语义不同，调用方据此决定是否重试

    n = 0
    failed = 0
    total = sum(1 for c in courses if c.get("status") == "published")
    for i, c in enumerate(courses, 1):
        if c.get("status") != "published":
            continue
        # 单门课失败不得掀翻整批：2026-09-17 曾因一门课的 audience 被 LLM 返成列表，
        # 在 Chroma upsert 抛 ValueError，导致 128 门课的重建在第 23 门整体退出，
        # 且耗时 20 分钟白跑。改为逐门隔离 + 逐门进度日志（长任务必须可观测）。
        try:
            # 注意：后端 Course 实体用 @JsonProperty("course_name") 序列化，故此处读 snake_case。
            # （MQ 路径不受影响，因为走独立的 CourseEvent DTO，字段是 camelCase。）
            tags = await _extract(c.get("course_name", ""), c.get("description", ""))
            body = {
                "courseId": c.get("id"),
                "courseName": c.get("course_name"),
                "category": c.get("category"),
                "level": c.get("level"),
                "price": c.get("price"),
                "description": c.get("description"),
                "status": c.get("status"),
                # PublishedCourseVO 自带 teacher_name（LEFT JOIN itcloud_user），映射到事件字段。
                "teacherName": c.get("teacher_name"),
                # 课程编码与评分：PublishedCourseVO 以 snake_case 返回，映射到事件 camelCase 字段，
                # 供 _upsert 写入 Chroma metadata（展示卡片用）。
                "courseCode": c.get("course_code"),
                "avgRating": c.get("avg_rating"),
            }
            synonyms = tags.get("synonyms") if isinstance(tags.get("synonyms"), list) else []
            aliases = tags.get("aliases") if isinstance(tags.get("aliases"), list) else []
            await asyncio.to_thread(_upsert, body, tags)
            await asyncio.to_thread(
                _write_intent, c.get("id"), tags, synonyms,
                aliases, c.get("course_name", ""),
            )
            n += 1
            logger.info("重建进度 %d/%d 完成 course_id=%s field=%s level=%s",
                        i, total, c.get("id"), tags.get("field") or "-", tags.get("level") or "-")
        except Exception as e:
            failed += 1
            logger.error("重建跳过 course_id=%s（%s: %s）",
                         c.get("id"), type(e).__name__, e)
    logger.info("全量重建完成，共处理 %d 门课程（失败 %d 门）", n, failed)
    # 记录重建时间戳：启动自举据此跳过新鲜索引（重启零 LLM 调用的依据）
    try:
        import datetime as _dt
        await asyncio.to_thread(
            _redis_intent().set, "idx:last_rebuild_at",
            _dt.datetime.now().isoformat(timespec="seconds"),
        )
    except Exception:
        pass
    return n


async def _index_is_fresh() -> bool:
    """启动自举跳过判定：有重建时间戳 且 Chroma 非空 → 索引新鲜。

    任一不满足（首次部署 / Redis 被清 / volume 被清）都视为不新鲜，
    保守触发重建——宁慢一次，不带陈旧索引静默跑。
    注意：启动瞬间 numpy/transformers 惰性导入与嵌入预热线程竞态，chroma_store.count
    失败时不抛异常而是返回 0（chroma_store 内部吞掉），会被误判为"索引空"。
    因此"时间戳在但 count=0"视为疑似竞态，重试等待预热完成（实测 ~20s）；
    时间戳缺失才是真正的不新鲜，立即重建。
    """
    for attempt in range(4):
        try:
            last = await asyncio.to_thread(_redis_intent().get, "idx:last_rebuild_at")
        except Exception as e:
            logger.warning("新鲜度检查读取时间戳失败（%s），10s 后重试", e)
            await asyncio.sleep(10)
            continue
        if not last:
            return False  # 从未重建过（首访部署/Redis 被清）→ 真不新鲜
        n = await asyncio.to_thread(chroma_store.count)
        if n > 0:
            return True
        logger.warning("时间戳在（%s）但 Chroma count=0（第 %d 次，疑似启动导入竞态），10s 后重试",
                       last, attempt + 1)
        await asyncio.sleep(10)
    return False


async def start_scheduler(interval_seconds: int = 86400):
    """每日全量重建调度（默认 24h）。后台任务，异常不退出，下个周期重试。

    启动自举：索引新鲜（上次重建过且 Chroma 非空）则直接跳过——重启从
    "90 次 LLM 调用 + 10 分钟高负载" 变为 0 调用；MQ 增量事件不受影响，
    仍实时消费。仅首次部署 / 数据丢失时才做启动全量（带退避重试）。
    注意：compose 中 ai-service 与 backend 同时启动，backend Spring Boot 完全就绪
    需 10s 左右，首次拉取大概率失败。失败后按 60s/120s/240s 退避重试（最多 3 次），
    而不是静默等 24h —— 否则索引整天缺失且日志只剩一条 error。
    """
    logger.info("课程索引定时重建调度已启动（间隔 %ss）", interval_seconds)
    if await _index_is_fresh():
        last = await asyncio.to_thread(_redis_intent().get, "idx:last_rebuild_at")
        logger.info("索引新鲜（上次重建于 %s），跳过启动自举", last)
    else:
        backoff_delays = [60, 120, 240]
        for attempt, delay in enumerate([0] + backoff_delays):
            if delay:
                logger.warning("初次全量重建第 %d 次重试（%ds 后）", attempt, delay)
                await asyncio.sleep(delay)
            try:
                done = await rebuild_all()
                if done >= 0:
                    logger.info("启动初次全量重建完成，处理 %d 门课程", done)
                    break
                # done == -1：拉取失败（rebuild_all 已记 error），继续退避重试
            except Exception as e:
                logger.exception("启动初次全量重建异常: %s", e)
        else:
            logger.error("初次全量重建重试耗尽，等待下个每日周期（%ss 后）兜底", interval_seconds)

    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await rebuild_all()
        except Exception as e:
            logger.exception("定时全量重建异常: %s", e)
