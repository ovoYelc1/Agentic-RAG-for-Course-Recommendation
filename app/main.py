# AI 客服服务 - 主入口
import asyncio
import logging

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 根日志默认 WARNING，会把 consumer/web_search 等模块的 info 日志
# （如"全量重建完成"）全部吞掉，排障时只能看到 error。配置为 INFO
# 让关键流程日志可见；uvicorn 用独立 logger，不受影响、不会重复输出。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="ITCloud AI 客服",
    description="基于 LLM 的智能课程推荐与问答服务",
    version="1.0.0"
)

# CORS 配置 - 允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ORIGINS", "http://localhost:5173")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 后台任务强引用集合。asyncio 内部只对 Task 持【弱引用】，若只调 create_task()
# 而不保存返回值，任务随时可能被 GC 掉，日志表现为
# "Task was destroyed but it is pending!"——MQ 消费者就是这样在重启后静默失联，
# queue.course.ai 的 consumers 变成 0、消息全部积压。必须显式保活。
_BG_TASKS: set = set()


def _spawn_background(coro) -> "asyncio.Task":
    task = asyncio.create_task(coro)
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)
    return task


@app.on_event("startup")
async def startup_event():
    """服务启动时初始化"""
    print("=" * 50)
    print("ITCloud AI 客服服务启动中...")
    print(
        f"LLM 模型: {os.getenv('LLM_MODEL', 'gpt-4o-mini')}  "
        f"(base_url={'已配置' if os.getenv('OPENAI_BASE_URL') else '默认'})"
    )
    print(f"Embedding 模型: {os.getenv('EMBEDDING_MODEL', 'shibing624/text2vec-base-chinese')}")
    print(f"Reranker 模型: {os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-v2-m3')}")
    print(f"Chroma: {os.getenv('CHROMA_HOST', 'chromadb')}:{os.getenv('CHROMA_PORT', 8000)}")
    print(f"Redis 对话记忆 DB: {os.getenv('REDIS_DB', 0)} | 意图缓存 DB: {os.getenv('REDIS_INTENT_DB', 2)}")
    _auth_on = os.getenv("AI_REQUIRE_AUTH", "false").lower() == "true"
    print(f"AI 端点鉴权: {'开启' if _auth_on else '关闭（本地联调）'}")
    if _auth_on:
        # 鉴权开启但密钥缺失会导致所有请求 401/500，启动即提示，避免上线后才发现
        print(f"  ├─ 网关令牌 AI_SERVICE_TOKEN: {'已配置' if os.getenv('AI_SERVICE_TOKEN') else '**缺失(将全部401)**'}")
        print(f"  └─ 用户 JWT 密钥 JWT_SECRET: {'已配置' if os.getenv('JWT_SECRET') else '**缺失(将全部500)**'}")
        print(f"限流: {os.getenv('AI_RATE_LIMIT_PER_MIN', 20)} 次/分钟 | 单用户并发 {os.getenv('AI_MAX_CONCURRENCY_PER_USER', 1)}")
    print("=" * 50)

    # 预热本地嵌入模型（HF_HUB_OFFLINE 强依赖宿主机缓存）：真实加载一次，
    # 缓存损坏/挂载失效在启动期即暴露，/health/ready 据此报 503，
    # 避免"容器 Running 但语义检索实际全挂"的隐性故障。
    # 必须在挂载 MQ 消费者/调度器【之前】完成：numpy/transformers 重导入若被
    # 多线程并发触发会产生 circular import，且失败的部分初始化模块会永久留在
    # sys.modules（进程内无法自愈，重试无用）。预热先行 = 只有一个线程首触导入。
    async def _warmup_embeddings():
        global _EMBEDDING_READY
        from app.rag import embeddings

        for attempt in range(4):
            try:
                await asyncio.to_thread(embeddings.embed, "预热")
                _EMBEDDING_READY = True
                print("[启动] 嵌入模型预热完成（语义检索就绪）")
                return
            except Exception as e:
                if attempt < 3:
                    print(
                        f"[启动] 嵌入模型预热失败（第 {attempt + 1} 次），"
                        f"10s 后重试 error_type={type(e).__name__} kind=embedding_warmup"
                    )
                    await asyncio.sleep(10)
        _EMBEDDING_READY = False
        print("[启动] 嵌入模型预热重试耗尽（语义检索不可用，结构化快查/兜底不受影响）")

    await _warmup_embeddings()

    # 启动课程事件消费者（"预制菜"离线模块）：后台任务，断线自动重连。
    # 延迟导入避免 aio_pika 未安装时影响其他导入；消费者内部已做重连。
    # 注意：必须在预热完成后挂载（见上方注释的导入串行化说明）。
    try:
        from app.mq import consumer as course_consumer

        _spawn_background(course_consumer.start_consumer())
        print("[启动] 课程事件消费者已挂载（RabbitMQ 不可达时会自动重试）")
        _spawn_background(course_consumer.start_scheduler())
        print("[启动] 课程索引每日全量重建调度已挂载（兜底用）")
    except Exception as e:
        print(
            f"[启动] 课程事件消费者挂载失败（可忽略，不影响对话功能） "
            f"error_type={type(e).__name__} kind=consumer_start"
        )


@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时清理"""
    print("ITCloud AI 客服服务关闭中...")
    # 释放异步 Redis 连接池，避免关闭时报 "Unclosed connection" 警告
    try:
        from app.services.cache import close_redis
        from app.core.security import close_blacklist_redis
        from app.services.memory import close_memory_redis

        await close_redis()
        await close_blacklist_redis()
        await close_memory_redis()
    except Exception as e:
        print(
            f"[关闭] Redis 连接释放失败（可忽略） "
            f"error_type={type(e).__name__} kind=redis_close"
        )


@app.get("/")
async def root():
    return {
        "service": "ITCloud AI 客服",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/api/ai/chat",
            "recommend": "/api/ai/recommend",
            "health": "/api/ai/health"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# 就绪标志：启动预热真实加载嵌入模型后置 True（见 startup_event）。
# HF 缓存损坏/挂载失效时保持 False，/health/ready 报 503，
# 供 docker healthcheck / K8s startupProbe 探测"Running 但实际不可用"的假活。
_EMBEDDING_READY = False


@app.get("/health/ready")
async def readiness_check():
    """就绪探测（无鉴权，供 compose healthcheck 使用，勿暴露业务信息）。"""
    if not _EMBEDDING_READY:
        return JSONResponse(status_code=503, content={"status": "not ready", "reason": "embedding model not loaded"})
    return {"status": "ready"}


# ===== AI 端点鉴权（双层信任边界）=====
# 原实现只校验 Nginx 注入的 X-API-Token，但 Nginx 对 /api/ai/ 是**无条件**盖章的
# （proxy_set_header X-API-Token），所以匿名请求到达时也自带合法令牌 → 免登录白嫖 LLM。
# 现拆成两层，均由 app.core.security.require_identity 完成：
#   1) X-API-Token   → 证明「来自内网网关」（防绕过 Nginx 直连）
#   2) Authorization → 证明「来自已登录用户」（防匿名滥用 / 伪造他人身份）
# AI_REQUIRE_AUTH != "true" 时两层都跳过，保留本地 uvicorn 直跑与 curl 调试能力。
from app.core.security import require_identity

# 导入并注册路由
from app.api.routes import router as ai_router
app.include_router(
    ai_router,
    prefix="",
    tags=["AI客服"],
    dependencies=[Depends(require_identity)],
)


if __name__ == "__main__":
    port = int(os.getenv("AI_SERVICE_PORT", 8000))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
