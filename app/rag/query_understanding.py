# 结构化 Query Understanding：把用户原始 Query 解析成可审计的结构化意图。
#
# 设计原则（对应 Phase 2）：
#   - 保留 original_query，绝不字符串替换/覆盖（禁止 alias substring rewrite）；
#   - 只输出结构化意图，不生成新的自然语言 Query；
#   - 词典全部来自「课程技术栈」或「通用中文技术学习用语」，可追踪、可人工修改、可关闭；
#   - 不做复杂数学公式，level 推断用「命中信号 + 简单计分」即可人工审计。
#
# 匹配策略：中文词用 substring（不依赖 jieba 分词粒度，可靠命中"调优/排障/会一点"等）；
#           英文词用词边界正则（避免 es/design 等子串误命中）。
#
# 输出 Structured Intent：
#   intent / domain / technology / goal / task / level /
#   explicit_level / inferred_level / level_confidence / level_evidence /
#   keywords / constraints / confidence
import os
import re

# ============================ 技术实体词典（可追踪：来自课程技术栈） ============================
# 中文技术词：substring 匹配。
TECH_CN = {
    # backend
    "微服务", "分布式", "高并发", "消息队列", "缓存", "数据库", "接口",
    "事务", "网关", "负载均衡", "服务端", "后端",
    # frontend
    "前端", "网页", "小程序", "跨端", "组件化", "界面",
    # data
    "数据分析", "数据可视化", "大数据", "爬虫", "数据结构", "算法", "数仓",
    "机器学习", "深度学习", "神经网络", "计算机视觉", "自然语言处理",
    # devops
    "运维", "云计算", "云原生", "可观测性", "容器", "服务网格", "服务器",
    # ai
    "大模型", "检索增强生成", "提示词", "智能体", "向量数据库", "人工智能",
    "自然语言", "模型",
}
# 英文技术词：词边界匹配，value 为规范输出形式。
TECH_EN = {
    # backend
    "java": "Java", "spring": "Spring", "springboot": "SpringBoot",
    "spring cloud": "Spring Cloud", "springcloud": "Spring Cloud",
    "jvm": "JVM", "mybatis": "MyBatis", "mysql": "MySQL", "redis": "Redis",
    "rabbitmq": "RabbitMQ", "kafka": "Kafka", "golang": "Go", "python": "Python",
    "node.js": "Node.js", "nodejs": "Node.js", "mongodb": "MongoDB",
    "elasticsearch": "Elasticsearch", "rpc": "RPC", "api": "API",
    # frontend
    "html": "HTML", "css": "CSS", "javascript": "JavaScript", "typescript": "TypeScript",
    "vue": "Vue", "vue3": "Vue3", "react": "React", "next.js": "Next.js", "nextjs": "Next.js",
    "flutter": "Flutter", "react native": "React Native", "angular": "Angular",
    "web": "Web", "h5": "H5", "uniapp": "uni-app", "electron": "Electron",
    # data
    "sql": "SQL", "pandas": "Pandas", "spark": "Spark", "flink": "Flink",
    "hadoop": "Hadoop", "etl": "ETL", "leetcode": "LeetCode", "nlp": "NLP", "bi": "BI",
    # devops
    "linux": "Linux", "docker": "Docker", "kubernetes": "Kubernetes", "k8s": "K8s",
    "ci/cd": "CI/CD", "cicd": "CI/CD", "git": "Git", "nginx": "Nginx",
    "prometheus": "Prometheus", "grafana": "Grafana", "jenkins": "Jenkins",
    "shell": "Shell", "devops": "DevOps", "ansible": "Ansible",
    # ai
    "llm": "LLM", "rag": "RAG", "prompt": "Prompt", "transformer": "Transformer",
    "embedding": "Embedding", "ai agent": "AI Agent", "agent": "Agent",
}

# ============================ 领域（domain）信号 → category ============================
# 与课程 category 枚举（backend/frontend/data/devops/ai）对齐；有限映射，可追踪、可关闭。
# 机器学习/深度学习只归 data（课程库中这类课在 data category）。
DOMAIN_SIGNALS = {
    "backend": ["后端", "服务端", "微服务", "接口", "缓存", "消息队列", "数据库",
                "高并发", "分布式", "后台", "事务", "网关", "搜索"],
    "frontend": ["前端", "网页", "小程序", "跨端", "界面", "手机应用", "h5", "ui"],
    "data": ["数据", "大数据", "数据分析", "数仓", "爬虫", "可视化", "报表", "算法",
             "建模", "机器学习", "深度学习", "神经网络"],
    "devops": ["运维", "部署", "容器", "监控", "服务器", "云原生", "上云", "发布",
               "持续集成", "持续部署", "自动化", "linux", "docker", "kubernetes", "k8s"],
    "ai": ["人工智能", "大模型", "智能体", "自然语言", "向量", "聊天机器人", "提示词",
           "rag", "ai", "llm", "推荐系统"],
}

# 技术词 → domain（当 query 没直接说领域、但提到技术栈时，反推 domain）。
TECH_TO_DOMAIN = {
    # backend
    "java": "backend", "spring": "backend", "springboot": "backend", "jvm": "backend",
    "mybatis": "backend", "mysql": "backend", "redis": "backend", "rabbitmq": "backend",
    "kafka": "backend", "mongodb": "backend", "elasticsearch": "backend", "go": "backend",
    "node.js": "backend", "微服务": "backend", "分布式": "backend", "高并发": "backend",
    "消息队列": "backend", "缓存": "backend", "数据库": "backend", "接口": "backend",
    # frontend
    "vue": "frontend", "react": "frontend", "angular": "frontend", "flutter": "frontend",
    "react native": "frontend", "next.js": "frontend", "html": "frontend", "css": "frontend",
    "javascript": "frontend", "typescript": "frontend", "前端": "frontend", "网页": "frontend",
    "小程序": "frontend", "跨端": "frontend",
    # data
    "sql": "data", "pandas": "data", "spark": "data", "flink": "data", "数据分析": "data",
    "数据可视化": "data", "机器学习": "data", "深度学习": "data", "大数据": "data",
    "爬虫": "data", "算法": "data", "数据结构": "data", "leetcode": "data",
    # devops
    "linux": "devops", "docker": "devops", "kubernetes": "devops", "k8s": "devops",
    "ci/cd": "devops", "git": "devops", "nginx": "devops", "运维": "devops", "部署": "devops",
    "云计算": "devops", "云原生": "devops",
    # ai
    "大模型": "ai", "rag": "ai", "prompt": "ai", "ai agent": "ai", "向量数据库": "ai",
    "检索增强生成": "ai", "llm": "ai",
}

# category → 该领域课程的技术栈词（用于 Intent Recall 的召回查询词）。
CATEGORY_TECH = {
    "backend": ["Java", "Spring", "SpringBoot", "SpringCloud", "MyBatis", "Redis",
                "RabbitMQ", "Kafka", "MySQL", "Python", "Node.js", "Go", "MongoDB",
                "Elasticsearch", "JVM", "微服务", "分布式", "高并发", "缓存", "消息队列"],
    "frontend": ["HTML", "CSS", "JavaScript", "TypeScript", "Vue", "React", "Next.js",
                 "微信小程序", "Flutter", "React Native", "Angular", "前端", "跨端"],
    "data": ["Python", "SQL", "Pandas", "数据分析", "数据可视化", "机器学习", "深度学习",
             "大数据", "Spark", "爬虫", "算法", "数据结构", "LeetCode"],
    "devops": ["Linux", "Docker", "Kubernetes", "K8s", "CI/CD", "Git", "Nginx",
               "云计算", "云原生", "监控", "可观测性", "运维", "部署"],
    "ai": ["大模型", "RAG", "Prompt", "AI Agent", "向量数据库", "检索增强生成",
           "机器学习", "深度学习", "自然语言处理", "LLM"],
}

# ============================ 目标（goal）/ 任务（task）词 ============================
GOAL_TERMS = {"入门", "零基础", "从零", "进阶", "精通", "掌握", "学会", "提升", "面试",
              "就业", "夯实", "打基础", "实战", "项目", "落地", "独立", "完整", "全栈",
              "搞定", "拿下", "会一点", "继续提高"}
TASK_TERMS = {"搭建", "开发", "设计", "优化", "调优", "排查", "部署", "整合", "集成",
              "实现", "编写", "配置", "监控", "测试", "重构", "缓存", "削峰", "限流",
              "持久化", "集群", "事务", "并发", "消息", "同步", "异步", "渲染", "路由",
              "状态管理", "组件化", "性能", "安全", "认证", "授权", "爬取", "抓取",
              "查询", "排障", "运维", "构建", "发布", "压测", "调参", "搜索"}

# ============================ 难度意图信号 ============================
# explicit：明确难度标签词（直接表达难度）。与 retriever._LEVEL_QUERY_SIGNALS 同源。
EXPLICIT_LEVEL_SIGNALS = {
    "beginner": ("零基础", "从零", "入门", "新手", "小白", "菜鸟", "初学", "打下基础"),
    "intermediate": ("进阶", "进一步提高", "提高", "拔高", "进一步提升"),
    "advanced": ("精通", "高阶", "高级", "底层", "底层原理", "原理", "源码", "调优",
                 "剖析", "极致", "骨灰级", "深度剖析", "深入理解", "深入"),
}

# inferred：无难度标签词时，由语气/意图词暗示难度。
IMPLICIT_LEVEL_SIGNALS = {
    "beginner": ("是什么", "是啥", "干嘛", "什么意思", "怎么开始", "怎么入门", "能听懂",
                 "听懂", "从头", "了解", "初步", "刚学", "不会", "不懂", "是干嘛的"),
    "intermediate": ("会一点", "会点", "继续提高", "做项目", "实际开发", "怎么设计", "怎么搭",
                     "搭起来", "进一步", "用起来", "怎么用", "实践", "落地"),
    "advanced": ("容灾", "排障", "线上", "生产环境", "压测", "调参", "高并发架构",
                 "性能优化", "内核", "架构演进", "治理", "全链路", "源码级", "极致性能"),
}

# ============================ 开关 ============================
def enabled() -> bool:
    """Query Understanding 是否启用（默认关闭，评测脚本直接调用函数绕过开关）。"""
    return os.getenv("RAG_QU", "0") == "1"


# ============================ 工具函数 ============================
def _match_cn(q, cn_set):
    """中文词 substring 匹配（不依赖分词粒度）。"""
    return [t for t in cn_set if t in q]


def _match_en(q, en_dict):
    """英文词词边界匹配，返回规范形式（去子串重叠：保留更长词）。

    负向边界只排除字母、不排除数字，使 vue3/k8s/es6 等带版本/数字的词能命中 vue/k8s。
    """
    ql = (q or "").lower()
    hits = {}
    for key, disp in en_dict.items():
        if re.search(r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z])", ql):
            hits[key] = disp
    kept = []
    for key in sorted(hits, key=len, reverse=True):
        if not any(key in k for k in hits if k != key and len(k) > len(key)):
            kept.append(hits[key])
    return kept


def _hit_level(signals, q):
    """命中且无冲突 → (level, evidence)；无命中或冲突 → (None, [])。"""
    present = {lv: [w for w in words if w in q] for lv, words in signals.items()}
    present = {lv: ev for lv, ev in present.items() if ev}
    if len(present) != 1:
        return None, []
    lv, ev = next(iter(present.items()))
    return lv, ev


def _dedup(seq):
    return list(dict.fromkeys(seq))


# ============================ 主入口 ============================
def understand(query: str) -> dict:
    """解析原始 query，返回 Structured Intent dict。original_query 原样保留。"""
    q = query or ""
    ql = q.lower()

    # 1) domain（有限映射，substring 匹配）
    domain = [cat for cat, words in DOMAIN_SIGNALS.items()
              if any(w in ql for w in words)]

    # 2) technology（英文词边界 + 中文 substring）
    tech_en = _match_en(q, TECH_EN)
    tech_cn = _match_cn(q, TECH_CN)
    technology = _dedup(tech_en + tech_cn)

    # 2b) 技术词反推 domain（query 没直接说领域、但提到技术栈时补充）
    if not domain:
        for t in technology:
            tl = t.lower()
            d = TECH_TO_DOMAIN.get(tl)
            if d and d not in domain:
                domain.append(d)

    # 3) goal / task
    goal = _match_cn(q, GOAL_TERMS)
    task = _match_cn(q, TASK_TERMS)

    # 4) explicit / inferred level
    explicit_level, explicit_ev = _hit_level(EXPLICIT_LEVEL_SIGNALS, q)
    inferred_level, inferred_ev = _hit_level(IMPLICIT_LEVEL_SIGNALS, q)

    if explicit_level:
        level = explicit_level
        level_confidence = 0.95
        level_evidence = explicit_ev
    elif inferred_level:
        level = inferred_level
        # 简单计分：1 条证据 0.6，每多一条 +0.1，封顶 0.85
        level_confidence = min(0.6 + 0.1 * (len(inferred_ev) - 1), 0.85)
        level_evidence = inferred_ev
    else:
        level = None
        level_confidence = 0.0
        level_evidence = []

    # 5) keywords（技术 + 任务词）
    keywords = _dedup(technology + task)

    # 6) Technical Query（只提技术实体/关键词，不造自然语言，去重）
    technical_query = " ".join(keywords)

    # 7) Intent Recall 召回词（命中的 category 技术栈，供审计）
    intent_tech = _dedup([t for cat in domain for t in CATEGORY_TECH.get(cat, [])])

    # 整体置信度：识别出实体（domain 或 technology）则较高，否则低
    confidence = 0.8 if (domain or technology) else 0.3

    return {
        "intent": "course_search",
        "domain": domain,
        "technology": technology,
        "goal": goal,
        "task": task,
        "level": level,
        "explicit_level": explicit_level,
        "inferred_level": inferred_level,
        "level_confidence": round(level_confidence, 2),
        "level_evidence": level_evidence,
        "keywords": keywords,
        "constraints": [],
        "confidence": confidence,
        # 附加（供 Intent Recall / 审计用，非规范字段）
        "technical_query": technical_query,
        "intent_tech": intent_tech,
        "original_query": q,
    }
