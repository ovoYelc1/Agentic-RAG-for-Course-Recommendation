# 意图提示器 - 轻量关键词匹配，仅作为 Agent 的"线索"，不再直接路由到死路
#
# 旧版问题：有一个"默认_faq"兜底规则，匹配 什么/为什么/如何/能不能/可以吗/请问
# 这类超高频词，导致几乎任何中文问题都被强行塞进 FAQ 分支，
# 而 FAQ 知识库只有 8 个主题，匹配不上就触发"抱歉我不确定"。
#
# 现在这个模块只做"意图提示"（hint_intent），给 Agent 一个初始方向；
# 真正决定怎么回答的是 Agent（它能调用工具查库/联网）。对于拿不准的，
# 直接返回 GENERIC，交给 Agent 用工具去解决，而不是推脱。
import re
from typing import Dict
from app.models.chat import Intent


FAQ_KEYWORDS = {
    "价格": [r"多少钱", r"价格", r"收费", r"费用", r"贵不贵", r"便宜", r"打折", r"优惠", r"免费"],
    "支付方式": [r"支付", r"付款", r"微信", r"支付宝", r"银行卡", r"怎么付"],
    "退款": [r"退款", r"退费", r"不满意", r"退钱", r"退款政策"],
    "证书": [r"证书", r"结业证", r"毕业证", r"认证"],
    "发票": [r"发票", r"报销", r"开票"],
    "试看": [r"试听", r"免费看", r"体验课", r"试看", r"先看"],
    "学习周期": [r"多久", r"多长时间", r"几天", r"几周", r"学时", r"课时"],
}

ORDER_KEYWORDS = [
    r"订单", r"购买记录", r"已购", r"我的课", r"买过", r"支付记录", r"下单"
]

ACCOUNT_KEYWORDS = [
    r"密码", r"登录", r"注册", r"账号", r"封号", r"禁言", r"找回", r"修改信息"
]

RECOMMEND_KEYWORDS = [
    r"推荐.*课", r"学.*什么", r"入门", r"新手", r"有没有.*课",
    r"想学", r"适合.*的课", r"有什么.*课程", r"自学", r"转行",
    r"进阶", r"提升", r"培训", r"教程",
    # 覆盖初学者常见说法（避免"前端课程/学编程"被误判为通用对话）
    r"前端", r"后端", r"编程", r"推荐课程", r"课程推荐",
    r"学编程", r"零基础", r"小白", r"入门课程", r"有.*课",
]


def hint_intent(message: str) -> Intent:
    """
    轻量意图提示（仅作为 Agent 的线索，不直接决定路由）。
    只匹配"明确属于某类别"的关键词，拿不准一律返回 GENERIC。
    """
    msg = (message or "").lower().strip()

    # 1. 课程推荐意图
    for pattern in RECOMMEND_KEYWORDS:
        if re.search(pattern, msg):
            return Intent.COURSE_RECOMMEND

    # 2. 订单查询意图
    for pattern in ORDER_KEYWORDS:
        if re.search(pattern, msg):
            return Intent.ORDER_QUERY

    # 3. 账号帮助意图
    for pattern in ACCOUNT_KEYWORDS:
        if re.search(pattern, msg):
            return Intent.ACCOUNT_HELP

    # 4. FAQ 意图（仅匹配明确的政策类关键词，不再用"什么/为什么"兜底）
    for patterns in FAQ_KEYWORDS.values():
        for pattern in patterns:
            if re.search(pattern, msg):
                return Intent.FAQ

    # 5. 其余交给 Agent 用工具解决
    return Intent.GENERIC


# 预定义映射表 - 关键词到课程字段的映射（模块级：extract_tech_keywords 也要用）
CATEGORY_MAP = {
    "java": "backend", "spring": "backend", "后端": "backend",
    "vue": "frontend", "react": "frontend", "前端": "frontend",
    "css": "frontend", "html": "frontend", "小程序": "frontend",
    "python": "data", "数据分析": "data", "机器学习": "data",
    "ai": "ai", "人工智能": "ai", "深度学习": "data", "爬虫": "data",
    "docker": "devops", "k8s": "devops", "运维": "devops",
    "kubernetes": "devops", "linux": "devops", "redis": "backend",
    "devops": "devops", "部署": "devops",
}

LEVEL_MAP = {
    "入门": "beginner", "初级": "beginner", "新手": "beginner",
    "初学者": "beginner", "零基础": "beginner", "基础": "beginner",
    "中级": "intermediate", "进阶": "intermediate", "提高": "intermediate",
    "高级": "advanced", "精通": "advanced", "深入": "advanced",
}

# 口语停用词（提取关键词时剔除，避免"有没有/以内"这类填充词污染信号）
_STOP_WORDS = {
    "我想", "学", "什么", "课程", "课", "有", "没", "的", "了", "啊", "呢", "吗", "吧",
    "有没有", "以内", "块钱", "左右", "大概", "推荐", "想要", "可以", "告诉", "一下",
    "换个", "换一批", "太", "了",
}

# 关键词提取正则：英文技术词（java/vue3/c++/node.js）+ 中文 2-4 字词。
# 原来只匹配中文，英文技术词只能靠 CATEGORY_MAP 顺带进 keywords，
# 不在映射表里的英文词（vue3、redis、c++）永远提取不到。
_WORD_PATTERN = r'[A-Za-z][A-Za-z0-9#+.]{1,10}|[\u4e00-\u9fa5]{2,4}'


def extract_tech_keywords(text: str) -> list:
    """
    从文本提取英文技术关键词（java/vue3/docker/redis...），供检索结果按
    "课程名包含"收窄——修复"问 java 推荐 Go"。

    只认英文词，中文词一律不参与名称收窄：
    - 中文词语义宽泛（"后端开发"），按名称收窄会误伤同类课
      （Spring Boot 名字里没有"后端"，会被滤掉）；
    - 英文技术词是具体技术，课程名即含该词（"Java高级开发实战"含 java），
      收窄安全。中文方向诉求由 category/level 结构化过滤 + 语义排序承担。
    """
    if not text:
        return []
    out, seen = [], set()
    for w in re.findall(r'[A-Za-z][A-Za-z0-9#+.]{1,10}', text):
        wl = w.lower()
        # AI/人工智能作为课程目录里的独立分类处理，不作为课名收窄词。
        if wl in seen or wl in _STOP_WORDS or wl == "ai":
            continue
        seen.add(wl)
        out.append(wl)
    return out[:3]


def extract_recommend_params(message: str) -> Dict:
    """
    从用户消息中提取课程推荐参数

    示例:
        输入: "我想学Java入门，预算200以内"
        输出: {"keywords": ["Java"], "category": "backend", "level": "beginner", "max_price": 200.0}
    """
    params = {"keywords": [], "category": None, "level": None, "max_price": None}
    msg_lower = (message or "").lower()

    # 提取难度/分类：取"最后出现"的词（支持"太入门了，换个进阶的"这类转向表达——
    # 先出现的"入门"是旧状态，应以靠后的"进阶"为准；分类同理，如"前端看腻了想学后端"）
    last_pos, last_level = -1, None
    for keyword, level in LEVEL_MAP.items():
        pos = msg_lower.rfind(keyword)
        if pos != -1 and pos > last_pos:
            last_pos, last_level = pos, level
    params["level"] = last_level

    # 先记录分类映射命中，后续会结合否定词和显式方向词计算有效分类。
    hit = []  # (pos, category, keyword)
    for keyword, category in CATEGORY_MAP.items():
        if keyword == "ai":
            matches = list(re.finditer(r"(?<![a-z0-9])ai(?![a-z0-9])", msg_lower))
            pos = matches[-1].start() if matches else -1
        else:
            pos = msg_lower.rfind(keyword)
        if pos != -1:
            hit.append((pos, category, keyword))

    # ---- 否定表达识别（v2）：排除用户明确不想要的领域/技术词 ----
    # "不想学 Java" / "不要前端" / "排除 Python"。命中则从正向关键词里剔除，
    # 并记录到 excluded。DB2 或单值 category 若被否定则不应作为硬约束。
    excluded = []
    # 正则：否定词 紧邻 目标技术词/领域词（允许中间穿插个别助词）
    neg_patterns = [
        r"(?:不想|不学|不要|不.*学|排除|别推荐|剔除|抛开|不用)"
        r".{0,6}?(java|python|vue|react|前端|后端|ai|人工智能|深度学习|机器学习|docker|k8s)",
        r"(java|python|vue|前端|后端|ai|docker)"
        r".{0,4}?(?:就算了|不要|不想|别学|排除)",
    ]
    for pat in neg_patterns:
        for m in re.finditer(pat, msg_lower):
            for g in m.groups():
                if g and g not in excluded:
                    excluded.append(g.lower())
    params["excluded"] = excluded
    params["excluded_tech_keywords"] = list(excluded)

    # 显式方向/分类词优先于技术名推导出的默认分类。
    # 例如“Python AI”应落 ai，而不是同时误带 Python 的 data 默认映射；
    # “Java 后端”仍落 backend，因为后端是明确表达的方向。
    explicit_category_terms = {
        "后端", "前端", "数据分析", "机器学习", "人工智能",
        "ai", "深度学习", "爬虫", "运维", "devops", "部署",
    }
    explicit_hits = [item for item in hit if item[2] in explicit_category_terms]
    domain_joiner = re.compile(r"和|与|还是|以及|vs|对比|比较", re.IGNORECASE)
    positive_hits = []
    for item in hit:
        pos, category, keyword = item
        if keyword in excluded:
            continue
        if keyword not in explicit_category_terms and explicit_hits:
            redundant = False
            for explicit_pos, explicit_category, _explicit_keyword in explicit_hits:
                if category == explicit_category:
                    redundant = True
                    break
                start, end = sorted((pos, explicit_pos))
                if domain_joiner.search(msg_lower[start:end]):
                    continue
                redundant = True
                break
            if redundant:
                continue
        positive_hits.append(item)

    # 被否定的技术词不再贡献其默认分类；但用户另有明确方向词时保留该方向。
    categories_ordered = []
    for _pos, category, _keyword in sorted(positive_hits, key=lambda x: x[0]):
        if category not in categories_ordered:
            categories_ordered.append(category)
    last_pos, last_cat, last_cat_kw = -1, None, None
    for pos, category, keyword in positive_hits:
        if pos > last_pos:
            last_pos, last_cat, last_cat_kw = pos, category, keyword
    if last_cat:
        params["category"] = last_cat
        if last_cat_kw != "ai" and last_cat_kw not in params["keywords"]:
            params["keywords"].append(last_cat_kw)
    params["categories"] = categories_ordered if len(categories_ordered) > 1 else []

    # 被否定的技术词不参与名称收窄；被否定的领域从 category/categories 剔除
    if excluded:
        neg_tech = set(excluded)  # 英文技术词
        params["tech_keywords"] = [
            k for k in extract_tech_keywords(message) if k not in neg_tech
        ]
        # 技术排除不等于方向排除：“不想学 Java”只排除 Java 课程，
        # 不应顺带清空用户同时明确提出的 backend 方向。
        neg_cat = {
            k for k, c in CATEGORY_MAP.items()
            if k in excluded and k in explicit_category_terms
        }
        neg_cat_vals = {CATEGORY_MAP[k] for k in neg_cat}
        if last_cat in neg_cat_vals:
            params["category"] = None
        if categories_ordered:
            params["categories"] = [
                c for c in categories_ordered if c not in neg_cat_vals
            ]
            # 只剩一个时退回单值
            if len(params["categories"]) == 1:
                params["category"] = params["categories"][0]
                params["categories"] = []
    else:
        params["tech_keywords"] = extract_tech_keywords(message)

    # 提取价格（"200元/100块/预算200以内"都算；"以内"不带单位也要认）
    price_match = re.search(r"(\d+)\s*(元|块|rmb|以内)", message or "")
    if price_match:
        params["max_price"] = float(price_match.group(1))

    # "免费"是明确的零元硬约束；"便宜点/太贵了"没有新金额时不生成新上限，
    # 由上下文继承已有 max_price，避免凭空把用户预算压到任意常数。
    if params["max_price"] is None:
        from app.services.query_signals import is_faq_trial_query
        if "免费" in msg_lower and not is_faq_trial_query(message):
            params["max_price"] = 0.0

    # 提取关键词（英文技术词 + 中文实词，去停用词）
    words = [
        w for w in re.findall(_WORD_PATTERN, message or "")
        if w not in _STOP_WORDS and w.lower() != "ai"
    ]
    params["keywords"] = list(set(params["keywords"] + words))[:5]

    # 否定技术词只能作为排除条件，不能同时进入正向关键词收窄。
    # 否则后续 keyword_narrow 可能先收窄到 Java，再由 excluded 过滤掉全部结果。
    if excluded:
        excluded_norm = {str(item).lower() for item in excluded}
        params["keywords"] = [
            word for word in params["keywords"]
            if str(word).lower() not in excluded_norm
        ]

    return params
