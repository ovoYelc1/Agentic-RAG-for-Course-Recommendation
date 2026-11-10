"""共享的查询语义信号。

路由、缓存跳过和上下文过滤必须使用同一套课程精化词表，避免同一句话
在不同层被判断成 FAQ 或课程请求。
"""
import re


_SPOKEN_FILTER_WORDS = {
    "便宜", "便宜点", "便宜些", "便宜一些", "便宜一点",
    "再便宜", "更便宜", "低价", "实惠", "划算",
    "太贵", "太贵了", "贵了", "嫌贵", "超预算",
    "能退吗", "可以退吗", "能退款吗", "可以退款吗",
    "有没有", "有没", "推荐", "想要", "告诉我", "谢谢",
}


def filter_search_keywords(values) -> list:
    """过滤不具备课程实体意义的口语词，供 search_filters 使用。

    课程槽位保留原始 keywords 以兼容历史状态；这里只清洗实际会进入
    keyword_narrow / 缓存上下文的过滤条件。
    """
    out = []
    seen = set()
    for value in values or []:
        word = str(value).strip()
        normalized = re.sub(r"\s+", "", word).lower()
        if not normalized or normalized in _SPOKEN_FILTER_WORDS:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(word)
    return out


COURSE_REFINEMENT_RE = re.compile(
    r"便宜点|便宜些|便宜一些|便宜一点|再便宜|更便宜|"
    r"低价|实惠|划算|预算|超预算|嫌贵|太贵|贵了|"
    r"免费课程|免费.*课|有没有.*课|有没.*课|找.*课|推荐.*课|"
    r"\d+(?:\.\d+)?\s*(?:元|块|rmb)?以内",
)
FAQ_TRIAL_RE = re.compile(r"免费章节|免费试学|免费试听|试听|先看再买")


def has_course_refinement_signal(query: str) -> bool:
    return bool(COURSE_REFINEMENT_RE.search((query or "").lower()))


def is_faq_trial_query(query: str) -> bool:
    """试看/免费章节政策问题不是零元课程价格过滤。"""
    return bool(FAQ_TRIAL_RE.search((query or "").lower()))
