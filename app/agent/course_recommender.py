# 课程推荐器 - 匹配理由生成（纯函数，无 DB 依赖）
#
# 历史说明：原 recommend_courses()（LLM 提参 + MySQL 直查）已在检索层
# 迁移到 RAG（app/rag/course_search.search_courses）后成为死代码并被移除；
# 结构化参数提取见 app/agent/router.extract_recommend_params。
# 本模块保留 _generate_match_reason，负责为推荐结果生成人类可读的匹配理由，
# 被 agent.py / tools.py / api/routes.py 三处复用。
from typing import Dict


def _generate_match_reason(course: Dict, params: Dict) -> str:
    """
    生成课程匹配理由

    示例: "匹配关键词: Java, 适合初学者, 分类: backend"
    """
    reasons = []

    # 关键词匹配
    keywords = params.get("keywords", [])
    if keywords:
        matched = [kw for kw in keywords if kw in course.get("course_name", "") or kw in course.get("description", "")]
        if matched:
            reasons.append(f"匹配关键词: {', '.join(matched)}")

    # 难度匹配
    if params.get("level") and course.get("level") == params["level"]:
        level_cn = {"beginner": "初学者", "intermediate": "进阶者", "advanced": "高级开发者"}
        reasons.append(f"适合{level_cn.get(params['level'], params['level'])}")

    # 分类匹配
    if params.get("category") and course.get("category") == params["category"]:
        cat_cn = {"backend": "后端开发", "frontend": "前端开发", "data": "数据分析", "devops": "运维部署"}
        reasons.append(f"属于{cat_cn.get(params['category'], params['category'])}领域")

    # 价格匹配
    if params.get("max_price"):
        try:
            if float(course.get("price") or 0) <= float(params["max_price"]):
                reasons.append(f"价格在预算内 (¥{course.get('price', 0)})")
        except (TypeError, ValueError):
            pass

    # 评分优先
    rating = course.get("avg_rating")
    if rating and rating >= 4.5:
        reasons.append(f"高分好评课程 ({rating}分)")

    return "；".join(reasons) if reasons else "综合匹配度较高"
