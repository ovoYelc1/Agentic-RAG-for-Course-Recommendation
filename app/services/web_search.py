# 联网搜索服务 - 多供应商支持（Tavily / SerpAPI / Serper / DuckDuckGo 兜底）
#
# 设计目标：让 AI 在"知识库/数据库没有答案"时，真正能去网上找资料，而不是说"抱歉我不确定"。
# 优先级：如果配置了任一搜索 API Key，则使用该供应商（结果最稳定）；
#         否则使用 DuckDuckGo HTML 搜索（无需 Key，尽力而为，可能受网络环境影响）。
#
# 环境变量：
#   WEB_SEARCH_PROVIDER  # 可选：tavily | serpapi | serper | duckduckgo（不填则自动按可用 Key 选择）
#   TAVILY_API_KEY       # Tavily 搜索（推荐，专为 LLM 设计）
#   SERPAPI_KEY          # SerpAPI（Google 搜索结果）
#   SERPER_API_KEY       # Serper.dev（Google 搜索结果）
import os
import re
import json
import logging

import httpx

logger = logging.getLogger("web_search")


async def web_search(query: str, num_results: int = 5) -> list:
    """
    联网搜索，返回结果列表：[{'title': str, 'url': str, 'content': str}]

    优先使用配置了 API Key 的供应商，否则用 DuckDuckGo（无需 Key，尽力而为）。
    任何异常都会被捕获并返回空列表（不会让上层崩溃或触发"抱歉"话术）。
    """
    if not query or not query.strip():
        return []

    provider = (os.getenv("WEB_SEARCH_PROVIDER") or "").lower().strip()
    if not provider:
        # 自动按可用 Key 选择供应商
        if os.getenv("TAVILY_API_KEY"):
            provider = "tavily"
        elif os.getenv("SERPAPI_KEY"):
            provider = "serpapi"
        elif os.getenv("SERPER_API_KEY"):
            provider = "serper"
        else:
            provider = "duckduckgo"

    try:
        if provider == "tavily":
            return await _search_tavily(query, num_results)
        elif provider == "serpapi":
            return await _search_serpapi(query, num_results)
        elif provider == "serper":
            return await _search_serper(query, num_results)
        else:
            return await _search_duckduckgo(query, num_results)
    except Exception as exc:
        status = (
            f", HTTP {exc.response.status_code}"
            if isinstance(exc, httpx.HTTPStatusError)
            else ""
        )
        logger.error(
            "%s 搜索失败 (%s%s)", provider, type(exc).__name__, status
        )
        return []


# ====================== 各供应商实现 ======================

async def _search_tavily(query: str, num_results: int) -> list:
    api_key = os.getenv("TAVILY_API_KEY")
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": num_results,
                "search_depth": "basic",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": (r.get("content") or "")[:400],
            })
        return results[:num_results]


async def _search_serpapi(query: str, num_results: int) -> list:
    api_key = os.getenv("SERPAPI_KEY")
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            "https://serpapi.com/search.json",
            params={"engine": "google", "q": query, "num": num_results, "api_key": api_key},
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("organic_results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "content": (r.get("snippet") or "")[:400],
            })
        return results[:num_results]


async def _search_serper(query: str, num_results: int) -> list:
    api_key = os.getenv("SERPER_API_KEY")
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("organic", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "content": (r.get("snippet") or "")[:400],
            })
        return results[:num_results]


async def _search_duckduckgo(query: str, num_results: int) -> list:
    """
    DuckDuckGo HTML 搜索（无需 Key，尽力而为）。
    注意：DuckDuckGo 对自动化请求有一定限制，结果可能不稳定或为空。
          生产环境建议配置 Tavily / SerpAPI / Serper 的 Key。
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Content-Type": "application/x-www-form-urlencoded",
    }
    async with httpx.AsyncClient(timeout=20, headers=headers, follow_redirects=True) as client:
        resp = await client.post("https://html.duckduckgo.com/html/", data={"q": query})
        resp.raise_for_status()
        html = resp.text

    results = []
    # 结果标题与链接
    title_blocks = re.findall(
        r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.S
    )
    snippet_blocks = re.findall(
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', html, re.S
    )
    for i, (href, title) in enumerate(title_blocks[:num_results]):
        snippet = snippet_blocks[i] if i < len(snippet_blocks) else ""
        results.append({
            "title": _strip_tags(title),
            "url": _clean_ddg_url(href),
            "content": _strip_tags(snippet)[:400],
        })
    return results


def _clean_ddg_url(href: str) -> str:
    """DuckDuckGo 结果链接是 302 跳转，尝试提取真实的 uddg 参数。"""
    m = re.search(r"uddg=([^&]+)", href)
    if m:
        from urllib.parse import unquote

        return unquote(m.group(1))
    return href


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


if __name__ == "__main__":
    # 简单的自检测试
    import asyncio

    async def _test():
        res = await web_search("Python 数据分析 入门", 3)
        print(json.dumps(res, ensure_ascii=False, indent=2))

    asyncio.run(_test())
