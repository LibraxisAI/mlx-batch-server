"""
Web Search Tool - Search the web using Brave Search API.

Provides web search capability as a hosted tool for the Responses API.
Requires BRAVE_API_KEY environment variable.

Created by M&K (c)2026 The LibraxisAI Team
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


async def execute_web_search(
    query: str,
    count: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Execute a web search using Brave Search API.

    Args:
        query: Search query string
        count: Number of results to return (default: 5)
        **kwargs: Additional parameters (ignored)

    Returns:
        Search results dict with 'results' list

    Raises:
        ValueError: If BRAVE_API_KEY is not configured
        httpx.HTTPError: If the API request fails
    """
    api_key = os.environ.get("BRAVE_API_KEY")

    if not api_key:
        logger.warning("BRAVE_API_KEY not set - web_search tool unavailable")
        return {
            "error": "Web search not configured (missing BRAVE_API_KEY)",
            "results": [],
        }

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
    }

    params: dict[str, str | int] = {
        "q": query,
        "count": min(count, 10),  # Cap at 10 results
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                BRAVE_SEARCH_URL,
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()

        # Extract web results
        web_results = data.get("web", {}).get("results", [])

        results = []
        for item in web_results[:count]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                }
            )

        logger.info(f"Web search for '{query}' returned {len(results)} results")

        return {
            "query": query,
            "results": results,
            "results_count": len(results),
        }

    except httpx.HTTPStatusError as e:
        logger.error(f"Brave Search API error: {e.response.status_code}")
        return {
            "error": f"Search API error: {e.response.status_code}",
            "results": [],
        }
    except httpx.RequestError as e:
        logger.error(f"Brave Search request failed: {e}")
        return {
            "error": f"Search request failed: {e}",
            "results": [],
        }
