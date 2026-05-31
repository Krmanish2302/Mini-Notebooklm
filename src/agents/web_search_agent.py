"""
web_search_agent.py — Web search powered by Tavily, wrapped as a LangChain tool.

LangChain upgrade notes
-----------------------
* Search     : LangChain TavilySearchResults tool
               (langchain-community >= 0.0.20 · pip install langchain-community tavily-python)
               Direct REST calls replaced — Tavily client handles retries,
               rate-limits, and response parsing.

* Content    : LangChain WebBaseLoader (BeautifulSoup-based, zero extra deps).
               Falls back to AsyncChromiumLoader for JS-heavy pages.
               trafilatura kept as final safety net (unchanged).

* Tool wrap  : web_search_tool() decorated with @tool so it's usable in
               any LangGraph ToolNode or create_react_agent() call.

* ToolNode   : build_search_node() returns a LangGraph ToolNode ready to
               wire into a StateGraph as a drop-in node.

Setup
-----
    pip install langchain-community tavily-python
    export TAVILY_API_KEY="tvly-..."

Result format (unchanged — UI expects these keys)
-------------------------------------------------
    id, title, url, snippet, score, source_type, selected
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RESULTS  = 10
_DEFAULT_SEARCH_DEPTH = "advanced"   # "basic" | "advanced"


# ── WebSearchAgent ────────────────────────────────────────────────────────────

class WebSearchAgent:
    """
    Web search + content fetcher backed by DuckDuckGo (no API keys needed).

    Parameters
    ----------
    api_key         : Unused. Kept for backward compatibility.
    max_results     : max search results to return  (default 10)
    search_depth    : Unused. Kept for backward compatibility.
    include_domains : allowlist of domains
    exclude_domains : blocklist of domains
    """

    def __init__(
        self,
        api_key:         Optional[str]       = None,
        max_results:     int                 = _DEFAULT_MAX_RESULTS,
        search_depth:    str                 = _DEFAULT_SEARCH_DEPTH,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> None:
        self.max_results     = max_results
        self.search_depth    = search_depth
        self.include_domains = include_domains or []
        self.exclude_domains = exclude_domains or []
        logger.info(
            "[WebSearchAgent] Ready with DuckDuckGo fallback, max=%d",
            max_results,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the web using DuckDuckGo and return results normalised for the UI.

        Returns an empty list (never raises) on error so the UI degrades
        gracefully.

        Each result: {id, title, url, snippet, score, source_type, selected}
        """
        if not query.strip():
            return []
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=self.max_results))
            return self._format_results(raw)
        except Exception as exc:
            logger.warning("[WebSearchAgent] DuckDuckGo search failed: %s", exc)
            return [{"error": str(exc)}]

    def fetch_content(self, url: str) -> str:
        """
        Fetch and extract clean text from a URL.

        Layered strategy:
          1. LangChain WebBaseLoader   — fast, BeautifulSoup-based, no extra deps
          2. AsyncChromiumLoader       — JS-heavy pages (requires playwright)
          3. trafilatura               — final fallback
        """
        # ── Layer 1: WebBaseLoader ────────────────────────────────────────────
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            loader = WebBaseLoader(web_paths=[url], requests_kwargs={"headers": headers})
            docs   = loader.load()
            if docs and docs[0].page_content.strip():
                logger.debug("[WebSearchAgent] WebBaseLoader success: %s", url)
                return docs[0].page_content.strip()
        except Exception as exc:
            logger.debug("[WebSearchAgent] WebBaseLoader failed (%s): %s", url, exc)

        # ── Layer 2: AsyncChromiumLoader (JS pages) ───────────────────────────
        try:
            from langchain_community.document_loaders import AsyncChromiumLoader
            from langchain_community.document_transformers import BeautifulSoupTransformer
            chromium_loader = AsyncChromiumLoader(urls=[url])
            raw_docs        = chromium_loader.load()
            bs_transformer  = BeautifulSoupTransformer()
            clean_docs      = bs_transformer.transform_documents(
                raw_docs, tags_to_extract=["p", "article", "section", "main"]
            )
            if clean_docs and clean_docs[0].page_content.strip():
                logger.debug("[WebSearchAgent] AsyncChromiumLoader success: %s", url)
                return clean_docs[0].page_content.strip()
        except Exception as exc:
            logger.debug("[WebSearchAgent] AsyncChromiumLoader failed (%s): %s", url, exc)

        # ── Layer 3: trafilatura ──────────────────────────────────────────────
        try:
            import requests
            import trafilatura
            page = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            page.raise_for_status()
            content = trafilatura.extract(page.text)
            if content:
                logger.debug("[WebSearchAgent] trafilatura success: %s", url)
                return content
        except Exception as exc:
            logger.warning("[WebSearchAgent] All fetch layers failed (%s): %s", url, exc)

        return f"Error: could not extract content from {url}"

    # ── LangChain tool accessor ───────────────────────────────────────────────

    @property
    def lc_tool(self) -> Any:
        return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _format_results(self, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalise DDG text search output into the shape the UI expects.
        DDGS returns: {'title', 'href', 'body'}
        Output shape: {id, title, url, snippet, score, source_type, selected}
        """
        formatted: List[Dict[str, Any]] = []
        for i, r in enumerate(raw):
            url   = r.get("href") or r.get("url") or ""
            title = r.get("title") or url or f"Result {i + 1}"
            uid   = f"web_{i}_{hashlib.md5(url.encode()).hexdigest()[:8]}"
            
            # Apply domain filters if any
            domain = url.split("//")[-1].split("/")[0]
            if self.include_domains and not any(d in domain for d in self.include_domains):
                continue
            if self.exclude_domains and any(d in domain for d in self.exclude_domains):
                continue

            formatted.append({
                "id":          uid,
                "title":       title,
                "url":         url,
                "snippet":     r.get("body") or r.get("content") or r.get("snippet") or "",
                "score":       round(1.0 - (i * 0.05), 3),
                "source_type": "website",
                "selected":    False,
            })
        return formatted


# ── @tool — usable in LangGraph ToolNode ─────────────────────────────────────

_default_agent: Optional[WebSearchAgent] = None


def _get_default_agent() -> WebSearchAgent:
    """Lazy singleton — only instantiated when the tool is first called."""
    global _default_agent
    if _default_agent is None:
        _default_agent = WebSearchAgent()
    return _default_agent


@tool
def web_search_tool(query: str) -> List[Dict[str, Any]]:
    """
    Search the web using Tavily and return structured results.

    Use this tool when the user asks about current events, recent news,
    or information that may not be in the ingested documents.

    Args:
        query: The search query string.

    Returns:
        List of result dicts with keys: id, title, url, snippet, score,
        source_type, selected.
    """
    return _get_default_agent().search(query)


@tool
def fetch_url_tool(url: str) -> str:
    """
    Fetch and extract clean text content from a URL.

    Use this tool when the user wants to ingest a specific web page
    or when a search result needs its full content loaded.

    Args:
        url: The URL to fetch content from.

    Returns:
        Extracted text content of the page.
    """
    return _get_default_agent().fetch_content(url)


# ── LangGraph ToolNode builder ────────────────────────────────────────────────

def build_search_node(
    api_key:         Optional[str]       = None,
    max_results:     int                 = _DEFAULT_MAX_RESULTS,
    search_depth:    str                 = _DEFAULT_SEARCH_DEPTH,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
) -> ToolNode:
    """
    Build a LangGraph ToolNode wrapping web_search_tool + fetch_url_tool.

    Wire this directly into a StateGraph:

        from src.agents import build_search_node
        from langgraph.graph import StateGraph

        graph = StateGraph(AgentState)
        graph.add_node("web_search", build_search_node())
        graph.add_edge("agent", "web_search")

    The ToolNode automatically:
      - Reads tool_calls from the last AIMessage in state["messages"]
      - Dispatches to the correct tool
      - Appends ToolMessages back to state["messages"]

    Parameters
    ----------
    api_key, max_results, search_depth, include_domains, exclude_domains
        Passed to WebSearchAgent — override env-var defaults here if needed.
    """
    # Configure the lazy singleton with explicit settings if provided
    global _default_agent
    _default_agent = WebSearchAgent(
        api_key=api_key,
        max_results=max_results,
        search_depth=search_depth,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
    )
    return ToolNode(tools=[web_search_tool, fetch_url_tool])