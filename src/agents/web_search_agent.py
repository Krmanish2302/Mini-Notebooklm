from typing import List, Dict, Any, Optional
import os
import requests


class WebSearchAgent:
    """
    Web search powered by Tavily API — purpose-built for RAG.

    Tavily returns clean, pre-extracted text (no raw HTML scraping needed),
    supports diverse result types (web pages, Wikipedia, news, GitHub, docs)
    and is free up to 1 000 searches / month.

    Setup
    -----
    1.  Sign up at https://tavily.com and grab your API key.
    2.  Set the environment variable::

            export TAVILY_API_KEY="tvly-..."

        Or pass it explicitly::

            agent = WebSearchAgent(api_key="tvly-...")

    Result format
    -------------
    Each result dict contains:
        id        – stable hash-based id for the UI
        title     – page title
        url       – canonical URL
        snippet   – clean extracted text snippet (ready for RAG)
        score     – Tavily relevance score (0-1)
        source_type – always "website" (so the pipeline routes it correctly)
    """

    TAVILY_SEARCH_URL = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 10,
        search_depth: str = "advanced",   # "basic" or "advanced"
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        self.max_results = max_results
        self.search_depth = search_depth
        self.include_domains = include_domains or []
        self.exclude_domains = exclude_domains or []

        if not self.api_key:
            raise ValueError(
                "Tavily API key not found. "
                "Set the TAVILY_API_KEY environment variable or pass api_key=\"tvly-...\"."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the web and return formatted results ready for display in the UI.

        Each result has the keys: id, title, url, snippet, score, source_type.
        Returns an empty list (not an exception) on error so the UI degrades
        gracefully.
        """
        if not query.strip():
            return []

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": self.max_results,
            "search_depth": self.search_depth,
            "include_answer": False,       # we only need result list
            "include_raw_content": False,  # snippet is enough for preview
        }
        if self.include_domains:
            payload["include_domains"] = self.include_domains
        if self.exclude_domains:
            payload["exclude_domains"] = self.exclude_domains

        try:
            resp = requests.post(
                self.TAVILY_SEARCH_URL,
                json=payload,
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            return [{"error": "Tavily search timed out. Please try again."}]
        except requests.exceptions.HTTPError as e:
            return [{"error": f"Tavily API error: {e.response.status_code} — {e.response.text}"}]
        except Exception as e:
            return [{"error": str(e)}]

        return self._format_results(data.get("results", []))

    def fetch_content(self, url: str) -> str:
        """
        Fetch and extract clean text content from a URL.
        Used when a user selects a search result and wants to ingest it.
        Tavily's /extract endpoint is used when available; falls back to
        trafilatura for direct scraping.
        """
        # Try Tavily extract first (returns clean text, no HTML parsing needed)
        try:
            resp = requests.post(
                "https://api.tavily.com/extract",
                json={"api_key": self.api_key, "urls": [url]},
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results and results[0].get("raw_content"):
                return results[0]["raw_content"]
        except Exception:
            pass  # Fall through to trafilatura

        # Fallback: direct HTTP + trafilatura
        try:
            import trafilatura
            page = requests.get(
                url,
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            page.raise_for_status()
            content = trafilatura.extract(page.text)
            return content or ""
        except Exception as e:
            return f"Error fetching content: {e}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_results(self, raw: List[Dict]) -> List[Dict[str, Any]]:
        """Normalise Tavily result objects into the shape expected by the UI."""
        formatted = []
        for i, r in enumerate(raw):
            title = r.get("title") or r.get("url", f"Result {i + 1}")
            formatted.append({
                "id": f"web_{i}_{abs(hash(r.get('url', str(i))))}",
                "title": title,
                "url": r.get("url", ""),
                "snippet": r.get("content") or r.get("snippet", ""),
                "score": round(r.get("score", 0.0), 3),
                "source_type": "website",
                "selected": False,
            })
        return formatted
