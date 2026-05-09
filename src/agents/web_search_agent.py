from typing import List, Dict, Any
import requests
from urllib.parse import quote

class WebSearchAgent:
    """
    Searches web using DuckDuckGo and returns formatted results.
    Results can be added as sources like uploaded files.
    """
    
    DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"
    
    def __init__(self, max_results: int = 10):
        self.max_results = max_results
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search."""
        try:
            response = requests.get(
                self.DUCKDUCKGO_URL,
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=30
            )
            response.raise_for_status()
            
            # Parse results (simplified - use BeautifulSoup in production)
            results = self._parse_results(response.text)
            return results[:self.max_results]
        except Exception as e:
            return [{"error": str(e)}]
    
    def _parse_results(self, html: str) -> List[Dict[str, Any]]:
        """Parse search results from HTML."""
        # In production, use BeautifulSoup for proper parsing
        # This is a simplified version
        results = []
        # TODO: Implement proper HTML parsing
        return results
    
    def search_and_format(self, query: str) -> List[Dict[str, Any]]:
        """Search and format for UI display."""
        raw_results = self.search(query)
        
        formatted = []
        for i, result in enumerate(raw_results):
            if "error" in result:
                continue
            formatted.append({
                "id": f"web_{i}_{hash(result.get('title', ''))}",
                "title": result.get("title", "Untitled"),
                "url": result.get("url", ""),
                "snippet": result.get("snippet", ""),
                "source_type": "website",
                "selected": False
            })
        
        return formatted
    
    def fetch_content(self, url: str) -> str:
        """Fetch full content from URL."""
        try:
            response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            
            # Use trafilatura for content extraction
            import trafilatura
            content = trafilatura.extract(response.text)
            return content or ""
        except Exception as e:
            return f"Error fetching content: {e}"