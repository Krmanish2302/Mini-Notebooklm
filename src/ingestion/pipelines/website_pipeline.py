import trafilatura
from crawl4ai import AsyncWebCrawler
from typing import Dict, Any

class WebsitePipeline:
    """Extracts clean text from websites."""
    
    @staticmethod
    async def process(url: str, source_id: str) -> Dict[str, Any]:
        # Try Crawl4AI first
        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)
                content = result.markdown
        except:
            # Fallback to trafilatura
            import requests
            response = requests.get(url, timeout=30)
            content = trafilatura.extract(response.text) or ""
        
        return {
            "content": content,
            "metadata": {
                "url": url,
                "word_count": len(content.split()),
                "source_id": source_id
            },
            "modality": "text"
        }