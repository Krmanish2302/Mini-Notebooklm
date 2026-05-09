"""
website_pipeline.py

Extracts clean article text from any URL.
Primary:  Crawl4AI  (handles JS-rendered pages, returns clean markdown)
Fallback: trafilatura (fast, lightweight, handles static HTML)
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class WebsitePipeline:
    """Extracts clean text from a website URL."""

    @staticmethod
    async def process(url: str, source_id: str) -> Dict[str, Any]:
        """
        Fetch and extract text from a URL.

        Strategy:
            1. Try Crawl4AI (async, handles JS-heavy pages).
            2. Fall back to trafilatura if Crawl4AI fails or returns empty content.

        Args:
            url:       The URL to scrape.
            source_id: Unique identifier for this source.

        Returns:
            dict with keys: content, metadata, modality

        Raises:
            ValueError: If both extraction methods fail or return empty content.
        """
        content = ""

        # ── 1. Try Crawl4AI ───────────────────────────────────────────────────
        try:
            from crawl4ai import AsyncWebCrawler
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(url=url)
                content = (result.markdown or "").strip()
                if content:
                    logger.debug("WebsitePipeline: Crawl4AI succeeded for %s", url)
        except Exception as crawl_err:
            logger.warning(
                "WebsitePipeline: Crawl4AI failed for %s — %s. Trying trafilatura.",
                url, crawl_err,
            )
            content = ""

        # ── 2. Fallback: trafilatura ───────────────────────────────────────────
        if not content:
            try:
                import requests
                import trafilatura

                resp = requests.get(
                    url,
                    timeout=30,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; MiniNotebookLM/1.0)"},
                )
                resp.raise_for_status()
                content = (trafilatura.extract(resp.text) or "").strip()
                if content:
                    logger.debug("WebsitePipeline: trafilatura succeeded for %s", url)
            except Exception as traf_err:
                logger.error(
                    "WebsitePipeline: trafilatura also failed for %s — %s", url, traf_err
                )
                content = ""

        # ── 3. Guard: both methods failed ─────────────────────────────────────
        if not content:
            raise ValueError(
                f"Could not extract readable content from {url}. "
                "The page may be behind a login, bot-protected, or empty."
            )

        return {
            "content": content,
            "metadata": {
                "url": url,
                "word_count": len(content.split()),
                "source_id": source_id,
            },
            "modality": "text",
        }
