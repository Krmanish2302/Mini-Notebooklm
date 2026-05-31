"""
website_pipeline.py

LangGraph ingestion pipeline for website/URL sources (PRD §2.6).

Stages:
  1. web_fetch      — Crawl4AI (JS rendering); fallback to WebBaseLoader
  2. web_clean      — strip nav/footer/sidebar/ads, keep article body
  3. web_chunk      — heading-based splits; recursive fallback
  4. web_embed      — embed + persist

Usage:
    from src.ingestion.website_pipeline import run_website_pipeline
    result = run_website_pipeline(url="https://...", source_id="web_001")
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from src.ingestion.state import IngestionState

from src.ingestion.nodes.utils import safe_node

logger = logging.getLogger(__name__)

# HTML elements / patterns to strip
_NOISE_PATTERNS = [
    re.compile(r"<nav[^>]*>.*?</nav>",         re.DOTALL | re.IGNORECASE),
    re.compile(r"<header[^>]*>.*?</header>",   re.DOTALL | re.IGNORECASE),
    re.compile(r"<footer[^>]*>.*?</footer>",   re.DOTALL | re.IGNORECASE),
    re.compile(r"<aside[^>]*>.*?</aside>",     re.DOTALL | re.IGNORECASE),
    re.compile(r"<script[^>]*>.*?</script>",   re.DOTALL | re.IGNORECASE),
    re.compile(r"<style[^>]*>.*?</style>",     re.DOTALL | re.IGNORECASE),
    re.compile(r"<!--.*?-->",                  re.DOTALL),
]
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
@safe_node("web_fetch")
def web_fetch(state: dict) -> dict:
    url       = state["file_path"]
    source_id = state["source_id"]
    domain    = urlparse(url).netloc

    # Try Crawl4AI first (JS rendering)
    try:
        import asyncio
        from crawl4ai import AsyncWebCrawler

        async def _crawl():
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(url=url)
                return result.markdown or result.cleaned_html or ""

        content = asyncio.run(_crawl())
        origin  = "crawl4ai"
        logger.info("[web_fetch] Crawl4AI success — %d chars from '%s'", len(content), url)

    except Exception as crawl_err:
        logger.warning("[web_fetch] Crawl4AI failed (%s), falling back to WebBaseLoader", crawl_err)
        try:
            from langchain_community.document_loaders import WebBaseLoader
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            docs    = WebBaseLoader(url, requests_kwargs={"headers": headers}).load()
            content = "\n\n".join(d.page_content for d in docs)
            origin  = "webbaseloader"
        except Exception as wb_err:
            raise RuntimeError(f"Both fetch methods failed: {wb_err}") from wb_err

    doc = Document(
        page_content=content,
        metadata={
            "source_id":   source_id,
            "source_type": "website",
            "url":         url,
            "domain":      domain,
            "fetch_origin": origin,
        },
    )
    logger.info("[web_fetch] Raw content %d chars", len(content))
    return {"raw_documents": [doc], "original_char_count": len(content)}


@safe_node("web_clean")
def web_clean(state: dict) -> dict:
    raw  = state.get("raw_documents", [])
    docs = []

    for doc in raw:
        text = doc.page_content

        # Strip known noise HTML patterns
        for pat in _NOISE_PATTERNS:
            text = pat.sub("", text)

        # Strip remaining HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Strip cookie banners / share widgets / comment sections
        noise_phrases = [
            r"share this article",
            r"related articles?",
            r"you might also like",
            r"about the author",
            r"leave a reply",
            r"accept (all )?cookies?",
            r"subscribe to our newsletter",
        ]
        for phrase in noise_phrases:
            text = re.sub(
                rf"(?i)({phrase}).*?(?=\n\n|$)", "", text, flags=re.DOTALL
            )

        # Normalize whitespace
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        after_chars = len(text)
        orig_chars  = state.get("original_char_count", len(text))
        reduction   = round(100 * (orig_chars - after_chars) / orig_chars, 1) if orig_chars else 0

        docs.append(Document(
            page_content=text,
            metadata={**doc.metadata, "after_char_count": after_chars},
        ))

        logger.info(
            "[web_clean] %d chars → %d chars (%s%% reduction)",
            orig_chars, after_chars, reduction,
        )

    return {"cleaned_documents": docs}


@safe_node("web_chunk")
def web_chunk(state: dict) -> dict:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs      = state.get("cleaned_documents", [])
    source_id = state["source_id"]
    chunks    = []
    chunk_idx = 0

    for doc in docs:
        text = doc.page_content
        url  = doc.metadata.get("url", "")

        # Attempt heading-based split
        heading_splits = _HEADING_RE.split(text)

        if len(heading_splits) > 3:  # headings found
            # heading_splits: [pre, level, heading, body, level, heading, body, ...]
            i = 0
            current_heading = "Introduction"
            current_level   = 1
            while i < len(heading_splits):
                part = heading_splits[i]
                if i + 2 < len(heading_splits) and heading_splits[i].startswith("#"):
                    current_level   = len(heading_splits[i])
                    current_heading = heading_splits[i + 1].strip()
                    body            = heading_splits[i + 2].strip()
                    i += 3
                else:
                    body            = part.strip()
                    i += 1

                if not body or len(body.split()) < 10:
                    continue

                # Sub-divide if section > 600 tokens (≈2400 chars)
                if len(body) > 2400:
                    sub_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=2400, chunk_overlap=200,
                        separators=["\n\n", "\n", ". ", " "],
                    )
                    sub_chunks = sub_splitter.split_text(body)
                else:
                    sub_chunks = [body]

                for sub in sub_chunks:
                    chunks.append(Document(
                        page_content=sub,
                        metadata={
                            **doc.metadata,
                            "chunk_id":        f"{source_id}_{chunk_idx}",
                            "chunk_index":     chunk_idx,
                            "section_heading": current_heading,
                            "heading_level":   current_level,
                            "url":             url,
                        },
                    ))
                    chunk_idx += 1
        else:
            # No headings — fallback to recursive text splitting
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            for sub in splitter.split_documents([doc]):
                sub.metadata["chunk_id"]    = f"{source_id}_{chunk_idx}"
                sub.metadata["chunk_index"] = chunk_idx
                sub.metadata["source_type"] = "website"
                chunks.append(sub)
                chunk_idx += 1

    logger.info("[web_chunk] %d chunks from %d doc(s)", len(chunks), len(docs))
    return {"chunks": chunks}


@safe_node("web_embed")
def web_embed(state: dict) -> dict:
    from src.ingestion.nodes.embed_node import embed_and_index
    return embed_and_index(state)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def _build_website_graph() -> StateGraph:
    g = StateGraph(IngestionState)
    g.add_node("web_fetch", web_fetch)
    g.add_node("web_clean", web_clean)
    g.add_node("web_chunk", web_chunk)
    g.add_node("web_embed", web_embed)

    g.set_entry_point("web_fetch")
    g.add_edge("web_fetch", "web_clean")
    g.add_edge("web_clean", "web_chunk")
    g.add_edge("web_chunk", "web_embed")
    g.add_edge("web_embed", END)
    return g.compile()


website_app = _build_website_graph()


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------
def run_website_pipeline(url: str, source_id: str, source_name: Optional[str] = None) -> Dict[str, Any]:
    init_state = {
        "file_path":   url,
        "source_id":   source_id,
        "source_type": "website",
        "source_name": source_name,
    }
    result = website_app.invoke(init_state)
    if result.get("error"):
        raise RuntimeError(f"Website pipeline failed: {result['error']}")
    logger.info("[run_website_pipeline] Done — %d chunks", result.get("num_chunks", 0))
    return result
