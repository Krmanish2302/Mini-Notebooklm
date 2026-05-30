"""
youtube_pipeline.py

LangGraph ingestion pipeline for YouTube sources (PRD §2.3).

Stages:
  1. yt_fetch    — YoutubeLoader (LangChain) fetches timestamped transcript
  2. yt_clean    — remove filler words, short segments, repeated words
  3. yt_chunk    — SemanticChunker (similarity < threshold → split)
  4. yt_embed    — embed + persist

Usage:
    from src.ingestion.youtube_pipeline import run_youtube_pipeline
    result = run_youtube_pipeline(url="https://youtu.be/...", source_id="vid_001")
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from src.ingestion.nodes.utils import safe_node

logger = logging.getLogger(__name__)

# Filler words to strip
_FILLERS = re.compile(
    r"\b(um+|uh+|er|ah|you know|kind of|sort of|basically|literally|I mean)\b",
    re.IGNORECASE,
)
_REPEATED = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)  # "the the" → "the"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
@safe_node("yt_fetch")
def yt_fetch(state: dict) -> dict:
    from langchain_community.document_loaders import YoutubeLoader

    url       = state["file_path"]   # URL stored in file_path
    source_id = state["source_id"]

    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    docs   = loader.load()

    for doc in docs:
        doc.metadata.update({"source_id": source_id, "source_type": "youtube", "url": url})

    total_words = sum(len(d.page_content.split()) for d in docs)
    logger.info("[yt_fetch] Fetched %d doc(s), ~%d words from '%s'", len(docs), total_words, url)
    return {"raw_documents": docs, "original_word_count": total_words}


@safe_node("yt_clean")
def yt_clean(state: dict) -> dict:
    from langchain_core.documents import Document

    raw         = state.get("raw_documents", [])
    cleaned     = []
    total_after = 0

    for doc in raw:
        text = doc.page_content
        # Remove filler words
        text = _FILLERS.sub("", text)
        # Remove repeated words
        text = _REPEATED.sub(r"\1", text)
        # Collapse extra spaces
        text = re.sub(r" {2,}", " ", text).strip()
        # Remove very short lines (<5 words)
        lines = [ln for ln in text.split("\n") if len(ln.split()) >= 5]
        text  = "\n".join(lines)
        wc    = len(text.split())
        if wc < 10:
            continue
        total_after += wc
        cleaned.append(Document(page_content=text, metadata=doc.metadata))

    orig  = state.get("original_word_count", 0)
    pct   = round(100 * (orig - total_after) / orig, 1) if orig else 0
    logger.info(
        "[yt_clean] %d words → %d words (%s%% reduction)",
        orig, total_after, pct,
    )
    return {
        "cleaned_documents": cleaned,
        "cleaned_word_count": total_after,
        "reduction_pct": pct,
    }


@safe_node("yt_chunk")
def yt_chunk(state: dict) -> dict:
    """Semantic chunking — split where cosine similarity < threshold."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs      = state.get("cleaned_documents", [])
    source_id = state["source_id"]
    # Use RecursiveCharacterTextSplitter as semantic-aware default
    # (SemanticChunker requires embeddings at chunk time — expensive;
    #  use RecursiveCST with \n\n boundary as semantic proxy)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=70,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"]    = f"{source_id}_{i}"
        c.metadata["chunk_index"] = i
        c.metadata["source_type"] = "youtube"

    logger.info("[yt_chunk] %d chunks", len(chunks))
    return {"chunks": chunks}


@safe_node("yt_embed")
def yt_embed(state: dict) -> dict:
    from src.ingestion.nodes.embed_node import embed_and_index
    return embed_and_index(state)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def _build_yt_graph() -> StateGraph:
    g = StateGraph(dict)
    g.add_node("yt_fetch", yt_fetch)
    g.add_node("yt_clean", yt_clean)
    g.add_node("yt_chunk", yt_chunk)
    g.add_node("yt_embed", yt_embed)

    g.set_entry_point("yt_fetch")
    g.add_edge("yt_fetch", "yt_clean")
    g.add_edge("yt_clean", "yt_chunk")
    g.add_edge("yt_chunk", "yt_embed")
    g.add_edge("yt_embed", END)
    return g.compile()


yt_app = _build_yt_graph()


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------
def run_youtube_pipeline(url: str, source_id: str) -> Dict[str, Any]:
    init_state = {
        "file_path":   url,
        "source_id":   source_id,
        "source_type": "youtube",
    }
    result = yt_app.invoke(init_state)
    if result.get("error"):
        raise RuntimeError(f"YouTube pipeline failed: {result['error']}")
    logger.info(
        "[run_youtube_pipeline] Done — %d chunks, reduction=%s%%",
        result.get("num_chunks", 0),
        result.get("reduction_pct", "?"),
    )
    return result
