"""
pdf_pipeline.py

LangGraph ingestion pipeline for PDF sources.

Stages (PRD §2.2):
  1. extract_node   — PyMuPDF page-by-page extraction
  2. analyze_node   — 20% sample → stats for every chunking strategy
  3. [PAUSE]        — caller presents stats, passes user choice back into state
  4. chunk_node     — chunk with chosen strategy
  5. embed_node     — embed + persist to MultiFAISSStore

Usage:
    from src.ingestion.pdf_pipeline import run_pdf_pipeline, analyze_pdf

    # Step 1 — get analysis stats (for UI to display)
    stats = analyze_pdf(file_path, source_id)

    # Step 2 — user picks strategy + model, then run full pipeline
    result = run_pdf_pipeline(
        file_path=file_path,
        source_id=source_id,
        strategy="paragraph_based",   # user choice
        embedding_dim=384,            # user choice
    )
"""
from __future__ import annotations

import logging
import os
import re
import statistics
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from src.ingestion.nodes.utils import safe_node

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "data/vectorstores")
ANALYSIS_SAMPLE  = float(os.getenv("PDF_ANALYSIS_SAMPLE", "0.20"))  # 20%

STRATEGY_DESCRIPTIONS = {
    "page_based":       "One chunk per page",
    "paragraph_based":  "Split at blank lines",
    "sentence_based":   "Each sentence is a chunk",
    "fixed_256":        "Fixed 256-token windows",
    "fixed_512":        "Fixed 512-token windows",
    "chapter_based":    "Split at detected headings",
    "semantic":         "Split where semantic similarity drops",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _token_estimate(text: str) -> int:
    return int(len(text.split()) * 1.33)


def _detect_headings(text: str) -> bool:
    patterns = [
        r"^(Chapter|CHAPTER|Section|SECTION|Part|PART)\s+\d+",
        r"^[A-Z][A-Z\s]{4,}$",
    ]
    for line in text.split("\n"):
        for p in patterns:
            if re.match(p, line.strip()):
                return True
    return False


def _chunks_for_strategy(docs: List[Document], strategy: str) -> List[str]:
    """Return list of chunk texts for a strategy — used only for stat computation."""
    texts = [d.page_content for d in docs]

    if strategy == "page_based":
        return texts

    if strategy == "paragraph_based":
        result = []
        for t in texts:
            result.extend([p.strip() for p in t.split("\n\n") if p.strip()])
        return result

    if strategy == "sentence_based":
        result = []
        for t in texts:
            result.extend([s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()])
        return result

    if strategy == "fixed_256":
        result = []
        for t in texts:
            words = t.split()
            for i in range(0, len(words), 256):
                chunk = " ".join(words[i:i+256])
                if chunk:
                    result.append(chunk)
        return result

    if strategy == "fixed_512":
        result = []
        for t in texts:
            words = t.split()
            for i in range(0, len(words), 512):
                chunk = " ".join(words[i:i+512])
                if chunk:
                    result.append(chunk)
        return result

    if strategy == "chapter_based":
        heading_re = re.compile(
            r"^(Chapter|CHAPTER|Section|SECTION|Part|PART)\s+\d+|^[A-Z][A-Z\s]{4,}$",
            re.MULTILINE,
        )
        full = "\n\n".join(texts)
        parts = heading_re.split(full)
        return [p.strip() for p in parts if p and len(p.split()) > 10]

    # semantic — approximate with paragraph_based for stat purposes
    result = []
    for t in texts:
        result.extend([p.strip() for p in t.split("\n\n") if p.strip()])
    return result


def _compute_stats(chunks: List[str]) -> Dict[str, Any]:
    if not chunks:
        return {}
    token_counts = [_token_estimate(c) for c in chunks]
    mean   = statistics.mean(token_counts)
    median = statistics.median(token_counts)
    stddev = statistics.stdev(token_counts) if len(token_counts) > 1 else 0.0
    under50  = sum(1 for t in token_counts if t < 50)
    over400  = sum(1 for t in token_counts if t > 400)
    return {
        "estimated_chunks": len(chunks),
        "avg_tokens":        round(mean, 1),
        "median_tokens":     round(median, 1),
        "std_dev":           round(stddev, 1),
        "min_tokens":        min(token_counts),
        "max_tokens":        max(token_counts),
        "pct_under_50":      round(100 * under50 / len(chunks), 1),
        "pct_over_400":      round(100 * over400  / len(chunks), 1),
    }


def _recommend(stats_by_strategy: Dict[str, Dict], has_headings: bool) -> str:
    if has_headings:
        return "chapter_based"
    para_avg = stats_by_strategy.get("paragraph_based", {}).get("avg_tokens", 150)
    if para_avg > 300:
        return "fixed_512"
    if para_avg < 80:
        return "sentence_based"
    return "paragraph_based"


def _recommend_embedding(avg_tokens: float) -> str:
    if avg_tokens < 200:
        return "all-MiniLM-L6-v2"
    return "nomic-embed-text-v1.5"


# ---------------------------------------------------------------------------
# Public: analyze_pdf (call before showing UI)
# ---------------------------------------------------------------------------
def analyze_pdf(file_path: str, source_id: str) -> Dict[str, Any]:
    """
    Extract + sample 20% of pages, compute stats for each strategy.
    Returns a dict the UI uses to populate the PDF Analysis Panel.
    """
    from langchain_community.document_loaders import PyMuPDFLoader

    docs: List[Document] = PyMuPDFLoader(file_path).load()
    total_pages = len(docs)
    sample_size = max(1, int(total_pages * ANALYSIS_SAMPLE))

    # Stratified sample — evenly spaced
    step  = max(1, total_pages // sample_size)
    sample_docs = docs[::step][:sample_size]

    has_headings = any(_detect_headings(d.page_content) for d in sample_docs)

    stats_by_strategy: Dict[str, Dict] = {}
    for strategy in STRATEGY_DESCRIPTIONS:
        chunks = _chunks_for_strategy(sample_docs, strategy)
        stats  = _compute_stats(chunks)
        # Extrapolate estimated_chunks to full doc
        if stats and sample_size < total_pages:
            ratio = total_pages / sample_size
            stats["estimated_chunks"] = int(stats["estimated_chunks"] * ratio)
        stats_by_strategy[strategy] = stats

    recommended_strategy  = _recommend(stats_by_strategy, has_headings)
    recommended_embedding = _recommend_embedding(
        stats_by_strategy.get(recommended_strategy, {}).get("avg_tokens", 150)
    )

    total_words = sum(len(d.page_content.split()) for d in docs)

    return {
        "source_id":             source_id,
        "file_path":             file_path,
        "total_pages":           total_pages,
        "total_words_estimated": total_words,
        "has_headings":          has_headings,
        "sample_pages":          sample_size,
        "strategies":            stats_by_strategy,
        "recommended_strategy":  recommended_strategy,
        "recommended_embedding": recommended_embedding,
        "strategy_descriptions": STRATEGY_DESCRIPTIONS,
    }


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------
@safe_node("pdf_extract")
def pdf_extract(state: dict) -> dict:
    from langchain_community.document_loaders import PyMuPDFLoader
    file_path = state["file_path"]
    docs = PyMuPDFLoader(file_path).load()
    # tag each page
    for i, doc in enumerate(docs):
        doc.metadata.update({
            "source_id":   state["source_id"],
            "source_type": "pdf",
            "page_number": i + 1,
        })
    logger.info("[pdf_extract] %d pages loaded from '%s'", len(docs), file_path)
    return {"raw_documents": docs, "total_pages": len(docs)}


@safe_node("pdf_chunk")
def pdf_chunk(state: dict) -> dict:
    """Chunk with strategy chosen by user (default: paragraph_based)."""
    docs      = state.get("raw_documents", [])
    strategy  = state.get("strategy", "paragraph_based")
    source_id = state["source_id"]

    if strategy == "paragraph_based":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=80,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)

    elif strategy == "fixed_512":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

    elif strategy == "fixed_256":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

    elif strategy == "sentence_based":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=30,
            separators=[". ", "! ", "? ", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)

    elif strategy == "chapter_based":
        from langchain_text_splitters import MarkdownHeaderTextSplitter
        # Fallback: treat ALL-CAPS lines as section boundaries via recursive
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, chunk_overlap=200,
            separators=["\n\n\n", "\n\n", "\n", ". ", " "],
        )
        chunks = splitter.split_documents(docs)

    elif strategy == "page_based":
        chunks = docs  # one chunk per page

    else:  # semantic or unknown — default to paragraph
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=80,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)

    for i, c in enumerate(chunks):
        c.metadata["chunk_id"]       = f"{source_id}_{i}"
        c.metadata["chunk_index"]    = i
        c.metadata["strategy_used"]  = strategy
        c.metadata["source_type"]    = "pdf"

    logger.info("[pdf_chunk] strategy=%s → %d chunks", strategy, len(chunks))
    return {"chunks": chunks}


@safe_node("pdf_embed")
def pdf_embed(state: dict) -> dict:
    from src.ingestion.nodes.embed_node import embed_and_index
    return embed_and_index(state)


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
def _build_pdf_graph() -> StateGraph:
    g = StateGraph(dict)
    g.add_node("pdf_extract", pdf_extract)
    g.add_node("pdf_chunk",   pdf_chunk)
    g.add_node("pdf_embed",   pdf_embed)

    g.set_entry_point("pdf_extract")
    g.add_edge("pdf_extract", "pdf_chunk")
    g.add_edge("pdf_chunk",   "pdf_embed")
    g.add_edge("pdf_embed",   END)
    return g.compile()


pdf_app = _build_pdf_graph()


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------
def run_pdf_pipeline(
    file_path:     str,
    source_id:     str,
    strategy:      str = "paragraph_based",
    embedding_dim: int = 384,
) -> Dict[str, Any]:
    """
    Run the full PDF ingestion pipeline.
    Call analyze_pdf() first to get stats, then pass user choices here.
    """
    init_state = {
        "file_path":      file_path,
        "source_id":      source_id,
        "strategy":       strategy,
        "embedding_dim":  embedding_dim,
        "source_type":    "pdf",
    }
    result = pdf_app.invoke(init_state)
    if result.get("error"):
        raise RuntimeError(f"PDF pipeline failed: {result['error']}")
    logger.info(
        "[run_pdf_pipeline] Done — %d chunks, store='%s'",
        result.get("num_chunks", 0),
        result.get("vectorstore_path", ""),
    )
    return result
