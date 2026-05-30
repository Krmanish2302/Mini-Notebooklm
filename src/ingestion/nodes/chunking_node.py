"""
chunking_node.py

LangGraph nodes: split cleaned Document objects into retrievable chunks.

Two chunkers available — the LangGraph router (choose_chunker) picks one:

  recursive_chunk  — RecursiveCharacterTextSplitter (default, fast)
  semantic_chunk   — SemanticChunker (embedding-based, higher quality)

The chunker is chosen based on state["source_type"]:
  - PDFs / text  → recursive_chunk  (unless SEMANTIC_CHUNKING_ENABLED=true)
  - Any source   → semantic_chunk   if env var SEMANTIC_CHUNKING_ENABLED="true"

Each output Document carries extra metadata:
  {
    "chunk_id":    "<source_id>_<index>",
    "chunk_index": <int>,
    "chunk_total": <int>,       # set in embed_node after all chunks known
    "chunker":     "recursive" | "semantic",
  }
"""
from __future__ import annotations

import os
import logging
from typing import List
from langchain_core.documents import Document
from .utils import safe_node

logger = logging.getLogger(__name__)

# ── Config (override via env) ──────────────────────────────────────────────────
CHUNK_SIZE              = int(os.getenv("CHUNK_SIZE",    "1000"))
CHUNK_OVERLAP           = int(os.getenv("CHUNK_OVERLAP", "200"))
SEMANTIC_CHUNKING       = os.getenv("SEMANTIC_CHUNKING_ENABLED", "false").lower() == "true"


def _tag_chunks(chunks: List[Document], source_id: str, chunker: str) -> List[Document]:
    """Add chunk_id / chunk_index metadata to every chunk."""
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"]    = f"{source_id}_{i}"
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunker"]     = chunker
    return chunks


# ── Router ─────────────────────────────────────────────────────────────────────

def choose_chunker(state: dict) -> str:
    """
    LangGraph conditional-edge router.
    Returns the name of the next node to execute.
    """
    if SEMANTIC_CHUNKING:
        logger.info("[choose_chunker] → semantic_chunk")
        return "semantic_chunk"
    logger.info("[choose_chunker] → recursive_chunk")
    return "recursive_chunk"


# ── Node: Recursive character splitter ─────────────────────────────────────────

@safe_node("recursive_chunk")
def recursive_chunk(state: dict) -> dict:
    """
    LangGraph node — split docs with RecursiveCharacterTextSplitter.

    Reads:  state["cleaned_documents"], state["source_id"]
    Writes: state["chunks"]
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs      = state.get("cleaned_documents", [])
    source_id = state.get("source_id", "unknown")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,   # adds char offset to metadata
    )
    chunks: List[Document] = splitter.split_documents(docs)
    chunks = _tag_chunks(chunks, source_id, "recursive")

    logger.info("[recursive_chunk] %d docs → %d chunks", len(docs), len(chunks))
    return {"chunks": chunks}


# ── Node: Semantic splitter (embedding-based) ───────────────────────────────────

@safe_node("semantic_chunk")
def semantic_chunk(state: dict) -> dict:
    """
    LangGraph node — split docs with SemanticChunker.

    Uses OpenAI embeddings to detect topic-shift breakpoints.
    Requires OPENAI_API_KEY in environment.

    Reads:  state["cleaned_documents"], state["source_id"]
    Writes: state["chunks"]
    """
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings

    docs      = state.get("cleaned_documents", [])
    source_id = state.get("source_id", "unknown")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    chunker    = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )
    chunks: List[Document] = chunker.split_documents(docs)
    chunks = _tag_chunks(chunks, source_id, "semantic")

    logger.info("[semantic_chunk] %d docs → %d chunks", len(docs), len(chunks))
    return {"chunks": chunks}
