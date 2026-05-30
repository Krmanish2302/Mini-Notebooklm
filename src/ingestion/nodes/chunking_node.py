"""
chunking_node.py

LangGraph nodes: split cleaned Documents into retrievable chunks.

Router (choose_chunker):
  SEMANTIC_CHUNKING_ENABLED=true  → semantic_chunk  (embedding-based breakpoints)
  default                         → recursive_chunk (fast, zero LLM calls)

Both nodes tag every chunk with:
  chunk_id, chunk_index, chunker name
"""
from __future__ import annotations
import os
import logging
from typing import List
from langchain_core.documents import Document
from .utils import safe_node

logger = logging.getLogger(__name__)

CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE",    "1000"))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP", "200"))
SEMANTIC_CHUNKING = os.getenv("SEMANTIC_CHUNKING_ENABLED", "false").lower() == "true"


def _tag(chunks: List[Document], source_id: str, chunker: str) -> List[Document]:
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"]    = f"{source_id}_{i}"
        c.metadata["chunk_index"] = i
        c.metadata["chunker"]     = chunker
    return chunks


def choose_chunker(state: dict) -> str:
    route = "semantic_chunk" if SEMANTIC_CHUNKING else "recursive_chunk"
    logger.info("[choose_chunker] → %s", route)
    return route


@safe_node("recursive_chunk")
def recursive_chunk(state: dict) -> dict:
    """
    Uses RecursiveCharacterTextSplitter — zero LLM calls.
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
        add_start_index=True,
    )
    chunks = _tag(splitter.split_documents(docs), source_id, "recursive")
    logger.info("[recursive_chunk] %d docs → %d chunks", len(docs), len(chunks))
    return {"chunks": chunks}


@safe_node("semantic_chunk")
def semantic_chunk(state: dict) -> dict:
    """
    Uses SemanticChunker — 1 embedding call per chunk, no LLM.
    Requires OPENAI_API_KEY (or swap embeddings for HuggingFace).
    Reads:  state["cleaned_documents"], state["source_id"]
    Writes: state["chunks"]
    """
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings

    docs      = state.get("cleaned_documents", [])
    source_id = state.get("source_id", "unknown")

    chunker = SemanticChunker(
        OpenAIEmbeddings(model="text-embedding-3-small"),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )
    chunks = _tag(chunker.split_documents(docs), source_id, "semantic")
    logger.info("[semantic_chunk] %d docs → %d chunks", len(docs), len(chunks))
    return {"chunks": chunks}