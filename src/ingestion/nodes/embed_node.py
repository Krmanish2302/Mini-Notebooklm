"""
embed_node.py

LangGraph node: embed chunks and persist to a FAISS vector store.

What this node does
-------------------
1. Deduplicates chunks by content hash (prevents re-indexing same text).
2. Builds / updates a FAISS vector store using LangChain's FAISS wrapper.
3. Persists the index to disk at VECTOR_STORE_DIR/<source_id>/.
4. Writes num_chunks and vectorstore_path back into state.

Embedding model is selected via env var EMBEDDING_PROVIDER:
  "openai"      → OpenAIEmbeddings  (text-embedding-3-small, default)
  "huggingface" → HuggingFaceEmbeddings  (all-MiniLM-L6-v2)

Requires:
    pip install langchain-openai faiss-cpu
    OR
    pip install langchain-community sentence-transformers faiss-cpu
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import List

from langchain_core.documents import Document
from .utils import safe_node

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
VECTOR_STORE_DIR   = os.getenv("VECTOR_STORE_DIR",   "data/vectorstores")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")


def _get_embeddings():
    """Return the configured LangChain Embeddings object."""
    if EMBEDDING_PROVIDER == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Default: OpenAI
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _dedup_chunks(chunks: List[Document]) -> List[Document]:
    """Remove duplicate chunks by SHA-256 of page_content."""
    seen: set = set()
    unique: List[Document] = []
    for chunk in chunks:
        h = hashlib.sha256(chunk.page_content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    removed = len(chunks) - len(unique)
    if removed:
        logger.info("[embed_and_index] Removed %d duplicate chunks", removed)
    return unique


@safe_node("embed_and_index")
def embed_and_index(state: dict) -> dict:
    """
    LangGraph node — embed + FAISS index.

    Reads:  state["chunks"], state["source_id"]
    Writes: state["vectorstore_path"], state["num_chunks"]
    """
    from langchain_community.vectorstores import FAISS

    chunks: List[Document] = state.get("chunks", [])
    source_id = state.get("source_id", "unknown")

    if not chunks:
        raise ValueError("[embed_and_index] No chunks to index — aborting.")

    # 1. Deduplicate
    chunks = _dedup_chunks(chunks)

    # 2. Stamp chunk_total now that we know the final count
    for chunk in chunks:
        chunk.metadata["chunk_total"] = len(chunks)

    # 3. Build embeddings
    embeddings = _get_embeddings()

    # 4. Build FAISS vectorstore from documents
    logger.info(
        "[embed_and_index] Embedding %d chunks with provider='%s'",
        len(chunks), EMBEDDING_PROVIDER,
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 5. Persist to disk
    store_path = os.path.join(VECTOR_STORE_DIR, source_id)
    os.makedirs(store_path, exist_ok=True)
    vectorstore.save_local(store_path)
    logger.info("[embed_and_index] FAISS index saved to '%s'", store_path)

    return {
        "vectorstore_path": store_path,
        "num_chunks":       len(chunks),
    }
