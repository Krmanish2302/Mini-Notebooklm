"""
embed_node.py

LangGraph node: embed child chunks into FAISS, then build a
persistent ParentDocumentRetriever (0 LLM calls).

What this node does
-------------------
1. Deduplicates chunks by SHA-256 content hash.
2. Builds a FAISS vectorstore from child chunks (400 chars) using
   LangChain FAISS.from_documents().
3. Persists FAISS index to VECTOR_STORE_DIR/<source_id>/.
4. Calls build_parent_retriever() to wire up parent chunks (2000 chars)
   in a persistent LocalFileStore docstore — 0 LLM calls.
5. Writes vectorstore_path and num_chunks back into LangGraph state.

Disk layout after this node
---------------------------
  data/vectorstores/<source_id>/
      index.faiss          ← FAISS child-chunk vectors
      index.pkl            ← FAISS metadata
      docstore/            ← LocalFileStore parent chunks
          <uuid>.json
          ...

Embedding model is selected via EMBEDDING_PROVIDER env var:
  "openai"       → OpenAIEmbeddings  (text-embedding-3-small, default)
  "huggingface"  → HuggingFaceEmbeddings  (all-MiniLM-L6-v2, free/offline)
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import List

from langchain_core.documents import Document
from .utils import safe_node

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
VECTOR_STORE_DIR    = os.getenv("VECTOR_STORE_DIR",   "data/vectorstores")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")


def _get_embeddings():
    """Return the configured LangChain Embeddings object."""
    if EMBEDDING_PROVIDER == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _dedup_chunks(chunks: List[Document]) -> List[Document]:
    """Remove duplicate chunks by SHA-256 of page_content."""
    seen:   set             = set()
    unique: List[Document]  = []
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
    LangGraph node — embed + FAISS index + ParentDocumentRetriever.

    Reads:  state["chunks"], state["source_id"]
    Writes: state["vectorstore_path"], state["num_chunks"]
    """
    from langchain_community.vectorstores import FAISS
    from src.ingestion.parent_retriever import build_parent_retriever

    chunks: List[Document] = state.get("chunks", [])
    source_id = state.get("source_id", "unknown")

    if not chunks:
        raise ValueError("[embed_and_index] No chunks to index — aborting.")

    # 1. Deduplicate
    chunks = _dedup_chunks(chunks)

    # 2. Stamp chunk_total
    for chunk in chunks:
        chunk.metadata["chunk_total"] = len(chunks)

    # 3. Build embeddings
    embeddings = _get_embeddings()

    # 4. Build initial FAISS from all chunks (used as base index)
    store_path = os.path.join(VECTOR_STORE_DIR, source_id)
    os.makedirs(store_path, exist_ok=True)

    logger.info(
        "[embed_and_index] Embedding %d chunks with provider='%s'",
        len(chunks), EMBEDDING_PROVIDER,
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(store_path)
    logger.info("[embed_and_index] FAISS index saved to '%s'", store_path)

    # 5. Build ParentDocumentRetriever on top (0 LLM calls)
    #    This re-splits chunks into parent (2000) + child (400) layers
    #    and persists both to disk.
    logger.info("[embed_and_index] Building ParentDocumentRetriever...")
    build_parent_retriever(
        chunks=chunks,
        vectorstore_path=store_path,
    )
    logger.info("[embed_and_index] ParentDocumentRetriever persisted to '%s/docstore/'", store_path)

    return {
        "vectorstore_path": store_path,
        "num_chunks":       len(chunks),
    }
