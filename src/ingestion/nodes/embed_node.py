"""
embed_node.py

LangGraph node: embed chunks into FAISS + build ParentDocumentRetriever.

Steps:
  1. SHA-256 dedup (remove identical chunks)
  2. FAISS.from_documents()  → persisted to VECTOR_STORE_DIR/<source_id>/
  3. build_parent_retriever() → child (400) + parent (2000) layers persisted
     0 LLM calls total.

Env vars:
  VECTOR_STORE_DIR    = data/vectorstores
  EMBEDDING_PROVIDER  = openai | huggingface
"""
from __future__ import annotations
import hashlib
import logging
import os
from typing import List
from langchain_core.documents import Document
from .utils import safe_node

logger = logging.getLogger(__name__)

VECTOR_STORE_DIR   = os.getenv("VECTOR_STORE_DIR",   "data/vectorstores")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")


def _get_embeddings():
    """Centralised embeddings factory — respects EMBEDDING_PROVIDER env var."""
    from src.ingestion.embedding.embedding_registry import EmbeddingRegistry
    return EmbeddingRegistry.get(EMBEDDING_PROVIDER)


def _dedup(chunks: List[Document]) -> List[Document]:
    seen, unique = set(), []
    for c in chunks:
        h = hashlib.sha256(c.page_content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(c)
    removed = len(chunks) - len(unique)
    if removed:
        logger.info("[embed_and_index] Removed %d duplicate chunks", removed)
    return unique


def _load_all_docs_safe(vectorstore) -> List[Document]:
    """
    Safely extract all stored documents from a FAISS vectorstore without
    relying on the internal _dict attribute (which is not part of the
    public LangChain API and may be None or absent).
    """
    # Preferred: public docstore interface
    try:
        docstore = vectorstore.docstore
        if hasattr(docstore, "_dict") and docstore._dict:
            return list(docstore._dict.values())
    except Exception:
        pass

    # Fallback: iterate via index_to_docstore_id
    try:
        ids  = list(vectorstore.index_to_docstore_id.values())
        docs = []
        for doc_id in ids:
            doc = vectorstore.docstore.search(doc_id)
            if doc and not isinstance(doc, str):   # FAISS returns str on miss
                docs.append(doc)
        return docs
    except Exception as exc:
        logger.warning("[embed_node] Could not extract docs from docstore: %s", exc)
        return []


@safe_node("embed_and_index")
def embed_and_index(state: dict) -> dict:
    """
    Reads:  state["chunks"], state["source_id"]
    Writes: state["vectorstore_path"], state["num_chunks"]
    """
    from langchain_community.vectorstores import FAISS
    from src.ingestion.parent_retriever import build_parent_retriever

    chunks: List[Document] = state.get("chunks", [])
    source_id = state.get("source_id", "unknown")

    if not chunks:
        raise ValueError("No chunks to index.")

    chunks = _dedup(chunks)
    for c in chunks:
        c.metadata["chunk_total"] = len(chunks)

    store_path = os.path.join(VECTOR_STORE_DIR, source_id)
    os.makedirs(store_path, exist_ok=True)

    embeddings  = _get_embeddings()
    logger.info("[embed_and_index] Embedding %d chunks (provider=%s)", len(chunks), EMBEDDING_PROVIDER)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(store_path)
    logger.info("[embed_and_index] FAISS saved → '%s'", store_path)

    build_parent_retriever(chunks=chunks, vectorstore_path=store_path)
    logger.info("[embed_and_index] ParentDocumentRetriever saved → '%s/docstore/'", store_path)

    return {"vectorstore_path": store_path, "num_chunks": len(chunks)}
