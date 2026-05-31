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
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")


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

    embedding_model_name = state.get("embedding_model")
    from src.ingestion.embedding.embedding_registry import EmbeddingRegistry
    embeddings_model = EmbeddingRegistry.get(embedding_model_name)
    resolved_model_name = getattr(embeddings_model, "model_name", getattr(embeddings_model, "model", "unknown"))
    logger.info("[embed_and_index] Embedding %d chunks (model=%s)", len(chunks), resolved_model_name)

    # Resolve dimension of embedding model
    sample_emb = embeddings_model.embed_query("test")
    dim = len(sample_emb)

    import faiss
    from langchain_community.docstore.in_memory import InMemoryDocstore

    # Create HNSW flat index with Inner Product (Cosine similarity) metric
    hnsw_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw_index.hnsw.efConstruction = 64
    hnsw_index.hnsw.efSearch = 16

    vectorstore = FAISS(
        embedding_function=embeddings_model,
        index=hnsw_index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vectorstore.add_documents(chunks)
    vectorstore.save_local(store_path)
    logger.info("[embed_and_index] FAISS (HNSW) saved → '%s'", store_path)

    build_parent_retriever(chunks=chunks, vectorstore_path=store_path)
    logger.info("[embed_and_index] ParentDocumentRetriever saved → '%s/docstore/'", store_path)

    # Register the source in the SQLite database
    try:
        from src.storage.sqlite_manager import SQLiteManager
        db = SQLiteManager()
        filename = state.get("source_name") or state.get("file_path", source_id)
        if filename and not state.get("source_name"):
            filename = os.path.basename(filename)
        db.save_source(
            source_id=source_id,
            name=filename,
            source_type=state.get("source_type", "pdf"),
            metadata={
                "total_pages": state.get("total_pages", 1),
                "num_chunks": len(chunks),
                "embedding_model": embedding_model_name or "all-MiniLM-L6-v2"
            }
        )
        logger.info("[embed_and_index] Source '%s' registered in SQLite database as '%s'", source_id, filename)
    except Exception as db_exc:
        logger.warning("[embed_and_index] Failed to register source in SQLite: %s", db_exc)

    return {"vectorstore_path": store_path, "num_chunks": len(chunks)}
