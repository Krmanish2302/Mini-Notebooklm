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
    
    source_type = state.get("source_type", "pdf")
    
    # ── Group chunks into parents ────────────────────────────────────────────
    from src.ingestion.parent_child_creator import group_chunks_into_parents
    parents = group_chunks_into_parents(chunks, source_id, source_type)
    
    # Map child IDs to parent_id and schema fields
    child_to_parent = {}
    for p in parents:
        for cid in p["child_ids"]:
            child_to_parent[cid] = p["parent_id"]

    for i, c in enumerate(chunks):
        cid = c.metadata.get("chunk_id") or f"{source_id}_{i}"
        c.metadata["chunk_id"] = cid
        c.metadata["child_id"] = cid
        c.metadata["source_id"] = source_id
        c.metadata["source_type"] = source_type
        c.metadata["document_id"] = state.get("file_path", source_id)
        c.metadata["chunking_strategy_used"] = c.metadata.get("strategy_used") or state.get("strategy", "unknown")
        c.metadata["position_in_source"] = c.metadata.get("chunk_index", i)
        
        # Parent mapping
        pid = child_to_parent.get(cid)
        c.metadata["parent_id"] = pid
        
        # Position in parent
        if pid:
            parent_record = next((p for p in parents if p["parent_id"] == pid), None)
            if parent_record:
                try:
                    c.metadata["position_in_parent"] = parent_record["child_ids"].index(cid)
                except ValueError:
                    c.metadata["position_in_parent"] = 0
            else:
                c.metadata["position_in_parent"] = 0
        else:
            c.metadata["position_in_parent"] = 0

        # Child type
        if "child_type" not in c.metadata:
            if source_type == "youtube":
                c.metadata["child_type"] = "transcript"
            elif source_type == "image":
                c.metadata["child_type"] = "caption"
            else:
                c.metadata["child_type"] = "text"

        # Page number
        c.metadata["page_number"] = c.metadata.get("page") or c.metadata.get("page_number")
        
        # Timestamps for video
        if source_type == "youtube":
            c.metadata["timestamps"] = {
                "start": c.metadata.get("start"),
                "end": c.metadata.get("end")
            }
            
        # Heading path
        c.metadata["heading_path"] = c.metadata.get("section_heading") or c.metadata.get("chapter")
        
        # Text representation
        c.metadata["text"] = c.page_content

    store_path = os.path.join(VECTOR_STORE_DIR, source_id)
    os.makedirs(store_path, exist_ok=True)

    # Force the use of exactly one main retrieval embedding model for FAISS
    from src.ingestion.embedding.embedding_registry import EmbeddingRegistry
    embeddings_model = EmbeddingRegistry.get()
    resolved_model_name = getattr(embeddings_model, "model_name", getattr(embeddings_model, "model", "unknown"))
    logger.info("[embed_and_index] Embedding %d chunks using main retrieval model: %s", len(chunks), resolved_model_name)

    # Resolve dimension of embedding model
    sample_emb = embeddings_model.embed_query("test")
    dim = len(sample_emb)

    # ── Save parents and child chunks in SQLite ──────────────────────────────
    try:
        from src.storage.sqlite_manager import SQLiteManager
        db = SQLiteManager()
        
        # 1. Register/Save source
        filename = state.get("source_name") or state.get("file_path", source_id)
        if filename and not state.get("source_name"):
            filename = os.path.basename(filename)
        db.save_source(
            source_id=source_id,
            name=filename,
            source_type=source_type,
            metadata={
                "total_pages": state.get("total_pages", 1),
                "num_chunks": len(chunks),
                "embedding_model": resolved_model_name
            }
        )
        logger.info("[embed_and_index] Source '%s' registered in SQLite", source_id)
        
        # 2. Save parents
        if parents:
            n_parents = db.save_parents_batch(parents)
            logger.info("[embed_and_index] Saved %d parents to SQLite", n_parents)
            
        # 3. Save child chunks to SQLite chunks table
        chunk_records = []
        for c in chunks:
            chunk_records.append({
                "chunk_id": c.metadata.get("chunk_id"),
                "source_id": source_id,
                "content": c.page_content,
                "metadata": c.metadata,
                "embedding_dim": dim
            })
        n_chunks = db.save_chunks_batch(chunk_records)
        logger.info("[embed_and_index] Saved %d child chunks to SQLite chunks table", n_chunks)
        
    except Exception as db_exc:
        logger.warning("[embed_and_index] Failed to register metadata or parents/chunks in SQLite: %s", db_exc)

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

    return {"vectorstore_path": store_path, "num_chunks": len(chunks)}

