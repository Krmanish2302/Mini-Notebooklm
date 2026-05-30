"""
source_manager.py — Source registry and chunk storage coordinator.

Acts as the single entry point for:
  1. Registering a new source (SQLiteManager)
  2. Storing chunks + embeddings (SQLiteManager + MultiFAISSStore)
  3. Querying active source IDs for pipeline filtering

LangChain integration:
    - store_chunks() accepts List[Document] directly.
    - get_active_documents() returns List[Document] for a source.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document

from src.storage.faiss_store import MultiFAISSStore
from src.storage.sqlite_manager import SQLiteManager

logger = logging.getLogger(__name__)


class SourceManager:
    """
    Coordinates source registration and chunk storage across SQLite + FAISS.

    Usage:
        sm = SourceManager(faiss_store, sqlite_manager)
        result = sm.store_chunks(
            documents=docs,         # List[Document]
            embeddings=vectors,     # List[List[float]]
            source_id="paper_01",
            dim=768,
        )
        print(result["chunks_stored"])
    """

    def __init__(
        self,
        faiss_store: MultiFAISSStore,
        sqlite:      SQLiteManager,
    ):
        self.faiss_store = faiss_store
        self.sqlite      = sqlite

    # ── Source registration ───────────────────────────────────────────────────

    def register_source(
        self,
        source_id:   str,
        name:        str,
        source_type: str,
        metadata:    Optional[Dict[str, Any]] = None,
    ) -> str:
        self.sqlite.save_source(
            source_id=source_id,
            name=name,
            source_type=source_type,
            metadata=metadata,
        )
        logger.info("[SourceManager] Registered source: %s (%s)", source_id, source_type)
        return source_id

    # ── Chunk storage ─────────────────────────────────────────────────────────

    def store_chunks(
        self,
        documents:  List[Document],
        embeddings: List[List[float]],
        source_id:  str,
        dim:        int,
        metadata:   Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Store LangChain Documents + their embeddings.

        Args:
            documents  : List[Document] — page_content + metadata
            embeddings : List[List[float]] — one per document
            source_id  : source identifier
            dim        : embedding dimension
            metadata   : extra metadata merged into each chunk record

        Returns:
            {source_id, chunks_stored, chunk_ids}
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"documents ({len(documents)}) and embeddings ({len(embeddings)}) must match."
            )

        chunk_records = []
        faiss_chunks  = []
        chunk_ids     = []

        for doc, emb in zip(documents, embeddings):
            cid = doc.metadata.get("chunk_id") or str(uuid.uuid4())
            chunk_ids.append(cid)

            # Merge metadata
            meta = {**(metadata or {}), **doc.metadata, "chunk_id": cid, "source_id": source_id}

            chunk_records.append({
                "chunk_id":     cid,
                "source_id":    source_id,
                "content":      doc.page_content,
                "metadata":     meta,
                "embedding_dim": dim,
            })
            faiss_chunks.append({"id": cid, "embedding": emb})

        # Persist to SQLite
        self.sqlite.save_chunks_batch(chunk_records)

        # Persist to FAISS
        self.faiss_store.add_chunks(faiss_chunks, dim)

        logger.info(
            "[SourceManager] Stored %d chunks for source=%s dim=%d",
            len(chunk_ids), source_id, dim,
        )
        return {
            "source_id":     source_id,
            "chunks_stored": len(chunk_ids),
            "chunk_ids":     chunk_ids,
        }

    # ── Source queries ────────────────────────────────────────────────────────

    def get_active_source_ids(
        self,
        requested: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Filter *requested* against active sources in SQLite.
        If requested is None, return all active source IDs.
        """
        all_active = {
            s["source_id"]
            for s in self.sqlite.list_sources(active_only=True)
        }
        if requested is None:
            return list(all_active)
        return [sid for sid in requested if sid in all_active]

    def get_all_active_source_ids(self) -> List[str]:
        return self.get_active_source_ids(requested=None)

    def get_active_documents(self, source_id: str) -> List[Document]:
        """Return all stored LangChain Documents for an active source."""
        source = self.sqlite.get_source(source_id)
        if not source or not source["active"]:
            return []
        return self.sqlite.get_documents_by_source(source_id)

    def delete_source(self, source_id: str, dim: int) -> Dict[str, Any]:
        """
        Delete a source and all its chunks from SQLite + FAISS.
        Returns {source_id, chunks_deleted}.
        """
        chunk_ids = self.sqlite.get_chunk_ids_by_source(source_id)
        if chunk_ids:
            self.faiss_store.delete_chunks(chunk_ids, dim)
        n_deleted = self.sqlite.delete_chunks_by_source(source_id)
        self.sqlite.delete_source(source_id)
        logger.info(
            "[SourceManager] Deleted source=%s (%d chunks)", source_id, n_deleted
        )
        return {"source_id": source_id, "chunks_deleted": n_deleted}

    def set_source_active(self, source_id: str, active: bool) -> None:
        self.sqlite.set_source_active(source_id, active)
        logger.info("[SourceManager] source=%s active=%s", source_id, active)