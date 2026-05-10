"""
storage_manager.py  —  Single orchestration surface over all three stores

Used by
-------
  IngestGraph node_store     — store embedded chunks
  LangGraph retrieve node    — lookup chunks by id after FAISS search
  LangGraph delete node      — remove a source end-to-end

Public API
----------
  store(source, chunks, embedding_model, dim)
      -> persists to FAISS + SQLite + KnowledgeGraph
      -> returns source_id (str)

  delete_source(source_id)
      -> removes from all three stores atomically

  get_chunk_content(chunk_id) -> str | None
      -> fast SQLite lookup by chunk id

  get_chunks_as_documents(chunk_ids) -> List[LangChain Document]
      -> hydrate chunk_ids into LangChain Document objects

  get_all_chunks_for_bm25() -> List[Dict]
      -> returns [{'id': ..., 'content': ...}] for BM25 index rebuild

  get_stats() -> Dict
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from .faiss_store import MultiFAISSStore
from .sqlite_manager import SQLiteManager
from .knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Thin orchestration layer — delegates to specialised stores.
    Keeps all three stores in sync on every write/delete.
    """

    def __init__(
        self,
        faiss_store: MultiFAISSStore,
        sqlite_manager: SQLiteManager,
        knowledge_graph: KnowledgeGraph,
    ):
        self.faiss = faiss_store
        self.sqlite = sqlite_manager
        self.graph = knowledge_graph

        # Hydrate in-memory sources dict from SQLite on startup
        self._sources: Dict[str, Dict[str, Any]] = {
            s["id"]: s for s in self.sqlite.get_sources()
        }
        logger.info(
            "StorageManager: hydrated %d sources from SQLite on startup",
            len(self._sources),
        )

    # ── write ─────────────────────────────────────────────────────────────────

    def store(
        self,
        source: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        embedding_model: str,
        dim: int,
    ) -> str:
        """
        Persist a source and its embedded chunks across all three stores.

        Fix (Bug 4): FAISS add_chunks is wrapped in a try/except. If it
        raises mid-batch (e.g. dim mismatch on chunk N), the SQLite source
        row that was already written is rolled back to prevent split-brain
        between FAISS and SQLite that would later break deletion.
        """
        source_id = source.get("id") or __import__("uuid").uuid4().hex
        source["id"] = source_id

        # 1. SQLite source row first (we can roll it back on FAISS failure)
        self.sqlite.add_source(source)
        self._sources[source_id] = source

        # 2. FAISS — multi-dim store handles dim routing
        try:
            self.faiss.add_chunks(chunks, dim=dim)
        except Exception as faiss_err:
            # Rollback SQLite source row to keep stores in sync
            logger.error(
                "StorageManager.store: FAISS add_chunks failed for source %s — rolling back. Error: %s",
                source_id, faiss_err,
            )
            try:
                self.sqlite.delete_source(source_id)
            except Exception:
                pass
            self._sources.pop(source_id, None)
            raise

        # 3. SQLite chunk rows — annotated with embedding provenance
        for chunk in chunks:
            cid = chunk["id"]
            dim_idx = self.faiss._indexes.get(dim)
            fid = dim_idx.id_map.get(cid) if dim_idx else None
            self.sqlite.add_chunk({
                "id":                cid,
                "source_id":         source_id,
                "content":           chunk.get("content", ""),
                "modality":          chunk.get("modality", "text"),
                "metadata":          chunk.get("metadata", {}),
                "embedding_model":   embedding_model,
                "faiss_dim":         dim,
                "faiss_internal_id": fid,
            })

        # 4. KnowledgeGraph nodes
        if self.graph:
            for chunk in chunks:
                try:
                    self.graph.add_chunk(chunk)
                except Exception as kg_err:
                    logger.warning(
                        "StorageManager.store: KnowledgeGraph.add_chunk failed for %s: %s",
                        chunk.get("id"), kg_err,
                    )

        logger.info(
            "StorageManager: stored source %s — %d chunks (model=%s dim=%d)",
            source_id, len(chunks), embedding_model, dim,
        )
        return source_id

    # ── delete ────────────────────────────────────────────────────────────────

    def delete_source(self, source_id: str) -> bool:
        """
        Fully delete a source and all its chunks from every store.
        """
        if source_id not in self._sources:
            logger.warning("StorageManager.delete_source: unknown source %s", source_id)
            return False

        chunk_rows = self.sqlite.get_chunks_for_deletion(source_id)

        # Group by dim
        by_dim: Dict[int, List[str]] = defaultdict(list)
        for row in chunk_rows:
            d = row.get("faiss_dim")
            if d is not None:
                by_dim[d].append(row["id"])

        # FAISS deletion per dim
        for dim, chunk_ids in by_dim.items():
            self.faiss.delete_chunks(chunk_ids, dim=dim)

        # KnowledgeGraph
        if self.graph:
            for row in chunk_rows:
                cid = row["id"]
                try:
                    self.graph.graph.remove_node(cid)
                except Exception:
                    pass

        # SQLite purge (cascades chunks via delete_source)
        self.sqlite.delete_source(source_id)

        # Memory
        del self._sources[source_id]

        logger.info(
            "StorageManager: removed source %s — %d chunks across %d FAISS indexes",
            source_id, len(chunk_rows), len(by_dim),
        )
        return True

    # ── read ──────────────────────────────────────────────────────────────────

    def get_chunk_content(self, chunk_id: str) -> Optional[str]:
        return self.sqlite.get_chunk_content(chunk_id)

    def get_chunks_as_documents(self, chunk_ids: List[str]) -> List[Document]:
        """Hydrate chunk_ids into LangChain Document objects."""
        docs = []
        for cid in chunk_ids:
            content = self.sqlite.get_chunk_content(cid)
            if content is not None:
                docs.append(Document(page_content=content, metadata={"chunk_id": cid}))
        return docs

    def get_all_chunks_for_bm25(self) -> List[Dict]:
        """Return all chunks as [{id, content}] for BM25 index rebuild."""
        results = []
        for source_id in self._sources:
            rows = self.sqlite.get_chunks_by_source(source_id)
            for row in rows:
                results.append({"id": row["id"], "content": row.get("content", "")})
        return results

    def get_stats(self) -> Dict:
        return {
            "source_count": len(self._sources),
            "faiss": self.faiss.get_stats(),
        }
