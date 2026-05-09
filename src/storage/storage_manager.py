"""
storage_manager.py  —  Single orchestration surface over all three stores

Used by
-------
  MasterPipeline.ingest()     — store embedded chunks
  LangGraph retrieve node     — lookup chunks by id after FAISS search
  LangGraph delete node       — remove a source end-to-end

Public API
----------
  store(source, chunks, embedding_model, dim)
      → persists to FAISS + SQLite + KnowledgeGraph
      → returns source_id (str)

  delete_source(source_id)
      → removes from all three stores atomically

  get_chunk_content(chunk_id) → str | None
      → fast SQLite lookup by chunk id

  get_chunks_as_documents(chunk_ids) → List[LangChain Document]
      → hydrate chunk_ids into LangChain Document objects
        (used by HybridRetriever to return LC-native objects)

  get_all_chunks_for_bm25() -> List[Dict]
      → returns [{'id': ..., 'content': ...}] for BM25 index rebuild

  get_stats() -> Dict
"""
from __future__ import annotations

import logging
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

        Parameters
        ----------
        source          : must have 'id', 'title', 'source_type'
        chunks          : each must have 'id', 'content', 'embedding'
        embedding_model : name used for provenance tracking
        dim             : embedding dimensionality (FAISS index routing key)

        Returns
        -------
        source_id : str
        """
        source_id = source["id"]

        # 1. SQLite — source row first (FK constraint)
        self.sqlite.add_source(source)
        self._sources[source_id] = source

        # 2. FAISS — multi-dim routing
        self.faiss.add_chunks(chunks, dim=dim)

        # 3. SQLite — chunk rows with embedding provenance
        for chunk in chunks:
            cid = chunk["id"]
            self.sqlite.add_chunk({
                "id":               cid,
                "source_id":        source_id,
                "content":          chunk.get("content", ""),
                "modality":         chunk.get("modality", "text"),
                "metadata":         chunk.get("metadata", {}),
                "embedding_model":  embedding_model,
                "faiss_dim":        dim,
                "faiss_internal_id": (
                    self.faiss._indexes[dim].id_map.get(cid)
                ),
            })

        # 4. KnowledgeGraph — nodes (auto-link disabled by default for speed;
        #    caller can call graph.add_edge() explicitly for cross-modal links)
        for chunk in chunks:
            self.graph.add_chunk(chunk, auto_link=False)

        logger.info(
            "StorageManager.store: source=%s  chunks=%d  model=%s  dim=%d",
            source_id, len(chunks), embedding_model, dim,
        )
        return source_id

    # ── delete ────────────────────────────────────────────────────────────────

    def delete_source(self, source_id: str) -> bool:
        """
        Remove a source and all its chunks from FAISS + SQLite + KnowledgeGraph.
        Returns True on success, False if source not found.
        """
        if source_id not in self._sources:
            logger.warning("StorageManager.delete_source: unknown source_id=%s", source_id)
            return False

        # Get chunk metadata before deleting SQLite rows
        chunk_rows = self.sqlite.get_chunks_for_deletion(source_id)

        # Group by dim for FAISS
        from collections import defaultdict
        by_dim: Dict[int, List[str]] = defaultdict(list)
        for row in chunk_rows:
            d = row.get("faiss_dim")
            if d is not None:
                by_dim[d].append(row["id"])

        for dim, chunk_ids in by_dim.items():
            self.faiss.delete_chunks(chunk_ids, dim=dim)

        # KnowledgeGraph
        removed = self.graph.remove_source(source_id)
        logger.debug("StorageManager: removed %d graph nodes for source=%s", removed, source_id)

        # SQLite (cascades chunks)
        self.sqlite.delete_source(source_id)

        del self._sources[source_id]
        logger.info("StorageManager.delete_source: removed source=%s", source_id)
        return True

    # ── read ──────────────────────────────────────────────────────────────────

    def get_chunk_content(self, chunk_id: str) -> Optional[str]:
        """Fast single-chunk content lookup via SQLite."""
        return self.sqlite.get_chunk_content(chunk_id)

    def get_chunks_as_documents(
        self,
        chunk_ids: List[str],
        include_context_window: bool = True,
    ) -> List[Document]:
        """
        Hydrate chunk_ids into LangChain Document objects.

        The 'context_window' stored during ContextualEnricher.enrich() is
        included in metadata so the LLM receives the surrounding context
        without it being embedded (as designed).
        """
        docs: List[Document] = []
        for cid in chunk_ids:
            content = self.sqlite.get_chunk_content(cid)
            if content is None:
                continue
            node_data = self.graph.graph.nodes.get(cid, {})
            metadata = dict(node_data.get("metadata", {}))
            metadata["chunk_id"] = cid
            metadata["source_id"] = node_data.get("source_id", "")
            metadata["modality"] = node_data.get("modality", "text")
            metadata["section"] = node_data.get("section", "")
            # context_window is stored in metadata by ContextualEnricher
            if not include_context_window:
                metadata.pop("context_window", None)
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def get_all_chunks_for_bm25(self) -> List[Dict[str, Any]]:
        """
        Return minimal chunk dicts for BM25 index rebuild.
        Called by MasterPipeline after every ingestion.
        """
        rows = []
        for source_id in self._sources:
            for row in self.sqlite.get_chunks_by_source(source_id):
                rows.append({"id": row["id"], "content": row.get("content", "")})
        return rows

    # ── sources ───────────────────────────────────────────────────────────────

    def get_all_sources(self) -> List[Dict[str, Any]]:
        return list(self._sources.values())

    def get_source_count(self) -> int:
        return len(self._sources)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "sources": self.get_source_count(),
            "faiss": self.faiss.get_stats(),
            "graph": self.graph.get_stats(),
        }
