"""
storage_manager.py — Orchestrator: keeps MultiFAISSStore, SQLiteManager,
                      KnowledgeGraph, and SourceManager in sync.

Responsibilities:
  1. Ingest pipeline: accept List[Document] + embeddings → persist everywhere
  2. Delete pipeline: remove from all three stores atomically
  3. Graph construction: auto-link chunks in KnowledgeGraph post-ingest
  4. LangChain callback support: fires on_ingest_start / on_ingest_end

LangChain integration:
    - All ingest/retrieval APIs use List[Document] as the canonical type.
    - on_ingest callbacks use LangChain BaseCallbackHandler protocol.
    - get_context_documents() returns List[Document] ready for ChatPromptTemplate.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document

from src.storage.faiss_store    import MultiFAISSStore
from src.storage.knowledge_graph import KnowledgeGraph
from src.storage.source_manager  import SourceManager
from src.storage.sqlite_manager  import SQLiteManager

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Top-level orchestrator for the storage layer.

    Usage:
        sm = StorageManager(faiss_store, sqlite, knowledge_graph)
        result = sm.ingest(
            documents=docs,
            embeddings=vectors,
            source_id="paper_01",
            dim=768,
        )
    """

    def __init__(
        self,
        faiss_store:     MultiFAISSStore,
        sqlite:          SQLiteManager,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        callbacks:       Optional[List[BaseCallbackHandler]] = None,
    ):
        self.faiss_store     = faiss_store
        self.sqlite          = sqlite
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.callbacks       = callbacks or []

        # SourceManager is built internally — single source of truth for chunk storage
        self.source_manager = SourceManager(faiss_store=faiss_store, sqlite=sqlite)

    # ── Ingest ────────────────────────────────────────────────────────────────

    def ingest(
        self,
        documents:   List[Document],
        embeddings:  List[List[float]],
        source_id:   str,
        dim:         int,
        source_name: str                     = "",
        source_type: str                     = "text",
        metadata:    Optional[Dict[str, Any]] = None,
        auto_link:   bool                    = False,
    ) -> Dict[str, Any]:
        """
        Full ingest pipeline:
          1. Register source in SQLite
          2. Store chunks + embeddings (SQLite + FAISS) via SourceManager
          3. Add chunks to KnowledgeGraph (optionally auto-link)
          4. Fire LangChain callbacks

        Returns:
            {source_id, chunks_stored, chunk_ids}
        """
        self._fire("on_ingest_start", {"source_id": source_id, "n_docs": len(documents)})

        # 1. Register source
        self.source_manager.register_source(
            source_id=source_id,
            name=source_name or source_id,
            source_type=source_type,
            metadata=metadata,
        )

        # 2. Store chunks + embeddings
        result = self.source_manager.store_chunks(
            documents=documents,
            embeddings=embeddings,
            source_id=source_id,
            dim=dim,
            metadata=metadata,
        )

        # 3. Add to KnowledgeGraph
        for doc, emb, cid in zip(documents, embeddings, result["chunk_ids"]):
            self.knowledge_graph.add_document(
                doc=doc,
                chunk_id=cid,
                embedding=emb,
                auto_link=auto_link,
            )

        self._fire("on_ingest_end", result)
        return result

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_source(self, source_id: str, dim: int) -> Dict[str, Any]:
        """
        Remove a source from SQLite, FAISS, and KnowledgeGraph.
        """
        # Remove from KnowledgeGraph first (needs chunk_ids from SQLite)
        chunk_ids = self.sqlite.get_chunk_ids_by_source(source_id)
        for cid in chunk_ids:
            self.knowledge_graph.remove_node(cid)

        # Remove from SQLite + FAISS via SourceManager
        result = self.source_manager.delete_source(source_id, dim)
        logger.info("[StorageManager] Deleted source=%s", source_id)
        return result

    # ── Context retrieval ─────────────────────────────────────────────────────

    def get_context_documents(
        self,
        chunk_ids: List[str],
        include_graph_neighbors: bool = False,
        graph_depth:             int  = 1,
    ) -> List[Document]:
        """
        Resolve chunk_ids to LangChain Documents, optionally expanding
        via KnowledgeGraph neighbours.

        Use this to build the context block for ChatPromptTemplate.
        """
        # Resolve via SQLiteManager
        docs: List[Document] = []
        seen: set = set()
        for cid in chunk_ids:
            doc = self.sqlite.get_chunk_as_document(cid)
            if doc and cid not in seen:
                docs.append(doc)
                seen.add(cid)

        # Optionally expand via graph neighbours
        if include_graph_neighbors:
            extra_ids: list = []
            for cid in chunk_ids:
                extra_ids.extend(
                    self.knowledge_graph.get_neighbors(cid, depth=graph_depth)
                )
            for eid in extra_ids:
                if eid not in seen:
                    doc = self.sqlite.get_chunk_as_document(eid)
                    if doc:
                        docs.append(doc)
                        seen.add(eid)

        return docs

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "faiss":  self.faiss_store.get_stats(),
            "sqlite": self.sqlite.get_stats(),
            "graph":  self.knowledge_graph.get_stats(),
        }

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def add_callback(self, handler: BaseCallbackHandler) -> None:
        """Add a LangChain BaseCallbackHandler for observability."""
        self.callbacks.append(handler)

    def _fire(self, event: str, data: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            fn: Optional[Callable] = getattr(cb, event, None)
            if callable(fn):
                try:
                    fn(**data)
                except Exception as exc:
                    logger.warning("[StorageManager] Callback %s error: %s", event, exc)