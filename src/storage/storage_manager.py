"""
storage_manager.py  —  Single orchestration surface over all three stores

Fixes applied
-------------
BUG-C01  get_chunks_as_documents now delegates to sqlite.get_chunks_by_ids()
         for a single batched query instead of N separate connections.
BUG-C05  Removed direct access to faiss._indexes.id_map; replaced with the
         public MultiFAISSStore.get_internal_id() method.
BUG-R03  get_chunks_as_documents now includes source_id + metadata in the
         Document object so PromptBuilder can render [S1] — filename headers.
BUG-Q03  get_all_chunks_for_bm25 now issues a single SELECT instead of
         one query per source (N+1 pattern).
"""
from __future__ import annotations

import json
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
        source_id = source.get("id") or __import__("uuid").uuid4().hex
        source["id"] = source_id

        self.sqlite.add_source(source)
        self._sources[source_id] = source

        try:
            self.faiss.add_chunks(chunks, dim=dim)
        except Exception as faiss_err:
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

        # BUG-C05: use public method instead of private _indexes.id_map
        for chunk in chunks:
            cid = chunk["id"]
            fid = self.faiss.get_internal_id(cid, dim)
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
        if source_id not in self._sources:
            logger.warning("StorageManager.delete_source: unknown source %s", source_id)
            return False

        chunk_rows = self.sqlite.get_chunks_for_deletion(source_id)

        by_dim: Dict[int, List[str]] = defaultdict(list)
        for row in chunk_rows:
            d = row.get("faiss_dim")
            if d is not None:
                by_dim[d].append(row["id"])

        for dim, chunk_ids in by_dim.items():
            self.faiss.delete_chunks(chunk_ids, dim=dim)

        if self.graph:
            for row in chunk_rows:
                cid = row["id"]
                try:
                    self.graph.graph.remove_node(cid)
                except Exception:
                    pass

        self.sqlite.delete_source(source_id)
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
        """
        Hydrate chunk_ids into LangChain Document objects.

        BUG-C01 fix: single batched IN() query instead of N separate connections.
        BUG-R03 fix: includes source_id + parsed metadata so PromptBuilder can
                     render [S1] — filename attribution headers.
        """
        if not chunk_ids:
            return []
        rows = self.sqlite.get_chunks_by_ids(chunk_ids)
        id_to_row = {r["id"]: r for r in rows}
        docs = []
        for cid in chunk_ids:
            row = id_to_row.get(cid)
            if row is None:
                continue
            try:
                meta_dict = json.loads(row.get("metadata") or "{}")
            except (json.JSONDecodeError, TypeError):
                meta_dict = {}
            meta_dict["chunk_id"] = cid
            meta_dict["source_id"] = row.get("source_id", "")
            # Provide a 'source' key so PromptBuilder shows the filename
            if "source" not in meta_dict and row.get("source_id"):
                meta_dict["source"] = meta_dict.get("title", row["source_id"])
            docs.append(Document(page_content=row["content"] or "", metadata=meta_dict))
        return docs

    def get_all_chunks_for_bm25(self) -> List[Dict]:
        """
        Return all chunks as [{id, content}] for BM25 index rebuild.

        BUG-Q03 fix: single SELECT instead of one query per source (N+1).
        """
        import sqlite3 as _sqlite3
        with _sqlite3.connect(self.sqlite.db_path, check_same_thread=False) as conn:
            conn.row_factory = _sqlite3.Row
            rows = conn.execute("SELECT id, content FROM chunks").fetchall()
        return [{"id": r["id"], "content": r["content"] or ""} for r in rows]

    def get_stats(self) -> Dict:
        return {
            "source_count": len(self._sources),
            "faiss": self.faiss.get_stats(),
        }
