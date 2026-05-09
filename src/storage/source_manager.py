"""
source_manager.py  —  Source lifecycle manager

Fixes vs original:
    - sources dict hydrated from SQLite on __init__ (was always empty on restart)
    - remove_source uses per-dim FAISS deletion via MultiFAISSStore
    - tracks embedding_model + faiss_dim per chunk for correct deletion routing
"""
from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from typing import Dict, List, Any, Optional

from .faiss_store import MultiFAISSStore
from .sqlite_manager import SQLiteManager

logger = logging.getLogger(__name__)


class SourceManager:
    """
    Manages source lifecycle across MultiFAISSStore + SQLite + KnowledgeGraph.
    """

    def __init__(
        self,
        faiss_store: MultiFAISSStore,
        sqlite_manager: SQLiteManager,
        graph_storage=None,
    ):
        self.faiss = faiss_store
        self.sqlite = sqlite_manager
        self.graph = graph_storage
        # Hydrate from DB — fixes the "sources always empty" bug
        self.sources: Dict[str, Dict[str, Any]] = {
            s["id"]: s for s in self.sqlite.get_sources()
        }
        logger.info("SourceManager: hydrated %d sources from DB", len(self.sources))

    # ── add ───────────────────────────────────────────────────────────────────

    def add_source(
        self,
        source: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        embedding_model: str,
        dim: int,
    ) -> str:
        """
        Persist a source + its embedded chunks.

        Args:
            source:          Source metadata dict (must have 'id').
            chunks:          List of chunk dicts, each with 'embedding' field.
            embedding_model: Name/id of the embedding model used.
            dim:             Embedding dimensionality.
        """
        source_id = source.get("id") or str(uuid.uuid4())
        source["id"] = source_id

        # 1. SQLite source row
        self.sqlite.add_source(source)
        self.sources[source_id] = source

        # 2. FAISS — multi-dim store handles dim routing
        self.faiss.add_chunks(chunks, dim=dim)

        # 3. SQLite chunk rows — annotated with embedding provenance
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
                "faiss_internal_id": self.faiss._indexes[dim].id_map.get(cid),
            })

        # 4. KnowledgeGraph nodes
        if self.graph:
            for chunk in chunks:
                self.graph.add_chunk(chunk)

        logger.info(
            "SourceManager: added source %s — %d chunks (model=%s dim=%d)",
            source_id, len(chunks), embedding_model, dim,
        )
        return source_id

    # ── delete ────────────────────────────────────────────────────────────────

    def remove_source(self, source_id: str) -> bool:
        """
        Fully delete a source and all its chunks from every store.

        Steps:
            1. Lookup chunk rows from SQLite (has faiss_dim + faiss_internal_id)
            2. Group chunk IDs by faiss_dim
            3. Call MultiFAISSStore.delete_chunks(chunk_ids, dim) per group
            4. Remove KnowledgeGraph nodes
            5. Purge SQLite rows
            6. Drop in-memory entry
        """
        if source_id not in self.sources:
            logger.warning("SourceManager.remove_source: unknown source %s", source_id)
            return False

        # Step 1: get chunk metadata
        chunk_rows = self.sqlite.get_chunks_for_deletion(source_id)

        # Step 2: group by dim
        by_dim: Dict[int, List[str]] = defaultdict(list)
        for row in chunk_rows:
            d = row.get("faiss_dim")
            if d is not None:
                by_dim[d].append(row["id"])

        # Step 3: FAISS deletion per dim
        for dim, chunk_ids in by_dim.items():
            self.faiss.delete_chunks(chunk_ids, dim=dim)

        # Step 4: KnowledgeGraph
        if self.graph:
            for row in chunk_rows:
                cid = row["id"]
                try:
                    self.graph.graph.remove_node(cid)
                except Exception:
                    pass

        # Step 5: SQLite purge
        self.sqlite.delete_source(source_id)

        # Step 6: memory
        del self.sources[source_id]

        logger.info(
            "SourceManager: removed source %s — %d chunks across %d FAISS indexes",
            source_id, len(chunk_rows), len(by_dim),
        )
        return True

    # ── utils ─────────────────────────────────────────────────────────────────

    def get_all_sources(self) -> List[Dict[str, Any]]:
        return list(self.sources.values())

    def get_source_count(self) -> int:
        return len(self.sources)

    def get_source_stats(self, source_id: str) -> Dict:
        chunks = self.sqlite.get_chunks_by_source(source_id)
        return {
            "chunk_count": len(chunks),
            "source_type": self.sources.get(source_id, {}).get("source_type", "unknown"),
        }
