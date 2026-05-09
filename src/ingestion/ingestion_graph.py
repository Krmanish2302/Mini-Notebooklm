"""
ingestion_graph.py

Tracks the lineage of every ingested source through the pipeline stages.
Stores a lightweight directed graph:  Source → Chunks → Embeddings → Index.

This is NOT the knowledge graph (that lives in src/graph/).
This is purely an audit / observability tool so the pipeline knows
which sources have completed which stages.

Usage (from master_pipeline.py):
    graph = IngestionGraph()
    graph.add_source(source_id, name="report.pdf", source_type="pdf")
    graph.mark_stage(source_id, "extracted")
    graph.mark_stage(source_id, "chunked", meta={"num_chunks": 42})
    graph.mark_stage(source_id, "embedded")
    graph.mark_stage(source_id, "indexed")
    status = graph.get_status(source_id)  # → {"stages": [...], "complete": True}
"""
import time
from typing import Dict, Any, List, Optional

# Ordered pipeline stages — a source progresses through these in sequence.
PIPELINE_STAGES = [
    "extracted",    # raw text pulled out of file / URL
    "preprocessed", # cleaned / normalised
    "chunked",      # split into chunks
    "embedded",     # vector embeddings computed
    "indexed",      # stored in FAISS + BM25
]


class IngestionGraph:
    """
    Lightweight in-memory lineage tracker for the ingestion pipeline.

    Each source is a node; completed pipeline stages are stored as
    timestamped edges so the UI and pipeline logic can query progress.
    """

    def __init__(self):
        # source_id → {"name", "type", "added_at", "stages": [{stage, ts, meta}]}
        self._nodes: Dict[str, Dict[str, Any]] = {}

    # ── source management ────────────────────────────────────────────────────

    def add_source(
        self,
        source_id: str,
        name: str = "",
        source_type: str = "unknown",
    ) -> None:
        """Register a new source at the start of ingestion."""
        if source_id in self._nodes:
            return  # idempotent
        self._nodes[source_id] = {
            "source_id": source_id,
            "name": name,
            "type": source_type,
            "added_at": time.time(),
            "stages": [],
            "error": None,
        }

    def mark_stage(
        self,
        source_id: str,
        stage: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record that *source_id* has completed *stage*.

        Args:
            source_id: The source to update.
            stage:     One of PIPELINE_STAGES.
            meta:      Optional extra info (e.g. {"num_chunks": 42}).
        """
        if source_id not in self._nodes:
            self.add_source(source_id)
        self._nodes[source_id]["stages"].append({
            "stage": stage,
            "completed_at": time.time(),
            "meta": meta or {},
        })

    def mark_error(
        self,
        source_id: str,
        stage: str,
        error: str,
    ) -> None:
        """Record a failure at *stage* for *source_id*."""
        if source_id not in self._nodes:
            self.add_source(source_id)
        self._nodes[source_id]["error"] = {"stage": stage, "message": error, "at": time.time()}

    # ── queries ───────────────────────────────────────────────────────────────

    def get_status(self, source_id: str) -> Dict[str, Any]:
        """
        Return the ingestion status of a source.

        Returns:
            {
                "source_id": str,
                "stages":    [{"stage", "completed_at", "meta"}, ...],
                "complete":  bool,   # True when all PIPELINE_STAGES done
                "error":     dict | None,
            }
        """
        node = self._nodes.get(source_id)
        if not node:
            return {"source_id": source_id, "stages": [], "complete": False, "error": None}

        completed = {s["stage"] for s in node["stages"]}
        return {
            **node,
            "complete": all(s in completed for s in PIPELINE_STAGES),
        }

    def get_all_statuses(self) -> List[Dict[str, Any]]:
        """Return status dicts for every registered source."""
        return [self.get_status(sid) for sid in self._nodes]

    def is_complete(self, source_id: str) -> bool:
        """True if every pipeline stage has been completed."""
        return self.get_status(source_id)["complete"]

    def completed_stages(self, source_id: str) -> List[str]:
        """List of stage names that have been marked complete."""
        node = self._nodes.get(source_id)
        if not node:
            return []
        return [s["stage"] for s in node["stages"]]

    def remove_source(self, source_id: str) -> None:
        """Remove a source from the lineage graph."""
        self._nodes.pop(source_id, None)

    def clear(self) -> None:
        """Reset the entire graph."""
        self._nodes.clear()

    def __repr__(self) -> str:
        return f"IngestionGraph(sources={len(self._nodes)})"
