"""
ingestion_runner.py

Convenience entry-point for running the full ingestion pipeline.

Can be called:
  - From master_pipeline.py  (programmatic)
  - As a CLI script:  python -m src.ingestion.ingestion_runner <file> <source_id>
  - With optional RAPTOR tree building

Examples
--------
Programmatic:
    from src.ingestion.ingestion_runner import run_ingestion

    result = run_ingestion("data/report.pdf", source_id="report_001")
    print(result["num_chunks"])  # 142
    print(result["vectorstore_path"])  # data/vectorstores/report_001

With RAPTOR:
    result = run_ingestion("data/book.pdf", source_id="book_001", build_raptor=True)

CLI:
    python -m src.ingestion.ingestion_runner data/report.pdf report_001
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional

from .ingestion_graph import ingestion_app
from .state          import IngestionState

logger = logging.getLogger(__name__)


def run_ingestion(
    file_path:    str,
    source_id:    str,
    source_type:  Optional[str] = None,
    build_raptor: bool = False,
) -> Dict[str, Any]:
    """
    Run the full LangGraph ingestion pipeline for a single file.

    Args:
        file_path:    Path to file or URL.
        source_id:    Unique ID for this source (used as FAISS folder name).
        source_type:  Optional override ('pdf', 'csv', 'text', 'website', 'youtube').
        build_raptor: If True, build a RAPTOR summary tree after indexing.

    Returns:
        Final IngestionState dict with keys:
          - num_chunks, vectorstore_path, metadata, error (if any)
    """
    initial_state: IngestionState = {
        "file_path":   file_path,
        "source_id":   source_id,
        "source_type": source_type or "",
        "error":       None,
        "failed_node": None,
    }

    logger.info("[run_ingestion] Starting pipeline for '%s' (id=%s)", file_path, source_id)
    result = ingestion_app.invoke(initial_state)

    if result.get("error"):
        logger.error(
            "[run_ingestion] Pipeline failed: %s (node: %s)",
            result["error"], result.get("failed_node"),
        )
        return result

    logger.info(
        "[run_ingestion] ✓ Done. %d chunks indexed at '%s'",
        result.get("num_chunks", 0),
        result.get("vectorstore_path", ""),
    )

    if build_raptor and not result.get("error"):
        from .raptor_builder import build_raptor_tree
        logger.info("[run_ingestion] Building RAPTOR tree…")
        build_raptor_tree(
            chunks=result["chunks"],
            vectorstore_path=result["vectorstore_path"],
            source_id=source_id,
        )

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python -m src.ingestion.ingestion_runner <file_path> <source_id> [--raptor]")
        sys.exit(1)

    _file    = sys.argv[1]
    _id      = sys.argv[2]
    _raptor  = "--raptor" in sys.argv

    _result = run_ingestion(_file, _id, build_raptor=_raptor)

    if _result.get("error"):
        print(f"FAILED: {_result['error']}")
        sys.exit(1)
    else:
        print(f"SUCCESS: {_result['num_chunks']} chunks → {_result['vectorstore_path']}")
