"""
ingestion_runner.py

Single entry point to run the full ingestion pipeline.

Programmatic usage:
    from src.ingestion import run_ingestion
    result = run_ingestion("data/report.pdf", source_id="rep_001")
    print(result["num_chunks"])           # 142
    print(result["vectorstore_path"])     # data/vectorstores/rep_001

CLI usage:
    python -m src.ingestion.ingestion_runner data/report.pdf rep_001
    python -m src.ingestion.ingestion_runner https://youtu.be/xyz yt_001 --source-type youtube
"""
from __future__ import annotations
import logging
import sys
from typing import Any, Dict, Optional

from .ingestion_graph import ingestion_app
from .state           import IngestionState

logger = logging.getLogger(__name__)


def run_ingestion(
    file_path:   str,
    source_id:   str,
    source_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full LangGraph ingestion pipeline.

    Returns state dict with:
        num_chunks, vectorstore_path, metadata, error, failed_node
    """
    initial: IngestionState = {
        "file_path":   file_path,
        "source_id":   source_id,
        "source_type": source_type or "",
        "error":       None,
        "failed_node": None,
    }

    logger.info("[run_ingestion] Starting: file='%s' id='%s'", file_path, source_id)
    result = ingestion_app.invoke(initial)

    if result.get("error"):
        logger.error(
            "[run_ingestion] FAILED at node='%s': %s",
            result.get("failed_node"), result["error"],
        )
    else:
        logger.info(
            "[run_ingestion] ✓ %d chunks → %s",
            result.get("num_chunks", 0),
            result.get("vectorstore_path", ""),
        )
    return result


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if len(sys.argv) < 3:
        print("Usage: python -m src.ingestion.ingestion_runner <file_path> <source_id> [--source-type pdf|csv|text|website|youtube]")
        sys.exit(1)

    _file = sys.argv[1]
    _id   = sys.argv[2]
    _type = None
    if "--source-type" in sys.argv:
        _type = sys.argv[sys.argv.index("--source-type") + 1]

    _r = run_ingestion(_file, _id, source_type=_type)
    if _r.get("error"):
        print(f"FAILED: {_r['error']} (node: {_r.get('failed_node')})")
        sys.exit(1)
    print(f"SUCCESS: {_r['num_chunks']} chunks → {_r['vectorstore_path']}")