"""
ingestion_runner.py

Convenience entry-point for running the full ingestion pipeline.

Can be called:
  - From master_pipeline.py  (programmatic)
  - As a CLI script:  python -m src.ingestion.ingestion_runner <file> <source_id>

Examples
--------
Ingest a PDF:
    from src.ingestion.ingestion_runner import run_ingestion

    result = run_ingestion("data/report.pdf", source_id="report_001")
    print(result["num_chunks"])         # 142
    print(result["vectorstore_path"])   # data/vectorstores/report_001

Load retriever at query time:
    from src.ingestion.parent_retriever import load_parent_retriever

    retriever = load_parent_retriever("data/vectorstores/report_001")
    docs = retriever.invoke("What is the main conclusion?")
    # docs → List[Document] of PARENT chunks (2000 chars) fed to LLM

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
    file_path:   str,
    source_id:   str,
    source_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full LangGraph ingestion pipeline for a single file.

    The pipeline automatically:
      - Loads the file (PDF, CSV, text, website, YouTube)
      - Detects scanned PDFs and falls back to OCR
      - Cleans and normalises text
      - Chunks with RecursiveCharacterTextSplitter (or SemanticChunker)
      - Embeds into FAISS (child chunks, 400 chars)
      - Builds persistent ParentDocumentRetriever (parent chunks, 2000 chars)
      → 0 LLM API calls at ingestion time

    Args:
        file_path:   Path to file or URL.
        source_id:   Unique ID for this source (used as FAISS folder name).
        source_type: Optional override ('pdf','csv','text','website','youtube').

    Returns:
        Final IngestionState dict:
          {
            "num_chunks":        int,
            "vectorstore_path": str,   # e.g. data/vectorstores/report_001
            "metadata":         dict,
            "error":            str | None,
            "failed_node":      str | None,
          }
    """
    initial_state: IngestionState = {
        "file_path":   file_path,
        "source_id":   source_id,
        "source_type": source_type or "",
        "error":       None,
        "failed_node": None,
    }

    logger.info(
        "[run_ingestion] Starting pipeline for '%s' (id=%s)",
        file_path, source_id,
    )
    result = ingestion_app.invoke(initial_state)

    if result.get("error"):
        logger.error(
            "[run_ingestion] Pipeline FAILED at node='%s': %s",
            result.get("failed_node"), result["error"],
        )
    else:
        logger.info(
            "[run_ingestion] ✓ Done. %d chunks indexed at '%s'",
            result.get("num_chunks", 0),
            result.get("vectorstore_path", ""),
        )

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if len(sys.argv) < 3:
        print(
            "Usage: python -m src.ingestion.ingestion_runner "
            "<file_path> <source_id> [--source-type pdf|csv|text|website|youtube]"
        )
        sys.exit(1)

    _file        = sys.argv[1]
    _id          = sys.argv[2]
    _source_type = None
    if "--source-type" in sys.argv:
        _idx         = sys.argv.index("--source-type")
        _source_type = sys.argv[_idx + 1]

    _result = run_ingestion(_file, _id, source_type=_source_type)

    if _result.get("error"):
        print(f"FAILED  : {_result['error']} (node: {_result.get('failed_node')})")
        sys.exit(1)
    else:
        print(
            f"SUCCESS : {_result['num_chunks']} chunks → "
            f"{_result['vectorstore_path']}"
        )
