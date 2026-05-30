"""
detect_node.py

LangGraph node: decide whether a PDF is text-native or scanned/image-based.

Heuristic: if the average word count per page is below a threshold
(default 30 words), the document is assumed to be scanned and the
pipeline will route to the OCR fallback node.

This node is a pure pass-through for non-PDF sources.
"""
from __future__ import annotations

import logging
from .utils import safe_node

logger = logging.getLogger(__name__)

# If average words-per-page falls below this, treat as scanned
_SCAN_THRESHOLD = 30


@safe_node("detect_scanned")
def detect_scanned(state: dict) -> dict:
    """
    LangGraph node — detect if PDF is scanned.

    Reads:  state["raw_documents"], state["source_type"]
    Writes: state["is_scanned"]
    """
    source_type = state.get("source_type", "")

    # Only meaningful for PDFs
    if source_type != "pdf":
        return {"is_scanned": False}

    docs = state.get("raw_documents", [])
    if not docs:
        return {"is_scanned": True}   # empty → assume scanned

    total_words = sum(len(d.page_content.split()) for d in docs)
    avg_words   = total_words / len(docs)

    is_scanned = avg_words < _SCAN_THRESHOLD
    logger.info(
        "[detect_scanned] avg_words_per_page=%.1f → is_scanned=%s",
        avg_words, is_scanned,
    )
    return {"is_scanned": is_scanned}
