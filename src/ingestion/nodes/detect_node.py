"""
detect_node.py

LangGraph node: detect if a PDF is scanned/image-based.
Heuristic: avg words per page < SCAN_THRESHOLD → scanned.
Pass-through for non-PDF sources.
"""
from __future__ import annotations
import logging
from .utils import safe_node

logger = logging.getLogger(__name__)
SCAN_THRESHOLD = 30


@safe_node("detect_scanned")
def detect_scanned(state: dict) -> dict:
    """
    Reads:  state["raw_documents"], state["source_type"]
    Writes: state["is_scanned"]
    """
    if state.get("source_type") != "pdf":
        return {"is_scanned": False}

    docs = state.get("raw_documents", [])
    if not docs:
        return {"is_scanned": True}

    avg_words = sum(len(d.page_content.split()) for d in docs) / len(docs)
    is_scanned = avg_words < SCAN_THRESHOLD
    logger.info("[detect_scanned] avg_words=%.1f → is_scanned=%s", avg_words, is_scanned)
    return {"is_scanned": is_scanned}