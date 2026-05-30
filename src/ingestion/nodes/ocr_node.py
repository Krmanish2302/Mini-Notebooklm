"""
ocr_node.py

LangGraph node: OCR fallback for scanned PDFs.

Fallback chain:
  1. UnstructuredPDFLoader (hi_res)  — needs poppler + tesseract
  2. UnstructuredPDFLoader (fast)    — layout only
  3. PyMuPDFLoader                   — last resort

Install: pip install unstructured[pdf] pytesseract
"""
from __future__ import annotations
import logging
from typing import List
from langchain_core.documents import Document
from .utils import safe_node

logger = logging.getLogger(__name__)


@safe_node("ocr_fallback")
def ocr_fallback(state: dict) -> dict:
    """
    Reads:  state["file_path"], state["source_id"]
    Writes: state["raw_documents"]
    """
    file_path = state["file_path"]
    source_id = state.get("source_id", "")
    docs: List[Document] = []

    for strategy in ("hi_res", "fast"):
        try:
            from langchain_community.document_loaders import UnstructuredPDFLoader
            docs = UnstructuredPDFLoader(file_path, strategy=strategy).load()
            logger.info("[ocr_fallback] strategy='%s' → %d docs", strategy, len(docs))
            if docs:
                break
        except Exception as exc:
            logger.warning("[ocr_fallback] strategy='%s' failed: %s", strategy, exc)

    if not docs:
        from langchain_community.document_loaders import PyMuPDFLoader
        docs = PyMuPDFLoader(file_path).load()
        logger.warning("[ocr_fallback] Fell back to PyMuPDFLoader → %d docs", len(docs))

    for doc in docs:
        doc.metadata.update({"source_id": source_id, "source_type": "pdf", "ocr": True})

    return {"raw_documents": docs}