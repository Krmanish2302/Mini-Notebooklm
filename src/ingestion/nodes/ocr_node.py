"""
ocr_node.py

LangGraph node: OCR fallback for scanned / image-only PDFs.

Uses LangChain's UnstructuredPDFLoader with strategy="hi_res" which
internally calls Tesseract OCR via the `unstructured` library.

Fallback chain:
  1. UnstructuredPDFLoader (hi_res)  — best quality, needs poppler + tesseract
  2. UnstructuredPDFLoader (fast)    — layout-only, no OCR
  3. PyMuPDFLoader                   — last resort text extraction

Requires:
    pip install unstructured[pdf] pytesseract poppler-utils
"""
from __future__ import annotations

import logging
from typing import List
from langchain_core.documents import Document
from .utils import safe_node

logger = logging.getLogger(__name__)


def _try_unstructured(file_path: str, strategy: str) -> List[Document]:
    from langchain_community.document_loaders import UnstructuredPDFLoader
    loader = UnstructuredPDFLoader(file_path, strategy=strategy)
    return loader.load()


def _try_pymupdf(file_path: str) -> List[Document]:
    from langchain_community.document_loaders import PyMuPDFLoader
    return PyMuPDFLoader(file_path).load()


@safe_node("ocr_fallback")
def ocr_fallback(state: dict) -> dict:
    """
    LangGraph node — OCR-based PDF loading fallback.

    Reads:  state["file_path"], state["source_id"]
    Writes: state["raw_documents"]
    """
    file_path = state["file_path"]
    source_id = state.get("source_id", "")

    docs: List[Document] = []

    for strategy in ("hi_res", "fast"):
        try:
            docs = _try_unstructured(file_path, strategy)
            logger.info(
                "[ocr_fallback] UnstructuredPDFLoader strategy='%s' → %d docs",
                strategy, len(docs),
            )
            if docs:
                break
        except Exception as exc:
            logger.warning("[ocr_fallback] strategy='%s' failed: %s", strategy, exc)

    if not docs:
        logger.warning("[ocr_fallback] Falling back to PyMuPDFLoader")
        docs = _try_pymupdf(file_path)

    for doc in docs:
        doc.metadata["source_id"]   = source_id
        doc.metadata["source_type"] = "pdf"
        doc.metadata["ocr"]         = True

    logger.info("[ocr_fallback] OCR produced %d documents", len(docs))
    return {"raw_documents": docs}
