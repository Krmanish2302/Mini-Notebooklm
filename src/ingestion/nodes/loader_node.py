"""
loader_node.py

LangGraph node: load source file into List[Document] using LangChain loaders.

Loader selection (by file extension / source_type):
  .pdf            → PyMuPDFLoader       (per-page metadata, fast)
  .csv            → CSVLoader
  .txt / .md      → TextLoader
  http(s):// URL  → WebBaseLoader
  youtube URL     → YoutubeLoader
"""
from __future__ import annotations
import logging
import os
from typing import List

from langchain_core.documents import Document
from .utils import safe_node

logger = logging.getLogger(__name__)


def _detect_type(file_path: str) -> str:
    if file_path.startswith(("http://", "https://")):
        if "youtube.com" in file_path or "youtu.be" in file_path:
            return "youtube"
        return "website"
    ext = os.path.splitext(file_path)[-1].lower()
    return {".pdf": "pdf", ".csv": "csv", ".txt": "text", ".md": "text", ".html": "website"}.get(ext, "text")


def _load_pdf(path: str) -> List[Document]:
    from langchain_community.document_loaders import PyMuPDFLoader
    return PyMuPDFLoader(path).load()


def _load_csv(path: str) -> List[Document]:
    from langchain_community.document_loaders.csv_loader import CSVLoader
    return CSVLoader(file_path=path).load()


def _load_text(path: str) -> List[Document]:
    from langchain_community.document_loaders import TextLoader
    return TextLoader(path, encoding="utf-8").load()


def _load_website(url: str) -> List[Document]:
    from langchain_community.document_loaders import WebBaseLoader
    return WebBaseLoader(url).load()


def _load_youtube(url: str) -> List[Document]:
    from langchain_community.document_loaders import YoutubeLoader
    return YoutubeLoader.from_youtube_url(url, add_video_info=False).load()


_LOADERS = {
    "pdf":     _load_pdf,
    "csv":     _load_csv,
    "text":    _load_text,
    "website": _load_website,
    "youtube": _load_youtube,
}


@safe_node("load_document")
def load_document(state: dict) -> dict:
    """
    Reads:  state["file_path"], state["source_id"], state["source_type"]
    Writes: state["raw_documents"], state["source_type"], state["metadata"]
    """
    file_path   = state["file_path"]
    source_id   = state.get("source_id", os.path.basename(file_path))
    source_type = state.get("source_type") or _detect_type(file_path)

    logger.info("[load_document] '%s' → type='%s'", file_path, source_type)

    docs: List[Document] = _LOADERS.get(source_type, _load_text)(file_path)

    for doc in docs:
        doc.metadata["source_id"]   = source_id
        doc.metadata["source_type"] = source_type

    logger.info("[load_document] Loaded %d pages/docs", len(docs))

    return {
        "raw_documents": docs,
        "source_type":   source_type,
        "metadata": {
            "source_id":   source_id,
            "source_type": source_type,
            "file_path":   file_path,
            "total_pages": len(docs),
        },
    }