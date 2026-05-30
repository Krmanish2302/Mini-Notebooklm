"""
loader_node.py

LangGraph node: load a source file into a list of LangChain Document objects.

Supported loaders (auto-selected by file extension / source_type):
  PDF     → PyMuPDFLoader  (fast, layout-aware, per-page metadata)
  CSV     → CSVLoader
  TXT/MD  → TextLoader
  Website → WebBaseLoader
  YouTube → YoutubeLoader

Each Document carries rich metadata:
  {
    "source":      <file_path or URL>,
    "page":        <int>,          # PDFs only
    "source_id":   <caller id>,
    "source_type": "pdf" | …,
  }
"""
from __future__ import annotations

import logging
import os
from typing import List

from langchain_core.documents import Document
from .utils import safe_node

logger = logging.getLogger(__name__)


def _detect_source_type(file_path: str) -> str:
    """Infer source type from file extension."""
    ext = os.path.splitext(file_path)[-1].lower()
    mapping = {
        ".pdf":  "pdf",
        ".csv":  "csv",
        ".txt":  "text",
        ".md":   "text",
        ".html": "website",
    }
    if file_path.startswith(("http://", "https://")):
        if "youtube.com" in file_path or "youtu.be" in file_path:
            return "youtube"
        return "website"
    return mapping.get(ext, "text")


def _load_pdf(file_path: str) -> List[Document]:
    from langchain_community.document_loaders import PyMuPDFLoader
    loader = PyMuPDFLoader(file_path)
    return loader.load()


def _load_csv(file_path: str) -> List[Document]:
    from langchain_community.document_loaders.csv_loader import CSVLoader
    loader = CSVLoader(file_path=file_path)
    return loader.load()


def _load_text(file_path: str) -> List[Document]:
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def _load_website(url: str) -> List[Document]:
    from langchain_community.document_loaders import WebBaseLoader
    loader = WebBaseLoader(url)
    return loader.load()


def _load_youtube(url: str) -> List[Document]:
    from langchain_community.document_loaders import YoutubeLoader
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    return loader.load()


_LOADER_MAP = {
    "pdf":     _load_pdf,
    "csv":     _load_csv,
    "text":    _load_text,
    "website": _load_website,
    "youtube": _load_youtube,
}


@safe_node("load_document")
def load_document(state: dict) -> dict:
    """
    LangGraph node — load source file into List[Document].

    Reads:  state["file_path"], state["source_id"]
    Writes: state["raw_documents"], state["source_type"], state["metadata"]
    """
    file_path   = state["file_path"]
    source_id   = state.get("source_id", os.path.basename(file_path))
    source_type = state.get("source_type") or _detect_source_type(file_path)

    logger.info("[load_document] Loading '%s' as type='%s'", file_path, source_type)

    loader_fn = _LOADER_MAP.get(source_type, _load_text)
    docs: List[Document] = loader_fn(file_path)

    # Enrich every document's metadata with source_id + source_type
    for doc in docs:
        doc.metadata["source_id"]   = source_id
        doc.metadata["source_type"] = source_type

    logger.info("[load_document] Loaded %d pages/documents", len(docs))

    return {
        "raw_documents": docs,
        "source_type":   source_type,
        "metadata": {
            "source_id":    source_id,
            "source_type":  source_type,
            "file_path":    file_path,
            "total_pages":  len(docs),
        },
    }
