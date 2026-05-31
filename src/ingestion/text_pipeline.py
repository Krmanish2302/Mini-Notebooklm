"""
text_pipeline.py

LangGraph ingestion pipeline for text sources: .txt / .md / .docx / paste (PRD §2.4).

Stages:
  1. text_extract    — detect format, extract with appropriate LangChain loader
  2. text_preprocess — normalize whitespace, unicode, remove noise
  3. text_chunk      — RecursiveCharacterTextSplitter (chunk_size=512 tokens ≈ 2000 chars)
  4. text_embed      — embed + persist

Usage:
    from src.ingestion.text_pipeline import run_text_pipeline
    result = run_text_pipeline(file_path="notes.md", source_id="notes_001")
    # Or for pasted text:
    result = run_text_pipeline(content="Raw pasted text...", source_id="paste_001")
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import unicodedata
from typing import Any, Dict, Optional

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from src.ingestion.state import IngestionState

from src.ingestion.nodes.utils import safe_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
@safe_node("text_extract")
def text_extract(state: dict) -> dict:
    source_id   = state["source_id"]
    file_path   = state.get("file_path")
    raw_content = state.get("content")   # pasted text shortcut

    if raw_content:
        # Pasted text — wrap as a single Document
        docs = [Document(
            page_content=raw_content,
            metadata={"source_id": source_id, "source_type": "text", "origin": "paste"},
        )]
        logger.info("[text_extract] Pasted text (%d chars)", len(raw_content))
        return {"raw_documents": docs}

    ext = os.path.splitext(file_path)[-1].lower()

    if ext in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader
        docs = TextLoader(file_path, encoding="utf-8").load()

    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        docs = Docx2txtLoader(file_path).load()

    elif ext == ".csv":
        from langchain_community.document_loaders.csv_loader import CSVLoader
        docs = CSVLoader(file_path=file_path).load()

    else:
        # Fallback — try TextLoader
        from langchain_community.document_loaders import TextLoader
        docs = TextLoader(file_path, encoding="utf-8").load()

    for doc in docs:
        doc.metadata.update({"source_id": source_id, "source_type": "text"})

    logger.info("[text_extract] Loaded %d doc(s) from '%s' (ext=%s)", len(docs), file_path, ext)
    return {"raw_documents": docs}


@safe_node("text_preprocess")
def text_preprocess(state: dict) -> dict:
    raw   = state.get("raw_documents", [])
    clean = []

    for doc in raw:
        t = doc.page_content
        # Unicode normalization
        t = unicodedata.normalize("NFKC", t)
        # Fancy quotes / dashes
        t = t.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
        t = t.replace("\u2013", "-").replace("\u2014", "-")
        # Multiple spaces / newlines
        t = re.sub(r"[ \t]{2,}", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        # Remove non-UTF-8 safe characters
        t = t.encode("utf-8", errors="ignore").decode("utf-8")
        t = t.strip()
        clean.append(Document(
            page_content=t,
            metadata={**doc.metadata, "word_count": len(t.split())},
        ))

    logger.info("[text_preprocess] %d → %d docs", len(raw), len(clean))
    return {"cleaned_documents": clean}


@safe_node("text_chunk")
def text_chunk(state: dict) -> dict:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs      = state.get("cleaned_documents", [])
    source_id = state["source_id"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,        # ≈ 512 tokens
        chunk_overlap=400,      # 20% overlap
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"]    = f"{source_id}_{i}"
        c.metadata["chunk_index"] = i
        c.metadata["source_type"] = "text"

    logger.info("[text_chunk] %d chunks", len(chunks))
    return {"chunks": chunks}


@safe_node("text_embed")
def text_embed(state: dict) -> dict:
    from src.ingestion.nodes.embed_node import embed_and_index
    return embed_and_index(state)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def _build_text_graph() -> StateGraph:
    g = StateGraph(IngestionState)
    g.add_node("text_extract",    text_extract)
    g.add_node("text_preprocess", text_preprocess)
    g.add_node("text_chunk",      text_chunk)
    g.add_node("text_embed",      text_embed)

    g.set_entry_point("text_extract")
    g.add_edge("text_extract",    "text_preprocess")
    g.add_edge("text_preprocess", "text_chunk")
    g.add_edge("text_chunk",      "text_embed")
    g.add_edge("text_embed",      END)
    return g.compile()


text_app = _build_text_graph()


# ---------------------------------------------------------------------------
# Public runners
# ---------------------------------------------------------------------------
def run_text_pipeline(
    source_id:  str,
    file_path:  Optional[str] = None,
    content:    Optional[str] = None,
    source_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ingest a text file (.txt/.md/.docx) or raw pasted content.
    Pass either `file_path` or `content`, not both.
    """
    if not file_path and not content:
        raise ValueError("Either file_path or content must be provided.")

    init_state = {
        "source_id":   source_id,
        "source_type": "text",
        "file_path":   file_path,
        "content":     content,
        "source_name": source_name,
    }
    result = text_app.invoke(init_state)
    if result.get("error"):
        raise RuntimeError(f"Text pipeline failed: {result['error']}")
    logger.info("[run_text_pipeline] Done — %d chunks", result.get("num_chunks", 0))
    return result
