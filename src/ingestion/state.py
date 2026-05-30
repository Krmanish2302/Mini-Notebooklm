"""
state.py

Central TypedDict that flows through every LangGraph node in the
ingestion pipeline.  All nodes receive this dict and return a partial
update — LangGraph merges the updates automatically.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class IngestionState(TypedDict, total=False):
    # ── inputs ────────────────────────────────────────────────────────────────
    file_path:   str                    # absolute path to the source file
    source_id:   str                    # caller-supplied unique identifier
    source_type: str                    # "pdf" | "csv" | "youtube" | "website" …

    # ── intermediate ──────────────────────────────────────────────────────────
    raw_documents:     List[Document]   # output of loader node
    cleaned_documents: List[Document]   # output of preprocess node
    chunks:            List[Document]   # output of chunking node
    is_scanned:        bool             # set by detect_scanned node

    # ── outputs ───────────────────────────────────────────────────────────────
    vectorstore_path: str               # persisted FAISS index directory
    num_chunks:       int               # total chunks indexed
    metadata:         Dict[str, Any]    # source-level metadata summary

    # ── error handling ────────────────────────────────────────────────────────
    error:       Optional[str]          # set on any node failure
    failed_node: Optional[str]          # which node raised the error
