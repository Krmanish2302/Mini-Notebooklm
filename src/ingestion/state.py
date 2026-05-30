"""
state.py

Single TypedDict that flows through every LangGraph node.
All nodes receive this dict and return partial updates —
LangGraph merges them automatically.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class IngestionState(TypedDict, total=False):
    # ── inputs ──────────────────────────────────────────────────────
    file_path:   str        # absolute or relative path / URL
    source_id:   str        # caller-supplied unique identifier
    source_type: str        # "pdf" | "csv" | "text" | "website" | "youtube"

    # ── intermediate ────────────────────────────────────────────────
    raw_documents:     List[Document]   # after loader node
    cleaned_documents: List[Document]   # after preprocess node
    chunks:            List[Document]   # after chunking node
    is_scanned:        bool             # set by detect_scanned node

    # ── outputs ─────────────────────────────────────────────────────
    vectorstore_path: str               # persisted FAISS directory
    num_chunks:       int               # total chunks indexed
    metadata:         Dict[str, Any]    # source-level summary

    # ── error handling ───────────────────────────────────────────────
    error:       Optional[str]          # set on node failure
    failed_node: Optional[str]          # which node raised the error