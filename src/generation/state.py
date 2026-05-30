"""
state.py — TypedDict flowing through every generation LangGraph node.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document
from src.generation.persona_config import PersonaConfig


class GenerationState(TypedDict, total=False):
    # ── inputs ─────────────────────────────────────────────────────────────────
    query:       str
    # FIX #1: documents was missing from GenerationState — caused extract_citations
    # to always get [] and produce empty citations/chunks_used on every response.
    documents:   List[Document]
    mode:        str                  # "chat" | "study" | "research"
    history:     str
    persona:     PersonaConfig
    stream:      bool

    # ── intermediate ─────────────────────────────────────────────────────────────
    prompt:      str                  # assembled prompt (after build_prompt_node)
    raw_output:  str                  # raw LLM text   (after generate_node)

    # ── output ─────────────────────────────────────────────────────────────────
    answer:          str
    citations:       List[Dict[str, Any]]
    follow_ups:      List[str]
    sources_used:    List[str]
    chunks_used:     List[Dict[str, Any]]
    tokens_estimate: int

    # ── error handling ────────────────────────────────────────────────────────────────
    error:       Optional[str]
    failed_node: Optional[str]
