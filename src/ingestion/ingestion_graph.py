"""
ingestion_graph.py

LangGraph StateGraph for the full ingestion pipeline.

Flow:
  load_document
       │
       ▼
  detect_scanned ──(scanned)──► ocr_fallback
       │                              │
       │(not scanned)                 │
       ▼                              │
  preprocess ◄──────────────────────┘
       │
       ▼
  choose_chunker ──(semantic)──► semantic_chunk
       │                               │
       │(recursive)                    │
       ▼                               │
  recursive_chunk ◄──────────────────┘
       │
       ▼
  embed_and_index
       │
       ▼
      END

Any node failure → handle_error → END
"""

from __future__ import annotations
import logging
from langgraph.graph import StateGraph, END

from .state import IngestionState
from .nodes.loader_node import load_document
from .nodes.detect_node import detect_scanned
from .nodes.ocr_node import ocr_fallback
from .nodes.preprocess_node import preprocess
from .nodes.chunking_node import recursive_chunk, semantic_chunk, choose_chunker
from .nodes.embed_node import embed_and_index
from .nodes.error_node import handle_error

logger = logging.getLogger(__name__)


def _err(s: dict) -> str:
    return "handle_error" if s.get("error") else "ok"


def build_ingestion_graph() -> StateGraph:
    wf = StateGraph(IngestionState)

    wf.add_node("load_document", load_document)
    wf.add_node("detect_scanned", detect_scanned)
    wf.add_node("ocr_fallback", ocr_fallback)
    wf.add_node("preprocess", preprocess)
    wf.add_node("recursive_chunk", recursive_chunk)
    wf.add_node("semantic_chunk", semantic_chunk)
    wf.add_node("embed_and_index", embed_and_index)
    wf.add_node("handle_error", handle_error)

    wf.set_entry_point("load_document")

    # load_document → detect_scanned | handle_error
    wf.add_conditional_edges(
        "load_document",
        lambda s: "handle_error" if s.get("error") else "detect_scanned",
        {"detect_scanned": "detect_scanned", "handle_error": "handle_error"},
    )

    # detect_scanned → ocr_fallback | preprocess
    wf.add_conditional_edges(
        "detect_scanned",
        lambda s: "ocr_fallback" if s.get("is_scanned") else "preprocess",
        {"ocr_fallback": "ocr_fallback", "preprocess": "preprocess"},
    )

    # ocr_fallback → preprocess | handle_error
    wf.add_conditional_edges(
        "ocr_fallback",
        lambda s: "handle_error" if s.get("error") else "preprocess",
        {"preprocess": "preprocess", "handle_error": "handle_error"},
    )

    # preprocess → choose_chunker
    wf.add_conditional_edges(
        "preprocess",
        choose_chunker,
        {"recursive_chunk": "recursive_chunk", "semantic_chunk": "semantic_chunk"},
    )

    # recursive_chunk → embed_and_index | handle_error
    wf.add_conditional_edges(
        "recursive_chunk",
        lambda s: "handle_error" if s.get("error") else "embed_and_index",
        {"embed_and_index": "embed_and_index", "handle_error": "handle_error"},
    )

    # semantic_chunk → embed_and_index | handle_error
    wf.add_conditional_edges(
        "semantic_chunk",
        lambda s: "handle_error" if s.get("error") else "embed_and_index",
        {"embed_and_index": "embed_and_index", "handle_error": "handle_error"},
    )

    # embed_and_index → END | handle_error
    wf.add_conditional_edges(
        "embed_and_index",
        lambda s: "handle_error" if s.get("error") else END,
        {END: END, "handle_error": "handle_error"},
    )

    wf.add_edge("handle_error", END)
    return wf


ingestion_app = build_ingestion_graph().compile()
