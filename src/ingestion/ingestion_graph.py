"""
ingestion_graph.py

Full LangGraph-powered ingestion pipeline for Mini-NotebookLM.

Flow
----
  load_document
       │
       ▼
  detect_scanned  ──(is_scanned=True)──►  ocr_fallback
       │                                       │
       │(is_scanned=False)                     │
       ▼                                       │
  preprocess  ◄──────────────────────────────┘
       │
       ▼
  choose_chunker  ──(semantic)──►  semantic_chunk
       │                                │
       │(recursive / default)           │
       ▼                                │
  recursive_chunk  ◄────────────────────┘
       │
       ▼
  embed_and_index
       │
       ▼
     END

Usage
-----
    from src.ingestion import ingestion_app, IngestionState

    state = ingestion_app.invoke({
        "file_path": "/data/report.pdf",
        "source_id": "report_001",
    })
    print(state["num_chunks"], "chunks indexed at", state["vectorstore_path"])
"""
from __future__ import annotations

import logging
from langgraph.graph import StateGraph, END

from .state import IngestionState
from .nodes.loader_node      import load_document
from .nodes.detect_node      import detect_scanned
from .nodes.ocr_node         import ocr_fallback
from .nodes.preprocess_node  import preprocess
from .nodes.chunking_node    import recursive_chunk, semantic_chunk, choose_chunker
from .nodes.embed_node       import embed_and_index
from .nodes.error_node       import handle_error

logger = logging.getLogger(__name__)

# ── Build graph ────────────────────────────────────────────────────────────────

def build_ingestion_graph() -> StateGraph:
    workflow = StateGraph(IngestionState)

    # Register nodes
    workflow.add_node("load_document",    load_document)
    workflow.add_node("detect_scanned",   detect_scanned)
    workflow.add_node("ocr_fallback",     ocr_fallback)
    workflow.add_node("preprocess",       preprocess)
    workflow.add_node("recursive_chunk",  recursive_chunk)
    workflow.add_node("semantic_chunk",   semantic_chunk)
    workflow.add_node("embed_and_index",  embed_and_index)
    workflow.add_node("handle_error",     handle_error)

    # Entry point
    workflow.set_entry_point("load_document")

    # load_document → detect_scanned  (or error)
    workflow.add_conditional_edges(
        "load_document",
        lambda s: "handle_error" if s.get("error") else "detect_scanned",
        {"detect_scanned": "detect_scanned", "handle_error": "handle_error"},
    )

    # detect_scanned → ocr_fallback  OR  preprocess
    workflow.add_conditional_edges(
        "detect_scanned",
        lambda s: "ocr_fallback" if s.get("is_scanned") else "preprocess",
        {"ocr_fallback": "ocr_fallback", "preprocess": "preprocess"},
    )

    # ocr_fallback → preprocess  (or error)
    workflow.add_conditional_edges(
        "ocr_fallback",
        lambda s: "handle_error" if s.get("error") else "preprocess",
        {"preprocess": "preprocess", "handle_error": "handle_error"},
    )

    # preprocess → choose chunker
    workflow.add_conditional_edges(
        "preprocess",
        choose_chunker,
        {"recursive_chunk": "recursive_chunk", "semantic_chunk": "semantic_chunk"},
    )

    # Both chunkers → embed_and_index
    workflow.add_conditional_edges(
        "recursive_chunk",
        lambda s: "handle_error" if s.get("error") else "embed_and_index",
        {"embed_and_index": "embed_and_index", "handle_error": "handle_error"},
    )
    workflow.add_conditional_edges(
        "semantic_chunk",
        lambda s: "handle_error" if s.get("error") else "embed_and_index",
        {"embed_and_index": "embed_and_index", "handle_error": "handle_error"},
    )

    # embed_and_index → END  (or error)
    workflow.add_conditional_edges(
        "embed_and_index",
        lambda s: "handle_error" if s.get("error") else END,
        {END: END, "handle_error": "handle_error"},
    )

    workflow.add_edge("handle_error", END)

    return workflow


# Compiled app — import this anywhere in the codebase
ingestion_app = build_ingestion_graph().compile()
