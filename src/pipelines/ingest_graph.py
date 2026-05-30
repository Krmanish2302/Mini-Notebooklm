"""
ingest_graph.py — LangGraph IngestGraph (v2)

Rewritten as a thin LangGraph wrapper over src.ingestion.ingestion_router.

Previous version had its own load/chunk/embed nodes using a generic
RecursiveCharacterTextSplitter(512) and a LoaderFactory that no longer
exists — bypassing all per-source pipeline logic.

New topology:
    validate → ingest → done
    validate → handle_error → END
    ingest   → handle_error → END  (on error key in state)

All routing (PDF analysis, YouTube cleaning, image captioning, etc.) is
delegated to ingestion_router.ingest(), which calls the correct pipeline.

State contract (input):
    source_type   : str   — pdf | youtube | text | image | website
    source_input  : str   — file path or URL
    source_id     : str   — unique ID for this source
    strategy      : str   — PDF only — chunking strategy, default paragraph_based
    analyze_only  : bool  — PDF only — return stats without full ingest
    content       : str   — text/paste only — raw text content
    metadata      : dict  — optional extra metadata

State contract (output, appended):
    ingest_result : dict  — full result from ingestion_router.ingest()
    chunks_stored : int
    vectorstore_path : str
    error         : str   — set on failure
    failed_node   : str   — set on failure
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TypedDict

from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────

class IngestState(TypedDict, total=False):
    # ── inputs ───────────────────────────────────────────────────
    source_type:      str
    source_input:     str
    source_id:        str
    strategy:         str
    analyze_only:     bool
    content:          Optional[str]
    metadata:         Optional[Dict[str, Any]]

    # ── outputs ────────────────────────────────────────────────
    ingest_result:    Dict[str, Any]
    chunks_stored:    int
    vectorstore_path: str

    # ── error ────────────────────────────────────────────────
    error:            Optional[str]
    failed_node:      Optional[str]


# ── Nodes ────────────────────────────────────────────────────────────────

SOURCE_TYPES = {"pdf", "youtube", "text", "image", "website"}


def _node_validate(state: IngestState) -> IngestState:
    """
    Guard: check required fields and source_type membership.
    Also maps legacy source_type aliases ("url" → "website") for back-compat.
    """
    try:
        src   = (state.get("source_input") or "").strip()
        stype = (state.get("source_type")  or "").lower().strip()
        sid   = (state.get("source_id")    or "").strip()

        if not sid:
            return {**state, "error": "source_id is required", "failed_node": "validate"}

        # Legacy alias
        if stype == "url":
            stype = "website"

        if stype not in SOURCE_TYPES:
            return {
                **state,
                "error": f"Unknown source_type '{stype}'. Valid: {SOURCE_TYPES}",
                "failed_node": "validate",
            }

        # text/paste — source_input can be empty if content is supplied
        if stype == "text" and not src and not state.get("content"):
            return {
                **state,
                "error": "Either source_input or content is required for text ingestion.",
                "failed_node": "validate",
            }

        if stype != "text" and not src:
            return {**state, "error": "source_input is required", "failed_node": "validate"}

        logger.info("[IngestGraph:validate] OK — type=%s src=%s", stype, src[:80] if src else "<paste>")
        return {**state, "source_type": stype}

    except Exception as exc:
        return {**state, "error": str(exc), "failed_node": "validate"}


def _node_ingest(state: IngestState) -> IngestState:
    """
    Delegate to ingestion_router.ingest() — handles all source-type logic.
    """
    if state.get("error"):
        return state
    try:
        from src.ingestion.ingestion_router import ingest

        result = ingest(
            source_type   = state["source_type"],
            source_id     = state["source_id"],
            file_path     = state.get("source_input") or None,
            content       = state.get("content"),
            analyze_only  = bool(state.get("analyze_only", False)),
            strategy      = state.get("strategy", "paragraph_based"),
            embedding_dim = 384,
        )

        chunks_stored    = result.get("num_chunks", 0)
        vectorstore_path = result.get("vectorstore_path", "")

        logger.info(
            "[IngestGraph:ingest] Done — type=%s id=%s chunks=%d",
            state["source_type"], state["source_id"], chunks_stored,
        )
        return {
            **state,
            "ingest_result":    result,
            "chunks_stored":    chunks_stored,
            "vectorstore_path": vectorstore_path,
        }

    except Exception as exc:
        logger.error("[IngestGraph:ingest] %s", exc)
        return {**state, "error": str(exc), "failed_node": "ingest"}


def _node_handle_error(state: IngestState) -> IngestState:
    logger.error(
        "[IngestGraph] FAILED at node='%s': %s",
        state.get("failed_node", "unknown"),
        state.get("error", "Unknown error"),
    )
    return state


# ── Conditional routing ────────────────────────────────────────────────────────

def _route(state: IngestState) -> str:
    return "handle_error" if state.get("error") else "ingest"


def _route_after_ingest(state: IngestState) -> str:
    return "handle_error" if state.get("error") else END


# ── Graph builder ─────────────────────────────────────────────────────────────────

def build_ingest_graph() -> Any:
    """
    Build and compile the ingestion LangGraph.
    No external dependencies injected — all delegation goes through ingestion_router.
    """
    builder = StateGraph(IngestState)

    builder.add_node("validate",     _node_validate)
    builder.add_node("ingest",       _node_ingest)
    builder.add_node("handle_error", _node_handle_error)

    builder.set_entry_point("validate")
    builder.add_conditional_edges("validate", _route, {"ingest": "ingest", "handle_error": "handle_error"})
    builder.add_conditional_edges("ingest",   _route_after_ingest, {"handle_error": "handle_error", END: END})
    builder.add_edge("handle_error", END)

    return builder.compile()


# Module-level compiled graph (import and reuse)
ingest_app = build_ingest_graph()


# ── Convenience runner ────────────────────────────────────────────────────────────────

def run_ingest(
    source_type:   str,
    source_id:     str,
    source_input:  str  = "",
    content:       str  = None,
    strategy:      str  = "paragraph_based",
    analyze_only:  bool = False,
    metadata:      dict = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper — invoke the ingest graph with keyword arguments.

    Returns the final IngestState dict.
    Raises RuntimeError if the graph ends in an error state.
    """
    state = ingest_app.invoke({
        "source_type":  source_type,
        "source_id":    source_id,
        "source_input": source_input,
        "content":      content,
        "strategy":     strategy,
        "analyze_only": analyze_only,
        "metadata":     metadata or {},
    })
    if state.get("error"):
        raise RuntimeError(
            f"[IngestGraph] {state['failed_node']}: {state['error']}"
        )
    return state
