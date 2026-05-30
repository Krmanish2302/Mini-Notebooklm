"""
ingest_graph.py — LangGraph IngestGraph

Graph topology:
    validate → load → chunk → embed → store → done
    Any node → handle_error → END

Handles all document types (PDF, web, YouTube, text) via
LangChain document loaders + text splitters from src/ingestion.
State carries both per-source metadata and chunk-level lists.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.storage.source_manager import SourceManager

logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

class IngestState(TypedDict, total=False):
    # ── inputs ──────────────────────────────────────────────────────────
    source_input:    str           # file path, URL, or YouTube URL
    source_type:     str           # "pdf" | "url" | "youtube" | "text"
    source_id:       str
    metadata:        Dict[str, Any]

    # ── intermediate ────────────────────────────────────────────────────
    raw_docs:        List[Any]     # List[Document] from loader
    chunks:          List[Any]     # List[Document] after splitting
    embeddings:      List[Any]     # List[List[float]]

    # ── output ──────────────────────────────────────────────────────────
    chunk_ids:       List[str]
    chunks_stored:   int
    source_record:   Dict[str, Any]

    # ── error ────────────────────────────────────────────────────────────
    error:           Optional[str]
    failed_node:     Optional[str]


# ── Nodes ─────────────────────────────────────────────────────────────────────

def _node_validate(state: IngestState) -> IngestState:
    """Validate that source_input + source_type are present and source_type is known."""
    try:
        src   = state.get("source_input", "").strip()
        stype = state.get("source_type", "").lower().strip()
        if not src:
            return {**state, "error": "source_input is empty", "failed_node": "validate"}
        valid_types = {"pdf", "url", "youtube", "text"}
        if stype not in valid_types:
            return {
                **state,
                "error": f"Unknown source_type '{stype}'. Choose: {valid_types}",
                "failed_node": "validate",
            }
        # Basic path check for PDF
        if stype == "pdf" and not Path(src).exists():
            return {
                **state,
                "error": f"PDF file not found: {src}",
                "failed_node": "validate",
            }
        logger.info("[IngestGraph:validate] OK — type=%s src=%s", stype, src[:80])
        return {**state, "source_type": stype}
    except Exception as exc:
        return {**state, "error": str(exc), "failed_node": "validate"}


def _node_load(state: IngestState) -> IngestState:
    """Load raw documents using the appropriate LangChain loader."""
    if state.get("error"):
        return state
    try:
        from src.ingestion.loader_factory import LoaderFactory
        loader   = LoaderFactory.get(state["source_type"], state["source_input"])
        raw_docs = loader.load()
        logger.info("[IngestGraph:load] Loaded %d docs", len(raw_docs))
        return {**state, "raw_docs": raw_docs}
    except Exception as exc:
        logger.error("[IngestGraph:load] %s", exc)
        return {**state, "error": str(exc), "failed_node": "load"}


def _node_chunk(state: IngestState) -> IngestState:
    """Split raw docs into chunks using RecursiveCharacterTextSplitter."""
    if state.get("error"):
        return state
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(state["raw_docs"])
        # Inject source_id into metadata for downstream citation
        source_id = state.get("source_id", state["source_input"][:40])
        for c in chunks:
            c.metadata.setdefault("source_id", source_id)
            c.metadata.update(state.get("metadata") or {})

        logger.info("[IngestGraph:chunk] %d raw_docs → %d chunks", len(state["raw_docs"]), len(chunks))
        return {**state, "chunks": chunks}
    except Exception as exc:
        logger.error("[IngestGraph:chunk] %s", exc)
        return {**state, "error": str(exc), "failed_node": "chunk"}


def _node_embed(state: IngestState) -> IngestState:
    """Embed chunks using EmbeddingRegistry."""
    if state.get("error"):
        return state
    try:
        from src.ingestion.embedding.embedding_registry import EmbeddingRegistry
        model   = EmbeddingRegistry.get_default()
        texts   = [c.page_content for c in state["chunks"]]
        vectors = model.embed_documents(texts)
        logger.info("[IngestGraph:embed] Embedded %d chunks", len(vectors))
        return {**state, "embeddings": vectors}
    except Exception as exc:
        logger.error("[IngestGraph:embed] %s", exc)
        return {**state, "error": str(exc), "failed_node": "embed"}


def _make_node_store(source_manager: SourceManager):
    def node(state: IngestState) -> IngestState:
        """Store chunks + vectors in FAISS + SQLite via SourceManager."""
        if state.get("error"):
            return state
        try:
            result = source_manager.store_chunks(
                chunks=state["chunks"],
                embeddings=state["embeddings"],
                source_id=state.get("source_id", state["source_input"][:40]),
                metadata=state.get("metadata") or {},
            )
            logger.info(
                "[IngestGraph:store] Stored %d chunks — source_id=%s",
                result["chunks_stored"], result["source_id"],
            )
            return {
                **state,
                "chunk_ids":     result.get("chunk_ids", []),
                "chunks_stored": result["chunks_stored"],
                "source_record": result,
            }
        except Exception as exc:
            logger.error("[IngestGraph:store] %s", exc)
            return {**state, "error": str(exc), "failed_node": "store"}
    return node


def _node_handle_error(state: IngestState) -> IngestState:
    logger.error(
        "[IngestGraph] FAILED at node='%s': %s",
        state.get("failed_node", "unknown"),
        state.get("error", "Unknown error"),
    )
    return state


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_ingest_graph(source_manager: SourceManager) -> Any:
    builder = StateGraph(IngestState)

    builder.add_node("validate",     _node_validate)
    builder.add_node("load",         _node_load)
    builder.add_node("chunk",        _node_chunk)
    builder.add_node("embed",        _node_embed)
    builder.add_node("store",        _make_node_store(source_manager))
    builder.add_node("handle_error", _node_handle_error)

    builder.set_entry_point("validate")

    def _check(s):
        return 