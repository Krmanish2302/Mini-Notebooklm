"""
ingest_graph.py  —  LangGraph IngestGraph

Fixes applied
-------------
BUG-C04  node_detect used nest_asyncio.apply() + get_event_loop().run_until_complete()
         for the website pipeline.  Inside a FastAPI / uvicorn async context this
         causes a deadlock.  Fixed: run WebsitePipeline in a ThreadPoolExecutor
         with asyncio.run() so it gets its own event loop.
BUG-R05  node_embed silently defaulted dim=384 when embed_chunks returned [].  A
         missing embedding is a hard error — the store would silently route chunks
         to the wrong FAISS index.  Now returns an error state instead.
BUG-F03  _build_background_graph was called inside the thread closure on every
         background ingest job — compiling a new StateGraph from scratch each time.
         IngestGraph.__init__ now pre-compiles and caches self._bg_graph.
BUG-Q04  Inline imports (asyncio, nest_asyncio inside node_detect) moved to the
         module top level.
"""
from __future__ import annotations

import asyncio                     # BUG-Q04: moved from inside node_detect
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor  # BUG-C04
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.ingestion.pipelines.pdf_pipeline import PDFPipeline
from src.ingestion.pipelines.website_pipeline import WebsitePipeline
from src.ingestion.pipelines.youtube_pipeline import YouTubePipeline
from src.ingestion.pipelines.csv_pipeline import CSVPipeline
from src.ingestion.preprocessing.adaptive_preprocessor import AdaptivePreprocessor
from src.ingestion.preprocessing.content_analyzer import ContentAnalyzer
from src.ingestion.preprocessing.contextual_enricher import ContextualEnricher
from src.ingestion.chunking.adaptive_chunker import AdaptiveChunker
from src.ingestion.embedding.embedding_registry import EmbeddingRegistry
from src.storage.source_manager import SourceManager

logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

class IngestState(TypedDict, total=False):
    source_id:         str
    file_path:         Optional[str]
    url:               Optional[str]
    source_type:       str
    raw_result:        Dict[str, Any]
    cleaned_text:      str
    analysis:          Dict[str, Any]
    chunking_strategy: str
    embedding_model:   str
    chunks:            List[Dict[str, Any]]
    enriched_chunks:   List[Dict[str, Any]]
    embedded_chunks:   List[Dict[str, Any]]
    embedding_dim:     int
    stored:            bool
    error:             Optional[str]


# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_detect(state: IngestState) -> IngestState:
    """Run the appropriate extraction pipeline based on source_type."""
    try:
        st  = state["source_type"]
        sid = state["source_id"]
        if st == "pdf":
            result = PDFPipeline.process(state["file_path"], sid)
        elif st == "website":
            # BUG-C04: use ThreadPoolExecutor + asyncio.run() instead of
            # nest_asyncio + run_until_complete() which deadlocks FastAPI.
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    WebsitePipeline.process(state["url"], sid),
                )
                result = future.result()
        elif st == "youtube":
            result = YouTubePipeline.process(state["url"], sid)
        elif st == "csv":
            result = CSVPipeline.process(state["file_path"], sid)
        else:
            raise ValueError(f"Unsupported source_type: {st}")
        return {**state, "raw_result": result}
    except Exception as exc:
        logger.error("IngestGraph[detect]: %s", exc)
        return {**state, "error": str(exc)}


def node_clean(state: IngestState) -> IngestState:
    if state.get("error"):
        return state
    try:
        preprocessor = AdaptivePreprocessor()
        raw = state["raw_result"]
        processed = preprocessor.process(
            raw.get("content", ""),
            state["source_type"],
            raw.get("metadata", {}),
        )
        return {**state, "cleaned_text": processed["cleaned_content"]}
    except Exception as exc:
        logger.error("IngestGraph[clean]: %s", exc)
        return {**state, "error": str(exc)}


def node_analyze(state: IngestState) -> IngestState:
    if state.get("error"):
        return state
    try:
        analyzer = ContentAnalyzer()
        analysis = analyzer.analyze(
            state["cleaned_text"], source_type=state["source_type"]
        )
        return {**state, "analysis": analysis}
    except Exception as exc:
        logger.error("IngestGraph[analyze]: %s", exc)
        return {**state, "error": str(exc)}


def node_chunk(state: IngestState) -> IngestState:
    if state.get("error"):
        return state
    try:
        strategy = state.get("chunking_strategy", "recursive")
        chunker  = AdaptiveChunker(default_strategy=strategy)
        meta = {
            "source_id":   state["source_id"],
            "source_type": state["source_type"],
            "modality":    state["raw_result"].get("modality", "text"),
            **state["raw_result"].get("metadata", {}),
        }
        chunks = chunker.chunk(state["cleaned_text"], strategy=strategy, metadata=meta)
        return {**state, "chunks": chunks}
    except Exception as exc:
        logger.error("IngestGraph[chunk]: %s", exc)
        return {**state, "error": str(exc)}


def node_enrich(state: IngestState) -> IngestState:
    if state.get("error"):
        return state
    try:
        enricher = ContextualEnricher(window_sentences=3)
        raw_meta = state["raw_result"].get("metadata", {})
        enriched = enricher.enrich(
            state["chunks"],
            metadata={
                "title":       raw_meta.get("title", state.get("file_path") or state.get("url", "")),
                "source_type": state["source_type"],
            },
        )
        return {**state, "enriched_chunks": enriched}
    except Exception as exc:
        logger.error("IngestGraph[enrich]: %s", exc)
        return {**state, "error": str(exc)}


def node_embed(state: IngestState) -> IngestState:
    """EmbeddingRegistry: embed with user-chosen model."""
    if state.get("error"):
        return state
    try:
        model_name = state.get("embedding_model", "all-MiniLM-L6-v2")
        pipeline   = EmbeddingRegistry.get(model_name)
        embedded   = pipeline.embed_chunks(state["enriched_chunks"])

        # BUG-R05: hard error on empty result — silent dim=384 fallback would
        # route chunks to the wrong FAISS index.
        if not embedded:
            return {**state, "error": "embed_chunks returned empty — all chunks failed to embed"}

        dim = len(embedded[0]["embedding"])
        return {**state, "embedded_chunks": embedded, "embedding_dim": dim}
    except Exception as exc:
        logger.error("IngestGraph[embed]: %s", exc)
        return {**state, "error": str(exc)}


def _make_node_store(source_manager: SourceManager):
    def node_store(state: IngestState) -> IngestState:
        if state.get("error"):
            return state
        try:
            raw_meta = state["raw_result"].get("metadata", {})
            source = {
                "id":          state["source_id"],
                "title":       raw_meta.get("title", state.get("file_path") or state.get("url", "Untitled")),
                "source_type": state["source_type"],
                "file_path":   state.get("file_path"),
                "url":         state.get("url"),
                "metadata":    raw_meta,
                "status":      "ready",
            }
            source_manager.add_source(
                source=source,
                chunks=state["embedded_chunks"],
                embedding_model=state.get("embedding_model", "all-MiniLM-L6-v2"),
                dim=state["embedding_dim"],
            )
            return {**state, "stored": True}
        except Exception as exc:
            logger.error("IngestGraph[store]: %s", exc)
            return {**state, "error": str(exc)}
    return node_store


# ── Graph builder ─────────────────────────────────────────────────────────────

def _should_abort(state: IngestState) -> str:
    return "abort" if state.get("error") else "continue"


def build_ingest_graph(source_manager: SourceManager) -> StateGraph:
    builder = StateGraph(IngestState)
    builder.add_node("detect",  node_detect)
    builder.add_node("clean",   node_clean)
    builder.add_node("analyze", node_analyze)
    builder.add_node("chunk",   node_chunk)
    builder.add_node("enrich",  node_enrich)
    builder.add_node("embed",   node_embed)
    builder.add_node("store",   _make_node_store(source_manager))
    builder.set_entry_point("detect")
    builder.add_edge("detect", "clean")
    builder.add_edge("clean",  "analyze")
    builder.add_conditional_edges(
        "analyze", _should_abort, {"abort": END, "continue": "chunk"},
    )
    builder.add_edge("chunk",  "enrich")
    builder.add_edge("enrich", "embed")
    builder.add_edge("embed",  "store")
    builder.add_edge("store",  END)
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer, interrupt_after=["analyze"])


def _build_background_graph(source_manager: SourceManager):
    """Same graph as build_ingest_graph but WITHOUT interrupt — for background use."""
    builder = StateGraph(IngestState)
    builder.add_node("detect",  node_detect)
    builder.add_node("clean",   node_clean)
    builder.add_node("analyze", node_analyze)
    builder.add_node("chunk",   node_chunk)
    builder.add_node("enrich",  node_enrich)
    builder.add_node("embed",   node_embed)
    builder.add_node("store",   _make_node_store(source_manager))
    builder.set_entry_point("detect")
    builder.add_edge("detect", "clean")
    builder.add_edge("clean",  "analyze")
    builder.add_conditional_edges(
        "analyze", _should_abort, {"abort": END, "continue": "chunk"}
    )
    builder.add_edge("chunk",  "enrich")
    builder.add_edge("enrich", "embed")
    builder.add_edge("embed",  "store")
    builder.add_edge("store",  END)
    return builder.compile()


# ── IngestGraph façade ────────────────────────────────────────────────────────

class IngestGraph:
    """
    High-level façade over the compiled LangGraph.

    BUG-F03 fix: both the interactive graph and the background graph are
    compiled ONCE at __init__ time.  run_background() reuses self._bg_graph
    instead of calling _build_background_graph() inside the thread closure.
    """

    def __init__(self, source_manager: SourceManager):
        self.source_manager = source_manager
        # BUG-F03: compile both graphs once
        self.graph    = build_ingest_graph(source_manager)       # interactive (interrupt)
        self._bg_graph = _build_background_graph(source_manager) # background (no interrupt)

    # ── two-phase interactive flow ────────────────────────────────────────────

    def phase1(
        self,
        file_path: Optional[str] = None,
        url: Optional[str] = None,
        source_type: str = "pdf",
    ) -> tuple:
        source_id = str(uuid.uuid4())
        thread_id = str(uuid.uuid4())
        initial_state: IngestState = {
            "source_id":   source_id,
            "file_path":   file_path,
            "url":         url,
            "source_type": source_type,
        }
        config = {"configurable": {"thread_id": thread_id}}
        state = self.graph.invoke(initial_state, config=config)
        if state.get("error"):
            raise RuntimeError(f"IngestGraph phase1 error: {state['error']}")
        return thread_id, state.get("analysis", {})

    def phase2(
        self,
        thread_id: str,
        chunking_strategy: str,
        embedding_model: str,
    ) -> Dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        self.graph.update_state(config, {
            "chunking_strategy": chunking_strategy,
            "embedding_model":   embedding_model,
        })
        state = self.graph.invoke(None, config=config)
        if state.get("error"):
            raise RuntimeError(f"IngestGraph phase2 error: {state['error']}")
        return state

    # ── background ingest (mid-chat) ──────────────────────────────────────────

    def run_background(
        self,
        source_config: Dict[str, Any],
        on_complete: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> threading.Thread:
        # Capture the pre-compiled graph to avoid recompilation in the thread
        bg_graph = self._bg_graph  # BUG-F03: cached at __init__

        def _run():
            source_id = str(uuid.uuid4())
            thread_id = str(uuid.uuid4())
            initial_state: IngestState = {
                "source_id":         source_id,
                "file_path":         source_config.get("file_path"),
                "url":               source_config.get("url"),
                "source_type":       source_config.get("source_type", "pdf"),
                "chunking_strategy": source_config.get("chunking_strategy", "recursive"),
                "embedding_model":   source_config.get("embedding_model", "all-MiniLM-L6-v2"),
            }
            config = {"configurable": {"thread_id": thread_id}}
            try:
                state = bg_graph.invoke(initial_state, config=config)
                if state.get("error"):
                    raise RuntimeError(state["error"])
                if on_complete:
                    on_complete(source_id)
            except Exception as exc:
                logger.error("IngestGraph background error: %s", exc)
                if on_error:
                    on_error(str(exc))

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t
