"""
ingest_graph.py  —  LangGraph IngestGraph

The per-document ingestion pipeline implemented as a LangGraph StateGraph.
Each node is a pure function operating on IngestState.

Graph topology:
    detect → clean → analyze → [PAUSE: user picks chunker + model] → chunk
    → enrich → embed → store → done

The PAUSE is implemented via LangGraph's interrupt_after=["analyze"] pattern.
The Streamlit/FastAPI layer calls graph.invoke() for the first half (up to
analyze), presents stats to the user, collects choices, then calls
graph.invoke() again with the updated state to complete the second half.

Mid-chat background ingest:
    Use IngestGraph.run_background(source_config) which dispatches the full
    graph in a daemon Thread and calls on_complete(source_id) when done.
"""
from __future__ import annotations

import logging
import threading
import uuid
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
    # Input
    source_id:      str
    file_path:      Optional[str]
    url:            Optional[str]
    source_type:    str            # pdf | website | youtube | csv

    # After detect
    raw_result:     Dict[str, Any]  # pipeline output dict

    # After clean
    cleaned_text:   str

    # After analyze  ← PAUSE POINT — shown to UI
    analysis:       Dict[str, Any]  # ContentAnalyzer output (stats + recommendation)

    # User choices (filled in by UI before second invoke)
    chunking_strategy: str          # recursive | semantic | paragraph | page | chapter | hierarchical
    embedding_model:   str          # e.g. "all-MiniLM-L6-v2" | "all-mpnet-base-v2"

    # After chunk
    chunks:         List[Dict[str, Any]]

    # After enrich
    enriched_chunks: List[Dict[str, Any]]

    # After embed
    embedded_chunks: List[Dict[str, Any]]
    embedding_dim:   int

    # After store
    stored:         bool
    error:          Optional[str]


# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_detect(state: IngestState) -> IngestState:
    """Run the appropriate extraction pipeline based on source_type."""
    try:
        st = state["source_type"]
        sid = state["source_id"]
        if st == "pdf":
            result = PDFPipeline.process(state["file_path"], sid)
        elif st == "website":
            import asyncio, nest_asyncio
            nest_asyncio.apply()
            result = asyncio.get_event_loop().run_until_complete(
                WebsitePipeline.process(state["url"], sid)
            )
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
    """AdaptivePreprocessor: clean and normalise raw content."""
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
    """
    ContentAnalyzer: produce stats + strategy recommendation.
    This is the PAUSE POINT — the UI reads state["analysis"] and presents
    it to the user before the graph continues.
    """
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
    """AdaptiveChunker: chunk with user-chosen strategy."""
    if state.get("error"):
        return state
    try:
        strategy = state.get("chunking_strategy", "recursive")
        chunker = AdaptiveChunker(default_strategy=strategy)
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
    """ContextualEnricher: add context windows + source headers."""
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
        pipeline = EmbeddingRegistry.get(model_name)
        embedded = pipeline.embed_chunks(state["enriched_chunks"])
        # Determine dim from first embedding
        dim = len(embedded[0]["embedding"]) if embedded else 384
        return {**state, "embedded_chunks": embedded, "embedding_dim": dim}
    except Exception as exc:
        logger.error("IngestGraph[embed]: %s", exc)
        return {**state, "error": str(exc)}


def _make_node_store(source_manager: SourceManager):
    """Factory — captures source_manager in closure."""
    def node_store(state: IngestState) -> IngestState:
        """SourceManager: write to FAISS + SQLite + KnowledgeGraph."""
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
    """
    Build and compile the LangGraph IngestGraph.

    The graph has an interrupt_after=["analyze"] so the UI can pause,
    show the analysis to the user, collect chunking_strategy + embedding_model,
    then resume.
    """
    builder = StateGraph(IngestState)

    # Register nodes
    builder.add_node("detect",  node_detect)
    builder.add_node("clean",   node_clean)
    builder.add_node("analyze", node_analyze)   # ← PAUSE after this
    builder.add_node("chunk",   node_chunk)
    builder.add_node("enrich",  node_enrich)
    builder.add_node("embed",   node_embed)
    builder.add_node("store",   _make_node_store(source_manager))

    # Edges
    builder.set_entry_point("detect")
    builder.add_edge("detect",  "clean")
    builder.add_edge("clean",   "analyze")
    # After analyze: if error → END, else → chunk
    builder.add_conditional_edges(
        "analyze",
        _should_abort,
        {"abort": END, "continue": "chunk"},
    )
    builder.add_edge("chunk",   "enrich")
    builder.add_edge("enrich",  "embed")
    builder.add_edge("embed",   "store")
    builder.add_edge("store",   END)

    checkpointer = MemorySaver()
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_after=["analyze"],   # PAUSE: wait for user choices
    )


# ── IngestGraph façade ────────────────────────────────────────────────────────

class IngestGraph:
    """
    High-level façade over the compiled LangGraph.

    Usage — two-phase (interactive UI):
        ig = IngestGraph(source_manager)

        # Phase 1: extract + clean + analyze
        thread_id, analysis = ig.phase1(file_path="doc.pdf", source_type="pdf")

        # ... UI shows analysis, user picks strategy + model ...

        # Phase 2: chunk + enrich + embed + store
        result = ig.phase2(thread_id, chunking_strategy="recursive",
                           embedding_model="all-MiniLM-L6-v2")

    Usage — background (mid-chat ingest, no UI pause):
        ig.run_background(
            source_config={"file_path": "new.pdf", "source_type": "pdf",
                           "chunking_strategy": "recursive",
                           "embedding_model": "all-MiniLM-L6-v2"},
            on_complete=lambda sid: print(f"Ready: {sid}"),
            on_error=lambda e: print(f"Failed: {e}"),
        )
    """

    def __init__(self, source_manager: SourceManager):
        self.source_manager = source_manager
        self.graph = build_ingest_graph(source_manager)

    # ── two-phase interactive flow ────────────────────────────────────────────

    def phase1(
        self,
        file_path: Optional[str] = None,
        url: Optional[str] = None,
        source_type: str = "pdf",
    ) -> tuple[str, Dict[str, Any]]:
        """
        Run detect → clean → analyze.  Returns (thread_id, analysis_dict).
        The graph is paused after analyze — call phase2() to continue.
        """
        source_id = str(uuid.uuid4())
        thread_id = str(uuid.uuid4())
        initial_state: IngestState = {
            "source_id":   source_id,
            "file_path":   file_path,
            "url":         url,
            "source_type": source_type,
        }
        config = {"configurable": {"thread_id": thread_id}}
        # Graph runs until interrupt_after=["analyze"] and suspends
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
        """
        Resume from the pause point with user-chosen parameters.
        Runs chunk → enrich → embed → store.
        Returns final state dict.
        """
        config = {"configurable": {"thread_id": thread_id}}
        # Resume: update state with user choices, then continue
        update = {
            "chunking_strategy": chunking_strategy,
            "embedding_model":   embedding_model,
        }
        self.graph.update_state(config, update)
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
        """
        Run the full ingest pipeline in a daemon thread.
        Ideal for mid-chat ingestion — chat continues unblocked.

        Args:
            source_config: dict with keys:
                file_path, url, source_type,
                chunking_strategy, embedding_model
            on_complete: called with source_id when done
            on_error:    called with error string on failure

        Returns:
            The daemon Thread (already started).
        """
        def _run():
            source_id = str(uuid.uuid4())
            thread_id = str(uuid.uuid4())
            initial_state: IngestState = {
                "source_id":         source_id,
                "file_path":         source_config.get("file_path"),
                "url":               source_config.get("url"),
                "source_type":       source_config.get("source_type", "pdf"),
                # Pre-fill user choices — no pause needed in background mode
                "chunking_strategy": source_config.get("chunking_strategy", "recursive"),
                "embedding_model":   source_config.get("embedding_model", "all-MiniLM-L6-v2"),
            }
            config = {"configurable": {"thread_id": thread_id}}
            try:
                # In background mode we skip the interrupt — invoke runs all nodes
                # Build a no-interrupt version for background use
                bg_graph = _build_background_graph(self.source_manager)
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
