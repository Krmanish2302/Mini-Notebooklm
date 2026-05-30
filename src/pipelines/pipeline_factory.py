"""
pipeline_factory.py — PipelineFactory

Single entry point for the UI (and tests) to obtain a fully-wired pipeline
without manually threading FAISS / SQLite / SourceManager / embedder through
every call site.

Usage (Streamlit app):
    from src.pipelines.pipeline_factory import PipelineFactory

    # Build once in st.session_state
    factory = PipelineFactory.from_session(st.session_state)

    # Get a pipeline by mode
    pipe = factory.get("chat")
    result = pipe.run("What is osmosis?", source_ids=["bio_101"])

    # Switch mode on the fly
    deep = factory.get("deep_research")
    result = deep.run("Explain CRISPR mechanisms in detail")

Modes:
    "chat"          → ChatPipeline
    "study"         → StudyPipeline
    "deep_research" → DeepResearchPipeline
    "ingest"        → run_ingest() helper (not a pipeline class)

All pipelines share:
    - The same MultiFAISSStore instance
    - The same SQLiteManager instance
    - The same SourceManager instance
    - The same Embedder (from EmbeddingRegistry.get_default())
    - session_id scoped to the Streamlit user session
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional, Union

from src.generation.llm_registry import LLMRegistry
from src.ingestion.embedding.embedding_registry import EmbeddingRegistry
from src.pipelines.chat_pipeline import ChatPipeline
from src.pipelines.deep_research_pipeline import DeepResearchPipeline
from src.pipelines.study_pipeline import StudyPipeline
from src.storage.faiss_store import MultiFAISSStore
from src.storage.source_manager import SourceManager
from src.storage.sqlite_manager import SQLiteManager

logger = logging.getLogger(__name__)

Mode = Literal["chat", "study", "deep_research"]


class PipelineFactory:
    """
    Holds shared infrastructure and lazily creates pipelines per mode.

    Pipelines are created once and cached — switching modes is O(1).
    Call reset_pipelines() if the underlying stores change (e.g. new source added).
    """

    def __init__(
        self,
        faiss_store:    MultiFAISSStore,
        sqlite:         SQLiteManager,
        source_manager: SourceManager,
        embedder:       Any,              # LangChain Embeddings
        llm:            Any = None,       # BaseChatModel (optional, falls back to LLMRegistry)
        session_id:     str = "default",
        compressor:     Any = None,
        reranker:       Any = None,
    ):
        self.faiss_store    = faiss_store
        self.sqlite         = sqlite
        self.source_manager = source_manager
        self.embedder       = embedder
        self.llm            = llm
        self.session_id     = session_id
        self.compressor     = compressor
        self.reranker       = reranker

        self._cache: Dict[str, Any] = {}

    # ── Classmethods ───────────────────────────────────────────────────────────────

    @classmethod
    def from_session(cls, session_state: Any, session_id: str = "default") -> "PipelineFactory":
        """
        Build a PipelineFactory from a Streamlit st.session_state dict-like object.

        Expected keys in session_state:
            faiss_store    : MultiFAISSStore
            sqlite         : SQLiteManager
            source_manager : SourceManager
            embedder       : LangChain Embeddings (optional, falls back to EmbeddingRegistry)
            llm            : BaseChatModel (optional, falls back to LLMRegistry)
            compressor     : contextual compressor (optional)
            reranker       : cross-encoder reranker (optional)
        """
        faiss_store    = session_state.get("faiss_store")
        sqlite         = session_state.get("sqlite")
        source_manager = session_state.get("source_manager")
        embedder       = session_state.get("embedder") or EmbeddingRegistry.get_default()
        llm            = session_state.get("llm")
        compressor     = session_state.get("compressor")
        reranker       = session_state.get("reranker")

        if faiss_store is None or sqlite is None or source_manager is None:
            raise ValueError(
                "session_state must contain 'faiss_store', 'sqlite', and 'source_manager'."
            )

        return cls(
            faiss_store    = faiss_store,
            sqlite         = sqlite,
            source_manager = source_manager,
            embedder       = embedder,
            llm            = llm,
            session_id     = session_id,
            compressor     = compressor,
            reranker       = reranker,
        )

    # ── Public API ────────────────────────────────────────────────────────────────

    def get(self, mode: Mode) -> Union[ChatPipeline, StudyPipeline, DeepResearchPipeline]:
        """
        Return the pipeline for `mode`, creating it on first call.

        Pipelines are cached per mode — same instance reused across turns.
        """
        if mode not in self._cache:
            self._cache[mode] = self._build(mode)
            logger.info("[PipelineFactory] Built pipeline mode='%s'", mode)
        return self._cache[mode]

    def set_session(self, session_id: str) -> None:
        """Switch session for all cached pipelines (e.g. multi-user Streamlit)."""
        self.session_id = session_id
        for pipe in self._cache.values():
            if hasattr(pipe, "set_session"):
                pipe.set_session(session_id)

    def set_llm(self, llm: Any) -> None:
        """Hot-swap the LLM for all cached pipelines (e.g. user switches model)."""
        self.llm = llm
        for pipe in self._cache.values():
            if hasattr(pipe, "set_llm"):
                pipe.set_llm(llm)

    def reset_pipelines(self) -> None:
        """
        Discard cached pipelines so the next get() rebuilds them.
        Call after adding/removing sources that change FAISS dimensions.
        """
        self._cache.clear()
        logger.info("[PipelineFactory] Pipeline cache cleared.")

    # ── Internal ────────────────────────────────────────────────────────────────

    def _build(self, mode: Mode) -> Any:
        """Instantiate and configure the pipeline for the given mode."""
        llm = self.llm or LLMRegistry.get()
        common = dict(
            faiss_store    = self.faiss_store,
            sqlite         = self.sqlite,
            source_manager = self.source_manager,
            embedder       = self.embedder,
            compressor     = self.compressor,
            reranker       = self.reranker,
            session_id     = self.session_id,
        )

        if mode == "chat":
            pipe = ChatPipeline(**common)

        elif mode == "study":
            pipe = StudyPipeline(**common)

        elif mode == "deep_research":
            pipe = DeepResearchPipeline(**common, multi_hop=False)

        else:
            raise ValueError(f"Unknown mode '{mode}'. Valid: chat, study, deep_research")

        pipe.set_llm(llm)
        return pipe
