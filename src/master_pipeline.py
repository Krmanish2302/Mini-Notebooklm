"""
master_pipeline.py  —  Top-level orchestrator for Mini NotebookLM

Fixes applied
-------------
1. HybridRetriever constructed with embedders={dim: embedder} dict (not dim int).
2. generate() delegates to ChatPipeline / DeepResearchPipeline / StudyPipeline
   depending on self.mode — BM25 sparse path is now always exercised.
3. ChatPipeline, DeepResearchPipeline, StudyPipeline are all instantiated at
   startup and re-wired in set_mode() so the LLM callable is always current.
4. set_mode() re-creates only the history objects; pipelines are updated in-place
   via _rebuild_mode_pipelines().
5. stream=True path in generate() is preserved for ChatPipeline only (Deep
   Research and Study are never streamed — they do multi-step LLM calls).
6. persona_config parameter wired through generate() → ChatPipeline.run().
7. Mode string "deep_research" now matched correctly (was only "research").
8. generate() (non-streaming) now returns the full result dict, not bare str.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
import time
from typing import Any, Dict, Iterator, List, Optional

import nest_asyncio
nest_asyncio.apply()

logger = logging.getLogger(__name__)

# ── Core ──────────────────────────────────────────────────────────────────────
from src.core.config import Config
from src.core.models import Query, LLMResponse

# ── Ingestion ─────────────────────────────────────────────────────────────────
from src.ingestion.file_detector import FileDetector
from src.ingestion.pipelines.pdf_pipeline import PDFPipeline
from src.ingestion.pipelines.image_pipeline import ImagePipeline
from src.ingestion.pipelines.website_pipeline import WebsitePipeline
from src.ingestion.pipelines.youtube_pipeline import YouTubePipeline
from src.ingestion.pipelines.csv_pipeline import CSVPipeline
from src.ingestion.preprocessing.adaptive_preprocessor import AdaptivePreprocessor
from src.ingestion.chunking.chunker_registry import ChunkerRegistry
from src.ingestion.embedding.embedding_pipeline import EmbeddingPipeline
from src.ingestion.embedding.embedding_registry import EmbeddingRegistry
from src.ingestion.merging.cross_modal_merger import CrossModalMerger
from src.ingestion.ingestion_graph import IngestionGraph

# ── Storage ───────────────────────────────────────────────────────────────────
from src.storage import (
    MultiFAISSStore,
    SQLiteManager,
    KnowledgeGraph,
    StorageManager,
)

# ── Graph ─────────────────────────────────────────────────────────────────────
from src.graph.graph_storage import GraphStorage
from src.graph.graph_retriever import GraphRetriever

# ── Retrieval ─────────────────────────────────────────────────────────────────
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.contextual_compressor import ContextualCompressor

# ── Mode Pipelines ────────────────────────────────────────────────────────────
from src.pipelines.chat_pipeline import ChatPipeline
from src.pipelines.deep_research_pipeline import DeepResearchPipeline
from src.pipelines.study_pipeline import StudyPipeline

# ── Generation ────────────────────────────────────────────────────────────────
from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import PromptBuilder
from src.generation.response_parser import ResponseParser

# ── Chat History ──────────────────────────────────────────────────────────────
from src.chat_history.chat_history_manager import ChatHistoryManager
from src.chat_history.rag_history import RAGChatHistory
from src.chat_history.graph_history import GraphHistory


class MasterPipeline:
    """
    Single orchestration surface for Mini NotebookLM.

    Ingestion flow:
        file/url
          -> FileDetector (auto-detect when source_type is None)
          -> Pipeline (PDF/Website/YouTube/CSV/Image)
          -> AdaptivePreprocessor
          -> ChunkerRegistry.get_chunker(strategy)
          -> EmbeddingPipeline.embed_chunks()
          -> CrossModalMerger.merge()
          -> StorageManager.store()               <- FAISS + SQLite + KnowledgeGraph
          -> HybridRetriever.build_sparse_index() <- BM25 rebuild

    Generation flow (mode-dispatched):
        query
          -> ChatPipeline.run(query)          [chat mode]
          -> DeepResearchPipeline.run(query)  [research / deep_research mode]
          -> StudyPipeline.run(query)         [study mode]
    """

    def __init__(self, mode: str = "chat", config_path: str = "config.yaml"):
        self.config = Config(config_path)
        self.mode = mode
        self.session_id = str(uuid.uuid4())

        # ── Storage ───────────────────────────────────────────────────────────
        _faiss  = MultiFAISSStore(
            base_dir=self.config.get("storage.faiss_dir", "./data/vector_store")
        )
        _sqlite = SQLiteManager(
            db_path=self.config.get("storage.db_path", "./data/metadata.db")
        )
        _graph  = KnowledgeGraph(
            edge_threshold=self.config.get("storage.graph_edge_threshold", 0.75)
        )
        self.storage_manager = StorageManager(_faiss, _sqlite, _graph)

        # Legacy GraphStorage — kept for GraphRetriever / StudyPipeline
        self.graph_storage   = GraphStorage()
        self.graph_retriever = GraphRetriever(self.graph_storage)

        # ── Ingestion ─────────────────────────────────────────────────────────
        self.file_detector      = FileDetector()
        self.preprocessor       = AdaptivePreprocessor()
        self._default_embed_model: str = self.config.get(
            "embedding.default_model", "all-MiniLM-L6-v2"
        )
        self.embedder           = EmbeddingPipeline(model_name=self._default_embed_model)
        self.cross_modal_merger = CrossModalMerger(embedder=self.embedder.embedder)
        self.ingestion_graph    = IngestionGraph()

        # Probe to get correct dim (never trust config.yaml alone)
        _probe = self.embedder.embed_query("probe")
        self._embed_dim: int   = int(_probe.shape[0])
        self._embed_model: str = self.embedder.model_name
        logger.info(
            "MasterPipeline: embedding dim=%d model=%s",
            self._embed_dim, self._embed_model,
        )

        # ── Retrieval ─────────────────────────────────────────────────────────
        self._embedders: Dict[int, Any] = {self._embed_dim: self.embedder}

        self.hybrid_retriever = HybridRetriever(
            faiss_store=self.storage_manager.faiss,
            storage_manager=self.storage_manager,
            embedders=self._embedders,
            top_k=self.config.get("retrieval.top_k", 5),
        )

        # Reranker + compressor (shared by DeepResearch & Study)
        self.reranker   = Reranker()
        self.compressor = ContextualCompressor()

        # ── Generation ────────────────────────────────────────────────────────
        self.llm: Optional[LLMClient] = None
        self.prompt_builder  = PromptBuilder()
        self.response_parser = ResponseParser()

        # ── Chat History (per-session) ────────────────────────────────────────
        self.rag_history   = RAGChatHistory(session_id=self.session_id)
        self.graph_history = GraphHistory(
            session_id=self.session_id,
            knowledge_graph=self.storage_manager.graph,
        )
        self.chat_history = ChatHistoryManager(
            self.session_id, mode, self.graph_storage
        )

        # ── Mode Pipelines ────────────────────────────────────────────────────
        self._chat_pipeline:     Optional[ChatPipeline]         = None
        self._research_pipeline: Optional[DeepResearchPipeline] = None
        self._study_pipeline:    Optional[StudyPipeline]        = None

        # Response cache
        self._response_cache: Dict[str, str] = {}

        # Warm up EmbeddingRegistry for all persisted FAISS dims
        self._warmup_registry()

    # ── LLM / Mode config ─────────────────────────────────────────────────────

    def set_llm(self, provider: str, model: str, api_key: str, **kwargs) -> None:
        """Configure (or reconfigure) the LLM client, then rebuild mode pipelines."""
        self.llm = LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            **kwargs,
        )
        logger.info("MasterPipeline: LLM set to %s/%s", provider, model)
        self._rebuild_mode_pipelines()

    def set_mode(self, mode: str) -> None:
        """
        Switch conversation mode.  Preserves session_id so history is NOT wiped.
        Only the active pipeline changes.
        """
        self.mode = mode
        self.rag_history   = RAGChatHistory(session_id=self.session_id)
        self.graph_history = GraphHistory(
            session_id=self.session_id,
            knowledge_graph=self.storage_manager.graph,
        )
        self.chat_history = ChatHistoryManager(
            self.session_id, mode, self.graph_storage
        )
        if self.llm:
            self._rebuild_mode_pipelines()
        logger.info(
            "MasterPipeline: mode switched to '%s' (session=%s)",
            mode, self.session_id,
        )

    def _rebuild_mode_pipelines(self) -> None:
        """
        (Re)build ChatPipeline, DeepResearchPipeline, StudyPipeline.
        Called after set_llm() or set_mode().
        """
        if self.llm is None:
            return

        llm_callable = self.llm.invoke

        # -- Chat -------------------------------------------------------
        self._chat_pipeline = ChatPipeline(
            hybrid_retriever=self.hybrid_retriever,
            rag_history=self.rag_history,
            llm=llm_callable,
            top_k=self.config.get("retrieval.top_k", 5),
            history_k=3,
        )

        # -- Deep Research ----------------------------------------------
        self._research_pipeline = DeepResearchPipeline(
            hybrid_retriever=self.hybrid_retriever,
            rag_history=self.rag_history,
            contextual_compressor=self.compressor,
            reranker=self.reranker,
            llm=llm_callable,
            raptor=None,
            top_k=self.config.get("retrieval.top_k", 8),
            history_k=5,
            expansion_n=3,
        )

        # -- Study ------------------------------------------------------
        self._study_pipeline = StudyPipeline(
            deep_research_pipeline=self._research_pipeline,
            graph_retriever=self.graph_retriever,
            graph_history=self.graph_history,
            llm=llm_callable,
        )

        logger.info("MasterPipeline: mode pipelines rebuilt for mode='%s'", self.mode)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(
        self,
        file_path: Optional[str] = None,
        url: Optional[str] = None,
        source_type: Optional[str] = None,
        chunking_strategy: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a file or URL into the knowledge base.

        Returns
        -------
        Dict with source_id, num_chunks, profile, chunking_strategy,
        embedding_model, embedding_dim.
        """
        source_id = str(uuid.uuid4())

        # ── Step 1: resolve source_type ───────────────────────────────────────
        if source_type:
            resolved_type = source_type.lower().strip()
        else:
            try:
                detection     = self.file_detector.detect(file_path=file_path, url=url)
                resolved_type = detection["source_type"]
            except Exception as exc:
                raise RuntimeError(f"FileDetector failed: {exc}") from exc

        self.ingestion_graph.add_source(
            source_id,
            name=file_path or url or "unknown",
            source_type=resolved_type,
        )

        # ── Step 2: extract ───────────────────────────────────────────────────
        try:
            if resolved_type == "pdf":
                result = PDFPipeline.process(file_path, source_id)
            elif resolved_type == "image":
                result = ImagePipeline.process(file_path, source_id)
            elif resolved_type in ("video", "audio"):
                from src.ingestion.pipelines.video_audio_pipeline import VideoAudioPipeline
                result = VideoAudioPipeline.process(file_path, source_id)
            elif resolved_type == "youtube":
                result = YouTubePipeline.process(url, source_id)
            elif resolved_type == "website":
                loop   = asyncio.get_event_loop()
                result = loop.run_until_complete(
                    WebsitePipeline.process(url, source_id)
                )
            elif resolved_type == "csv":
                result = CSVPipeline.process(file_path, source_id)
            else:
                raise ValueError(f"Unknown source_type: '{resolved_type}'")
        except NotImplementedError:
            raise
        except Exception as exc:
            self.ingestion_graph.mark_error(source_id, stage="extract", error=str(exc))
            raise RuntimeError(f"Extraction failed for '{resolved_type}': {exc}") from exc

        raw_content = result.get("content", "")
        self.ingestion_graph.mark_stage(source_id, "extract")

        # ── Step 3: preprocess ────────────────────────────────────────────────
        try:
            preprocessed = self.preprocessor.preprocess(
                raw_content, source_type=resolved_type
            )
        except Exception as exc:
            self.ingestion_graph.mark_error(source_id, stage="preprocess", error=str(exc))
            raise RuntimeError(f"Preprocessing failed: {exc}") from exc
        self.ingestion_graph.mark_stage(source_id, "preprocess")

        # ── Step 4: chunk ─────────────────────────────────────────────────────
        try:
            effective_strategy = chunking_strategy or resolved_type
            chunker = ChunkerRegistry.get_chunker(effective_strategy)
            chunks  = chunker.chunk(preprocessed, source_id=source_id)
        except Exception as exc:
            self.ingestion_graph.mark_error(source_id, stage="chunk", error=str(exc))
            raise RuntimeError(f"Chunking failed: {exc}") from exc
        self.ingestion_graph.mark_stage(source_id, "chunk")

        # ── Step 5: embed ─────────────────────────────────────────────────────
        try:
            if embedding_model and embedding_model != self._embed_model:
                active_embedder = EmbeddingRegistry.get(embedding_model)
                _probe          = active_embedder.embed_query("probe")
                active_dim      = int(_probe.shape[0])
                active_model    = embedding_model
                logger.info(
                    "ingest: user-selected embedder %s (dim=%d)",
                    active_model, active_dim,
                )
                # Register in HybridRetriever so future queries search this dim too
                self._embedders[active_dim] = active_embedder
            else:
                active_embedder = self.embedder
                active_dim      = self._embed_dim
                active_model    = self._embed_model

            embedded_chunks = active_embedder.embed_chunks(chunks)
        except Exception as exc:
            self.ingestion_graph.mark_error(source_id, stage="embed", error=str(exc))
            raise RuntimeError(f"Embedding failed: {exc}") from exc
        self.ingestion_graph.mark_stage(source_id, "embed")

        # ── Step 6: cross-modal merge (no-op for single-source) ───────────────
        try:
            merged_chunks = self.cross_modal_merger.merge([embedded_chunks])
        except Exception as exc:
            logger.warning("CrossModalMerger failed (non-fatal): %s", exc)
            merged_chunks = embedded_chunks

        # ── Step 7: store ─────────────────────────────────────────────────────
        source_record = {
            "id":         source_id,
            "name":       file_path or url or "unknown",
            "type":       resolved_type,
            "created_at": time.time(),
        }
        try:
            self.storage_manager.store(
                source_record,
                merged_chunks,
                embedding_model=active_model,
                dim=active_dim,
            )
        except Exception as exc:
            self.ingestion_graph.mark_error(source_id, stage="store", error=str(exc))
            raise RuntimeError(f"Storage failed: {exc}") from exc
        self.ingestion_graph.mark_stage(source_id, "store")

        # ── Step 8: rebuild BM25 sparse index ─────────────────────────────────
        try:
            all_chunks = self.storage_manager.get_all_chunks_for_bm25()
            self.hybrid_retriever.build_sparse_index(all_chunks)
        except Exception as exc:
            logger.warning("BM25 rebuild failed (non-fatal): %s", exc)

        self.ingestion_graph.mark_stage(source_id, "complete")
        return {
            "source_id":         source_id,
            "num_chunks":        len(merged_chunks),
            "profile":           result.get("profile"),
            "chunking_strategy": effective_strategy,
            "embedding_model":   active_model,
            "embedding_dim":     active_dim,
        }

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        stream: bool = False,
        persona_config=None,          # FIX 6: wired through to ChatPipeline
    ):
        """
        Route query to the correct mode pipeline.

        Chat mode supports stream=True.
        Deep Research and Study mode always return a full dict
        (they make multiple internal LLM calls; streaming is not meaningful).

        Returns
        -------
        Dict[str, Any]   (stream=False — full result with answer + citations)
        Iterator[str]    (stream=True, Chat mode only)
        """
        if not self.llm:
            raise RuntimeError("LLM not configured. Call set_llm() first.")
        if self._chat_pipeline is None:
            raise RuntimeError(
                "Mode pipelines not initialised. "
                "This should not happen — call set_llm() before generate()."
            )

        # ── Chat mode ────────────────────────────────────────────────────────
        if self.mode == "chat":
            if stream:
                # Stream path: retrieve → build prompt → stream tokens → record history
                chunks = self.hybrid_retriever.retrieve(
                    query,
                    top_k=self.config.get("retrieval.top_k", 5),
                )
                history_context = self.rag_history.format_for_prompt(query, k=3)
                prompt = self._chat_pipeline._build_prompt(
                    query, chunks, history_context, persona_config=persona_config
                )

                def _stream_and_record() -> Iterator[str]:
                    full = ""
                    for token in self.llm.stream(prompt):
                        full += token
                        yield token
                    self.rag_history.add_message("user", query)
                    self.rag_history.add_message("assistant", full)

                return _stream_and_record()
            else:
                # FIX 6 + FIX 8: pass persona_config, return full dict
                return self._chat_pipeline.run(query, persona_config=persona_config)

        # ── Deep Research mode (FIX 7: accept both "research" and "deep_research") ─
        elif self.mode in ("research", "deep_research"):
            return self._research_pipeline.run(query)

        # ── Study mode ───────────────────────────────────────────────────────
        elif self.mode == "study":
            return self._study_pipeline.run(query)

        else:
            raise ValueError(f"Unknown mode: '{self.mode}'")

    def generate_full(
        self,
        query: str,
        persona_config=None,
    ) -> Dict[str, Any]:
        """
        Like generate() but always non-streaming and always returns the full
        result dict including sources, learning_path, sub_queries etc.
        """
        if not self.llm:
            raise RuntimeError("LLM not configured. Call set_llm() first.")
        if self._chat_pipeline is None:
            raise RuntimeError("Mode pipelines not initialised.")

        if self.mode == "chat":
            return self._chat_pipeline.run(query, persona_config=persona_config)
        elif self.mode in ("research", "deep_research"):
            return self._research_pipeline.run(query)
        elif self.mode == "study":
            return self._study_pipeline.run(query)
        else:
            raise ValueError(f"Unknown mode: '{self.mode}'")

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _warmup_registry(self) -> None:
        """
        Pre-load EmbeddingPipeline for every FAISS dim that was persisted
        in a previous session so generate() can embed queries correctly
        after a server restart.
        """
        _DIM_TO_MODEL: Dict[int, str] = {
            384:  "all-MiniLM-L6-v2",
            768:  "all-mpnet-base-v2",
            1024: "e5-large-v2",
            1536: "text-embedding-3-small",
            3072: "text-embedding-3-large",
        }
        for dim in self.storage_manager.faiss.active_dims():
            if dim in self._embedders:
                continue
            model_name = _DIM_TO_MODEL.get(dim)
            if not model_name:
                logger.warning(
                    "_warmup_registry: no model known for dim=%d — skipping", dim
                )
                continue
            try:
                ep = EmbeddingRegistry.get(model_name)
                self._embedders[dim] = ep
                logger.info(
                    "_warmup_registry: registered dim=%d -> %s", dim, model_name
                )
            except Exception as exc:
                logger.warning(
                    "_warmup_registry: failed for dim=%d (%s): %s",
                    dim, model_name, exc,
                )

    def delete_source(self, source_id: str) -> bool:
        """Remove a source and all its chunks from every store."""
        try:
            self.storage_manager.delete_source(source_id)
            all_chunks = self.storage_manager.get_all_chunks_for_bm25()
            self.hybrid_retriever.build_sparse_index(all_chunks)
            return True
        except Exception as exc:
            logger.error("delete_source(%s) failed: %s", source_id, exc)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Return a snapshot of current state."""
        return {
            "session_id":   self.session_id,
            "mode":         self.mode,
            "embed_dim":    self._embed_dim,
            "embed_model":  self._embed_model,
            "active_dims":  list(self._embedders.keys()),
            "storage":      self.storage_manager.get_stats(),
        }
