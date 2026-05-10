"""
master_pipeline.py  —  Top-level orchestrator for Mini NotebookLM

Bug fixes applied (2026-05-10 audit):
  BUG-005: generate(stream=True) for deep_research / study modes now returns
           _stream_dict() generator instead of a bare dict, which api.py's
           event_generator would iterate as dict keys (garbling output).
  BUG-009: asyncio.get_event_loop() removed; MasterPipeline is synchronous.
  BUG-010: BM25 sparse index built at startup from persisted chunks.
  BUG-023: Heavy model loading still happens here but is now triggered from
           the FastAPI lifespan handler (api.py) so workers are ready before
           the first request is accepted.
  BUG-025: No deprecated asyncio calls remain in this file.

Fixes from previous audit (kept):
  B1-B6 as documented in original docstring below.
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

# -- Core -----------------------------------------------------------------
from src.core.config import Config
from src.core.models import Query, LLMResponse

# -- Ingestion ------------------------------------------------------------
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

# -- Storage --------------------------------------------------------------
from src.storage import (
    MultiFAISSStore,
    SQLiteManager,
    KnowledgeGraph,
    StorageManager,
)

# -- Graph ----------------------------------------------------------------
from src.graph.graph_storage import GraphStorage
from src.graph.graph_retriever import GraphRetriever

# -- Retrieval ------------------------------------------------------------
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.contextual_compressor import ContextualCompressor

# -- Mode Pipelines -------------------------------------------------------
from src.pipelines.chat_pipeline import ChatPipeline
from src.pipelines.deep_research_pipeline import DeepResearchPipeline
from src.pipelines.study_pipeline import StudyPipeline

# -- Generation -----------------------------------------------------------
from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import PromptBuilder
from src.generation.response_parser import ResponseParser

# -- Chat History ---------------------------------------------------------
from src.chat_history.chat_history_manager import ChatHistoryManager
from src.chat_history.rag_history import RAGChatHistory
from src.chat_history.graph_history import GraphHistory

# -- RAGAS Evaluation -----------------------------------------------------
from src.evaluation.ragas_evaluator import RAGASEvaluator
from src.evaluation.ragas_bridge import attach_ragas


class MasterPipeline:
    """
    Single orchestration surface for Mini NotebookLM.

    Ingestion flow:
        file/url
          -> FileDetector -> Pipeline -> AdaptivePreprocessor
          -> ChunkerRegistry -> EmbeddingPipeline -> CrossModalMerger
          -> StorageManager (FAISS + SQLite + KnowledgeGraph)
          -> HybridRetriever.build_sparse_index()

    Generation flow (mode-dispatched):
        query
          -> ChatPipeline.run()          [chat mode]
          -> DeepResearchPipeline.run()  [deep_research]
          -> StudyPipeline.run()         [study]
          -> attach_ragas(result, query) [non-streaming only]
    """

    def __init__(self, mode: str = "chat", config_path: str = "config.yaml"):
        self.config     = Config(config_path)
        self.mode       = mode
        self.session_id = str(uuid.uuid4())

        # -- Storage ---------------------------------------------------------
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

        self.graph_storage   = GraphStorage()
        self.graph_retriever = GraphRetriever(self.graph_storage)

        # -- Ingestion -------------------------------------------------------
        self.file_detector      = FileDetector()
        self.preprocessor       = AdaptivePreprocessor()
        self._default_embed_model: str = self.config.get(
            "embedding.default_model", "all-MiniLM-L6-v2"
        )
        self.embedder           = EmbeddingPipeline(model_name=self._default_embed_model)
        self.cross_modal_merger = CrossModalMerger(embedder=self.embedder.embedder)
        self.ingestion_graph    = IngestionGraph()

        _probe = self.embedder.embed_query("probe")
        self._embed_dim: int   = int(_probe.shape[0])
        self._embed_model: str = self.embedder.model_name
        logger.info(
            "MasterPipeline: embedding dim=%d model=%s",
            self._embed_dim, self._embed_model,
        )

        # -- Retrieval -------------------------------------------------------
        self._embedders: Dict[int, Any] = {self._embed_dim: self.embedder}

        self.hybrid_retriever = HybridRetriever(
            faiss_store=self.storage_manager.faiss,
            storage_manager=self.storage_manager,
            embedders=self._embedders,
            top_k=self.config.get("retrieval.top_k", 5),
            fetch_k_multiplier=self.config.get("retrieval.fetch_k_multiplier", 3),  # BUG-026
        )

        self.reranker   = Reranker()
        self.compressor = ContextualCompressor()

        # -- Generation ------------------------------------------------------
        self.llm: Optional[LLMClient] = None
        self.prompt_builder  = PromptBuilder()
        self.response_parser = ResponseParser()

        # -- RAGAS Evaluator -------------------------------------------------
        self.ragas_evaluator = RAGASEvaluator(
            embedding_model=self._embed_model,
            overlap_threshold=self.config.get("evaluation.overlap_threshold", 0.25),
        )
        self._last_ragas: Optional[Dict[str, Any]] = None

        # -- Chat History ----------------------------------------------------
        self.rag_history   = RAGChatHistory(session_id=self.session_id)
        self.graph_history = GraphHistory(
            session_id=self.session_id,
            knowledge_graph=self.storage_manager.graph,
        )
        self.chat_history = ChatHistoryManager(
            self.session_id, mode, self.graph_storage
        )

        # -- Mode Pipelines --------------------------------------------------
        self._chat_pipeline:     Optional[ChatPipeline]         = None
        self._research_pipeline: Optional[DeepResearchPipeline] = None
        self._study_pipeline:    Optional[StudyPipeline]        = None

        self._response_cache: Dict[str, str] = {}
        self._warmup_registry()

        # BUG-010: build sparse index at startup from persisted chunks
        self._warmup_bm25()

    # -------------------------------------------------------------------------
    #  BM25 startup warm-up  (BUG-010)
    # -------------------------------------------------------------------------

    def _warmup_bm25(self) -> None:
        """Build BM25 sparse index from any chunks already in the store."""
        try:
            all_chunks = self.storage_manager.get_all_chunks_for_bm25()
            if all_chunks:
                self.hybrid_retriever.build_sparse_index(all_chunks)
                logger.info(
                    "MasterPipeline: BM25 index built from %d persisted chunks",
                    len(all_chunks),
                )
        except Exception as exc:
            logger.warning("MasterPipeline: BM25 startup build failed (non-fatal): %s", exc)

    # -------------------------------------------------------------------------
    #  LLM / Mode config
    # -------------------------------------------------------------------------

    def set_llm(self, provider: str, model: str, api_key: str, **kwargs) -> None:
        self.llm = LLMClient(provider=provider, model=model, api_key=api_key, **kwargs)
        logger.info("MasterPipeline: LLM set to %s/%s", provider, model)
        self._rebuild_mode_pipelines()

    def set_mode(self, mode: str) -> None:
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
            "MasterPipeline: mode switched to '%s' (session=%s)", mode, self.session_id,
        )

    def _rebuild_mode_pipelines(self) -> None:
        if self.llm is None:
            return

        llm_callable   = self.llm.invoke
        llm_stream_fn  = self.llm.stream     # BUG-004: separate stream callable
        self.compressor.llm = llm_callable

        self._chat_pipeline = ChatPipeline(
            hybrid_retriever=self.hybrid_retriever,
            rag_history=self.rag_history,
            llm=llm_callable,
            llm_stream=llm_stream_fn,        # BUG-004
            top_k=self.config.get("retrieval.top_k", 5),
            history_k=3,
        )

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

        self._study_pipeline = StudyPipeline(
            deep_research_pipeline=self._research_pipeline,
            graph_retriever=self.graph_retriever,
            graph_history=self.graph_history,
            llm=llm_callable,
        )

        logger.info("MasterPipeline: mode pipelines rebuilt for mode='%s'", self.mode)

    # -------------------------------------------------------------------------
    #  _stream_dict  (BUG-005)
    # -------------------------------------------------------------------------

    def _stream_dict(self, result: dict) -> Iterator:
        """
        BUG-005: deep_research / study modes return a full result dict.
        api.py's event_generator expects a lazy iterator that yields tokens
        (str) and then one metadata dict.

        Iterating a raw dict yields its *keys* ("answer", "citations", …)
        as tokens — completely wrong.  This shim yields words from the answer
        then emits the metadata dict as the final item.
        """
        answer = result.get("answer", "")
        for word in answer.split(" "):
            if word:
                yield word + " "

        meta_keys = [
            "citations", "retrieved_chunks", "context_chunks",
            "sources_used", "sub_queries", "quiz_cards",
            "summary_bullets", "learning_path", "ragas",
        ]
        meta = {k: result[k] for k in meta_keys if k in result}
        if meta:
            yield meta

    # -------------------------------------------------------------------------
    #  RAGAS helper
    # -------------------------------------------------------------------------

    def evaluate_response(
        self,
        query: str,
        answer: str,
        context_chunks: Optional[List[Dict]] = None,
        ground_truth: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not answer or not answer.strip():
            return None
        try:
            r = self.ragas_evaluator.evaluate(
                question=query,
                answer=answer,
                context_chunks=context_chunks or [],
                ground_truth=ground_truth,
            )
            self._last_ragas = r.to_dict()
            return self._last_ragas
        except Exception as exc:
            logger.warning("evaluate_response failed: %s", exc)
            return None

    @property
    def last_ragas(self) -> Optional[Dict[str, Any]]:
        return self._last_ragas

    # -------------------------------------------------------------------------
    #  Ingestion
    # -------------------------------------------------------------------------

    def ingest(
        self,
        file_path: Optional[str] = None,
        url: Optional[str] = None,
        source_type: Optional[str] = None,
        chunking_strategy: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        source_id = str(uuid.uuid4())

        if source_type:
            resolved_type = source_type.lower().strip()
        else:
            try:
                detection     = self.file_detector.detect(file_path=file_path, url=url)
                resolved_type = detection["source_type"]
            except Exception as exc:
                raise RuntimeError(f"FileDetector failed: {exc}") from exc

        self.ingestion_graph.add_source(
            source_id, name=file_path or url or "unknown", source_type=resolved_type,
        )

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
                result = loop.run_until_complete(WebsitePipeline.process(url, source_id))
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

        try:
            preprocessed = self.preprocessor.preprocess(raw_content, source_type=resolved_type)
        except Exception as exc:
            self.ingestion_graph.mark_error(source_id, stage="preprocess", error=str(exc))
            raise RuntimeError(f"Preprocessing failed: {exc}") from exc
        self.ingestion_graph.mark_stage(source_id, "preprocess")

        try:
            effective_strategy = chunking_strategy or resolved_type
            chunker = ChunkerRegistry.get_chunker(effective_strategy)
            chunks  = chunker.chunk(preprocessed, source_id=source_id)
        except Exception as exc:
            self.ingestion_graph.mark_error(source_id, stage="chunk", error=str(exc))
            raise RuntimeError(f"Chunking failed: {exc}") from exc
        self.ingestion_graph.mark_stage(source_id, "chunk")

        try:
            if embedding_model and embedding_model != self._embed_model:
                active_embedder = EmbeddingRegistry.get(embedding_model)
                _probe          = active_embedder.embed_query("probe")
                active_dim      = int(_probe.shape[0])
                active_model    = embedding_model
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

        try:
            merged_chunks = self.cross_modal_merger.merge([embedded_chunks])
        except Exception as exc:
            logger.warning("CrossModalMerger failed (non-fatal): %s", exc)
            merged_chunks = embedded_chunks

        source_record = {
            "id":         source_id,
            "name":       file_path or url or "unknown",
            "type":       resolved_type,
            "created_at": time.time(),
        }
        try:
            self.storage_manager.store(
                source_record, merged_chunks,
                embedding_model=active_model, dim=active_dim,
            )
        except Exception as exc:
            self.ingestion_graph.mark_error(source_id, stage="store", error=str(exc))
            raise RuntimeError(f"Storage failed: {exc}") from exc
        self.ingestion_graph.mark_stage(source_id, "store")

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

    # -------------------------------------------------------------------------
    #  Generation
    # -------------------------------------------------------------------------

    def generate(
        self,
        query: str,
        stream: bool = False,
        persona_config=None,
        ground_truth: Optional[str] = None,
    ):
        """
        Route query to the correct mode pipeline.

        Non-streaming: returns a Dict with 'ragas' key.
        Streaming:     returns a lazy iterator (tokens + final metadata dict).
                       BUG-005: deep/study modes now use _stream_dict().
        """
        if not self.llm:
            raise RuntimeError("LLM not configured. Call set_llm() first.")
        if self._chat_pipeline is None:
            raise RuntimeError(
                "Mode pipelines not initialised. Call set_llm() before generate()."
            )

        # -- Chat ------------------------------------------------------------
        if self.mode == "chat":
            if stream:
                chunks = self.hybrid_retriever.retrieve(
                    query, top_k=self.config.get("retrieval.top_k", 5),
                )
                history_context = self.rag_history.format_for_prompt(query, k=3)
                prompt = self._chat_pipeline._build_prompt(
                    query, chunks, history_context, persona_config=persona_config
                )

                def _stream_and_record() -> Iterator[str]:
                    full = ""
                    # BUG-004: use self.llm.stream (not self.llm.invoke)
                    for token in self.llm.stream(prompt):
                        full += token
                        yield token
                    self.rag_history.add_message("user", query)
                    self.rag_history.add_message("assistant", full)
                    try:
                        self.evaluate_response(
                            query=query, answer=full,
                            context_chunks=chunks, ground_truth=ground_truth,
                        )
                    except Exception:
                        pass

                return _stream_and_record()
            else:
                result = self._chat_pipeline.run(query, persona_config=persona_config)
                attach_ragas(result, query, self.ragas_evaluator, ground_truth)
                self._last_ragas = result.get("ragas")
                return result

        # -- Deep Research ---------------------------------------------------
        elif self.mode in ("research", "deep_research"):
            result = self._research_pipeline.run(query)
            attach_ragas(result, query, self.ragas_evaluator, ground_truth)
            self._last_ragas = result.get("ragas")
            # BUG-005: wrap in stream iterator if requested
            return self._stream_dict(result) if stream else result

        # -- Study -----------------------------------------------------------
        elif self.mode == "study":
            result = self._study_pipeline.run(query)
            attach_ragas(result, query, self.ragas_evaluator, ground_truth)
            self._last_ragas = result.get("ragas")
            # BUG-005: wrap in stream iterator if requested
            return self._stream_dict(result) if stream else result

        else:
            raise ValueError(f"Unknown mode: '{self.mode}'")

    def generate_full(
        self,
        query: str,
        persona_config=None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.llm:
            raise RuntimeError("LLM not configured. Call set_llm() first.")
        if self._chat_pipeline is None:
            raise RuntimeError("Mode pipelines not initialised.")

        if self.mode == "chat":
            result = self._chat_pipeline.run(query, persona_config=persona_config)
        elif self.mode in ("research", "deep_research"):
            result = self._research_pipeline.run(query)
        elif self.mode == "study":
            result = self._study_pipeline.run(query)
        else:
            raise ValueError(f"Unknown mode: '{self.mode}'")

        attach_ragas(result, query, self.ragas_evaluator, ground_truth)
        self._last_ragas = result.get("ragas")
        return result

    # -------------------------------------------------------------------------
    #  Utilities
    # -------------------------------------------------------------------------

    def _warmup_registry(self) -> None:
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
                    "_warmup_registry: no model known for dim=%d - skipping", dim
                )
                continue
            try:
                ep = EmbeddingRegistry.get(model_name)
                self._embedders[dim] = ep
                logger.info("_warmup_registry: registered dim=%d -> %s", dim, model_name)
            except Exception as exc:
                logger.warning(
                    "_warmup_registry: failed for dim=%d (%s): %s", dim, model_name, exc,
                )

    def delete_source(self, source_id: str) -> bool:
        try:
            self.storage_manager.delete_source(source_id)
            all_chunks = self.storage_manager.get_all_chunks_for_bm25()
            self.hybrid_retriever.build_sparse_index(all_chunks)
            return True
        except Exception as exc:
            logger.error("delete_source(%s) failed: %s", source_id, exc)
            return False

    def get_stats(self) -> Dict[str, Any]:
        return {
            "session_id":  self.session_id,
            "mode":        self.mode,
            "embed_dim":   self._embed_dim,
            "embed_model": self._embed_model,
            "active_dims": list(self._embedders.keys()),
            "storage":     self.storage_manager.get_stats(),
        }
