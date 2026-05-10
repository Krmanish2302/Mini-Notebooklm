"""
master_pipeline.py  —  Top-level orchestrator for Mini NotebookLM

Phase-1 fixes
-------------
- MultiFAISSStore (was FAISSStore — wrong class name)
- StorageManager as orchestrator (was SourceManager — wrong class)
- storage.store() called with correct 4-arg signature
- BM25 rebuild uses storage_manager.get_all_chunks_for_bm25()
- nest_asyncio applied at module top (fixes asyncio.run inside Streamlit)
- set_mode() preserves session_id so history is NOT wiped on mode switch
- ingest() has per-stage try/except + IngestionGraph stage tracking
- generate() hydrates chunk_ids into LangChain Documents via StorageManager
- EmbeddingPipeline dim inferred from embed_query() result shape
- VideoAudioPipeline import removed (tombstoned — raises NotImplementedError)
- WebSearchAgent import removed (file not present yet — Phase 5)

Phase-2 fix
-----------
- HybridRetriever constructor now passes storage_manager + dim
"""
from __future__ import annotations

import asyncio
import logging
import uuid
import time
from typing import Any, Dict, Iterator, List, Optional

import nest_asyncio
nest_asyncio.apply()  # must be applied before any event-loop usage in Streamlit

logger = logging.getLogger(__name__)

# ── Core ─────────────────────────────────────────────────────────────────────
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
from src.ingestion.merging.cross_modal_merger import CrossModalMerger
from src.ingestion.ingestion_graph import IngestionGraph

# ── Storage (corrected class names + orchestrator) ────────────────────────────
from src.storage import (
    MultiFAISSStore,
    SQLiteManager,
    KnowledgeGraph,
    StorageManager,
)

# ── Graph (legacy — kept for GraphRetriever / StudyMode) ─────────────────────
from src.graph.graph_storage import GraphStorage
from src.graph.graph_retriever import GraphRetriever

# ── Retrieval ─────────────────────────────────────────────────────────────────
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker

# ── Generation ────────────────────────────────────────────────────────────────
from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import PromptBuilder
from src.generation.response_parser import ResponseParser

# ── Chat History ──────────────────────────────────────────────────────────────
from src.chat_history.chat_history_manager import ChatHistoryManager


class MasterPipeline:
    """
    Single orchestration surface for Mini NotebookLM.

    Ingestion flow:
        file/url
          → FileDetector (auto-detect when source_type is None)
          → Pipeline (PDF/Website/YouTube/CSV/Image)
          → AdaptivePreprocessor
          → ChunkerRegistry.get_chunker(strategy)
          → EmbeddingPipeline.embed_chunks()
          → CrossModalMerger.merge()
          → StorageManager.store()    ← FAISS + SQLite + KnowledgeGraph
          → HybridRetriever.build_sparse_index()  ← BM25 rebuild

    Generation flow (phases 2-5 will upgrade this to LangGraph):
        query
          → EmbeddingPipeline.embed_query()
          → HybridRetriever.retrieve()
          → StorageManager.get_chunks_as_documents()  ← LC Documents
          → ChatHistoryManager.get_history_context()
          → PromptBuilder.build_*_prompt()
          → LLMClient.invoke() / .stream()
          → ResponseParser.parse()

    FileDetector policy:
        - source_type explicitly passed → FileDetector SKIPPED (trust caller).
        - source_type is None           → FileDetector used for auto-detection.
    """

    def __init__(self, mode: str = "chat", config_path: str = "config.yaml"):
        self.config = Config(config_path)
        self.mode = mode
        self.session_id = str(uuid.uuid4())

        # ── Storage ──────────────────────────────────────────────────────────
        _faiss   = MultiFAISSStore(base_dir=self.config.get("storage.faiss_dir", "./data/vector_store"))
        _sqlite  = SQLiteManager(db_path=self.config.get("storage.db_path", "./data/metadata.db"))
        _graph   = KnowledgeGraph(
            edge_threshold=self.config.get("storage.graph_edge_threshold", 0.75)
        )
        self.storage_manager = StorageManager(_faiss, _sqlite, _graph)

        # Legacy GraphStorage — kept for GraphRetriever / StudyModeRetriever
        # until those are migrated to use KnowledgeGraph (Phase 3).
        self.graph_storage = GraphStorage()
        self.graph_retriever = GraphRetriever(self.graph_storage)

        # ── Ingestion ─────────────────────────────────────────────────────────
        self.file_detector  = FileDetector()
        self.preprocessor   = AdaptivePreprocessor()
        self._default_embed_model: str = self.config.get("embedding.default_model", "all-MiniLM-L6-v2")
        self.embedder       = EmbeddingPipeline(model_name=self._default_embed_model)
        self.cross_modal_merger = CrossModalMerger(embedder=self.embedder.embedder)
        self.ingestion_graph    = IngestionGraph()

        # Resolve embedding dimension once from a dummy query so MultiFAISSStore
        # always gets the right dim even if config.yaml is stale.
        _probe = self.embedder.embed_query("probe")
        self._embed_dim: int = int(_probe.shape[0])
        self._embed_model: str = self.embedder.model_name
        logger.info("MasterPipeline: embedding dim=%d model=%s", self._embed_dim, self._embed_model)

        # ── Retrieval ─────────────────────────────────────────────────────────
        # Phase-2 fix: pass storage_manager + dim so HybridRetriever can call
        # faiss_store.search({dim: qvec}) and hydrate results via StorageManager.
        self.hybrid_retriever = HybridRetriever(
            self.storage_manager.faiss,
            storage_manager=self.storage_manager,
            dim=self._embed_dim,
        )

        # ── Generation ────────────────────────────────────────────────────────
        self.llm: Optional[LLMClient] = None
        self.prompt_builder  = PromptBuilder()
        self.response_parser = ResponseParser()

        # ── Chat History ──────────────────────────────────────────────────────
        self.chat_history = ChatHistoryManager(
            self.session_id, mode, self.graph_storage
        )

        # Response cache for repeated queries (same mode + query text)
        self._response_cache: Dict[str, str] = {}

    # ── Mode / LLM config ────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        """
        Switch conversation mode.

        Preserves session_id so chat history is NOT wiped on mode switch.
        Only the backend selector inside ChatHistoryManager changes.
        """
        self.mode = mode
        self.chat_history = ChatHistoryManager(
            self.session_id,  # <-- same session, NOT a new uuid
            mode,
            self.graph_storage,
        )
        logger.info("MasterPipeline: mode switched to '%s' (session=%s)", mode, self.session_id)

    def set_llm(self, provider: str, model: str, api_key: str, **kwargs) -> None:
        """Configure (or reconfigure) the LLM client."""
        self.llm = LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            **kwargs,
        )
        logger.info("MasterPipeline: LLM set to %s/%s", provider, model)

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

        Parameters
        ----------
        file_path   : local path to a file (PDF, CSV, image …)
        url         : remote URL (website or YouTube link)
        source_type : explicit type string — when given the FileDetector is
                      skipped.  Valid values:
                      'pdf' | 'image' | 'website' | 'youtube' | 'csv'

        Returns
        -------
        Dict with:
            source_id : str   — unique ID assigned to this source
            num_chunks: int   — number of chunks stored
            profile   : dict  — DocumentProfiler output (PDF only, else None)
        """
        source_id = str(uuid.uuid4())

        # ── Step 1: resolve source_type ───────────────────────────────────────
        if source_type:
            resolved_type = source_type.lower().strip()
        else:
            try:
                detection = self.file_detector.detect(file_path=file_path, url=url)
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
                # Tombstoned: VideoAudioPipeline raises NotImplementedError.
                # Lazy import so the dead module never loads at startup.
                from src.ingestion.pipelines.video_audio_pipeline import VideoAudioPipeline
                result = VideoAudioPipeline.process(file_path, source_id)
            elif resolved_type == "youtube":
                result = YouTubePipeline.process(url, source_id)
            elif resolved_type == "website":
                loop = asyncio.get_event_loop()
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
            preprocessed = self.preprocessor.preprocess(raw_content, source_type=resolved_type)
        except Exception as exc:
            self.ingestion_graph.mark_error(source_id, stage="preprocess", error=str(exc))
            raise RuntimeError(f"Preprocessing failed: {exc}") from exc
        self.ingestion_graph.mark_stage(source_id, "preprocess")

        # ── Step 4: chunk ─────────────────────────────────────────────────────
        try:
            # Use user-selected strategy if provided, else default for source type
            effective_strategy = chunking_strategy or resolved_type
            chunker = ChunkerRegistry.get_chunker(effective_strategy)
            chunks  = chunker.chunk(preprocessed, source_id=source_id)
        except Exception as exc:
            self.ingestion_graph.mark_error(source_id, stage="chunk", error=str(exc))
            raise RuntimeError(f"Chunking failed: {exc}") from exc
        self.ingestion_graph.mark_stage(source_id, "chunk")
        
        # ── Step 5: embed ─────────────────────────────────────────────────────
        try:
            # Use user-selected embedding model if provided, else pipeline default
            if embedding_model and embedding_model != self._embed_model:
                from src.ingestion.embedding.embedding_registry import EmbeddingRegistry
                active_embedder = EmbeddingRegistry.get(embedding_model)
                _probe = active_embedder.embed_query("probe")
                active_dim   = int(_probe.shape[0])
                active_model = embedding_model
                logger.info("ingest: using user-selected embedder %s (dim=%d)", active_model, active_dim)
            else:
                active_embedder = self.embedder
                active_dim      = self._embed_dim
                active_model    = self._embed_model

            embedded_chunks = active_embedder.embed_chunks(chunks)
        except Exception as exc:
            self.ingestion_graph.mark_error(source_id, stage="embed", error=str(exc))
            raise RuntimeError(f"Embedding failed: {exc}") from exc
        self.ingestion_graph.mark_stage(source_id, "embed")

        # ── Step 6: merge cross-modal (no-op for single-source ingestion) ─────
        try:
            merged_chunks = self.cross_modal_merger.merge([embedded_chunks])
        except Exception as exc:
            # Non-fatal — fall back to un-merged chunks
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
    ):
        """
        Generate a response for *query* against the stored knowledge base.

        Parameters
        ----------
        query  : the user's question
        stream : if True yield token-by-token strings, else return full string

        Returns
        -------
        str  (stream=False)
        Iterator[str]  (stream=True)
        """
        if not self.llm:
            raise RuntimeError("LLM not configured.  Call set_llm() first.")

        # ── Embed query ───────────────────────────────────────────────────────
        query_embedding = self.embedder.embed_query(query)

        # ── Retrieve ──────────────────────────────────────────────────────────
        raw_results = self.storage_manager.faiss.search(
            query_vectors={self._embed_dim: query_embedding},
            k=self.config.get("retrieval.top_k", 10),
        )
        chunk_ids = [cid for cid, _score in raw_results.get(self._embed_dim, [])]

        # Hydrate chunk IDs → LangChain Documents
        documents = self.storage_manager.get_chunks_as_documents(chunk_ids)

        # ── Chat history ──────────────────────────────────────────────────────
        history_context = self.chat_history.get_history_context(
            query, max_messages=self.config.get("chat.history_window", 6)
        )

        # ── Build prompt ──────────────────────────────────────────────────────
        build_fn = {
            "chat":     self.prompt_builder.build_chat_prompt,
            "study":    self.prompt_builder.build_study_prompt,
            "research": self.prompt_builder.build_research_prompt,
        }.get(self.mode, self.prompt_builder.build_chat_prompt)

        prompt = build_fn(
            query=query,
            documents=documents,
            history=history_context,
        )

        # ── Call LLM ──────────────────────────────────────────────────────────
        if stream:
            def _stream_and_record() -> Iterator[str]:
                full = ""
                for token in self.llm.stream(prompt):
                    full += token
                    yield token
                self.chat_history.add_message("user",      query)
                self.chat_history.add_message("assistant", full)
            return _stream_and_record()
        else:
            response = self.llm.invoke(prompt)
            parsed   = self.response_parser.parse(response)
            self.chat_history.add_message("user",      query)
            self.chat_history.add_message("assistant", parsed)
            return parsed

    # ── Utilities ─────────────────────────────────────────────────────────────

    def delete_source(self, source_id: str) -> bool:
        """Remove a source and all its chunks from every store."""
        try:
            self.storage_manager.delete_source(source_id)
            # Rebuild BM25 after deletion
            all_chunks = self.storage_manager.get_all_chunks_for_bm25()
            self.hybrid_retriever.build_sparse_index(all_chunks)
            return True
        except Exception as exc:
            logger.error("delete_source(%s) failed: %s", source_id, exc)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Return a snapshot of what is currently stored."""
        return {
            "session_id": self.session_id,
            "mode":       self.mode,
            "embed_dim":  self._embed_dim,
            "embed_model": self._embed_model,
            "storage":    self.storage_manager.get_stats(),
        }
