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
        self.embedder       = EmbeddingPipeline(
            model_name=self.config.get("embedding.default_model", "all-MiniLM-L6-v2")
        )
        self.cross_modal_merger = CrossModalMerger(embedder=self.embedder.embedder)
        self.ingestion_graph    = IngestionGraph()

        # Resolve embedding dimension once from a dummy query so MultiFAISSStore
        # always gets the right dim even if config.yaml is stale.
        _probe = self.embedder.embed_query("probe")
        self._embed_dim: int = int(_probe.shape[0])
        self._embed_model: str = self.embedder.model_name
        logger.info("MasterPipeline: embedding dim=%d model=%s", self._embed_dim, self._embed_model)

        # ── Retrieval ─────────────────────────────────────────────────────────
        # HybridRetriever takes StorageManager — Phase 3 will rewrite internals
        # to use EnsembleRetriever; for now pass storage_manager so it can call
        # get_chunks_as_documents() after FAISS results come back.
        self.hybrid_retriever = HybridRetriever(self.storage_manager.faiss)

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
                # Tombstoned — raises NotImplementedError by design.
                from src.ingestion.pipelines.video_audio_pipeline import VideoAudioPipeline
                result = VideoAudioPipeline.process(file_path, source_id)
            elif resolved_type == "website":
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(WebsitePipeline.process(url, source_id))
            elif resolved_type == "youtube":
                result = YouTubePipeline.process(url, source_id)
            elif resolved_type == "csv":
                result = CSVPipeline.process(file_path, source_id)
            else:
                raise ValueError(
                    f"Unsupported source_type: '{resolved_type}'. "
                    "Valid: pdf, image, website, youtube, csv."
                )
        except Exception as exc:
            self.ingestion_graph.mark_error(source_id, "extracted", str(exc))
            raise RuntimeError(
                f"Ingestion failed at extraction stage (type='{resolved_type}'): {exc}"
            ) from exc

        self.ingestion_graph.mark_stage(source_id, "extracted",
                                        meta={"modality": result.get("modality")})

        # ── Step 3: preprocess ────────────────────────────────────────────────
        try:
            preprocessed = self.preprocessor.process(
                result["content"],
                resolved_type,
                result.get("metadata", {}),
            )
        except Exception as exc:
            self.ingestion_graph.mark_error(source_id, "preprocessed", str(exc))
            raise RuntimeError(f"Ingestion failed at preprocessing stage: {exc}") from exc

        self.ingestion_graph.mark_stage(source_id, "preprocessed")

        # ── Step 4: chunk ─────────────────────────────────────────────────────
        strategy = preprocessed.get("recommendation", {}).get("strategy", "recursive")
        try:
            chunker = ChunkerRegistry.get_chunker(strategy)
            chunks  = chunker.chunk(
                preprocessed["cleaned_content"],
                {
                    "source_id": source_id,
                    "modality":  result["modality"],
                    **result.get("metadata", {}),
                },
            )
        except Exception as exc:
            self.ingestion_graph.mark_error(source_id, "chunked", str(exc))
            raise RuntimeError(
                f"Ingestion failed at chunking stage (strategy='{strategy}'): {exc}"
            ) from exc

        self.ingestion_graph.mark_stage(source_id, "chunked", meta={"num_chunks": len(chunks)})

        # ── Step 5: embed ─────────────────────────────────────────────────────
        try:
            chunks = self.embedder.embed_chunks(chunks)
        except Exception as exc:
            self.ingestion_graph.mark_error(source_id, "embedded", str(exc))
            raise RuntimeError(f"Ingestion failed at embedding stage: {exc}") from exc

        self.ingestion_graph.mark_stage(source_id, "embedded")

        # ── Step 6: cross-modal merge ─────────────────────────────────────────
        chunks = self.cross_modal_merger.merge(chunks)

        # ── Step 7: store ─────────────────────────────────────────────────────
        source_meta = result.get("metadata", {})
        source = {
            "id":          source_id,
            "title":       source_meta.get("title", file_path or url or "Untitled"),
            "source_type": resolved_type,
            "file_path":   file_path,
            "url":         url,
            "metadata":    source_meta,
            "status":      "ready",
        }
        try:
            self.storage_manager.store(
                source=source,
                chunks=chunks,
                embedding_model=self._embed_model,
                dim=self._embed_dim,
            )
        except Exception as exc:
            self.ingestion_graph.mark_error(source_id, "indexed", str(exc))
            raise RuntimeError(f"Ingestion failed at storage stage: {exc}") from exc

        # ── Step 8: rebuild BM25 sparse index ────────────────────────────────
        # get_all_chunks_for_bm25() returns [{id, content}, ...] from SQLite
        # — no dependency on faiss.metadata which doesn't exist.
        try:
            all_chunks = self.storage_manager.get_all_chunks_for_bm25()
            self.hybrid_retriever.build_sparse_index(all_chunks)
        except Exception as exc:
            logger.warning("BM25 rebuild failed (non-fatal): %s", exc)

        self.ingestion_graph.mark_stage(source_id, "indexed", meta={"num_chunks": len(chunks)})

        logger.info(
            "MasterPipeline.ingest: source_id=%s type=%s chunks=%d",
            source_id, resolved_type, len(chunks),
        )
        return {
            "source_id":  source_id,
            "num_chunks": len(chunks),
            "profile":    None,
        }

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, query: str, stream: bool = False):
        """
        Generate a response for a user query.

        Retrieves relevant chunks, hydrates them into LangChain Documents,
        builds the prompt, calls the LLM, parses the response.

        Phase 5 will replace this method body with a LangGraph graph invocation.
        """
        if not self.llm:
            raise ValueError("LLM not configured — call set_llm() first.")

        # Cache check
        cache_key = f"{self.mode}:{query}"
        if cache_key in self._response_cache and not stream:
            return self._response_cache[cache_key]

        # Step 1: embed query
        query_embedding = self.embedder.embed_query(query)

        # Step 2: retrieve — dim-keyed dict for MultiFAISSStore
        raw_results = self.storage_manager.faiss.search(
            query_vectors={self._embed_dim: query_embedding},
            k=self.config.get("retrieval.top_k", 10),
        )
        chunk_ids = [cid for cid, _score in raw_results.get(self._embed_dim, [])]

        # Step 3: hydrate to LangChain Documents
        documents = self.storage_manager.get_chunks_as_documents(chunk_ids)

        # Step 4: BM25 fusion (hybrid)
        retrieved_dicts = [
            {"content": doc.page_content, "id": doc.metadata.get("chunk_id", ""), **doc.metadata}
            for doc in documents
        ]

        # Step 5: chat history context
        history_context = self.chat_history.get_history_context(query)

        # Step 6: build prompt
        context = self.prompt_builder.format_context(retrieved_dicts)

        if self.mode == "chat":
            prompt = self.prompt_builder.build_chat_prompt(query, context, history_context)
        elif self.mode == "deep_research":
            prompt = self.prompt_builder.build_deep_research_prompt(query, context, history_context)
        elif self.mode == "study":
            prompt = self.prompt_builder.build_study_mode_prompt(query, context, [], history_context)
        else:
            raise ValueError(f"Unknown mode: '{self.mode}'")

        # Step 7: generate
        if stream:
            return self._stream_response(prompt, query, retrieved_dicts)

        response_text = self.llm.invoke(prompt)
        parsed = self.response_parser.parse(response_text)

        # Step 8: persist to chat history
        source_ids = list({d.get("source_id", "") for d in retrieved_dicts})
        self.chat_history.add_message("user",      query,                   sources_used=source_ids)
        self.chat_history.add_message("assistant", parsed["content"],       sources_used=source_ids)

        self._response_cache[cache_key] = parsed["content"]
        return parsed["content"]

    def _stream_response(
        self,
        prompt: str,
        query: str,
        retrieved: List[Dict[str, Any]],
    ) -> Iterator[str]:
        """Yield response tokens one at a time."""
        full: List[str] = []
        for token in self.llm.stream(prompt):
            full.append(token)
            yield token
        complete = "".join(full)
        self.chat_history.add_message("user",      query)
        self.chat_history.add_message("assistant", complete)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def clear_cache(self) -> None:
        """Clear the in-memory response cache."""
        self._response_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return pipeline statistics from all stores."""
        return {
            "mode":       self.mode,
            "session_id": self.session_id,
            "storage":    self.storage_manager.get_stats(),
            "ingestion":  self.ingestion_graph.get_all_statuses(),
            "cache_size": len(self._response_cache),
        }
