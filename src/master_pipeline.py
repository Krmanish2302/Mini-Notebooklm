from typing import List, Dict, Any, Optional, Iterator
import uuid
import time

# Core
from src.core.config import Config
from src.core.models import Query, LLMResponse

# Ingestion
from src.ingestion.file_detector import FileDetector
from src.ingestion.pipelines.pdf_pipeline import PDFPipeline
from src.ingestion.pipelines.image_pipeline import ImagePipeline
from src.ingestion.pipelines.video_audio_pipeline import VideoAudioPipeline
from src.ingestion.pipelines.website_pipeline import WebsitePipeline
from src.ingestion.pipelines.youtube_pipeline import YouTubePipeline
from src.ingestion.pipelines.csv_pipeline import CSVPipeline
from src.ingestion.preprocessing.adaptive_preprocessor import AdaptivePreprocessor
from src.ingestion.preprocessing.document_profiler import DocumentProfiler
from src.ingestion.chunking.chunker_registry import ChunkerRegistry
from src.ingestion.embedding.embedding_pipeline import EmbeddingPipeline
from src.ingestion.merging.cross_modal_merger import CrossModalMerger

# Storage
from src.storage.faiss_store import FAISSStore
from src.storage.sqlite_manager import SQLiteManager
from src.storage.source_manager import SourceManager

# Graph
from src.graph.graph_storage import GraphStorage
from src.graph.graph_retriever import GraphRetriever

# Retrieval
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.contextual_compressor import ContextualCompressor
from src.retrieval.reranker import Reranker
from src.retrieval.advanced_retriever import AdvancedRetriever
from src.retrieval.study_mode import StudyModeRetriever

# Generation
from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import PromptBuilder
from src.generation.response_parser import ResponseParser

# Chat History
from src.chat_history.chat_history_manager import ChatHistoryManager

# Agents
from src.agents.web_search_agent import WebSearchAgent


class MasterPipeline:
    """
    Main orchestrator connecting all components.
    Provides unified interface for ingestion, retrieval, and generation.

    FileDetector usage policy:
    -  When source_type is explicitly provided (user clicked PDF / YouTube /
       Web / Text button in UI), FileDetector is SKIPPED — we trust the caller.
    -  When source_type is None (programmatic / bulk / CLI ingestion where the
       caller does not know the type), FileDetector is used as a fallback to
       auto-detect the correct pipeline.
    """

    def __init__(self, mode: str = "chat", config_path: str = "config.yaml"):
        self.config = Config(config_path)
        self.mode = mode
        self.session_id = str(uuid.uuid4())

        # Initialize storage
        self.faiss_store = FAISSStore(dimension=self.config.get("embedding.dimension", 384))
        self.sqlite = SQLiteManager()
        self.graph_storage = GraphStorage()

        # Initialize ingestion components
        self.file_detector = FileDetector()
        self.preprocessor = AdaptivePreprocessor()
        self.document_profiler = DocumentProfiler(chunk_overlap=50)
        self.embedder = EmbeddingPipeline(
            model_name=self.config.get("embedding.default_model", "all-MiniLM-L6-v2")
        )
        self.cross_modal_merger = CrossModalMerger(embedder=self.embedder.embedder)

        # Initialize retrieval components
        self.hybrid_retriever = HybridRetriever(self.faiss_store)
        self.compressor = ContextualCompressor()
        self.reranker = Reranker()
        self.advanced_retriever = AdvancedRetriever(
            self.hybrid_retriever, self.compressor, self.reranker
        )
        self.graph_retriever = GraphRetriever(self.graph_storage)
        self.study_retriever = StudyModeRetriever(
            self.advanced_retriever, self.graph_storage, self.graph_retriever
        )

        # Initialize generation
        self.llm: Optional[LLMClient] = None
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser()

        # Initialize chat history
        self.chat_history = ChatHistoryManager(
            self.session_id, mode, self.graph_storage
        )

        # Initialize source manager
        self.source_manager = SourceManager(
            self.faiss_store, self.sqlite, self.graph_storage
        )

        # Initialize web search
        self.web_search = WebSearchAgent()

        # Response cache for repeated queries
        self._response_cache: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Mode / LLM config
    # ------------------------------------------------------------------

    def set_mode(self, mode: str):
        """Switch conversation mode (preserves existing session_id)."""
        self.mode = mode
        # Re-use the same session_id so chat history is NOT wiped on mode switch.
        self.chat_history = ChatHistoryManager(
            self.session_id, mode, self.graph_storage
        )

    def set_llm(self, provider: str, model: str, api_key: str, **kwargs):
        """Configure LLM."""
        self.llm = LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            **kwargs
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        file_path: str = None,
        url: str = None,
        source_type: str = None,
    ) -> Dict[str, Any]:
        """
        Ingest a file or URL into the knowledge base.

        Parameters
        ----------
        file_path : str, optional
            Local path to a file (PDF, image, video, CSV, …).
        url : str, optional
            Remote URL (website or YouTube link).
        source_type : str, optional
            Explicit type: 'pdf' | 'image' | 'video' | 'audio' | 'website' |
            'youtube' | 'csv'.  When provided the FileDetector is SKIPPED.
            When None, FileDetector auto-detects the type.

        Returns
        -------
        Dict with keys:
            source_id : str   — unique ID for the ingested source
            profile   : dict  — DocumentProfiler output (PDF only, else None)
                               Use master_pipeline.document_profiler.get_ui_summary(profile)
                               to render in Streamlit.
        """
        # ── Step 1: Resolve source_type ──────────────────────────────────────
        if source_type:
            resolved_type = source_type.lower().strip()
        else:
            detection = self.file_detector.detect(file_path=file_path, url=url)
            resolved_type = detection["source_type"]

        source_id = str(uuid.uuid4())

        # ── Step 2: Extract content via the matching pipeline ────────────────
        try:
            if resolved_type == "pdf":
                result = PDFPipeline.process(file_path, source_id)
            elif resolved_type == "image":
                result = ImagePipeline.process(file_path, source_id)
            elif resolved_type in ("video", "audio"):
                result = VideoAudioPipeline.process(file_path, source_id)
            elif resolved_type == "website":
                import asyncio
                import nest_asyncio
                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(WebsitePipeline.process(url, source_id))
            elif resolved_type == "youtube":
                result = YouTubePipeline.process(url, source_id)
            elif resolved_type == "csv":
                result = CSVPipeline.process(file_path, source_id)
            else:
                raise ValueError(
                    f"Unsupported source type: '{resolved_type}'. "
                    "Valid values: pdf, image, video, audio, website, youtube, csv."
                )
        except Exception as exc:
            raise RuntimeError(
                f"Ingestion failed at pipeline stage for '{resolved_type}': {exc}"
            ) from exc

        # ── Step 2b: Document Profiling (PDF only) ───────────────────────────
        # Run before chunking so the profile can inform strategy selection.
        # For non-PDF sources the profile is None — callers should check.
        document_profile: Optional[Dict[str, Any]] = None
        if resolved_type == "pdf":
            try:
                raw_text = result.get("content", "")
                document_profile = self.document_profiler.profile(
                    text=raw_text,
                    source_type="pdf",
                    file_path=file_path,
                )
            except Exception:
                # Profiling failure is non-fatal — continue with defaults
                document_profile = None

        # ── Step 3: Preprocess ───────────────────────────────────────────────
        try:
            preprocessed = self.preprocessor.process(
                result["content"], resolved_type, result.get("metadata", {})
            )
        except Exception as exc:
            raise RuntimeError(
                f"Ingestion failed at preprocessing stage: {exc}"
            ) from exc

        # ── Step 4: Chunk ────────────────────────────────────────────────────
        # If DocumentProfiler ran and produced a recommendation, use it.
        # Otherwise fall back to AdaptivePreprocessor's recommendation.
        if document_profile and document_profile.get("recommendation"):
            strategy = document_profile["recommendation"]["strategy"]
        else:
            strategy = preprocessed["recommendation"]["strategy"]

        try:
            chunker = ChunkerRegistry.get_chunker(strategy)
            chunks = chunker.chunk(
                preprocessed["cleaned_content"],
                {
                    "source_id": source_id,
                    "modality": result["modality"],
                    **result.get("metadata", {}),
                    # embed the profiler recommendation in chunk metadata
                    "profiler_strategy": strategy,
                    "profiler_avg_tokens": (
                        document_profile["recommendation"]["avg_tokens"]
                        if document_profile else None
                    ),
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"Ingestion failed at chunking stage (strategy='{strategy}'): {exc}"
            ) from exc

        # ── Step 5: Embed ────────────────────────────────────────────────────
        try:
            chunks = self.embedder.embed_chunks(chunks)
        except Exception as exc:
            raise RuntimeError(
                f"Ingestion failed at embedding stage: {exc}"
            ) from exc

        # ── Step 6: Cross-modal merge ────────────────────────────────────────
        chunks = self.cross_modal_merger.merge(chunks)

        # ── Step 7: Store ────────────────────────────────────────────────────
        source_meta = result.get("metadata", {})
        if document_profile:
            source_meta["document_profile"] = document_profile

        source = {
            "id": source_id,
            "title": source_meta.get("title", "Untitled"),
            "source_type": resolved_type,
            "file_path": file_path,
            "url": url,
            "metadata": source_meta,
        }
        self.source_manager.add_source(source, chunks)

        # ── Step 8: Rebuild sparse BM25 index ────────────────────────────────
        self.hybrid_retriever.build_sparse_index(
            list(self.faiss_store.metadata.values())
        )

        return {
            "source_id": source_id,
            "profile":   document_profile,
        }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, query: str, stream: bool = False) -> str:
        """
        Generate response for user query.
        Uses mode-specific retrieval and prompting.
        """
        if not self.llm:
            raise ValueError("LLM not configured. Call set_llm() first.")

        start_time = time.time()

        # Check cache
        cache_key = f"{self.mode}:{query}"
        if cache_key in self._response_cache and not stream:
            return self._response_cache[cache_key]

        # Step 1: Embed query
        query_embedding = self.embedder.embed_query(query)

        # Step 2: Retrieve based on mode
        if self.mode == "chat":
            retrieved = self.hybrid_retriever.retrieve(query, query_embedding)
            learning_path = []
        elif self.mode == "deep_research":
            retrieved = self.advanced_retriever.retrieve(query, query_embedding)
            learning_path = []
        elif self.mode == "study":
            study_result = self.study_retriever.retrieve(query, query_embedding)
            retrieved = study_result["chunks"]
            learning_path = study_result["learning_path"]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Step 3: Get chat history context
        history_context = self.chat_history.get_history_context(query)

        # Step 4: Build prompt
        context = self.prompt_builder.format_context(retrieved)

        if self.mode == "chat":
            prompt = self.prompt_builder.build_chat_prompt(query, context, history_context)
        elif self.mode == "deep_research":
            prompt = self.prompt_builder.build_deep_research_prompt(query, context, history_context)
        elif self.mode == "study":
            prompt = self.prompt_builder.build_study_mode_prompt(
                query, context, learning_path, history_context
            )

        # Step 5: Generate
        if stream:
            return self._stream_response(prompt, query, retrieved)

        response_text = self.llm.invoke(prompt)

        # Parse response
        parsed = self.response_parser.parse(response_text)

        # Validate grounding
        self.response_parser.validate_grounding(parsed["content"], context)

        # Store in chat history
        self.chat_history.add_message(
            "user", query,
            sources_used=[c.get("source_id") for c in retrieved]
        )
        self.chat_history.add_message(
            "assistant", parsed["content"],
            sources_used=[c.get("source_id") for c in retrieved]
        )

        # Cache response
        self._response_cache[cache_key] = parsed["content"]

        return parsed["content"]

    def _stream_response(
        self, prompt: str, query: str, retrieved: List[Dict]
    ) -> Iterator[str]:
        """Stream response tokens."""
        full_response = []
        for token in self.llm.stream(prompt):
            full_response.append(token)
            yield token

        complete = "".join(full_response)
        self.chat_history.add_message("user", query)
        self.chat_history.add_message("assistant", complete)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear_cache(self):
        """Clear response cache."""
        self._response_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "mode": self.mode,
            "session_id": self.session_id,
            "sources": self.source_manager.get_source_count(),
            "chunks": self.faiss_store.get_stats(),
            "graph": self.graph_storage.get_stats(),
            "cache_size": len(self._response_cache),
        }
