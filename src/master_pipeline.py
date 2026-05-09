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
        self.embedder = EmbeddingPipeline(model_name=self.config.get("embedding.default_model", "all-MiniLM-L6-v2"))
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
    
    def set_mode(self, mode: str):
        """Switch conversation mode."""
        self.mode = mode
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
    
    def ingest(self, file_path: str = None, url: str = None, 
               source_type: str = None) -> str:
        """
        Ingest file or URL into knowledge base.
        Returns source_id.
        """
        # Step 1: Detect type
        if not source_type:
            detection = self.file_detector.detect(file_path=file_path, url=url)
            source_type = detection["source_type"]
        
        source_id = str(uuid.uuid4())
        
        # Step 2: Extract content based on type
        if source_type == "pdf":
            result = PDFPipeline.process(file_path, source_id)
        elif source_type == "image":
            result = ImagePipeline.process(file_path, source_id)
        elif source_type in ["video", "audio"]:
            result = VideoAudioPipeline.process(file_path, source_id)
        elif source_type == "website":
            import asyncio
            result = asyncio.run(WebsitePipeline.process(url, source_id))
        elif source_type == "youtube":
            result = YouTubePipeline.process(url, source_id)
        elif source_type == "csv":
            result = CSVPipeline.process(file_path, source_id)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        # Step 3: Preprocess
        preprocessed = self.preprocessor.process(
            result["content"], source_type, result.get("metadata", {})
        )
        
        # Step 4: Chunk
        strategy = preprocessed["recommendation"]["strategy"]
        chunker = ChunkerRegistry.get_chunker(strategy)
        chunks = chunker.chunk(preprocessed["cleaned_content"], {
            "source_id": source_id,
            "modality": result["modality"],
            **result.get("metadata", {})
        })
        
        # Step 5: Embed
        chunks = self.embedder.embed_chunks(chunks)
        
        # Step 6: Cross-modal merge
        chunks = self.cross_modal_merger.merge(chunks)
        
        # Step 7: Store
        source = {
            "id": source_id,
            "title": result.get("metadata", {}).get("title", "Untitled"),
            "source_type": source_type,
            "file_path": file_path,
            "url": url,
            "metadata": result.get("metadata", {})
        }
        self.source_manager.add_source(source, chunks)
        
        # Step 8: Update sparse index for BM25
        self.hybrid_retriever.build_sparse_index(list(self.faiss_store.metadata.values()))
        
        return source_id
    
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
        else:
            response_text = self.llm.invoke(prompt)
            
            # Parse response
            parsed = self.response_parser.parse(response_text)
            
            # Validate grounding
            is_grounded = self.response_parser.validate_grounding(
                parsed["content"], context
            )
            
            # Store in chat history
            self.chat_history.add_message(
                "user", query, sources_used=[c.get("source_id") for c in retrieved]
            )
            self.chat_history.add_message(
                "assistant", parsed["content"],
                sources_used=[c.get("source_id") for c in retrieved]
            )
            
            # Cache response
            self._response_cache[cache_key] = parsed["content"]
            
            latency = (time.time() - start_time) * 1000
            
            return parsed["content"]
    
    def _stream_response(self, prompt: str, query: str, retrieved: List[Dict]) -> Iterator[str]:
        """Stream response tokens."""
        full_response = []
        for token in self.llm.stream(prompt):
            full_response.append(token)
            yield token
        
        # Store complete response after streaming
        complete = "".join(full_response)
        self.chat_history.add_message("user", query)
        self.chat_history.add_message("assistant", complete)
    
    def clear_cache(self):
        """Clear response cache."""
        self._response_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "mode": self.mode,
            "session_id": self.session_id,
            "sources": len(self.source_manager.sources),
            "chunks": self.faiss_store.get_stats(),
            "graph": self.graph_storage.get_stats(),
            "cache_size": len(self._response_cache)
        }