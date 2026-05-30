"""
chat_pipeline.py — High-level Chat pipeline.

Orchestrates:
  1. Source filtering (active source IDs)
  2. ChatGraph invocation (embed → retrieve → compress → rerank → prompt → generate)
  3. Response + citation assembly via ResponseGenerator
  4. RAG-based history via RAGHistoryStore (no ConversationBufferWindowMemory)

History strategy
  - Every completed turn is embedded and stored in SQLite by RAGHistoryStore.
  - On each query, the top-k semantically-similar past turns are retrieved
    and injected into the prompt as plain text.
  - No MemorySaver or ConversationBufferWindowMemory is used anywhere.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.generation.llm_registry import LLMRegistry
from src.generation.persona_config import PersonaConfig
from src.generation.response_generator import ResponseGenerator
from src.pipelines.chat_graph import ChatGraph
from src.storage.faiss_store import MultiFAISSStore
from src.storage.source_manager import SourceManager
from src.storage.sqlite_manager import SQLiteManager
from src.storage.rag_history_store import RAGHistoryStore

logger = logging.getLogger(__name__)


class ChatPipeline:
    """
    End-to-end chat pipeline.

    Usage:
        pipe = ChatPipeline(faiss_store, sqlite, source_manager, embedder)
        pipe.set_persona(PersonaConfig(persona="professor"))
        result = pipe.run("Explain osmosis", source_ids=["bio_101"])
        print(result["answer"])
        print(result["citations"])
    """

    def __init__(
        self,
        faiss_store:    MultiFAISSStore,
        sqlite:         SQLiteManager,
        source_manager: SourceManager,
        embedder:       Any,            # any LangChain Embeddings (embed_query)
        compressor=None,
        reranker=None,
        history_top_k:  int = 4,        # past turns injected per query
        session_id:     str = "default",
    ):
        self.source_manager = source_manager
        self.sqlite         = sqlite
        self._persona: PersonaConfig = PersonaConfig()
        self._llm: Optional[Any]     = None
        self._history_top_k          = history_top_k

        # RAG-based history store
        self.history_store = RAGHistoryStore(sqlite, embedder)

        self.chat_graph = ChatGraph(
            faiss_store=faiss_store,
            sqlite=sqlite,
            source_manager=source_manager,
            rag_history_store=self.history_store,
            compressor=compressor,
            reranker=reranker,
        )
        self.chat_graph.set_session(session_id)

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_persona(self, persona: PersonaConfig) -> None:
        self._persona = persona

    def set_llm(self, llm: Any) -> None:
        self._llm = llm
        self.chat_graph.set_llm(llm)

    def set_session(self, session_id: str) -> None:
        """Switch to a different user/session context."""
        self.chat_graph.set_session(session_id)

    # ── Public API ──────────────────────────────────────────────────────────────

    def run(
        self,
        query:           str,
        source_ids:      Optional[List[str]] = None,
        top_k:           int                 = 8,
        score_threshold: float               = 0.0,
    ) -> Dict[str, Any]:
        """
        Run a single chat turn.

        Returns:
            answer, citations, follow_ups, sources_used, retrieved, error
        """
        llm = self._llm or LLMRegistry.get()
        self.chat_graph.set_llm(llm)

        # Filter active sources if specified
        if source_ids:
            active = self.source_manager.get_active_source_ids(source_ids)
        else:
            active = self.source_manager.get_all_active_source_ids()

        logger.info("[ChatPipeline] query='%s' sources=%s top_k=%d", query, active, top_k)

        # chat_graph handles RAG history retrieval + persistence internally
        result = self.chat_graph.chat(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            history_top_k=self._history_top_k,
        )

        # Assemble response + citations
        retrieved = result.get("retrieved", [])
        chunks = [
            {
                "citation_label": f"S{i+1}",
                "content":        c["content"],
                "source_id":      c.get("id", ""),
            }
            for i, c in enumerate(retrieved)
        ]
        generator = ResponseGenerator(context_chunks=chunks)
        assembled = generator.assemble(
            raw_llm_output=result.get("response", ""),
            query=query,
        )

        return {
            "answer":       assembled["answer"],
            "citations":    assembled["citations"],
            "follow_ups":   assembled["follow_ups"],
            "sources_used": assembled["sources_used"],
            "retrieved":    retrieved,
            "error":        result.get("error"),
        }

    def clear_history(self) -> None:
        """Wipe the current session's turn history from SQLite."""
        self.history_store.clear_session(self.chat_graph._session_id)

    def get_history(self) -> List[Dict[str, Any]]:
        """Return all turns for the current session as a list of dicts."""
        return self.history_store.session_turns(self.chat_graph._session_id)
