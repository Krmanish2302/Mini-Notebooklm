"""
chat_pipeline.py — High-level Chat pipeline.

Orchestrates:
  1. Source filtering (active source IDs)
  2. ChatGraph invocation (embed → retrieve → compress → rerank → prompt → generate)
  3. Response + citation assembly via ResponseGenerator
  4. Chat history management via LangChain ConversationBufferWindowMemory

LangChain components used:
  - ConversationBufferWindowMemory  (chat history)
  - LLMRegistry                     (model factory)
  - ChatGraph                       (LangGraph sub-graph)
  - ResponseGenerator               (citation resolver)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import AIMessage, HumanMessage

from src.generation.llm_registry import LLMRegistry
from src.generation.persona_config import PersonaConfig
from src.generation.response_generator import ResponseGenerator
from src.pipelines.chat_graph import ChatGraph
from src.storage.faiss_store import MultiFAISSStore
from src.storage.source_manager import SourceManager
from src.storage.sqlite_manager import SQLiteManager

logger = logging.getLogger(__name__)


class ChatPipeline:
    """
    End-to-end chat pipeline.

    Usage:
        pipe = ChatPipeline(faiss_store, sqlite, source_manager)
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
        compressor=None,
        reranker=None,
        window_k:       int = 6,
    ):
        self.source_manager = source_manager
        self.sqlite         = sqlite

        # LangChain memory — retains last window_k human/AI turn pairs
        self.memory = ConversationBufferWindowMemory(
            k=window_k,
            return_messages=True,
            memory_key="chat_history",
        )

        self._persona: PersonaConfig = PersonaConfig()
        self._llm: Optional[Any]     = None

        self.chat_graph = ChatGraph(
            faiss_store=faiss_store,
            sqlite=sqlite,
            source_manager=source_manager,
            compressor=compressor,
            reranker=reranker,
        )

    # ── Configuration ─────────────────────────────────────────────────────

    def set_persona(self, persona: PersonaConfig) -> None:
        self._persona = persona

    def set_llm(self, llm: Any) -> None:
        self._llm = llm
        self.chat_graph.set_llm(llm)

    def set_thread(self, thread_id: str) -> None:
        self.chat_graph.set_thread(thread_id)

    # ── Public API ────────────────────────────────────────────────────────

    def run(
        self,
        query:          str,
        source_ids:     Optional[List[str]] = None,
        top_k:          int                 = 8,
        score_threshold: float              = 0.0,
    ) -> Dict[str, Any]:
        """
        Run a single chat turn.

        Returns:
            answer, citations, follow_ups, sources_used, retrieved, error
        """
        llm = self._llm or LLMRegistry.get()
        self.chat_graph.set_llm(llm)

        # Load LangChain memory as LangChain message history
        mem_vars = self.memory.load_memory_variables({})
        history  = mem_vars.get("chat_history", [])

        # Filter active sources if specified
        if source_ids:
            active = self.source_manager.get_active_source_ids(source_ids)
        else:
            active = self.source_manager.get_all_active_source_ids()

        logger.info("[ChatPipeline] query='%s' sources=%s top_k=%d", query, active, top_k)

        result = self.chat_graph.chat(
            query=query,
            history=history,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        # Save to LangChain memory
        self.memory.save_context(
            {"input": query},
            {"output": result.get("response", "")},
        )

        # Assemble response + citations
        retrieved = result.get("retrieved", [])
        chunks    = [
            {"citation_label": f"S{i+1}", "content": c["content"], "source_id": c.get("id", "")}
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
        self.memory.clear()

    def get_history(self) -> List[Dict[str, str]]:
        """Return history as list of {role, content} dicts."""
        mem_vars = self.memory.load_memory_variables({})
        messages = mem_vars.get("chat_history", [])
        out = []
        for m in messages:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            out.append({"role": role, "content": m.content})
        return out