"""
study_pipeline.py — Study mode pipeline.

Features:
  - Concept-path guided retrieval (learning_path injects topic ordering)
  - Socratic follow-up generation via a lightweight LLM chain
  - RAG-based session history via RAGHistoryStore
  - LangGraph ChatGraph for retrieval + generation
  - ResponseGenerator for citation assembly

History strategy: RAG-based (no ConversationBufferWindowMemory).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.generation.llm_registry import LLMRegistry
from src.generation.response_generator import ResponseGenerator
from src.pipelines.chat_graph import ChatGraph
from src.storage.faiss_store import MultiFAISSStore
from src.storage.source_manager import SourceManager
from src.storage.sqlite_manager import SQLiteManager
from src.storage.rag_history_store import RAGHistoryStore

logger = logging.getLogger(__name__)

_SOCRATIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Socratic tutor. "
     "Given the answer below, generate 2-3 thought-provoking follow-up questions "
     "that deepen understanding. "
     "Return ONLY the questions as a bullet list — one per line."),
    ("human", "ANSWER:\n{answer}\n\nCONCEPT: {concept}"),
])


class StudyPipeline:
    """
    Study mode pipeline — intuition-building, Socratic follow-ups, concept-path awareness.

    Usage:
        pipe = StudyPipeline(faiss_store, sqlite, source_manager, embedder)
        result = pipe.run("What is entropy?", learning_path=["thermodynamics", "entropy", "heat death"])
        print(result["answer"])
        print(result["follow_ups"])
    """

    def __init__(
        self,
        faiss_store:     MultiFAISSStore,
        sqlite:          SQLiteManager,
        source_manager:  SourceManager,
        embedder:        Any,           # LangChain Embeddings
        compressor=None,
        reranker=None,
        top_k:           int   = 10,
        score_threshold: float = 0.0,
        history_top_k:   int   = 4,
        session_id:      str   = "default",
    ):
        self.source_manager  = source_manager
        self.sqlite          = sqlite
        self.top_k           = top_k
        self.score_threshold = score_threshold
        self._llm: Optional[Any] = None

        # RAG-based history
        self.history_store = RAGHistoryStore(sqlite, embedder)
        self._history_top_k = history_top_k

        self.chat_graph = ChatGraph(
            faiss_store=faiss_store,
            sqlite=sqlite,
            source_manager=source_manager,
            rag_history_store=self.history_store,
            mode="chat",
            compressor=compressor,
            reranker=reranker,
        )
        self.chat_graph.set_session(session_id)

    def set_llm(self, llm: Any) -> None:
        self._llm = llm
        self.chat_graph.set_llm(llm)

    def set_session(self, session_id: str) -> None:
        self.chat_graph.set_session(session_id)

    # ── Socratic follow-up generation ────────────────────────────────────────

    def _socratic_followups(self, answer: str, concept: str) -> List[str]:
        try:
            llm   = self._llm or LLMRegistry.get()
            chain = _SOCRATIC_PROMPT | llm | StrOutputParser()
            raw   = chain.invoke({"answer": answer[:800], "concept": concept})
            lines = [
                re.sub(r"^[\-\*\d.)\s]+", "", l).strip()
                for l in raw.splitlines() if l.strip()
            ]
            return [l for l in lines if len(l) > 8][:3]
        except Exception as exc:
            logger.warning("[StudyPipeline:socratic] %s — skipping", exc)
            return []

    # ── Concept-path query enrichment ──────────────────────────────────────

    @staticmethod
    def _enrich_query(query: str, learning_path: Optional[List[str]]) -> str:
        if not learning_path:
            return query
        path_str = " → ".join(learning_path[:4])
        return f"{query} [concept path: {path_str}]"

    # ── Public API ──────────────────────────────────────────────────────────────

    def run(
        self,
        query:          str,
        learning_path:  Optional[List[str]] = None,
        source_ids:     Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run a study-mode turn.

        Returns:
            answer, citations, follow_ups (Socratic), sources_used, retrieved, error
        """
        llm = self._llm or LLMRegistry.get()
        self.chat_graph.set_llm(llm)

        enriched = self._enrich_query(query, learning_path)
        logger.info("[StudyPipeline] query='%s' path=%s", query, learning_path)

        # chat_graph handles RAG history retrieval + persistence internally
        result = self.chat_graph.chat(
            query=enriched,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
            history_top_k=self._history_top_k,
        )

        response  = result.get("response", "")
        retrieved = result.get("retrieved", [])

        # Build citations
        chunks = [
            {
                "citation_label": f"S{i+1}",
                "content":        c["content"],
                "source_id":      c.get("id", ""),
            }
            for i, c in enumerate(retrieved)
        ]
        generator = ResponseGenerator(context_chunks=chunks)
        assembled = generator.assemble(raw_llm_output=response, query=query)

        # Socratic follow-ups (override ResponseGenerator follow_ups)
        concept   = (learning_path[-1] if learning_path else query)
        followups = self._socratic_followups(assembled["answer"], concept)
        if not followups:
            followups = assembled["follow_ups"]

        return {
            "answer":       assembled["answer"],
            "citations":    assembled["citations"],
            "follow_ups":   followups,
            "sources_used": assembled["sources_used"],
            "retrieved":    retrieved,
            "error":        result.get("error"),
        }

    def clear_history(self) -> None:
        self.history_store.clear_session(self.chat_graph._session_id)

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history_store.session_turns(self.chat_graph._session_id)
