"""
study_pipeline.py — Study mode pipeline.

Features:
  - Concept-path guided retrieval (learning_path injects topic ordering)
  - Socratic follow-up generation via a lightweight LLMChain
  - ConversationBufferWindowMemory for session history
  - LangGraph ChatGraph for retrieval + generation
  - ResponseGenerator for citation assembly

LangChain components:
  - ConversationBufferWindowMemory
  - LLMChain (Socratic follow-up extractor)
  - ChatGraph (LangGraph)
  - ResponseGenerator
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.generation.llm_registry import LLMRegistry
from src.generation.response_generator import ResponseGenerator
from src.pipelines.chat_graph import ChatGraph
from src.storage.faiss_store import MultiFAISSStore
from src.storage.source_manager import SourceManager
from src.storage.sqlite_manager import SQLiteManager

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
    Study mode pipeline — builds intuition, Socratic follow-ups, concept-path awareness.

    Usage:
        pipe = StudyPipeline(faiss_store, sqlite, source_manager)
        result = pipe.run("What is entropy?", learning_path=["thermodynamics", "entropy", "heat death"])
        print(result["answer"])
        print(result["follow_ups"])
    """

    def __init__(
        self,
        faiss_store:    MultiFAISSStore,
        sqlite:         SQLiteManager,
        source_manager: SourceManager,
        compressor=None,
        reranker=None,
        window_k:       int   = 4,
        top_k:          int   = 10,
        score_threshold: float = 0.0,
    ):
        self.source_manager  = source_manager
        self.sqlite          = sqlite
        self.top_k           = top_k
        self.score_threshold = score_threshold
        self._llm: Optional[Any] = None

        self.memory = ConversationBufferWindowMemory(
            k=window_k,
            return_messages=True,
            memory_key="chat_history",
        )

        self.chat_graph = ChatGraph(
            faiss_store=faiss_store,
            sqlite=sqlite,
            source_manager=source_manager,
            mode="chat",
            compressor=compressor,
            reranker=reranker,
        )

    def set_llm(self, llm: Any) -> None:
        self._llm = llm
        self.chat_graph.set_llm(llm)

    # ── Socratic follow-up generation ─────────────────────────────────────

    def _socratic_followups(self, answer: str, concept: str) -> List[str]:
        try:
            llm   = self._llm or LLMRegistry.get()
            chain = _SOCRATIC_PROMPT | llm | StrOutputParser()
            raw   = chain.invoke({"answer": answer[:800], "concept": concept})
            lines = [
                re.sub(r"^[\-\*\d.)\s]+", "", l).strip()
                for l in raw.splitlines()
                if l.strip()
            ]
            return [l for l in lines if len(l) > 8][:3]
        except Exception as exc:
            logger.warning("[StudyPipeline:socratic] %s — skipping", exc)
            return []

    # ── Concept-path query enrichment ─────────────────────────────────────

    @staticmethod
    def _enrich_query(query: str, learning_path: Optional[List[str]]) -> str:
        if not learning_path:
            return query
        path_str = " → ".join(learning_path[:4])
        return f"{query} [concept path: {path_str}]"

    # ── Public API ────────────────────────────────────────────────────────

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

        # Load history
        mem_vars = self.memory.load_memory_variables({})
        history  = mem_vars.get("chat_history", [])

        enriched = self._enrich_query(query, learning_path)
        logger.info("[StudyPipeline] query='%s' path=%s", query, learning_path)

        result = self.chat_graph.chat(
            query=enriched,
            history=history,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )

        response = result.get("response", "")

        # Save to memory (use original query, not enriched)
        self.memory.save_context({"input": query}, {"output": response})

        # Build citations
        retrieved = result.get("retrieved", [])
        chunks    = [
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
        concept    = (learning_path[-1] if learning_path else query)
        followups  = self._socratic_followups(assembled["answer"], concept)
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
        self.memory.clear()