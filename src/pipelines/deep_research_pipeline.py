"""
deep_research_pipeline.py — Deep Research mode pipeline.

Extends the base pipeline with:
  - mode="deep_research" (deeper, more structured persona prompt)
  - Larger top_k (20 default)
  - Optional multi-hop: re-queries with extracted sub-topics
  - Returns full chunks_used for front-end source panel
  - RAG-based history via RAGHistoryStore

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

_SUBTOPIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Extract 2-3 distinct sub-topics from the research question below. "
     "Return ONLY a bullet list: one sub-topic per line, no preamble."),
    ("human", "{query}"),
])


class DeepResearchPipeline:
    """
    Deep research pipeline — more chunks, structured answer, optional multi-hop.

    Usage:
        pipe = DeepResearchPipeline(faiss_store, sqlite, source_manager, embedder)
        result = pipe.run("How does CRISPR edit DNA?")
        print(result["answer"])
    """

    def __init__(
        self,
        faiss_store:     MultiFAISSStore,
        sqlite:          SQLiteManager,
        source_manager:  SourceManager,
        embedder:        Any,           # LangChain Embeddings
        compressor=None,
        reranker=None,
        top_k:           int   = 20,
        score_threshold: float = 0.0,
        multi_hop:       bool  = False,
        history_top_k:   int   = 3,    # fewer turns — deep research is mostly stateless
        session_id:      str   = "default",
    ):
        self.source_manager  = source_manager
        self.sqlite          = sqlite
        self.top_k           = top_k
        self.score_threshold = score_threshold
        self.multi_hop       = multi_hop
        self._llm: Optional[Any] = None
        self._history_top_k      = history_top_k

        # RAG-based history
        self.history_store = RAGHistoryStore(sqlite, embedder)

        self.chat_graph = ChatGraph(
            faiss_store=faiss_store,
            sqlite=sqlite,
            source_manager=source_manager,
            rag_history_store=self.history_store,
            mode="deep_research",
            compressor=compressor,
            reranker=reranker,
        )
        self.chat_graph.set_session(session_id)

    def set_llm(self, llm: Any) -> None:
        self._llm = llm
        self.chat_graph.set_llm(llm)

    def set_session(self, session_id: str) -> None:
        self.chat_graph.set_session(session_id)

    # ── Multi-hop sub-topic extraction ───────────────────────────────────────

    def _extract_subtopics(self, query: str) -> List[str]:
        try:
            llm   = self._llm or LLMRegistry.get()
            chain = _SUBTOPIC_PROMPT | llm | StrOutputParser()
            raw   = chain.invoke({"query": query})
            lines = [
                re.sub(r"^[\-\*\d.)\s]+", "", l).strip()
                for l in raw.splitlines() if l.strip()
            ]
            return [l for l in lines if len(l) > 4][:3]
        except Exception as exc:
            logger.warning("[DeepResearch:subtopics] %s — skipping multi-hop", exc)
            return []

    # ── Public API ──────────────────────────────────────────────────────────────

    def run(
        self,
        query:      str,
        source_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run deep research on a query.

        Returns:
            answer, citations, follow_ups, sources_used, chunks_used, tokens_estimate, error
        """
        llm = self._llm or LLMRegistry.get()
        self.chat_graph.set_llm(llm)
        self.chat_graph.set_mode("deep_research")

        # Multi-hop: gather additional context from sub-topics (no history save for sub-hops)
        extra_chunks: List[Dict] = []
        if self.multi_hop:
            for subtopic in self._extract_subtopics(query):
                logger.info("[DeepResearch] sub-hop: '%s'", subtopic)
                sub_result = self.chat_graph.chat(
                    query=subtopic,
                    top_k=self.top_k // 2,
                    score_threshold=self.score_threshold,
                    history_top_k=0,    # sub-hops don't need history
                )
                extra_chunks.extend(sub_result.get("retrieved", []))

        # Main query — chat_graph handles RAG history internally
        result = self.chat_graph.chat(
            query=query,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
            history_top_k=self._history_top_k,
        )

        # Deduplicate retrieved chunks
        seen: set = set()
        all_chunks: List[Dict] = []
        for c in result.get("retrieved", []) + extra_chunks:
            cid = c.get("id", c.get("content", "")[:40])
            if cid not in seen:
                seen.add(cid)
                all_chunks.append(c)

        # Assemble citations
        context_chunks = [
            {
                "citation_label": f"S{i+1}",
                "content":        c["content"],
                "source_id":      c.get("source_id", ""),
                "source_name":    c.get("source_name", ""),
                "page":           c.get("page", ""),
            }
            for i, c in enumerate(all_chunks)
        ]
        generator = ResponseGenerator(context_chunks=context_chunks)
        assembled = generator.assemble(
            raw_llm_output=result.get("response", ""),
            query=query,
        )

        return {
            "answer":          assembled["answer"],
            "citations":       assembled["citations"],
            "follow_ups":      assembled["follow_ups"],
            "sources_used":    assembled["sources_used"],
            "chunks_used":     assembled["chunks_used"],
            "tokens_estimate": assembled["tokens_estimate"],
            "error":           result.get("error"),
        }

    def clear_history(self) -> None:
        self.history_store.clear_session(self.chat_graph._session_id)

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history_store.session_turns(self.chat_graph._session_id)
