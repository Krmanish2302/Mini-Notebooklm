"""
deep_research_pipeline.py — Deep Research mode pipeline.

Extends ChatPipeline with:
  - mode="deep_research" (deeper, more structured persona prompt)
  - Larger top_k (20 default)
  - Lower score_threshold (accepts more candidates)
  - Returns full chunks_used for front-end source panel
  - Optional multi-hop: re-queries with extracted sub-topics

LangChain components:
  - LLMChain with StrOutputParser for sub-topic extraction
  - ChatGraph (LangGraph) for retrieval + generation
  - ResponseGenerator for citation assembly
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

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
        pipe = DeepResearchPipeline(faiss_store, sqlite, source_manager)
        result = pipe.run("How does CRISPR edit DNA?")
        print(result["answer"])
    """

    def __init__(
        self,
        faiss_store:    MultiFAISSStore,
        sqlite:         SQLiteManager,
        source_manager: SourceManager,
        compressor=None,
        reranker=None,
        top_k:          int   = 20,
        score_threshold: float = 0.0,
        multi_hop:      bool  = False,
    ):
        self.source_manager  = source_manager
        self.sqlite          = sqlite
        self.top_k           = top_k
        self.score_threshold = score_threshold
        self.multi_hop       = multi_hop
        self._llm: Optional[Any] = None

        self.chat_graph = ChatGraph(
            faiss_store=faiss_store,
            sqlite=sqlite,
            source_manager=source_manager,
            mode="deep_research",
            compressor=compressor,
            reranker=reranker,
        )

    def set_llm(self, llm: Any) -> None:
        self._llm = llm
        self.chat_graph.set_llm(llm)

    # ── Multi-hop sub-topic extraction ────────────────────────────────────

    def _extract_subtopics(self, query: str) -> List[str]:
        """Use a lightweight LLMChain to extract sub-topics for multi-hop retrieval."""
        try:
            llm   = self._llm or LLMRegistry.get()
            chain = _SUBTOPIC_PROMPT | llm | StrOutputParser()
            raw   = chain.invoke({"query": query})
            lines = [
                re.sub(r"^[\-\*\d.)\s]+", "", l).strip()
                for l in raw.splitlines()
                if l.strip()
            ]
            return [l for l in lines if len(l) > 4][:3]
        except Exception as exc:
            logger.warning("[DeepResearch:subtopics] %s — skipping multi-hop", exc)
            return []

    # ── Public API ────────────────────────────────────────────────────────

    def run(
        self,
        query:       str,
        source_ids:  Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run deep research on a query.

        Returns:
            answer, citations, follow_ups, sources_used, chunks_used, tokens_estimate, error
        """
        llm = self._llm or LLMRegistry.get()
        self.chat_graph.set_llm(llm)
        self.chat_graph.set_mode("deep_research")

        # Multi-hop: gather additional context from sub-topics
        extra_chunks: List[Dict] = []
        if self.multi_hop:
            for subtopic in self._extract_subtopics(query):
                logger.info("[DeepResearch] sub-hop: '%s'", subtopic)
                sub_result = self.chat_graph.chat(
                    query=subtopic,
                    top_k=self.top_k // 2,
                    score_threshold=self.score_threshold,
                )
                extra_chunks.extend(sub_result.get("retrieved", []))

        # Main query
        result = self.chat_graph.chat(
            query=query,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
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
                "source_id":      c.get("id", ""),
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