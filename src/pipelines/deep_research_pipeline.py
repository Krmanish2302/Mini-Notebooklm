"""
deep_research_pipeline.py  —  Deep Research mode pipeline.

Fixes applied
-------------
BUG-R01  SubQueryDecomposer was constructed with self.llm which is None by
         default.  If master_pipeline sets llm= after construction the
         decomposer's captured reference stays None.  Fixed: _decomposer is
         now created lazily on first run() call so it always uses the live llm.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from src.generation.prompt_builder import PromptBuilder
from src.generation.response_generator import ResponseGenerator
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.query_expander import SubQueryDecomposer
from src.chat_history.rag_history import RAGChatHistory

logger = logging.getLogger(__name__)


class DeepResearchPipeline:
    """
    Parameters
    ----------
    hybrid_retriever      : HybridRetriever
    rag_history           : RAGChatHistory
    contextual_compressor : ContextualCompressor (may be None)
    reranker              : Reranker             (may be None)
    llm                   : callable (str) -> str
    raptor                : unused — kept for API compat
    top_k                 : chunks per sub-query
    history_k             : turns in prompt
    expansion_n           : number of sub-queries to decompose into
    max_ctx_tokens        : token budget for final context block
    """

    def __init__(
        self,
        hybrid_retriever,
        rag_history: RAGChatHistory,
        contextual_compressor=None,
        reranker=None,
        llm: Optional[Callable[[str], str]] = None,
        raptor=None,
        top_k: int = 8,
        history_k: int = 5,
        expansion_n: int = 3,
        max_ctx_tokens: int = 4000,
    ):
        self.retriever    = hybrid_retriever
        self.history      = rag_history
        self.compressor   = contextual_compressor
        self.reranker     = reranker
        self.llm          = llm
        self.top_k        = top_k
        self.history_k    = history_k
        self.expansion_n  = expansion_n
        self._ctx_builder = ContextBuilder(max_tokens=max_ctx_tokens)
        # BUG-R01: _decomposer is built lazily in run() so it always captures
        # the live self.llm rather than whatever None was at __init__ time.
        self._decomposer: Optional[SubQueryDecomposer] = None

    def _get_decomposer(self) -> SubQueryDecomposer:
        """Lazily build / rebuild the decomposer so it always has the live LLM."""
        if self._decomposer is None or self._decomposer.llm is not self.llm:
            self._decomposer = SubQueryDecomposer(llm=self.llm, n=self.expansion_n)
        return self._decomposer

    # ─────────────────────────────────────────────────────────────────────
    #  Main entry point
    # ─────────────────────────────────────────────────────────────────────

    def run(self, query: str) -> Dict[str, Any]:
        if not self.llm:
            raise RuntimeError("LLM not configured for DeepResearchPipeline")

        # 1. Decompose — BUG-R01: use lazy getter
        decomposer  = self._get_decomposer()
        sub_queries = decomposer.decompose(query)
        logger.info(
            "DeepResearchPipeline: %d sub-queries for '%s'",
            len(sub_queries), query,
        )

        # 2. Retrieve per sub-query, union results
        all_chunks: List[Dict] = []
        seen_ids: set = set()
        for sq in sub_queries:
            try:
                hits = self.retriever.retrieve(sq, top_k=self.top_k)
            except Exception as exc:
                logger.warning("retrieve failed for sub-query '%s': %s", sq, exc)
                hits = []
            for c in hits:
                cid = c.get("id", "")
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_chunks.append(c)

        # 3. Optional compression
        if self.compressor and all_chunks:
            try:
                if self.compressor.llm is None:
                    self.compressor.llm = self.llm
                all_chunks = self.compressor.compress(all_chunks, query)
            except Exception as exc:
                logger.warning("compressor failed (non-fatal): %s", exc)

        # 4. Optional global rerank
        if self.reranker and all_chunks:
            try:
                all_chunks = self.reranker.rerank(query, all_chunks, top_k=None)
            except Exception as exc:
                logger.warning("reranker failed (non-fatal): %s", exc)

        # 5. Context assembly
        context_chunks, _sources = self._ctx_builder.build(all_chunks, query=query)

        # 6. Prompt — rewrite=False because we already decomposed the query
        history_str = self.history.format_for_prompt(query, k=self.history_k)
        prompt = PromptBuilder.build_research_prompt(
            query, context_chunks, history=history_str, rewrite=False,
        )

        # 7. LLM synthesis
        raw_output = self.llm(prompt)

        # 8. Structure response
        gen    = ResponseGenerator(context_chunks=context_chunks)
        result = gen.assemble(raw_output, query=query, generate_follow_ups=True)
        result["retrieved_chunks"] = all_chunks
        result["context_chunks"]   = context_chunks
        result["sub_queries"]      = sub_queries

        # 9. Persist history
        self.history.add_message("user",      query)
        self.history.add_message("assistant", result["answer"])

        return result
