"""
deep_research_pipeline.py  —  Deep Research mode pipeline.

Bug fixes (2026-05-10 audit)
----------------------------
1. ContextualCompressor was constructed with  `compressor.compress(all_chunks, query)`
   but the compressor injected from master_pipeline was built BEFORE the LLM was
   available (ContextualCompressor() no-arg call).  The pipeline now injects the
   live llm_callable into the compressor at run-time before calling compress().

2. Reranker.rerank() default top_k=5 silently truncated results.  Pipeline now
   explicitly passes top_k=None to let ContextBuilder apply the budget instead.

3. context_chunks were not stored on the result dict.  StudyPipeline was then
   unable to find them under 'context_chunks' key and silently fell back to
   'retrieved_chunks', losing the deduped/annotated context.
   Fixed: result["context_chunks"] = context_chunks added.

Multi-hop research flow
-----------------------
1. SubQueryDecomposer   : break query into N focused sub-queries (LLM call)
2. Per-sub-query retrieval via HybridRetriever
3. ContextualCompressor : compress each sub-result to fit token budget
4. Reranker             : global rerank across all sub-results
5. ContextBuilder       : dedup + budget + citation labels
6. PromptBuilder        : research prompt (fixed Sagan researcher persona)
7. LLM synthesis call
8. ResponseGenerator    : structured output with citations
9. RAGChatHistory       : persist turns
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
        self.retriever   = hybrid_retriever
        self.history     = rag_history
        self.compressor  = contextual_compressor
        self.reranker    = reranker
        self.llm         = llm
        self.top_k       = top_k
        self.history_k   = history_k
        self.expansion_n = expansion_n
        self._ctx_builder  = ContextBuilder(max_tokens=max_ctx_tokens)
        self._decomposer   = SubQueryDecomposer(llm=self.llm, n=expansion_n)

    # ─────────────────────────────────────────────────────────────────────
    #  Main entry point
    # ─────────────────────────────────────────────────────────────────────

    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute a full deep-research turn.

        Returns
        -------
        dict with: answer, citations, follow_ups, sources_used,
                   chunks_used, context_chunks, retrieved_chunks,
                   sub_queries, tokens_estimate
        """
        # 1. Decompose
        sub_queries = self._decomposer.decompose(query)
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

        # 3. Optional compression per chunk
        # BUG FIX: inject live llm into compressor right before calling compress()
        # so that the no-arg ContextualCompressor() from master_pipeline gets an LLM.
        if self.compressor and all_chunks and self.llm:
            try:
                if self.compressor.llm is None:
                    self.compressor.llm = self.llm   # late-inject LLM
                all_chunks = self.compressor.compress(all_chunks, query)
            except Exception as exc:
                logger.warning("compressor failed (non-fatal): %s", exc)

        # 4. Optional global rerank
        # BUG FIX: pass top_k=None so we don't silently truncate before ContextBuilder
        if self.reranker and all_chunks:
            try:
                all_chunks = self.reranker.rerank(query, all_chunks, top_k=None)
            except Exception as exc:
                logger.warning("reranker failed (non-fatal): %s", exc)

        # 5. Context assembly
        context_chunks, _sources = self._ctx_builder.build(all_chunks, query=query)

        # 6. Prompt
        history_str = self.history.format_for_prompt(query, k=self.history_k)
        prompt = PromptBuilder.build_research_prompt(
            query,
            context_chunks,
            history=history_str,
            rewrite=False,
        )

        # 7. LLM synthesis
        if not self.llm:
            raise RuntimeError("LLM not configured for DeepResearchPipeline")
        raw_output = self.llm(prompt)

        # 8. Structure response
        gen = ResponseGenerator(context_chunks=context_chunks)
        result = gen.assemble(raw_output, query=query, generate_follow_ups=True)
        result["retrieved_chunks"]  = all_chunks
        result["context_chunks"]    = context_chunks   # BUG FIX: StudyPipeline needs this
        result["sub_queries"]       = sub_queries

        # 9. Persist history
        self.history.add_message("user", query)
        self.history.add_message("assistant", result["answer"])

        return result
