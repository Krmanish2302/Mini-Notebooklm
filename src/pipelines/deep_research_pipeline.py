"""Deep Research Mode Pipeline

Retrieval strategy
------------------
* History  : RAG-based chat history (same as Chat mode, k=5 for broader context).
* Step 1   : LLM-based query expansion → N sub-queries.
* Step 2   : Hybrid retrieval (Dense ALL-dims + BM25 + RRF) for EACH sub-query.
* Step 3   : Deduplicate across all sub-query result sets.
* Step 4   : Contextual compression — LLM extracts only query-relevant
             sentences from each chunk (reduces token noise before reranking).
* Step 5   : Cross-encoder rerank (BAAI/bge-reranker-base) — most accurate
             relevance scoring, worth the latency for deep research.
* Step 6   : RAPTOR retrieval (if RAPTOR tree built at ingestion time) —
             adds 2-3 high-level summary nodes for macro context.
* Step 7   : Final merge: reranked leaves + RAPTOR summaries.

Token budget awareness
----------------------
Contextual compression (Step 4) ensures chunk tokens stay within budget
before reranking.  RAPTOR nodes are capped at 2 to avoid summary bloat.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional


class DeepResearchPipeline:
    """
    Entry point for Deep Research Mode.

    Parameters
    ----------
    hybrid_retriever     : HybridRetriever
    rag_history          : RAGChatHistory
    contextual_compressor: ContextualCompressor
    reranker             : Reranker   (BAAI/bge-reranker-base, lazy-loaded)
    llm                  : callable  (used for query expansion + final answer)
    raptor               : RaptorRetriever | None
    top_k                : final chunks to send to LLM (default 8)
    history_k            : history messages (default 5, broader than chat)
    expansion_n          : number of sub-queries to generate (default 3)
    """

    def __init__(
        self,
        hybrid_retriever,
        rag_history,
        contextual_compressor,
        reranker,
        llm,
        raptor=None,
        top_k: int = 8,
        history_k: int = 5,
        expansion_n: int = 3,
    ):
        self.retriever = hybrid_retriever
        self.history = rag_history
        self.compressor = contextual_compressor
        self.reranker = reranker
        self.llm = llm
        self.raptor = raptor
        self.top_k = top_k
        self.history_k = history_k
        self.expansion_n = expansion_n

    def run(self, query: str) -> Dict[str, Any]:
        """
        Full deep research turn.

        Returns
        -------
        {
          "answer"        : str,
          "sources"       : List[Dict],
          "sub_queries"   : List[str],
          "raptor_used"   : bool,
          "history_used"  : List[Dict],
        }
        """
        # ── 1. History ────────────────────────────────────────────────
        history_context = self.history.format_for_prompt(
            query, k=self.history_k
        )   # top-5 RAG + last-2 recency anchor

        # ── 2. LLM Query Expansion ───────────────────────────────────
        sub_queries: List[str] = self._expand_query(query)
        # e.g. "How does backprop work?" ->
        #   ["How does backpropagation work?",
        #    "What is the chain rule in neural networks?",
        #    "Explain gradient computation in deep learning"]

        # ── 3. Hybrid retrieval for EACH sub-query ───────────────────
        # Each call to HybridRetriever:
        #   a) embed sub_query with ALL active models (all dims)
        #   b) FAISS search on ALL indexes simultaneously
        #   c) BM25 search
        #   d) RRF fusion (dense 0.7 + sparse 0.3)
        all_results: List[Dict] = []
        seen_ids = set()
        for sq in sub_queries:
            hits = self.retriever.retrieve(sq, top_k=self.top_k)
            for chunk in hits:
                if chunk["id"] not in seen_ids:
                    seen_ids.add(chunk["id"])
                    all_results.append(chunk)

        # ── 4. Contextual Compression ────────────────────────────────
        # LLM strips each chunk down to only query-relevant sentences.
        # This trims token noise before the expensive cross-encoder pass.
        compressed: List[Dict] = self.compressor.compress(all_results, query)

        # ── 5. Cross-encoder Rerank ───────────────────────────────────
        # BAAI/bge-reranker-base scores each (query, chunk) pair jointly
        # — far more accurate than bi-encoder cosine similarity.
        reranked: List[Dict] = self.reranker.rerank(
            query, compressed, top_k=self.top_k
        )

        # ── 6. RAPTOR (optional) ─────────────────────────────────────
        # RAPTOR tree was built at ingestion time.
        # It returns 1-2 high-level summary nodes providing macro context
        # that individual chunk retrieval often misses.
        raptor_used = False
        if self.raptor:
            raptor_nodes = self.raptor.retrieve(query, top_k=2)
            for node in raptor_nodes:
                if node["id"] not in seen_ids:
                    node["retrieval_method"] = "raptor_summary"
                    reranked.append(node)
                    raptor_used = True

        # ── 7. Build prompt & Generate ───────────────────────────────
        prompt = self._build_prompt(query, reranked, history_context, sub_queries)
        answer = self.llm(prompt)

        # ── 8. Update history ────────────────────────────────────────
        self.history.add_message("user", query)
        self.history.add_message(
            "assistant", answer, sources_used=[c["id"] for c in reranked]
        )

        return {
            "answer": answer,
            "sources": reranked,
            "sub_queries": sub_queries,
            "raptor_used": raptor_used,
            "history_used": self.history.get_relevant_history(query, k=self.history_k),
        }

    # ------------------------------------------------------------------
    # LLM-based Query Expansion
    # ------------------------------------------------------------------

    def _expand_query(
        self,
        query: str,
        n: Optional[int] = None,
    ) -> List[str]:
        """
        Ask the LLM to generate N semantically distinct sub-queries.
        Falls back to the original query if LLM call fails.
        """
        n = n or self.expansion_n
        expansion_prompt = (
            f"Generate {n} distinct search queries that together fully cover "
            f"the following question. Output ONLY the queries, one per line, "
            f"no numbering.\n\nQuestion: {query}\n\nQueries:"
        )
        try:
            raw = self.llm(expansion_prompt)
            lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
            # Always prepend the original query
            all_queries = [query] + [l for l in lines if l != query]
            return all_queries[:n + 1]
        except Exception:
            return [query]

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        query: str,
        chunks: List[Dict],
        history_context: str,
        sub_queries: List[str],
    ) -> str:
        context_block = "\n\n".join(
            f"[{'SUMMARY' if c.get('retrieval_method') == 'raptor_summary' else 'Source'} "
            f"{i+1} | page {c.get('page_number', '?')}]\n{c['content']}"
            for i, c in enumerate(chunks)
        )
        history_block = (
            f"\nCONVERSATION HISTORY:\n{history_context}" if history_context else ""
        )
        sub_q_block = (
            "\nSEARCH ANGLES EXPLORED:\n"
            + "\n".join(f"  - {q}" for q in sub_queries)
        )
        return (
            f"You are a deep research assistant. Provide a thorough, well-structured "
            f"answer using ONLY the provided sources. Cite source numbers.\n"
            f"{history_block}"
            f"{sub_q_block}\n\n"
            f"SOURCES:\n{context_block}\n\n"
            f"RESEARCH QUESTION: {query}\n\n"
            f"COMPREHENSIVE ANSWER:"
        )
