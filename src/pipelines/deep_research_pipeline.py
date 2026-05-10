"""Deep Research Mode Pipeline

Retrieval strategy
------------------
* Step 1 : RAG-based chat history (k=5).
* Step 2 : LLM query expansion → N sub-queries.
* Step 3 : Hybrid retrieval (Dense ALL-dims + BM25 + RRF) per sub-query.
* Step 4 : Deduplicate.
* Step 5 : Contextual compression.
* Step 6 : Cross-encoder rerank + score threshold.
* Step 7 : RAPTOR summary nodes (optional macro-context).
* Step 8 : Build prompt → generate.

Persona: Carl Sagan as a chill classmate. Strictly source-grounded.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional

# ── Persona + grounding (token-efficient) ────────────────────────────────
_SYSTEM = (
    "You're Carl Sagan if he were a chill classmate. "
    "Go deep — structured, thorough, cite everything as [S1], [S2]… "
    "Use ONLY the sources below. Never invent facts. "
    "If it’s not there, say: 'Not in my notes, bro.'"
)


class DeepResearchPipeline:
    """
    Parameters
    ----------
    hybrid_retriever      : HybridRetriever
    rag_history           : RAGChatHistory
    contextual_compressor : ContextualCompressor
    reranker              : Reranker (BAAI/bge-reranker-base, lazy-loaded)
    llm                   : callable(prompt) -> str
    raptor                : RaptorRetriever | None
    top_k                 : final chunks to LLM (default 8)
    score_threshold       : drop chunks below this rerank score (default 0.0)
    history_k             : history messages (default 5)
    expansion_n           : sub-queries to generate (default 3)
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
        score_threshold: float = 0.0,
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
        self.score_threshold = score_threshold
        self.history_k = history_k
        self.expansion_n = expansion_n

    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        effective_top_k = top_k if top_k is not None else self.top_k
        effective_threshold = score_threshold if score_threshold is not None else self.score_threshold

        # 1. History
        history_context = self.history.format_for_prompt(query, k=self.history_k)

        # 2. Query expansion
        sub_queries: List[str] = self._expand_query(query)

        # 3. Hybrid retrieve + deduplicate
        all_results: List[Dict] = []
        seen_ids = set()
        for sq in sub_queries:
            for chunk in self.retriever.retrieve(sq, top_k=effective_top_k):
                if chunk["id"] not in seen_ids:
                    seen_ids.add(chunk["id"])
                    all_results.append(chunk)

        # 4. Contextual compression
        compressed: List[Dict] = self.compressor.compress(all_results, query)

        # 5. Cross-encoder rerank + threshold
        reranked: List[Dict] = self.reranker.rerank(query, compressed, top_k=len(compressed))
        reranked = [c for c in reranked if c.get("rerank_score", 1.0) >= effective_threshold]
        reranked = reranked[:effective_top_k]

        # 6. RAPTOR macro-context nodes (optional)
        # RAPTOR builds a recursive cluster-summary tree at ingest time.
        # Each node summarises a group of related chunks; the root is a
        # full-document abstract.  Retrieving 1-2 nodes adds macro-context
        # that individual paragraph chunks cannot surface.
        raptor_used = False
        if self.raptor:
            for node in self.raptor.retrieve(query, top_k=2):
                if node["id"] not in seen_ids:
                    node["retrieval_method"] = "raptor_summary"
                    reranked.append(node)
                    raptor_used = True

        # 7. Prompt + generate
        prompt = self._build_prompt(query, reranked, history_context, sub_queries)
        answer = self.llm(prompt)

        # 8. Update history
        self.history.add_message("user", query)
        self.history.add_message("assistant", answer, sources_used=[c["id"] for c in reranked])

        return {
            "answer": answer,
            "sources": reranked,
            "sub_queries": sub_queries,
            "raptor_used": raptor_used,
            "history_used": self.history.get_relevant_history(query, k=self.history_k),
        }

    def _expand_query(self, query: str, n: Optional[int] = None) -> List[str]:
        n = n or self.expansion_n
        prompt = (
            f"Generate {n} distinct search queries covering this question. "
            f"One per line, no numbering.\n\nQuestion: {query}\n\nQueries:"
        )
        try:
            raw = self.llm(prompt)
            lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
            return ([query] + [l for l in lines if l != query])[:n + 1]
        except Exception:
            return [query]

    def _build_prompt(
        self,
        query: str,
        chunks: List[Dict],
        history_context: str,
        sub_queries: List[str],
    ) -> str:
        sources = "\n\n".join(
            f"[S{i+1}]{' [SUMMARY]' if c.get('retrieval_method') == 'raptor_summary' else ''} "
            f"(p.{c.get('page_number', '?')})\n{c['content']}"
            for i, c in enumerate(chunks)
        )
        hist = f"HISTORY:\n{history_context}\n\n" if history_context else ""
        angles = "  • " + "\n  • ".join(sub_queries)
        return (
            f"{_SYSTEM}\n\n"
            f"{hist}"
            f"SEARCH ANGLES:\n{angles}\n\n"
            f"SOURCES:\n{sources}\n\n"
            f"Q: {query}\nA:"
        )
