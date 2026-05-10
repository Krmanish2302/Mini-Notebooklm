"""
advanced_retriever.py  —  Used by StudyModePipeline (via StudyModeRetriever)

Fixes
-----
* Stub query expander replaced with LLM-based expansion (same logic as
  DeepResearchPipeline._expand_query).  Accepts an optional `llm` param;
  falls back to the old keyword heuristics only if no LLM is provided.
* `retrieve()` now accepts `top_k` and `score_threshold` and passes them
  through to HybridRetriever and Reranker respectively.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AdvancedRetriever:
    """
    Advanced retrieval used by StudyModeRetriever (Study Mode broad pass).

    Pipeline
    --------
    1. LLM-based query expansion  → N semantically distinct sub-queries
    2. HybridRetriever.retrieve()  per expanded query (Dense + BM25 + RRF)
    3. Deduplicate across sub-query results
    4. Contextual compression      — strip irrelevant sentences
    5. Cross-encoder rerank        — score + threshold + top_k trim
    6. Optional RAPTOR summary nodes appended for macro-context

    Parameters
    ----------
    hybrid_retriever     : HybridRetriever
    contextual_compressor: ContextualCompressor
    reranker             : Reranker
    llm                  : callable(prompt: str) -> str  (optional)
                           When provided, used for LLM-based query expansion.
                           When None, falls back to keyword heuristics.
    raptor               : RaptorRetriever | None
    top_k                : default number of final chunks (default 8)
    score_threshold      : drop reranked chunks below this score (default 0.0)
    expansion_n          : number of LLM sub-queries to generate (default 3)
    """

    def __init__(
        self,
        hybrid_retriever,
        contextual_compressor,
        reranker,
        llm: Optional[Callable[[str], str]] = None,
        raptor=None,
        top_k: int = 8,
        score_threshold: float = 0.0,
        expansion_n: int = 3,
    ):
        self.hybrid = hybrid_retriever
        self.compressor = contextual_compressor
        self.reranker = reranker
        self.llm = llm
        self.raptor = raptor
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.expansion_n = expansion_n

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Full advanced retrieval pipeline.

        Args:
            query          : user query or study topic
            top_k          : override instance default for this call
            score_threshold: override instance default for this call
        """
        effective_top_k = top_k if top_k is not None else self.top_k
        effective_threshold = score_threshold if score_threshold is not None else self.score_threshold
        # Over-fetch candidates so compress+rerank have headroom
        candidate_k = effective_top_k * 3

        # ── Step 1: Query expansion ──────────────────────────────────
        expanded_queries = self._expand_query(query)

        # ── Step 2+3: Hybrid retrieve + deduplicate ───────────────────
        all_results: List[Dict[str, Any]] = []
        seen: set = set()
        for q in expanded_queries:
            for chunk in self.hybrid.retrieve(q, top_k=candidate_k):
                cid = chunk.get("id")
                if cid and cid not in seen:
                    seen.add(cid)
                    all_results.append(chunk)

        # ── Step 4: Contextual compression ───────────────────────────
        try:
            compressed = self.compressor.compress(all_results, query)
        except Exception as exc:
            logger.warning("AdvancedRetriever[compress] failed, using raw: %s", exc)
            compressed = all_results

        # ── Step 5: Cross-encoder rerank + threshold + top_k trim ─────
        try:
            reranked = self.reranker.rerank(query, compressed, top_k=len(compressed))
            reranked = [
                c for c in reranked
                if c.get("rerank_score", 1.0) >= effective_threshold
            ]
            reranked = reranked[:effective_top_k]
        except Exception as exc:
            logger.warning("AdvancedRetriever[rerank] failed, using compressed: %s", exc)
            reranked = compressed[:effective_top_k]

        logger.debug(
            "AdvancedRetriever: %d raw → %d compressed → %d reranked (threshold=%.2f)",
            len(all_results), len(compressed), len(reranked), effective_threshold,
        )

        # ── Step 6: Optional RAPTOR summary nodes ────────────────────
        if self.raptor:
            raptor_seen = {c["id"] for c in reranked if "id" in c}
            try:
                for r in self.raptor.retrieve(query, top_k=2):
                    if r.get("id") not in raptor_seen:
                        r["retrieval_method"] = "raptor_summary"
                        reranked.append(r)
            except Exception as exc:
                logger.warning("AdvancedRetriever[raptor] failed: %s", exc)

        return reranked

    # ------------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------------

    def _expand_query(
        self,
        query: str,
        n: Optional[int] = None,
    ) -> List[str]:
        """
        LLM-based query expansion when an LLM is available.
        Generates N semantically distinct sub-queries, always prepends the
        original query so it is always in the search set.

        Falls back to simple keyword heuristics when no LLM is provided
        (e.g. during unit tests or lightweight deployments).
        """
        n = n or self.expansion_n

        # ── LLM path ──────────────────────────────────────────────────
        if self.llm is not None:
            prompt = (
                f"Generate {n} distinct search queries that together fully cover "
                f"the following question. Output ONLY the queries, one per line, "
                f"no numbering.\n\nQuestion: {query}\n\nQueries:"
            )
            try:
                raw = self.llm(prompt)
                lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
                all_queries = [query] + [l for l in lines if l != query]
                logger.debug(
                    "AdvancedRetriever: LLM expanded to %d sub-queries", len(all_queries)
                )
                return all_queries[:n + 1]
            except Exception as exc:
                logger.warning(
                    "AdvancedRetriever: LLM query expansion failed (%s), "
                    "falling back to heuristics", exc
                )

        # ── Heuristic fallback (no LLM) ─────────────────────────────
        expansions = [query]
        lowered = query.lower()
        if "how" in lowered:
            expansions.append(lowered.replace("how", "method to", 1))
        if "why" in lowered:
            expansions.append(lowered.replace("why", "reason for", 1))
        if "what is" in lowered:
            expansions.append(lowered.replace("what is", "define", 1))

        seen: set = set()
        unique = []
        for q in expansions:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        return unique
