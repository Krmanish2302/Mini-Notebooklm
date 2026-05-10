from typing import List, Dict, Any
import numpy as np


class AdvancedRetriever:
    """Deep Research mode: Query expansion + RAPTOR + full pipeline."""

    def __init__(self, hybrid_retriever, contextual_compressor, reranker, raptor=None):
        self.hybrid = hybrid_retriever
        self.compressor = contextual_compressor
        self.reranker = reranker
        self.raptor = raptor

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Full advanced retrieval pipeline.

        Steps
        -----
        1. Query expansion
        2. HybridRetriever.retrieve() per expanded query (embedding handled internally)
        3. Deduplicate
        4. Contextual compression
        5. Cross-encoder rerank
        6. Merge optional RAPTOR summary nodes
        """
        # Step 1: Query expansion
        expanded_queries = self._expand_query(query)

        # Step 2: Retrieve for each expanded query
        all_results: List[Dict[str, Any]] = []
        seen: set = set()
        for q in expanded_queries:
            for chunk in self.hybrid.retrieve(q):
                cid = chunk.get("id")
                if cid and cid not in seen:
                    seen.add(cid)
                    all_results.append(chunk)

        # Step 3: Contextual compression (drop irrelevant chunks)
        compressed = self.compressor.compress(all_results, query)

        # Step 4: Rerank with cross-encoder
        reranked = self.reranker.rerank(query, compressed)

        # Step 5: Merge RAPTOR summary nodes (if available)
        if self.raptor:
            raptor_results = self.raptor.retrieve(query)
            raptor_seen = {c["id"] for c in reranked}
            for r in raptor_results:
                if r.get("id") not in raptor_seen:
                    reranked.append(r)

        return reranked

    # ------------------------------------------------------------------
    # Query expansion (stub — replace with LLM expansion in production)
    # ------------------------------------------------------------------

    def _expand_query(self, query: str) -> List[str]:
        """
        Returns original query plus simple keyword variants.
        TODO: replace with LLM-generated sub-questions for better coverage.
        """
        expansions = [query]
        lowered = query.lower()
        if "how" in lowered:
            expansions.append(query.lower().replace("how", "method to", 1))
        if "why" in lowered:
            expansions.append(query.lower().replace("why", "reason for", 1))
        if "what is" in lowered:
            expansions.append(query.lower().replace("what is", "define", 1))
        # Deduplicate while preserving order
        seen: set = set()
        unique = []
        for q in expansions:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        return unique
