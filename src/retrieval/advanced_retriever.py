from typing import List, Dict, Any
import numpy as np

class AdvancedRetriever:
    """Deep Research mode: Query expansion + RAPTOR + full pipeline."""
    
    def __init__(self, hybrid_retriever, contextual_compressor, reranker, raptor=None):
        self.hybrid = hybrid_retriever
        self.compressor = contextual_compressor
        self.reranker = reranker
        self.raptor = raptor
    
    def retrieve(self, query: str, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """Full advanced retrieval pipeline."""
        # Step 1: Query expansion (simple synonym expansion)
        expanded_queries = self._expand_query(query)
        
        # Step 2: Retrieve for each expanded query
        all_results = []
        for q in expanded_queries:
            # Note: In production, embed each expanded query
            results = self.hybrid.retrieve(q, query_embedding)
            all_results.extend(results)
        
        # Step 3: Deduplicate
        seen = set()
        unique = []
        for r in all_results:
            if r["id"] not in seen:
                seen.add(r["id"])
                unique.append(r)
        
        # Step 4: Contextual compression
        compressed = self.compressor.compress(unique, query)
        
        # Step 5: Rerank
        reranked = self.reranker.rerank(query, compressed)
        
        # Step 6: RAPTOR (if available)
        if self.raptor:
            raptor_results = self.raptor.retrieve(query)
            # Merge RAPTOR results
            for r in raptor_results:
                if r["id"] not in seen:
                    reranked.append(r)
        
        return reranked
    
    def _expand_query(self, query: str) -> List[str]:
        """Simple query expansion."""
        # In production, use LLM or synonym database
        expansions = [query]
        # Add variations
        if "how" in query.lower():
            expansions.append(query.replace("how", "what is the way"))
        if "why" in query.lower():
            expansions.append(query.replace("why", "what is the reason"))
        return expansions