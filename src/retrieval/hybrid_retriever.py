from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi

class HybridRetriever:
    """Combines FAISS (dense) + BM25 (sparse) with Reciprocal Rank Fusion."""
    
    def __init__(self, faiss_store, top_k: int = 5, 
                 dense_weight: float = 0.7, sparse_weight: float = 0.3):
        self.faiss_store = faiss_store
        self.top_k = top_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.bm25 = None
        self.corpus = []
        self.corpus_ids = []
    
    def build_sparse_index(self, chunks: List[Dict[str, Any]]):
        """Build BM25 index from chunks."""
        self.corpus = [c["content"].split() for c in chunks]
        self.corpus_ids = [c["id"] for c in chunks]
        self.bm25 = BM25Okapi(self.corpus)
    
    def retrieve(self, query: str, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """
        Retrieve using both dense and sparse methods.
        Returns fused results.
        """
        # Dense retrieval
        dense_results = self.faiss_store.search(query_embedding, k=self.top_k * 2)
        
        # Sparse retrieval
        sparse_results = []
        if self.bm25:
            tokenized_query = query.split()
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[-self.top_k * 2:][::-1]
            for idx in top_indices:
                if scores[idx] > 0:
                    sparse_results.append({
                        "chunk_id": self.corpus_ids[idx],
                        "score": scores[idx],
                        "method": "sparse"
                    })
        
        # Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)
        return fused[:self.top_k]
    
    def _reciprocal_rank_fusion(self, dense: List[Dict], sparse: List[Dict], k: int = 60) -> List[Dict]:
        """RRF: score = sum(1 / (k + rank))"""
        scores = {}
        
        for rank, item in enumerate(dense):
            cid = item.get("id", item.get("chunk_id"))
            scores[cid] = scores.get(cid, 0) + self.dense_weight * (1 / (k + rank))
        
        for rank, item in enumerate(sparse):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0) + self.sparse_weight * (1 / (k + rank))
        
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return full chunk data
        results = []
        for cid, score in sorted_scores:
            chunk = self.faiss_store.metadata.get(cid, {})
            chunk["score"] = score
            chunk["retrieval_method"] = "hybrid"
            results.append(chunk)
        
        return results