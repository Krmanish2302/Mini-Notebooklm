from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class Reranker:
    """Cross-encoder reranker for precise scoring."""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, chunks: List[Dict[str, Any]], 
               top_k: int = 5) -> List[Dict[str, Any]]:
        """Rerank chunks by query relevance."""
        if not chunks:
            return []
        
        pairs = [(query, c["content"]) for c in chunks]
        scores = self.model.predict(pairs)
        
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)
        
        # Sort by rerank score
        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return chunks[:top_k]