from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np

class ContextualCompressor:
    """Extracts only relevant sentences from retrieved chunks."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 relevance_threshold: float = 0.6):
        self.model = SentenceTransformer(model_name)
        self.threshold = relevance_threshold
    
    def compress(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Compress chunks by keeping only relevant sentences."""
        query_emb = self.model.encode(query)
        compressed = []
        
        for chunk in chunks:
            sentences = chunk["content"].split(". ")
            if len(sentences) <= 2:
                compressed.append(chunk)
                continue
            
            sent_embeddings = self.model.encode(sentences)
            similarities = np.dot(sent_embeddings, query_emb) / (
                np.linalg.norm(sent_embeddings, axis=1) * np.linalg.norm(query_emb)
            )
            
            relevant_sentences = [
                s for s, sim in zip(sentences, similarities) 
                if sim >= self.threshold
            ]
            
            if relevant_sentences:
                chunk["content"] = ". ".join(relevant_sentences)
                chunk["compressed"] = True
                compressed.append(chunk)
        
        return compressed