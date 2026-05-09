from typing import List, Dict, Any, Optional
import numpy as np

class CrossModalMerger:
    """Links chunks from different modalities using embedding similarity."""
    
    def __init__(self, similarity_threshold: float = 0.75, embedder=None):
        self.similarity_threshold = similarity_threshold
        self.embedder = embedder  # Must be provided (from EmbeddingPipeline)
    
    def merge(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add related_chunks field to each chunk based on embedding similarity.
        Only links image_caption and transcript chunks to text chunks.
        """
        text_chunks = [c for c in chunks if c.get("modality") == "text"]
        
        for chunk in chunks:
            if chunk.get("modality") in ["image_caption", "transcript"]:
                related = self._find_related(chunk, text_chunks)
                chunk["related_chunks"] = related
        
        return chunks
    
    def _find_related(self, chunk: Dict, text_chunks: List[Dict]) -> List[str]:
        """Find related text chunks using cosine similarity."""
        if not text_chunks or self.embedder is None:
            return []
        
        chunk_emb = self.embedder.embed_single(chunk["content"])
        text_embs = self.embedder.embed([t["content"] for t in text_chunks])
        
        # Cosine similarity
        similarities = np.dot(text_embs, chunk_emb) / (
            np.linalg.norm(text_embs, axis=1) * np.linalg.norm(chunk_emb)
        )
        
        related = []
        for i, sim in enumerate(similarities):
            if sim >= self.similarity_threshold:
                related.append({
                    "chunk_id": text_chunks[i]["id"],
                    "similarity": float(sim)
                })
        
        # Sort by similarity, take top 3
        related.sort(key=lambda x: x["similarity"], reverse=True)
        return related[:3]