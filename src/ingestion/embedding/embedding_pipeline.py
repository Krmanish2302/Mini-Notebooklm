from typing import List, Dict, Any, Optional
import hashlib
import numpy as np
from .text_embedder import TextEmbedder

class EmbeddingPipeline:
    """Manages embedding with semantic caching."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_cache: bool = True):
        self.embedder = TextEmbedder(model_name)
        self.use_cache = use_cache
        self._cache: Dict[str, np.ndarray] = {}
    
    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Embed list of chunks, using cache where possible."""
        texts = [c["content"] for c in chunks]
        embeddings = self.embed_batch(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()
        
        return chunks
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed batch with cache lookup."""
        if not self.use_cache:
            return self.embedder.embed(texts)
        
        results = []
        texts_to_embed = []
        indices = []
        
        for i, text in enumerate(texts):
            key = self._get_cache_key(text)
            if key in self._cache:
                results.append((i, self._cache[key]))
            else:
                texts_to_embed.append(text)
                indices.append(i)
        
        if texts_to_embed:
            new_embeddings = self.embedder.embed(texts_to_embed)
            for idx, text, emb in zip(indices, texts_to_embed, new_embeddings):
                key = self._get_cache_key(text)
                self._cache[key] = emb
                results.append((idx, emb))
        
        # Sort by original index
        results.sort(key=lambda x: x[0])
        return np.array([r[1] for r in results])
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed user query."""
        return self.embedder.embed_single(query)
    
    def clear_cache(self):
        self._cache.clear()