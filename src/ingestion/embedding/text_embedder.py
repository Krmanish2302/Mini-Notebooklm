from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class TextEmbedder:
    """Embedding with configurable models."""
    
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "nomic-embed-text-v1.5": 768,
        "jina-embeddings-v3": 1024
    }
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.MODEL_DIMENSIONS.get(model_name, 384)
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Batch embed texts."""
        if not texts:
            return np.array([])
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed single text."""
        return self.embed([text])[0]
    
    def get_dimension(self) -> int:
        return self.dimension