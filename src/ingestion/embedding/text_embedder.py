from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class TextEmbedder:
    """Embedding with configurable models.

    Fix (Bug 3): dimension is probed from the actual model output instead of
    a stale hard-coded MODEL_DIMENSIONS dict, so any model — including ones
    not in the old dict — always reports the correct dimensionality.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # Probe real dimension — never trust a stale registry dict
        probe = self.model.encode(["probe"], convert_to_numpy=True, show_progress_bar=False)
        self.dimension: int = int(np.atleast_2d(probe).shape[1])

    def embed(self, texts: List[str]) -> np.ndarray:
        """Batch embed texts. Returns shape (N, dim)."""
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        return self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns 1-D array of shape (dim,)."""
        result = self.embed([text])
        return result[0]

    def get_dimension(self) -> int:
        return self.dimension
