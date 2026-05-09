from typing import List, Dict, Any, Optional
import numpy as np


class ContextualCompressor:
    """Extracts only relevant sentences from retrieved chunks.

    Accepts an optional ``embedder`` parameter so the caller can pass the
    already-initialised EmbeddingPipeline model instead of loading a
    second SentenceTransformer.  If no embedder is provided a local
    SentenceTransformer is created on first use (also lazy-loaded).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        relevance_threshold: float = 0.6,
        embedder=None,
    ):
        self.model_name = model_name
        self.threshold = relevance_threshold
        # If an external embedder is provided, use it directly.
        # Otherwise _model stays None and is lazy-loaded on first compress().
        self._model = embedder
        self._owns_model = embedder is None  # True → we created it, we manage it

    def _load_model(self):
        """Lazy-load a local SentenceTransformer if no external embedder was given."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # deferred
            self._model = SentenceTransformer(self.model_name)

    def _encode(self, texts):
        """Encode texts using whichever model is available."""
        self._load_model()
        # Support both raw SentenceTransformer and EmbeddingPipeline wrappers
        if hasattr(self._model, "embed_documents"):
            # LangChain-style embedder (e.g. EmbeddingPipeline)
            return np.array(self._model.embed_documents(texts))
        # Raw SentenceTransformer
        return self._model.encode(texts)

    def _encode_query(self, query: str):
        """Encode a single query string."""
        self._load_model()
        if hasattr(self._model, "embed_query"):
            return np.array(self._model.embed_query(query))
        return self._model.encode(query)

    def compress(
        self, chunks: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Compress chunks by keeping only relevant sentences."""
        query_emb = self._encode_query(query)
        compressed: List[Dict[str, Any]] = []

        for chunk in chunks:
            sentences = chunk["content"].split(". ")
            if len(sentences) <= 2:
                compressed.append(chunk)
                continue

            sent_embeddings = self._encode(sentences)
            similarities = np.dot(sent_embeddings, query_emb) / (
                np.linalg.norm(sent_embeddings, axis=1) * np.linalg.norm(query_emb)
                + 1e-10  # guard against zero-norm edge case
            )

            relevant_sentences = [
                s for s, sim in zip(sentences, similarities)
                if sim >= self.threshold
            ]

            if relevant_sentences:
                chunk = dict(chunk)  # don't mutate the original
                chunk["content"] = ". ".join(relevant_sentences)
                chunk["compressed"] = True
                compressed.append(chunk)

        return compressed
