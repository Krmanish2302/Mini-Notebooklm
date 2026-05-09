from typing import List, Dict, Any, Optional


class Reranker:
    """Cross-encoder reranker for precise scoring.

    The CrossEncoder model is lazy-loaded: it is NOT downloaded at
    __init__ time.  The first call to rerank() triggers the download.
    This prevents blocking app startup for users who never use reranking.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None  # lazy-loaded on first rerank() call

    def _load_model(self):
        """Download and cache the CrossEncoder on first use."""
        if self._model is None:
            from sentence_transformers import CrossEncoder  # deferred import
            self._model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rerank chunks by query relevance using the cross-encoder."""
        if not chunks:
            return []

        self._load_model()  # no-op after first call

        pairs = [(query, c["content"]) for c in chunks]
        scores = self._model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return chunks[:top_k]
