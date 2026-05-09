from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi


class HybridRetriever:
    """Combines FAISS (dense) + BM25 (sparse) with Reciprocal Rank Fusion."""

    def __init__(
        self,
        faiss_store,
        storage_manager=None,
        dim: int = 768,
        top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ):
        self.faiss_store = faiss_store
        self.storage_manager = storage_manager  # required for chunk hydration
        self.dim = dim
        self.top_k = top_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[List[str]] = []
        self.corpus_ids: List[str] = []

    def build_sparse_index(self, chunks: List[Dict[str, Any]]):
        """Build BM25 index from chunks."""
        self.corpus = [c["content"].split() for c in chunks]
        self.corpus_ids = [c["id"] for c in chunks]
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        dim: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using dense + sparse methods and return fused results.

        Args:
            query:           Raw query string (used for BM25).
            query_embedding: 1-D numpy array for the query.
            dim:             Embedding dimension key.  Falls back to self.dim.
        """
        embed_dim = dim if dim is not None else self.dim

        # ── Dense retrieval ────────────────────────────────────────────────
        # MultiFAISSStore.search() requires Dict[int, np.ndarray]
        raw_dense = self.faiss_store.search(
            query_vectors={embed_dim: query_embedding},
            k=self.top_k * 2,
        )
        # raw_dense → {dim: [(chunk_id, score), ...]}
        dense_hits = raw_dense.get(embed_dim, [])
        dense_results = [
            {"chunk_id": cid, "score": score, "method": "dense"}
            for cid, score in dense_hits
        ]

        # ── Sparse retrieval ───────────────────────────────────────────────
        sparse_results = []
        if self.bm25:
            tokenized = query.split()
            scores = self.bm25.get_scores(tokenized)
            top_indices = np.argsort(scores)[-(self.top_k * 2):][::-1]
            for idx in top_indices:
                if scores[idx] > 0:
                    sparse_results.append(
                        {
                            "chunk_id": self.corpus_ids[idx],
                            "score": float(scores[idx]),
                            "method": "sparse",
                        }
                    )

        # ── Reciprocal Rank Fusion ─────────────────────────────────────────
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)
        return fused[: self.top_k]

    def _reciprocal_rank_fusion(
        self,
        dense: List[Dict],
        sparse: List[Dict],
        k: int = 60,
    ) -> List[Dict]:
        """RRF: score = sum(weight / (k + rank))"""
        rrf_scores: Dict[str, float] = {}

        for rank, item in enumerate(dense):
            cid = item.get("chunk_id", item.get("id"))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.dense_weight / (k + rank + 1)

        for rank, item in enumerate(sparse):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.sparse_weight / (k + rank + 1)

        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # ── Hydrate chunk data via StorageManager ──────────────────────────
        results: List[Dict] = []
        for cid, score in sorted_ids:
            if self.storage_manager is not None:
                docs = self.storage_manager.get_chunks_as_documents([cid])
                if docs:
                    chunk = {
                        "id": cid,
                        "content": docs[0].page_content,
                        **docs[0].metadata,
                    }
                else:
                    chunk = {"id": cid}
            else:
                chunk = {"id": cid}

            chunk["score"] = score
            chunk["retrieval_method"] = "hybrid"
            results.append(chunk)

        return results
