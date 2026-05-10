from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi


class HybridRetriever:
    """
    Core retriever used by ALL three modes.

    Bug fixes applied (2026-05-10 audit):
      BUG-003: _hydrate() now fetches all chunk IDs in a single batch call
               instead of N separate get_chunks_as_documents([cid]) calls.
      BUG-026: fetch_k_multiplier is a named constructor parameter (was magic 3).

    Strategy
    --------
    1. Dense  : Search EVERY active FAISS index (one per embedding dim).
    2. Sparse : BM25Okapi over the tokenised corpus.
    3. Fusion  : Reciprocal Rank Fusion (RRF).
    4. Hydrate : chunk_ids -> full chunk dicts via StorageManager (single batch).
    """

    def __init__(
        self,
        faiss_store,
        storage_manager,
        embedders: Dict[int, Any],          # {dim: EmbeddingPipeline}
        top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        fetch_k_multiplier: int = 3,        # BUG-026: named param, was hardcoded
    ):
        self.faiss_store        = faiss_store
        self.storage_manager    = storage_manager
        self.embedders          = embedders
        self.top_k              = top_k
        self.dense_weight       = dense_weight
        self.sparse_weight      = sparse_weight
        self.fetch_k_multiplier = fetch_k_multiplier  # BUG-026
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[List[str]]   = []
        self.corpus_ids: List[str]     = []

    # ------------------------------------------------------------------
    # Index maintenance
    # ------------------------------------------------------------------

    def build_sparse_index(self, chunks: List[Dict[str, Any]]):
        """
        (Re)build BM25 index.

        BUG-024 (partial): For large corpora consider incremental updates.
        This method is called after every ingest / delete, which is acceptable
        for corpora up to ~50k chunks.  For larger datasets, switch to an
        incremental append strategy.
        """
        if not chunks:
            return
        self.corpus     = [c["content"].split() for c in chunks]
        self.corpus_ids = [c["id"] for c in chunks]
        self.bm25       = BM25Okapi(self.corpus)

    # ------------------------------------------------------------------
    # Main retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        k       = top_k or self.top_k
        fetch_k = k * self.fetch_k_multiplier   # BUG-026

        # ── Dense ──────────────────────────────────────────────────────
        query_vectors: Dict[int, np.ndarray] = {}
        for dim, embedder in self.embedders.items():
            query_vectors[dim] = embedder.embed_query(query)

        raw_dense = self.faiss_store.search(query_vectors, k=fetch_k)

        dense_ranked: List[Dict] = []
        for dim, hits in raw_dense.items():
            for rank, (cid, score) in enumerate(hits):
                dense_ranked.append({"chunk_id": cid, "rank": rank,
                                     "score": score, "dim": dim})

        # ── Sparse ─────────────────────────────────────────────────────
        sparse_ranked: List[Dict] = []
        if self.bm25:
            scores  = self.bm25.get_scores(query.split())
            top_idx = np.argsort(scores)[-fetch_k:][::-1]
            for rank, idx in enumerate(top_idx):
                if scores[idx] > 0:
                    sparse_ranked.append({
                        "chunk_id": self.corpus_ids[idx],
                        "rank":     rank,
                        "score":    float(scores[idx]),
                    })

        # ── RRF Fusion ─────────────────────────────────────────────────
        fused = self._rrf_fuse(dense_ranked, sparse_ranked, k=60)

        # ── Hydrate (single batch) ─────────────────────────────────────
        return self._hydrate(fused[:k])

    # ------------------------------------------------------------------
    # RRF
    # ------------------------------------------------------------------

    def _rrf_fuse(
        self,
        dense: List[Dict],
        sparse: List[Dict],
        k: int = 60,
    ) -> List[Dict]:
        rrf: Dict[str, float] = {}

        best_dense_rank: Dict[str, int] = {}
        for item in dense:
            cid = item["chunk_id"]
            if cid not in best_dense_rank or item["rank"] < best_dense_rank[cid]:
                best_dense_rank[cid] = item["rank"]

        for cid, rank in best_dense_rank.items():
            rrf[cid] = rrf.get(cid, 0.0) + self.dense_weight / (k + rank + 1)

        for item in sparse:
            cid = item["chunk_id"]
            rrf[cid] = rrf.get(cid, 0.0) + self.sparse_weight / (k + item["rank"] + 1)

        return [
            {"chunk_id": cid, "rrf_score": score}
            for cid, score in sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        ]

    # ------------------------------------------------------------------
    # Hydration  (BUG-003: single batch call, not N individual calls)
    # ------------------------------------------------------------------

    def _hydrate(self, fused: List[Dict]) -> List[Dict]:
        """Resolve chunk_ids -> full content dicts via StorageManager (one batch)."""
        if not fused:
            return []

        ids  = [item["chunk_id"] for item in fused]
        docs = self.storage_manager.get_chunks_as_documents(ids)  # single call

        # Build lookup: doc id -> Document object
        doc_map: Dict[str, Any] = {}
        for doc in docs:
            doc_id = doc.metadata.get("id") or doc.metadata.get("chunk_id", "")
            if doc_id:
                doc_map[doc_id] = doc

        results = []
        for item in fused:
            cid = item["chunk_id"]
            doc = doc_map.get(cid)
            if doc:
                chunk = {
                    "id":      cid,
                    "content": doc.page_content,
                    **doc.metadata,
                }
            else:
                chunk = {"id": cid, "content": ""}
            chunk["rrf_score"]        = item["rrf_score"]
            chunk["retrieval_method"] = "hybrid"
            results.append(chunk)
        return results
