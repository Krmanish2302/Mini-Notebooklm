from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi


class HybridRetriever:
    """
    Core retriever used by ALL three modes.

    Strategy
    --------
    1. Dense  : Search EVERY active FAISS index (one per embedding dim).
                Each index was built with a different embedding model, so
                we embed the query with ALL registered models and fire one
                search per (dim, query_vector) pair — in parallel via a
                simple loop (FAISS releases the GIL; add ThreadPoolExecutor
                if you want true parallelism later).

    2. Sparse : BM25Okapi over the tokenised corpus of all stored chunks
                (built once after ingestion, rebuilt on source add/delete).

    3. Fusion  : Reciprocal Rank Fusion across ALL dense results + sparse
                 results.  RRF score = sum( weight / (k + rank) ) per chunk.
                 dense_weight=0.7, sparse_weight=0.3 (tunable).

    4. Hydrate : chunk_ids → full chunk dicts via StorageManager.
    """

    def __init__(
        self,
        faiss_store,          # MultiFAISSStore — holds all dim-keyed indexes
        storage_manager,      # StorageManager  — SQLite hydration
        embedders: Dict[int, Any],  # {dim: SentenceTransformer | OpenAIEmbedder}
        top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ):
        self.faiss_store = faiss_store
        self.storage_manager = storage_manager
        self.embedders = embedders          # ALL active embedding models keyed by dim
        self.top_k = top_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[List[str]] = []
        self.corpus_ids: List[str] = []

    # ------------------------------------------------------------------
    # Index maintenance
    # ------------------------------------------------------------------

    def build_sparse_index(self, chunks: List[Dict[str, Any]]):
        """(Re)build BM25 index.  Call after every ingestion or deletion."""
        self.corpus = [c["content"].split() for c in chunks]
        self.corpus_ids = [c["id"] for c in chunks]
        self.bm25 = BM25Okapi(self.corpus)

    # ------------------------------------------------------------------
    # Main retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Full hybrid retrieval.

        Steps
        -----
        1. Embed query with EVERY active model  → query_vectors {dim: vec}
        2. Dense search on EVERY FAISS index    → dense_hits {dim: [(id,score)]}
        3. BM25 search                          → sparse_hits [(id,score)]
        4. RRF fusion over all hits
        5. Hydrate top_k chunk_ids from SQLite
        """
        k = top_k or self.top_k
        fetch_k = k * 3   # over-fetch before fusion

        # ── Step 1 & 2 : Dense (all dims in parallel) ─────────────────────
        query_vectors: Dict[int, np.ndarray] = {}
        for dim, embedder in self.embedders.items():
            query_vectors[dim] = embedder.encode(query)

        raw_dense = self.faiss_store.search(query_vectors, k=fetch_k)
        # raw_dense → {dim: [(chunk_id, score), ...]}

        # Flatten: collect all (chunk_id, rank, weight) tuples for RRF
        dense_ranked: List[Dict] = []
        for dim, hits in raw_dense.items():
            for rank, (cid, score) in enumerate(hits):
                dense_ranked.append({"chunk_id": cid, "rank": rank,
                                     "score": score, "dim": dim})

        # ── Step 3 : Sparse ───────────────────────────────────────────────
        sparse_ranked: List[Dict] = []
        if self.bm25:
            scores = self.bm25.get_scores(query.split())
            top_idx = np.argsort(scores)[-fetch_k:][::-1]
            for rank, idx in enumerate(top_idx):
                if scores[idx] > 0:
                    sparse_ranked.append({"chunk_id": self.corpus_ids[idx],
                                          "rank": rank, "score": float(scores[idx])})

        # ── Step 4 : RRF Fusion ───────────────────────────────────────────
        fused = self._rrf_fuse(dense_ranked, sparse_ranked, k=60)

        # ── Step 5 : Hydrate ─────────────────────────────────────────────
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
        """
        RRF score = dense_weight/(k+rank_dense) + sparse_weight/(k+rank_sparse)

        For dense hits from multiple dims we keep the BEST rank per chunk_id
        across all dimensions (a chunk present in two indexes gets a bonus).
        """
        rrf: Dict[str, float] = {}

        # Aggregate dense: best rank per chunk across all active dims
        best_dense_rank: Dict[str, int] = {}
        for item in dense:
            cid = item["chunk_id"]
            if cid not in best_dense_rank or item["rank"] < best_dense_rank[cid]:
                best_dense_rank[cid] = item["rank"]

        for cid, rank in best_dense_rank.items():
            rrf[cid] = rrf.get(cid, 0.0) + self.dense_weight / (k + rank + 1)

        # Aggregate sparse
        for item in sparse:
            cid = item["chunk_id"]
            rrf[cid] = rrf.get(cid, 0.0) + self.sparse_weight / (k + item["rank"] + 1)

        return [
            {"chunk_id": cid, "rrf_score": score}
            for cid, score in sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        ]

    # ------------------------------------------------------------------
    # Hydration
    # ------------------------------------------------------------------

    def _hydrate(self, fused: List[Dict]) -> List[Dict]:
        """Resolve chunk_ids → full content dicts via StorageManager."""
        results = []
        for item in fused:
            cid = item["chunk_id"]
            docs = self.storage_manager.get_chunks_as_documents([cid])
            if docs:
                chunk = {
                    "id": cid,
                    "content": docs[0].page_content,
                    **docs[0].metadata,
                }
            else:
                chunk = {"id": cid, "content": ""}
            chunk["rrf_score"] = item["rrf_score"]
            chunk["retrieval_method"] = "hybrid"
            results.append(chunk)
        return results
