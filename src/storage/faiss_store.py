"""
faiss_store.py — Multi-dimensional FAISS store.

Design:
    One IndexIDMap2(IndexFlatIP) per embedding dimension.
    IndexIDMap2 + DirectMap.Hashtable enables O(1) deletion by FAISS internal ID.
    dim_registry maps: dim (int) → _DimIndex dataclass.
    Each index persisted at: <base_dir>/faiss_<dim>.index

LangChain integration:
    - Chunks are passed as List[Document] throughout (page_content + metadata).
    - add_documents() is the primary ingest API, mirroring LangChain VectorStore protocol.
    - similarity_search() returns List[Document] with score in metadata["score"].
    - LangChain CallbackManager is supported for observability hooks.

BUG-C05 (retained): get_internal_id() is the only public accessor for internal FAISS IDs.
"""
from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_BASE_DIR = "./data/vector_store"


@dataclass
class _DimIndex:
    """One FAISS index for a specific embedding dimension."""
    dim:      int
    index:    faiss.Index
    id_map:   Dict[str, int] = field(default_factory=dict)   # chunk_id → faiss int id
    rev_map:  Dict[int, str] = field(default_factory=dict)   # faiss int id → chunk_id
    _next_id: int = 0

    def next_id(self) -> int:
        fid = self._next_id
        self._next_id += 1
        return fid


class MultiFAISSStore:
    """
    Multi-dimensional FAISS vector store.

    Implements a subset of the LangChain VectorStore protocol:
        add_documents(docs, embeddings, dim)
        similarity_search(query_vectors, k) → List[Document]
        delete(chunk_ids, dim)

    Also exposes low-level add_chunks() / search() for pipeline compatibility.
    """

    def __init__(self, base_dir: str = _BASE_DIR):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self._indexes: Dict[int, _DimIndex] = {}
        self._load_all()

    # ── LangChain VectorStore protocol ────────────────────────────────────────

    def add_documents(
        self,
        documents:  List[Document],
        embeddings: List[List[float]],
        dim:        int,
    ) -> List[str]:
        """
        Add LangChain Documents with pre-computed embeddings.
        Returns list of chunk_ids stored.
        Mirrors langchain_core.vectorstores.VectorStore.add_documents().
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"documents ({len(documents)}) and embeddings ({len(embeddings)}) must match."
            )
        chunks = [
            {
                "id":        doc.metadata.get("chunk_id", doc.metadata.get("id", f"chunk_{i}")),
                "embedding": emb,
                "content":   doc.page_content,
                "metadata":  doc.metadata,
            }
            for i, (doc, emb) in enumerate(zip(documents, embeddings))
        ]
        self.add_chunks(chunks, dim)
        return [c["id"] for c in chunks]

    def similarity_search(
        self,
        query_vectors: Dict[int, np.ndarray],
        k:             int = 10,
    ) -> List[Document]:
        """
        Search across all active dims, RRF-fuse, return top-k as LangChain Documents.
        Score is stored in doc.metadata["score"].
        """
        raw = self.search(query_vectors, k=k)
        # RRF fusion
        scores: Dict[str, float] = {}
        for dim_results in raw.values():
            for rank, (chunk_id, score) in enumerate(dim_results):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (60 + rank + 1)

        top_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:k]
        docs = []
        for cid in top_ids:
            docs.append(Document(
                page_content=cid,   # caller resolves content from SQLiteManager
                metadata={"chunk_id": cid, "score": scores[cid]},
            ))
        return docs

    def delete(self, chunk_ids: List[str], dim: int) -> None:
        """LangChain VectorStore protocol: delete by IDs."""
        self.delete_chunks(chunk_ids, dim)

    # ── Low-level API (pipeline-compatible) ───────────────────────────────────

    def add_chunks(self, chunks: List[Dict[str, Any]], dim: int) -> None:
        """
        Add raw dicts [{id, embedding, ...}] to the index for *dim*.
        Preferred internal API — use add_documents() for LangChain Document flow.
        """
        if not chunks:
            return
        idx = self._get_or_create(dim)
        vectors, fids = [], []
        for chunk in chunks:
            emb = np.array(chunk["embedding"], dtype="float32")
            if emb.shape[0] != dim:
                raise ValueError(
                    f"Embedding dim mismatch: expected {dim}, got {emb.shape[0]}"
                )
            fid = idx.next_id()
            idx.id_map[chunk["id"]] = fid
            idx.rev_map[fid]        = chunk["id"]
            vectors.append(emb)
            fids.append(fid)

        mat = np.stack(vectors).astype("float32")
        faiss.normalize_L2(mat)
        idx.index.add_with_ids(mat, np.array(fids, dtype=np.int64))
        self._save(dim)
        logger.debug("[MultiFAISSStore] Added %d chunks to dim=%d", len(chunks), dim)

    def search(
        self,
        query_vectors: Dict[int, np.ndarray],
        k:             int = 10,
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Search each dim independently.
        Returns {dim: [(chunk_id, score), …]}.
        """
        results: Dict[int, List[Tuple[str, float]]] = {}
        for dim, qvec in query_vectors.items():
            if dim not in self._indexes:
                results[dim] = []
                continue
            idx = self._indexes[dim]
            if idx.index.ntotal == 0:
                results[dim] = []
                continue
            q = np.array(qvec, dtype="float32").reshape(1, -1)
            faiss.normalize_L2(q)
            actual_k = min(k, idx.index.ntotal)
            distances, indices = idx.index.search(q, actual_k)
            hits = []
            for fid, score in zip(indices[0], distances[0]):
                if fid < 0:
                    continue
                cid = idx.rev_map.get(int(fid))
                if cid:
                    hits.append((cid, float(score)))
            results[dim] = hits
        return results

    def delete_chunks(self, chunk_ids: List[str], dim: int) -> None:
        if dim not in self._indexes:
            return
        idx = self._indexes[dim]
        keep_cids = [cid for cid in idx.id_map.keys() if cid not in chunk_ids]
        if len(keep_cids) == len(idx.id_map):
            return
        
        if not keep_cids:
            hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            hnsw.hnsw.efConstruction = 64
            hnsw.hnsw.efSearch = 16
            idx.index = faiss.IndexIDMap2(hnsw)
            idx.index.set_direct_map(faiss.DirectMap.Hashtable)
            idx.id_map.clear()
            idx.rev_map.clear()
            idx._next_id = 0
        else:
            vectors = []
            new_id_map = {}
            new_rev_map = {}
            for i, cid in enumerate(keep_cids):
                fid = idx.id_map[cid]
                vec = idx.index.reconstruct(fid)
                vectors.append(vec)
                new_id_map[cid] = i
                new_rev_map[i] = cid
            
            hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            hnsw.hnsw.efConstruction = 64
            hnsw.hnsw.efSearch = 16
            new_index = faiss.IndexIDMap2(hnsw)
            new_index.set_direct_map(faiss.DirectMap.Hashtable)
            
            mat = np.stack(vectors).astype("float32")
            fids = np.array(range(len(keep_cids)), dtype=np.int64)
            new_index.add_with_ids(mat, fids)
            
            idx.index = new_index
            idx.id_map = new_id_map
            idx.rev_map = new_rev_map
            idx._next_id = len(keep_cids)
            
        self._save(dim)
        logger.info("[MultiFAISSStore] Deleted %d vectors from dim=%d", len(chunk_ids), dim)

    # BUG-C05: only public accessor for internal FAISS IDs
    def get_internal_id(self, chunk_id: str, dim: int) -> Optional[int]:
        """Return FAISS internal int64 ID for chunk_id in index dim, or None."""
        idx = self._indexes.get(dim)
        return idx.id_map.get(chunk_id) if idx else None

    def active_dims(self) -> List[int]:
        return [d for d, idx in self._indexes.items() if idx.index.ntotal > 0]

    def get_stats(self) -> Dict[str, Any]:
        return {
            str(d): {"total": idx.index.ntotal, "tracked": len(idx.id_map)}
            for d, idx in self._indexes.items()
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _index_path(self, dim: int) -> str:
        return os.path.join(self.base_dir, f"faiss_{dim}.index")

    def _meta_path(self, dim: int) -> str:
        return os.path.join(self.base_dir, f"faiss_{dim}_meta.pkl")

    def _get_or_create(self, dim: int) -> _DimIndex:
        if dim not in self._indexes:
            hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            hnsw.hnsw.efConstruction = 64
            hnsw.hnsw.efSearch = 16
            idmap = faiss.IndexIDMap2(hnsw)
            idmap.set_direct_map(faiss.DirectMap.Hashtable)
            self._indexes[dim] = _DimIndex(dim=dim, index=idmap)
        return self._indexes[dim]

    def _save(self, dim: int) -> None:
        idx = self._indexes[dim]
        faiss.write_index(idx.index, self._index_path(dim))
        with open(self._meta_path(dim), "wb") as f:
            pickle.dump(
                {
                    "id_map":   idx.id_map,
                    "rev_map":  idx.rev_map,
                    "next_id":  idx._next_id,
                },
                f,
            )

    def _load_all(self) -> None:
        for fname in os.listdir(self.base_dir):
            if not fname.startswith("faiss_") or not fname.endswith(".index"):
                continue
            try:
                dim   = int(fname.replace("faiss_", "").replace(".index", ""))
                index = faiss.read_index(self._index_path(dim))
                try:
                    index.set_direct_map(faiss.DirectMap.Hashtable)
                except Exception as dm_err:
                    logger.warning(
                        "[MultiFAISSStore] DirectMap unavailable dim=%d: %s", dim, dm_err
                    )
                meta_path = self._meta_path(dim)
                if os.path.exists(meta_path):
                    with open(meta_path, "rb") as f:
                        meta = pickle.load(f)
                    di = _DimIndex(
                        dim=dim, index=index,
                        id_map=meta.get("id_map", {}),
                        rev_map=meta.get("rev_map", {}),
                        _next_id=meta.get("next_id", 0),
                    )
                else:
                    di = _DimIndex(dim=dim, index=index)
                self._indexes[dim] = di
                logger.info("[MultiFAISSStore] Loaded dim=%d (%d vectors)", dim, index.ntotal)
            except Exception as exc:
                logger.warning("[MultiFAISSStore] Failed to load %s — %s", fname, exc)