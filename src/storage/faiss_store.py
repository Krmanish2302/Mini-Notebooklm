"""
faiss_store.py  —  Multi-dimensional FAISS store

Fixes applied
-------------
BUG-C05  Added public get_internal_id(chunk_id, dim) -> Optional[int] so
         StorageManager never accesses private _indexes or _DimIndex.id_map
         directly.  The old code would AttributeError on any index type that
         doesn't expose id_map as a Python dict.

Design:
    One IndexIDMap2(IndexFlatIP) per embedding dimension.
    IndexIDMap2 + DirectMap.Hashtable enables O(1) deletion by FAISS internal
    ID without scanning the full index.

    dim_registry maps:  dim (int) -> FAISSIndex dataclass
    Each index is persisted at:  <base_dir>/faiss_<dim>.index
"""
from __future__ import annotations

import os
import pickle
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)

_BASE_DIR = "./data/vector_store"


@dataclass
class _DimIndex:
    """One FAISS index for a specific embedding dimension."""
    dim: int
    index: faiss.Index
    id_map: Dict[str, int] = field(default_factory=dict)
    rev_map: Dict[int, str] = field(default_factory=dict)
    _next_id: int = 0

    def next_id(self) -> int:
        fid = self._next_id
        self._next_id += 1
        return fid


class MultiFAISSStore:
    """
    Multi-dimensional FAISS store backed by LangChain-compatible metadata.
    """

    def __init__(self, base_dir: str = _BASE_DIR):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self._indexes: Dict[int, _DimIndex] = {}
        self._load_all()

    # ── public API ────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Dict], dim: int) -> None:
        if not chunks:
            return
        idx = self._get_or_create(dim)
        vectors, ids = [], []
        for chunk in chunks:
            emb = np.array(chunk["embedding"], dtype="float32")
            if emb.shape[0] != dim:
                raise ValueError(
                    f"Embedding dim mismatch: expected {dim}, got {emb.shape[0]}"
                )
            fid = idx.next_id()
            idx.id_map[chunk["id"]] = fid
            idx.rev_map[fid] = chunk["id"]
            vectors.append(emb)
            ids.append(fid)

        mat = np.stack(vectors).astype("float32")
        faiss.normalize_L2(mat)
        idx.index.add_with_ids(mat, np.array(ids, dtype=np.int64))
        self._save(dim)
        logger.debug("MultiFAISSStore: added %d chunks to dim=%d", len(chunks), dim)

    def search(
        self,
        query_vectors: Dict[int, np.ndarray],
        k: int = 10,
    ) -> Dict[int, List[Tuple[str, float]]]:
        results: Dict[int, List[Tuple[str, float]]] = {}
        for dim, qvec in query_vectors.items():
            if dim not in self._indexes:
                results[dim] = []
                continue
            idx = self._indexes[dim]
            if idx.index.ntotal == 0:
                results[dim] = []
                continue
            q = qvec.astype("float32").reshape(1, -1)
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
        fids = [idx.id_map[cid] for cid in chunk_ids if cid in idx.id_map]
        if not fids:
            return
        sel = faiss.IDSelectorArray(np.array(fids, dtype=np.int64))
        idx.index.remove_ids(sel)
        for cid in chunk_ids:
            fid = idx.id_map.pop(cid, None)
            if fid is not None:
                idx.rev_map.pop(fid, None)
        self._save(dim)
        logger.info("MultiFAISSStore: deleted %d vectors from dim=%d", len(fids), dim)

    # BUG-C05: public method — never let callers reach into _indexes directly
    def get_internal_id(self, chunk_id: str, dim: int) -> Optional[int]:
        """
        Return the FAISS internal int64 ID for *chunk_id* in index *dim*,
        or None if not found.  Replaces the old pattern of directly accessing
        self._indexes[dim].id_map from StorageManager.
        """
        idx = self._indexes.get(dim)
        if idx is None:
            return None
        return idx.id_map.get(chunk_id)

    def active_dims(self) -> List[int]:
        return [d for d, idx in self._indexes.items() if idx.index.ntotal > 0]

    def get_stats(self) -> Dict:
        return {
            str(d): {"total": idx.index.ntotal, "tracked": len(idx.id_map)}
            for d, idx in self._indexes.items()
        }

    # ── persistence ───────────────────────────────────────────────────────────

    def _index_path(self, dim: int) -> str:
        return os.path.join(self.base_dir, f"faiss_{dim}.index")

    def _meta_path(self, dim: int) -> str:
        return os.path.join(self.base_dir, f"faiss_{dim}_meta.pkl")

    def _get_or_create(self, dim: int) -> _DimIndex:
        if dim not in self._indexes:
            flat = faiss.IndexFlatIP(dim)
            idmap = faiss.IndexIDMap2(flat)
            idmap.set_direct_map(faiss.DirectMap.Hashtable)
            self._indexes[dim] = _DimIndex(dim=dim, index=idmap)
        return self._indexes[dim]

    def _save(self, dim: int) -> None:
        idx = self._indexes[dim]
        faiss.write_index(idx.index, self._index_path(dim))
        with open(self._meta_path(dim), "wb") as f:
            pickle.dump(
                {"id_map": idx.id_map, "rev_map": idx.rev_map, "next_id": idx._next_id},
                f,
            )

    def _load_all(self) -> None:
        for fname in os.listdir(self.base_dir):
            if not fname.startswith("faiss_") or not fname.endswith(".index"):
                continue
            try:
                dim = int(fname.replace("faiss_", "").replace(".index", ""))
                index = faiss.read_index(self._index_path(dim))
                try:
                    index.set_direct_map(faiss.DirectMap.Hashtable)
                except Exception as dm_err:
                    logger.warning(
                        "MultiFAISSStore: could not set DirectMap for dim=%d: %s",
                        dim, dm_err,
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
                logger.info(
                    "MultiFAISSStore: loaded dim=%d (%d vectors)", dim, index.ntotal
                )
            except Exception as e:
                logger.warning("MultiFAISSStore: failed to load %s — %s", fname, e)
