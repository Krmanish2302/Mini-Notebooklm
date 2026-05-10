"""
RaptorBuilder  —  builds a RAPTOR summarisation tree at ingestion time.

RAPTOR (Recursive Abstractive Processing for Tree-Organised Retrieval)
-----------------------------------------------------------------------
Paper : https://arxiv.org/abs/2401.18059

How it works
------------
1. Start with leaf chunks (paragraph / sentence chunks).
2. Embed every leaf and cluster them with a lightweight k-means.
3. Ask the LLM to write a summary of each cluster  →  "Level-1 nodes".
4. Embed Level-1 summaries and repeat until only 1 cluster remains
   →  root node (whole-document abstract).
5. Store EVERY node (leaves + all summary levels) in:
     - FAISS  (via MultiFAISSStore, dim = embedding model chosen at ingest)
     - SQLite (chunks table, strategy='raptor_L{level}')
     - KnowledgeGraph  (with parent–child edges)

At query time
-------------
  RaptorRetriever.retrieve(query, top_k=2) does a normal dense FAISS
  search restricted to chunk_ids where strategy LIKE 'raptor_%'.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kmeans_cluster(embeddings: np.ndarray, n_clusters: int) -> List[List[int]]:
    rng = np.random.default_rng(42)
    centroid_idx = rng.choice(len(embeddings), size=n_clusters, replace=False)
    centroids = embeddings[centroid_idx].copy()

    for _ in range(30):
        dists = np.stack([np.linalg.norm(embeddings - c, axis=1) for c in centroids])
        labels = np.argmin(dists, axis=0)
        new_centroids = np.array([
            embeddings[labels == k].mean(axis=0) if (labels == k).any() else centroids[k]
            for k in range(n_clusters)
        ])
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    clusters: List[List[int]] = [[] for _ in range(n_clusters)]
    for i, lbl in enumerate(labels):
        clusters[lbl].append(i)
    return [c for c in clusters if c]


# ---------------------------------------------------------------------------
# RaptorBuilder
# ---------------------------------------------------------------------------

class RaptorBuilder:
    """
    Build the RAPTOR tree for a set of leaf chunks.

    Parameters
    ----------
    embedder        : object with .encode(text) -> np.ndarray
    llm             : callable(prompt: str) -> str
    storage_manager : StorageManager
    knowledge_graph : KnowledgeGraph
    source_id       : str
    faiss_dim       : int
    min_cluster_size: int  (default 3)
    max_levels      : int  (default 3)
    """

    STRATEGY_PREFIX = "raptor_L"

    def __init__(
        self,
        embedder,
        llm: Callable[[str], str],
        storage_manager,
        knowledge_graph,
        source_id: str,
        faiss_dim: int,
        min_cluster_size: int = 3,
        max_levels: int = 3,
    ):
        self.embedder = embedder
        self.llm = llm
        self.storage_manager = storage_manager
        self.kg = knowledge_graph
        self.source_id = source_id
        self.faiss_dim = faiss_dim
        self.min_cluster_size = min_cluster_size
        self.max_levels = max_levels

    def build(self, leaf_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build full RAPTOR tree. Returns all summary nodes created."""
        if not leaf_chunks:
            return []
        logger.info("RaptorBuilder: source=%s  leaves=%d", self.source_id, len(leaf_chunks))

        current_level_nodes = leaf_chunks
        current_level_ids = [c["id"] for c in current_level_nodes]
        all_summary_nodes: List[Dict[str, Any]] = []

        for level in range(1, self.max_levels + 1):
            if len(current_level_nodes) < self.min_cluster_size:
                logger.info("RaptorBuilder: stopping at level %d (too few nodes)", level)
                break
            summary_nodes, parent_map = self._build_level(current_level_nodes, level)
            if not summary_nodes:
                break
            self._store_summary_nodes(summary_nodes, parent_map, current_level_ids, level)
            all_summary_nodes.extend(summary_nodes)
            current_level_nodes = summary_nodes
            current_level_ids = [s["id"] for s in summary_nodes]

        logger.info("RaptorBuilder: complete  summary_nodes=%d", len(all_summary_nodes))
        return all_summary_nodes

    def _build_level(self, nodes: List[Dict[str, Any]], level: int):
        texts = [n["content"] for n in nodes]
        embeddings = np.stack(
            [self.embedder.encode(t, normalize_embeddings=True) for t in texts]
        )
        n_clusters = max(2, int(np.sqrt(len(nodes))))
        clusters = _kmeans_cluster(embeddings, n_clusters)

        summary_nodes: List[Dict[str, Any]] = []
        parent_map: Dict[str, List[str]] = {}

        for cluster_indices in clusters:
            child_nodes = [nodes[i] for i in cluster_indices]
            child_ids = [n["id"] for n in child_nodes]
            summary_text = self._summarise_cluster(child_nodes, level)
            if not summary_text:
                continue
            summary_id = f"raptor_{self.source_id}_L{level}_{uuid.uuid4().hex[:8]}"
            summary_embedding = self.embedder.encode(summary_text, normalize_embeddings=True)
            node: Dict[str, Any] = {
                "id": summary_id,
                "content": summary_text,
                "strategy": f"{self.STRATEGY_PREFIX}{level}",
                "source_id": self.source_id,
                "faiss_dim": self.faiss_dim,
                "embedding": summary_embedding,
                "token_count": len(summary_text.split()),
                "metadata": {"raptor_level": level, "child_count": len(child_ids), "child_ids": child_ids},
            }
            summary_nodes.append(node)
            parent_map[summary_id] = child_ids

        return summary_nodes, parent_map

    def _summarise_cluster(self, nodes: List[Dict[str, Any]], level: int) -> str:
        joined = "\n\n".join(n["content"] for n in nodes)
        instruction = "Write a detailed summary" if level == 1 else "Write a high-level abstract"
        prompt = (
            f"{instruction} of the following passages. "
            f"Be concise but preserve all key facts, entities and relationships. "
            f"Output ONLY the summary, no preamble.\n\n---\n{joined}\n---\n\nSUMMARY:"
        )
        try:
            return self.llm(prompt).strip()
        except Exception as exc:
            logger.warning("RaptorBuilder._summarise_cluster failed: %s", exc)
            return joined[:500] + " [...]"

    def _store_summary_nodes(self, summary_nodes, parent_map, child_ids, level):
        for node in summary_nodes:
            emb = node["embedding"].reshape(1, -1).astype("float32")
            self.storage_manager.faiss_store.add(
                vectors=emb, chunk_ids=[node["id"]], dim=self.faiss_dim
            )
            self.storage_manager.sqlite.insert_chunk(
                chunk_id=node["id"], source_id=self.source_id,
                content=node["content"], token_count=node["token_count"],
                strategy=node["strategy"], faiss_dim=self.faiss_dim,
                page_number=None, metadata=node["metadata"],
            )
            self.kg.add_chunk({
                "id": node["id"], "content": node["content"],
                "source_id": self.source_id, "modality": "raptor_summary",
                "embedding": node["embedding"].tolist(), "metadata": node["metadata"],
            })
            for child_id in parent_map.get(node["id"], []):
                self.kg.add_edge(node["id"], child_id, weight=1.0, relation="raptor_parent_of")


# ---------------------------------------------------------------------------
# RaptorRetriever
# ---------------------------------------------------------------------------

class RaptorRetriever:
    """Query-time: search FAISS restricted to raptor_* strategy nodes."""

    def __init__(self, faiss_store, storage_manager, embedder, faiss_dim: int):
        self.faiss_store = faiss_store
        self.storage_manager = storage_manager
        self.embedder = embedder
        self.faiss_dim = faiss_dim

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        query_vec = self.embedder.encode(query, normalize_embeddings=True)
        raw = self.faiss_store.search(
            query_vectors={self.faiss_dim: query_vec}, k=top_k * 5
        )
        hits = raw.get(self.faiss_dim, [])
        results: List[Dict[str, Any]] = []
        for chunk_id, score in hits:
            if len(results) >= top_k:
                break
            docs = self.storage_manager.get_chunks_as_documents([chunk_id])
            if not docs:
                continue
            meta = docs[0].metadata or {}
            if not meta.get("strategy", "").startswith("raptor_"):
                continue
            results.append({
                "id": chunk_id, "content": docs[0].page_content,
                "retrieval_method": "raptor_summary",
                "raptor_level": meta.get("raptor_level"),
                "score": score, **meta,
            })
        return results
