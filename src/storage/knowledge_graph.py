"""
knowledge_graph.py  —  NetworkX knowledge graph over LangChain Documents

Design
------
Nodes : chunk_id  (str)  — attributes mirror LangChain Document fields
          content, metadata, source_id, modality
Edges : (chunk_a, chunk_b, weight=cosine_similarity)
          Added when similarity >= edge_threshold (default 0.75)
          Also added explicitly by StorageManager for cross-modal merges.

Public API
----------
    add_chunk(chunk)           — add/upsert a node
    add_edge(a, b, weight)     — add weighted edge
    get_neighbors(chunk_id, n) — BFS up to depth n
    find_path(src, dst)        — shortest path (Dijkstra, weight=1-w)
    remove_node(chunk_id)      — delete node + incident edges
    get_stats()                — node / edge counts
    export_pyvis(path)         — HTML visualisation (optional)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Chunk-level knowledge graph backed by NetworkX."""

    def __init__(self, edge_threshold: float = 0.75):
        self.graph: nx.Graph = nx.Graph()
        self.edge_threshold = edge_threshold

    # ── nodes ─────────────────────────────────────────────────────────────────

    def add_chunk(
        self,
        chunk: Dict[str, Any],
        *,
        auto_link: bool = False,
        existing_embeddings: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        """
        Add or update a chunk node.

        Parameters
        ----------
        chunk           : must have 'id'; 'content', 'metadata', 'source_id',
                          'modality', and optionally 'embedding' are stored.
        auto_link       : if True, compare against existing nodes and add
                          edges where cosine similarity >= edge_threshold.
        existing_embeddings: pre-loaded {chunk_id: embedding} dict; used when
                          auto_link=True to avoid re-scanning node attributes.
        """
        cid = chunk.get("id")
        if not cid:
            logger.warning("KnowledgeGraph.add_chunk: chunk missing 'id', skipping")
            return

        self.graph.add_node(
            cid,
            content=chunk.get("content", ""),
            metadata=chunk.get("metadata", {}),
            source_id=chunk.get("source_id", ""),
            modality=chunk.get("modality", "text"),
            embedding=chunk.get("embedding"),
            section=chunk.get("metadata", {}).get("section_heading", ""),
        )

        if auto_link and chunk.get("embedding") is not None:
            self._auto_link(cid, chunk["embedding"], existing_embeddings or {})

    def _auto_link(
        self,
        new_id: str,
        new_emb: List[float],
        existing: Dict[str, List[float]],
    ) -> None:
        """Add edges to nodes whose cosine similarity meets the threshold."""
        v = np.array(new_emb, dtype="float32")
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            return
        for nid, data in self.graph.nodes(data=True):
            if nid == new_id:
                continue
            emb = existing.get(nid) or data.get("embedding")
            if emb is None:
                continue
            u = np.array(emb, dtype="float32")
            norm_u = np.linalg.norm(u)
            if norm_u == 0:
                continue
            sim = float(np.dot(v, u) / (norm_v * norm_u))
            if sim >= self.edge_threshold:
                self.graph.add_edge(new_id, nid, weight=sim, relation="semantic")

    # ── edges ─────────────────────────────────────────────────────────────────

    def add_edge(
        self,
        chunk_id_a: str,
        chunk_id_b: str,
        weight: float = 1.0,
        relation: str = "related",
    ) -> None:
        """Add a weighted directed edge (stored as undirected)."""
        if chunk_id_a not in self.graph or chunk_id_b not in self.graph:
            logger.debug(
                "KnowledgeGraph.add_edge: one or both nodes missing (%s, %s)",
                chunk_id_a, chunk_id_b,
            )
            return
        self.graph.add_edge(chunk_id_a, chunk_id_b, weight=weight, relation=relation)

    # ── retrieval ─────────────────────────────────────────────────────────────

    def get_neighbors(
        self,
        chunk_id: str,
        depth: int = 1,
        min_weight: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        BFS neighbourhood up to *depth* hops.

        Returns list of node attribute dicts (with 'id' injected).
        Edges below *min_weight* are ignored.
        """
        if chunk_id not in self.graph:
            return []

        visited: set = {chunk_id}
        frontier: set = {chunk_id}
        result: List[Dict[str, Any]] = []

        for _ in range(depth):
            next_frontier: set = set()
            for node in frontier:
                for nbr, edata in self.graph[node].items():
                    if nbr in visited:
                        continue
                    if edata.get("weight", 1.0) >= min_weight:
                        next_frontier.add(nbr)
            for nbr in next_frontier:
                attrs = dict(self.graph.nodes[nbr])
                attrs["id"] = nbr
                result.append(attrs)
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

        return result

    def find_path(
        self,
        source_id: str,
        target_id: str,
    ) -> List[str]:
        """
        Shortest path between two chunk nodes.
        Uses Dijkstra with cost = 1 - weight (higher weight ⟹ shorter cost).
        Returns [] if no path exists.
        """
        if source_id not in self.graph or target_id not in self.graph:
            return []
        try:
            path = nx.shortest_path(
                self.graph,
                source=source_id,
                target=target_id,
                weight=lambda u, v, d: 1.0 - d.get("weight", 0.5),
            )
            return path
        except nx.NetworkXNoPath:
            return []

    def get_chunks_by_source(self, source_id: str) -> List[str]:
        """Return all chunk_ids belonging to a given source."""
        return [
            nid
            for nid, data in self.graph.nodes(data=True)
            if data.get("source_id") == source_id
        ]

    # ── mutation ──────────────────────────────────────────────────────────────

    def remove_node(self, chunk_id: str) -> None:
        if chunk_id in self.graph:
            self.graph.remove_node(chunk_id)

    def remove_source(self, source_id: str) -> int:
        """Remove all nodes belonging to *source_id*. Returns count removed."""
        to_remove = self.get_chunks_by_source(source_id)
        for cid in to_remove:
            self.graph.remove_node(cid)
        return len(to_remove)

    # ── stats / export ────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "components": nx.number_connected_components(self.graph),
        }

    def export_pyvis(self, output_path: str = "knowledge_graph.html") -> None:
        """
        Render an interactive Pyvis HTML visualisation.
        Silently skips if pyvis is not installed.
        """
        try:
            from pyvis.network import Network  # type: ignore
        except ImportError:
            logger.info("KnowledgeGraph.export_pyvis: pyvis not installed, skipping")
            return

        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
        for nid, data in self.graph.nodes(data=True):
            label = (data.get("content") or nid)[:40]
            net.add_node(nid, label=label, title=data.get("content", "")[:200])
        for u, v, edata in self.graph.edges(data=True):
            net.add_edge(u, v, value=edata.get("weight", 0.5))
        net.show(output_path, notebook=False)
        logger.info("KnowledgeGraph: exported visualisation → %s", output_path)
