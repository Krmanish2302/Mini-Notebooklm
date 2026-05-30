"""
knowledge_graph.py — NetworkX knowledge graph over LangChain Documents.

Nodes : chunk_id (str) — attributes mirror LangChain Document fields:
          page_content, metadata, source_id, modality
Edges : (chunk_a, chunk_b, weight=cosine_similarity)
          Added when similarity >= edge_threshold (default 0.75)

LangChain integration:
    - add_document() accepts a LangChain Document directly.
    - get_neighbors_as_documents() returns List[Document].
    - All internal node attributes match Document field names
      (page_content, metadata) for drop-in compatibility.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Chunk-level semantic knowledge graph backed by NetworkX."""

    def __init__(self, edge_threshold: float = 0.75):
        self.graph: nx.Graph = nx.Graph()
        self.edge_threshold  = edge_threshold

    # ── Node management ───────────────────────────────────────────────────────

    def add_document(
        self,
        doc:        Document,
        chunk_id:   str,
        embedding:  Optional[List[float]] = None,
        auto_link:  bool = False,
    ) -> None:
        """
        Add a LangChain Document as a graph node.
        If auto_link=True, compute cosine similarity against all existing nodes
        and add edges where similarity >= edge_threshold.
        """
        self.graph.add_node(
            chunk_id,
            page_content = doc.page_content,
            metadata     = doc.metadata,
            source_id    = doc.metadata.get("source_id", ""),
            modality     = doc.metadata.get("modality", "text"),
            embedding    = embedding,
        )
        if auto_link and embedding:
            self._auto_link(chunk_id, embedding)

    def add_chunk(
        self,
        chunk:     Dict[str, Any],
        auto_link: bool = False,
    ) -> None:
        """
        Backward-compat: add a raw dict chunk as a node.
        Converts to Document internally.
        """
        doc = Document(
            page_content=chunk.get("content", ""),
            metadata={
                k: v for k, v in chunk.items()
                if k not in ("content", "embedding")
            },
        )
        self.add_document(
            doc=doc,
            chunk_id=chunk["id"],
            embedding=chunk.get("embedding"),
            auto_link=auto_link,
        )

    def add_edge(self, chunk_a: str, chunk_b: str, weight: float = 1.0) -> None:
        if self.graph.has_node(chunk_a) and self.graph.has_node(chunk_b):
            self.graph.add_edge(chunk_a, chunk_b, weight=weight)

    def remove_node(self, chunk_id: str) -> None:
        if self.graph.has_node(chunk_id):
            self.graph.remove_node(chunk_id)

    # ── Query API ─────────────────────────────────────────────────────────────

    def get_neighbors(self, chunk_id: str, depth: int = 1) -> List[str]:
        """BFS up to *depth* hops. Returns list of neighbour chunk_ids."""
        if not self.graph.has_node(chunk_id):
            return []
        visited  = {chunk_id}
        frontier = {chunk_id}
        for _ in range(depth):
            next_frontier: set = set()
            for node in frontier:
                for nbr in self.graph.neighbors(node):
                    if nbr not in visited:
                        visited.add(nbr)
                        next_frontier.add(nbr)
            frontier = next_frontier
        visited.discard(chunk_id)
        return list(visited)

    def get_neighbors_as_documents(
        self,
        chunk_id: str,
        depth:    int = 1,
    ) -> List[Document]:
        """Return neighbours as LangChain Documents."""
        nbr_ids = self.get_neighbors(chunk_id, depth)
        docs = []
        for nid in nbr_ids:
            attrs = self.graph.nodes[nid]
            docs.append(Document(
                page_content=attrs.get("page_content", ""),
                metadata=attrs.get("metadata", {"chunk_id": nid}),
            ))
        return docs

    def find_path(
        self,
        src: str,
        dst: str,
    ) -> Optional[List[str]]:
        """Shortest path (Dijkstra, weight=1-similarity). Returns None if unreachable."""
        if not (self.graph.has_node(src) and self.graph.has_node(dst)):
            return None
        try:
            return nx.shortest_path(
                self.graph, src, dst,
                weight=lambda u, v, d: 1.0 - d.get("weight", 0.0),
            )
        except nx.NetworkXNoPath:
            return None

    def get_subgraph_documents(
        self,
        chunk_ids: List[str],
    ) -> List[Document]:
        """Return LangChain Documents for a list of chunk_ids."""
        docs = []
        for cid in chunk_ids:
            if self.graph.has_node(cid):
                attrs = self.graph.nodes[cid]
                docs.append(Document(
                    page_content=attrs.get("page_content", ""),
                    metadata=attrs.get("metadata", {"chunk_id": cid}),
                ))
        return docs

    # ── Stats & export ────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
        }

    def export_pyvis(self, output_path: str = "knowledge_graph.html") -> str:
        """Export an interactive HTML visualisation (requires pyvis)."""
        try:
            from pyvis.network import Network
            net = Network(height="750px", width="100%", notebook=False)
            for node, attrs in self.graph.nodes(data=True):
                label = (attrs.get("page_content", node) or node)[:40]
                net.add_node(node, label=label, title=label)
            for u, v, data in self.graph.edges(data=True):
                net.add_edge(u, v, value=data.get("weight", 1.0))
            net.save_graph(output_path)
            return output_path
        except ImportError:
            logger.warning("[KnowledgeGraph] pyvis not installed — export skipped.")
            return ""

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        va, vb = np.array(a, dtype="float32"), np.array(b, dtype="float32")
        denom  = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 1e-9 else 0.0

    def _auto_link(self, new_id: str, embedding: List[float]) -> None:
        for node, attrs in self.graph.nodes(data=True):
            if node == new_id:
                continue
            emb = attrs.get("embedding")
            if emb is None:
                continue
            sim = self._cosine_similarity(embedding, emb)
            if sim >= self.edge_threshold:
                self.graph.add_edge(new_id, node, weight=sim)