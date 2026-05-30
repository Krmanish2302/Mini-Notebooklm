"""
graph_store.py — NetworkX DiGraph store for the knowledge graph.

Replaces graph_storage.py (renamed for clarity — storage/ already has
SQLiteManager; this is the graph topology layer).

Design:
    - Nodes  = chunk IDs (str).  Node attrs: content snippet, source_id,
               modality, embedding (List[float] optional), metadata dict.
    - Edges  = directed relationships.  Edge attrs: relation (str), weight (float).
    - Persisted to <graph_path> via pickle (safe: DiGraph is the full state).
    - auto_save=True (default) calls save() after every mutation.

LangChain integration:
    - get_documents(chunk_ids) returns List[Document] for pipeline use.
    - Node metadata is stored flat and returned in doc.metadata.
    - No LangChain class is subclassed here — GraphStore is a pure data store.
      LangChain retrieval protocol is handled by GraphRetriever (see below).
"""
from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import networkx as nx
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_DEFAULT_PATH = "./data/knowledge_graph/graph.pkl"

_RELATION_PRIORITY: Dict[str, int] = {
    "prerequisite_of": 0,
    "causes":          1,
    "is_a_type_of":    2,
    "raptor_parent_of": 3,
    "semantic":        4,
    "led_to":          5,
    "related":         6,
    "followed_by":     99,
    "mentions":        99,
}


class GraphStore:
    """
    Persistent NetworkX DiGraph store.

    Public API
    ----------
    add_node(chunk_id, **attrs)
    add_chunk(chunk)               ← dict-based ingest (pipeline compat)
    add_relationship(from_id, to_id, relation, weight)
    get_related(chunk_id, depth)   → List[Document]
    get_documents(chunk_ids)       → List[Document]
    find_path(start, end)          → Optional[List[str]]
    remove_node(chunk_id)
    remove_source(source_id)
    save() / load()
    get_stats()
    """

    def __init__(
        self,
        graph_path: str  = _DEFAULT_PATH,
        auto_save:  bool = True,
    ):
        self.graph_path = graph_path
        self.auto_save  = auto_save
        self.graph: nx.DiGraph = nx.DiGraph()
        self._load_or_create()

    # ── Node mutation ──────────────────────────────────────────────────────────

    def add_node(self, chunk_id: str, **attrs: Any) -> None:
        """
        Add or update a node.  Pass any keyword attrs — all land in node data.
        content is truncated to 300 chars for graph memory efficiency.
        """
        if "content" in attrs:
            attrs["content"] = attrs["content"][:300]
        self.graph.add_node(chunk_id, **attrs)
        if self.auto_save:
            self.save()

    def add_chunk(self, chunk: Dict[str, Any]) -> None:
        """
        Dict-based ingest API for pipeline compatibility.
        chunk keys: id (required), content, modality, source_id,
                    embedding, metadata (dict)
        """
        chunk_id = chunk.get("id") or chunk.get("chunk_id")
        if not chunk_id:
            raise ValueError("chunk must have 'id' or 'chunk_id'")
        self.add_node(
            chunk_id,
            content=chunk.get("content", "")[:300],
            modality=chunk.get("modality", "text"),
            source_id=chunk.get("source_id", ""),
            embedding=chunk.get("embedding"),
            metadata=chunk.get("metadata", {}),
        )

    def add_relationship(
        self,
        from_id:       str,
        to_id:         str,
        relation_type: str   = "related",
        weight:        float = 1.0,
    ) -> None:
        """Add a directed edge. Both nodes are created if absent."""
        if from_id not in self.graph:
            self.graph.add_node(from_id)
        if to_id not in self.graph:
            self.graph.add_node(to_id)
        self.graph.add_edge(from_id, to_id, relation=relation_type, weight=weight)
        if self.auto_save:
            self.save()

    def remove_node(self, chunk_id: str) -> None:
        if chunk_id in self.graph:
            self.graph.remove_node(chunk_id)
            if self.auto_save:
                self.save()

    def remove_source(self, source_id: str) -> int:
        """Remove all nodes belonging to source_id. Returns count removed."""
        to_remove = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("source_id") == source_id
        ]
        for n in to_remove:
            self.graph.remove_node(n)
        if to_remove and self.auto_save:
            self.save()
        logger.info("[GraphStore] Removed %d nodes for source_id=%s", len(to_remove), source_id)
        return len(to_remove)

    # ── Document retrieval ────────────────────────────────────────────────────

    def get_documents(self, chunk_ids: List[str]) -> List[Document]:
        """
        Return LangChain Documents for the given chunk_ids.
        Documents whose node doesn't exist are silently skipped.
        """
        docs = []
        for cid in chunk_ids:
            if cid not in self.graph:
                continue
            data = dict(self.graph.nodes[cid])
            content = data.pop("content", "")
            docs.append(Document(
                page_content=content,
                metadata={
                    "chunk_id":  cid,
                    "source_id": data.get("source_id", ""),
                    "modality":  data.get("modality", "text"),
                    **data.get("metadata", {}),
                },
            ))
        return docs

    def get_related(
        self,
        chunk_id: str,
        depth:    int = 1,
        exclude_low_priority: bool = True,
    ) -> List[Document]:
        """
        BFS to *depth* hops from chunk_id.
        Returns List[Document] with relation + weight in metadata.
        Low-priority relations (followed_by, mentions) skipped when
        exclude_low_priority=True.
        """
        if chunk_id not in self.graph:
            return []

        seen  = {chunk_id}
        docs  = []
        layer = [chunk_id]

        for _ in range(depth):
            next_layer = []
            for nid in layer:
                for nbr, edata in self.graph[nid].items():
                    if nbr in seen:
                        continue
                    rel = edata.get("relation", "related")
                    if exclude_low_priority and _RELATION_PRIORITY.get(rel, 50) >= 99:
                        continue
                    seen.add(nbr)
                    next_layer.append(nbr)
                    ndata = dict(self.graph.nodes[nbr])
                    content = ndata.pop("content", "")
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "chunk_id":  nbr,
                            "source_id": ndata.get("source_id", ""),
                            "relation":  rel,
                            "weight":    edata.get("weight", 1.0),
                            **ndata.get("metadata", {}),
                        },
                    ))
            layer = next_layer
            if not layer:
                break

        # Sort by relation priority
        docs.sort(
            key=lambda d: _RELATION_PRIORITY.get(d.metadata.get("relation", "related"), 50)
        )
        return docs

    # ── Path finding ──────────────────────────────────────────────────────────

    def find_path(self, start: str, end: str) -> Optional[List[str]]:
        """Shortest path between two nodes. Returns None if not reachable."""
        try:
            return nx.shortest_path(self.graph, start, end)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        with open(self.graph_path, "wb") as f:
            pickle.dump(self.graph, f)
        logger.debug("[GraphStore] Saved graph → %s", self.graph_path)

    def load(self) -> None:
        self._load_or_create()

    def _load_or_create(self) -> None:
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, "rb") as f:
                    self.graph = pickle.load(f)
                logger.info(
                    "[GraphStore] Loaded graph: %d nodes, %d edges",
                    self.graph.number_of_nodes(),
                    self.graph.number_of_edges(),
                )
            except Exception as exc:
                logger.warning("[GraphStore] Corrupt graph file, starting fresh: %s", exc)
                self.graph = nx.DiGraph()
        else:
            os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
            self.graph = nx.DiGraph()

    # ── Stats ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "nodes":   self.graph.number_of_nodes(),
            "edges":   self.graph.number_of_edges(),
            "sources": len({
                d.get("source_id")
                for _, d in self.graph.nodes(data=True)
                if d.get("source_id")
            }),
            "relation_counts": self._relation_counts(),
        }

    def _relation_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _, _, edata in self.graph.edges(data=True):
            r = edata.get("relation", "related")
            counts[r] = counts.get(r, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))