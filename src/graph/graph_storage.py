import networkx as nx
import pickle
import os
from typing import List, Dict, Any, Optional

class GraphStorage:
    """NetworkX knowledge graph for relationships."""
    
    def __init__(self, graph_path: str = "./data/knowledge_graph/graph.pkl"):
        self.graph_path = graph_path
        self.graph = nx.DiGraph()
        self._load_or_create()
    
    def _load_or_create(self):
        if os.path.exists(self.graph_path):
            with open(self.graph_path, 'rb') as f:
                self.graph = pickle.load(f)
        else:
            os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
    
    def add_chunk(self, chunk: Dict[str, Any]):
        """Add chunk as node with metadata."""
        self.graph.add_node(
            chunk["id"],
            content=chunk["content"][:200],  # Truncate for storage
            modality=chunk.get("modality", "text"),
            source_id=chunk.get("source_id", ""),
            metadata=chunk.get("metadata", {})
        )
    
    def add_relationship(self, from_id: str, to_id: str, relation_type: str, weight: float = 1.0):
        """Add directed edge between chunks."""
        self.graph.add_edge(from_id, to_id, relation=relation_type, weight=weight)
    
    def get_related(self, chunk_id: str, depth: int = 1) -> List[Dict[str, Any]]:
        """Get related chunks up to N hops."""
        if chunk_id not in self.graph:
            return []
        
        related = []
        for neighbor in nx.single_source_shortest_path_length(self.graph, chunk_id, cutoff=depth):
            if neighbor != chunk_id:
                node_data = self.graph.nodes[neighbor]
                edge_data = self.graph.edges[chunk_id, neighbor] if self.graph.has_edge(chunk_id, neighbor) else {}
                related.append({
                    "chunk_id": neighbor,
                    "content": node_data.get("content", ""),
                    "relation": edge_data.get("relation", "related"),
                    "weight": edge_data.get("weight", 1.0)
                })
        
        return related
    
    def find_path(self, start: str, end: str) -> Optional[List[str]]:
        """Find shortest path between two chunks."""
        try:
            return nx.shortest_path(self.graph, start, end)
        except nx.NetworkXNoPath:
            return None
    
    def save(self):
        with open(self.graph_path, 'wb') as f:
            pickle.dump(self.graph, f)
    
    def get_stats(self) -> Dict[str, int]:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges()
        }