from typing import List, Dict, Any, Optional
from .graph_storage import GraphStorage

class GraphRetriever:
    """Retrieves information using graph relationships."""
    
    def __init__(self, graph_storage: GraphStorage):
        self.graph = graph_storage
    
    def find_related_concepts(self, query: str, depth: int = 2) -> List[Dict[str, Any]]:
        """Find concepts related to query through graph traversal."""
        # In production, extract entities from query first
        # For now, search all nodes
        results = []
        
        for node_id in self.graph.graph.nodes():
            node_data = self.graph.graph.nodes[node_id]
            content = node_data.get("content", "").lower()
            
            # Simple keyword matching (replace with semantic search in production)
            if any(word in content for word in query.lower().split()[:3]):
                related = self.graph.get_related(node_id, depth=depth)
                results.append({
                    "id": node_id,
                    "content": node_data.get("content", ""),
                    "related": related,
                    "path": [node_id] + [r["chunk_id"] for r in related[:2]]
                })
        
        return results
    
    def get_learning_sequence(self, start_concept: str, end_concept: str) -> Optional[List[str]]:
        """Get learning sequence from concept A to concept B."""
        # Find nodes matching concepts
        start_nodes = self._find_nodes(start_concept)
        end_nodes = self._find_nodes(end_concept)
        
        if not start_nodes or not end_nodes:
            return None
        
        # Find shortest path
        path = self.graph.find_path(start_nodes[0], end_nodes[0])
        return path
    
    def _find_nodes(self, concept: str) -> List[str]:
        """Find nodes matching concept."""
        matches = []
        for node_id, data in self.graph.graph.nodes(data=True):
            if concept.lower() in data.get("content", "").lower():
                matches.append(node_id)
        return matches