from typing import List, Dict, Any
import numpy as np

class StudyModeRetriever:
    """
    Study Mode: Graph-Augmented Retrieval with Smart Fusion.
    Combines broad retrieval (quality) + graph retrieval (relationships).
    """
    
    def __init__(self, advanced_retriever, graph_storage, graph_retriever):
        self.advanced = advanced_retriever
        self.graph_storage = graph_storage
        self.graph_retriever = graph_retriever
    
    def retrieve(self, query: str, query_embedding: np.ndarray) -> Dict[str, Any]:
        """
        Returns: {"chunks": List[Dict], "learning_path": List[Dict]}
        """
        # Step 1: Broad retrieval (Advanced Pipeline)
        broad_results = self.advanced.retrieve(query, query_embedding)
        
        # Step 2: Graph retrieval (find relationship paths)
        graph_results = self.graph_retriever.find_related_concepts(query)
        
        # Step 3: Smart Fusion
        final_chunks = self._smart_fusion(broad_results, graph_results)
        
        # Step 4: Extract learning path from graph
        learning_path = self._extract_learning_path(graph_results)
        
        return {
            "chunks": final_chunks,
            "learning_path": learning_path
        }
    
    def _smart_fusion(self, broad: List[Dict], graph: List[Dict], max_chunks: int = 8) -> List[Dict]:
        """
        Combine broad retrieval (quality) with graph (relationships).
        Priority: Keep top broad results, add graph relationships.
        """
        seen = set()
        final = []
        
        # Add top 6 from broad retrieval (quality first)
        for chunk in broad[:6]:
            cid = chunk["id"]
            if cid not in seen:
                seen.add(cid)
                chunk["source"] = "broad"
                final.append(chunk)
        
        # Add up to 3 from graph (relationships)
        for chunk in graph:
            cid = chunk["id"]
            if cid not in seen and len(final) < max_chunks:
                seen.add(cid)
                chunk["source"] = "graph"
                final.append(chunk)
        
        return final
    
    def _extract_learning_path(self, graph_results: List[Dict]) -> List[Dict]:
        """Extract concept relationships for learning path display."""
        paths = []
        for result in graph_results:
            if "path" in result:
                paths.append({
                    "from": result["path"][0],
                    "to": result["path"][-1],
                    "steps": result["path"],
                    "relationship": result.get("relation", "related")
                })
        return paths