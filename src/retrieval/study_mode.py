from typing import List, Dict, Any


class StudyModeRetriever:
    """
    Study Mode: Graph-Augmented Retrieval with Smart Fusion.

    Combines:
    - AdvancedRetriever  → broad, high-quality chunk retrieval
    - GraphRetriever     → relationship-aware concept traversal
    """

    def __init__(self, advanced_retriever, graph_retriever):
        self.advanced = advanced_retriever
        self.graph_retriever = graph_retriever

    def retrieve(self, query: str) -> Dict[str, Any]:
        """
        Returns
        -------
        {
            "chunks":        List[Dict]  — merged & de-duplicated chunks,
            "learning_path": List[Dict]  — concept relationship steps for UI
        }
        """
        # Step 1: Broad retrieval (AdvancedRetriever handles embedding internally)
        broad_results = self.advanced.retrieve(query)

        # Step 2: Graph retrieval (concept BFS traversal)
        graph_results = self.graph_retriever.retrieve(query)

        # Step 3: Smart fusion — quality first, relationships second
        final_chunks = self._smart_fusion(broad_results, graph_results)

        # Step 4: Extract learning path from graph results
        learning_path = self._extract_learning_path(graph_results)

        return {
            "chunks": final_chunks,
            "learning_path": learning_path,
        }

    # ------------------------------------------------------------------
    # Fusion helpers
    # ------------------------------------------------------------------

    def _smart_fusion(
        self,
        broad: List[Dict],
        graph: List[Dict],
        max_chunks: int = 8,
    ) -> List[Dict]:
        """
        Keep top-6 broad results (quality), then fill remaining slots with
        unique graph results (relationships) up to max_chunks.
        """
        seen: set = set()
        final: List[Dict] = []

        for chunk in broad[:6]:
            cid = chunk.get("id")
            if cid and cid not in seen:
                seen.add(cid)
                chunk["source"] = "broad"
                final.append(chunk)

        for chunk in graph:
            cid = chunk.get("id")
            if cid and cid not in seen and len(final) < max_chunks:
                seen.add(cid)
                chunk["source"] = "graph"
                final.append(chunk)

        return final

    def _extract_learning_path(self, graph_results: List[Dict]) -> List[Dict]:
        """Extract concept hop sequences for the learning-path UI panel."""
        paths = []
        for result in graph_results:
            if "path" in result and len(result["path"]) >= 2:
                paths.append({
                    "from": result["path"][0],
                    "to": result["path"][-1],
                    "steps": result["path"],
                    "relationship": result.get("relation", "related"),
                })
        return paths
