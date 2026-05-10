"""Study Mode Pipeline

Retrieval strategy
------------------
* History  : Graph-based history — tracks CONCEPT NODES visited, not raw
             messages.  When you ask about "backpropagation" and then ask
             about "vanishing gradients", the system knows you've already
             covered prerequisite concepts and skips re-explaining them.

* Path A   : Full Deep Research pipeline
             (query expansion → hybrid ALL-dims → compress → rerank → RAPTOR)
             = quality, detail, factual grounding.

* Path B   : Graph Retrieval
             1. Extract entities from query.
             2. Find node(s) in KnowledgeGraph.
             3. Traverse prerequisite_of / causes / is_a_type_of edges.
             4. Return ordered concept path + chunk_ids for each node.
             = relationships, learning order, conceptual breadth.

* Fusion   : Smart Fusion
             top-6 from Path A (quality first) +
             up to 2 from Path B not already in A (relationship context).
             Deduplicated by chunk_id.

* Output   : chunks (for LLM) + learning_path (for UI to render as graph)

Chat History note
-----------------
Study mode uses GraphHistory (tracks concept nodes visited per session)
instead of RAGChatHistory.  This prevents re-teaching already-covered
concepts and builds a personalised learning path across the session.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional


class StudyPipeline:
    """
    Entry point for Study Mode.

    Parameters
    ----------
    deep_research_pipeline : DeepResearchPipeline
        Already-configured deep research pipeline (reused for Path A).
    graph_retriever        : GraphRetriever
        Traverses KnowledgeGraph for concept paths.
    graph_history          : GraphHistory
        Tracks visited concept nodes per session.
    llm                    : callable
    top_k_broad            : chunks from deep research to keep (default 6).
    top_k_graph            : graph relationship chunks to add (default 2).
    """

    # Max chunks from each path
    TOP_K_BROAD = 6
    TOP_K_GRAPH = 2

    def __init__(
        self,
        deep_research_pipeline,
        graph_retriever,
        graph_history,
        llm,
        top_k_broad: int = TOP_K_BROAD,
        top_k_graph: int = TOP_K_GRAPH,
    ):
        self.deep_pipeline = deep_research_pipeline
        self.graph_retriever = graph_retriever
        self.graph_history = graph_history
        self.llm = llm
        self.top_k_broad = top_k_broad
        self.top_k_graph = top_k_graph

    def run(self, query: str) -> Dict[str, Any]:
        """
        Full study turn.

        Returns
        -------
        {
          "answer"        : str,
          "sources"       : List[Dict],   # final merged chunks
          "learning_path" : List[Dict],   # concept relationship graph for UI
          "new_concepts"  : List[str],    # concepts introduced this turn
          "skipped_concepts": List[str],  # concepts already covered, skipped
        }
        """
        # ── Path A : Deep Research (quality + RAPTOR) ─────────────────
        # Run the full deep research pipeline — query expansion, hybrid
        # ALL-dims retrieval, contextual compression, cross-encoder rerank,
        # + optional RAPTOR summary nodes.
        dr_result = self.deep_pipeline.run(query)
        broad_chunks: List[Dict] = dr_result["sources"][: self.top_k_broad]
        broad_ids = {c["id"] for c in broad_chunks}

        # ── Path B : Graph Retrieval ──────────────────────────────────
        # find_related_concepts() traverses the KnowledgeGraph:
        #   1. NER: extract entities from query
        #   2. Node lookup in KnowledgeGraph
        #   3. Edge traversal: prerequisite_of, causes, is_a_type_of
        #   4. Return [{id, content, path, relation}, ...]
        graph_results: List[Dict] = self.graph_retriever.find_related_concepts(query)

        # ── Smart Fusion ──────────────────────────────────────────────
        final_chunks = self._smart_fusion(
            broad_chunks, graph_results, broad_ids
        )

        # ── Learning Path Extraction ──────────────────────────────────
        learning_path = self._extract_learning_path(graph_results)

        # ── Concept tracking (graph history) ─────────────────────────
        new_concepts, skipped = self._update_concept_history(graph_results, query)

        # ── Prompt with learning context ──────────────────────────────
        prompt = self._build_prompt(
            query, final_chunks, learning_path, new_concepts, skipped
        )
        answer = self.llm(prompt)

        return {
            "answer": answer,
            "sources": final_chunks,
            "learning_path": learning_path,
            "new_concepts": new_concepts,
            "skipped_concepts": skipped,
        }

    # ------------------------------------------------------------------
    # Smart Fusion
    # ------------------------------------------------------------------

    def _smart_fusion(
        self,
        broad: List[Dict],
        graph: List[Dict],
        broad_ids: set,
    ) -> List[Dict]:
        final = [c | {"source_path": "broad"} for c in broad]
        count = 0
        for chunk in graph:
            if chunk["id"] not in broad_ids and count < self.top_k_graph:
                final.append(chunk | {"source_path": "graph"})
                count += 1
        return final

    # ------------------------------------------------------------------
    # Learning Path
    # ------------------------------------------------------------------

    def _extract_learning_path(self, graph_results: List[Dict]) -> List[Dict]:
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

    # ------------------------------------------------------------------
    # Concept History
    # ------------------------------------------------------------------

    def _update_concept_history(
        self,
        graph_results: List[Dict],
        query: str,
    ):
        """Mark new concepts visited.  Return (new, skipped) lists."""
        concepts_this_turn = [
            r.get("concept", "") for r in graph_results if r.get("concept")
        ]
        new_concepts, skipped = [], []
        for concept in concepts_this_turn:
            if self.graph_history.has_visited(concept):
                skipped.append(concept)
            else:
                self.graph_history.mark_visited(concept, query)
                new_concepts.append(concept)
        return new_concepts, skipped

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        query: str,
        chunks: List[Dict],
        learning_path: List[Dict],
        new_concepts: List[str],
        skipped: List[str],
    ) -> str:
        context_block = "\n\n".join(
            f"[{'GRAPH-RELATION' if c.get('source_path') == 'graph' else 'Source'} "
            f"{i+1} | page {c.get('page_number','?')}]\n{c['content']}"
            for i, c in enumerate(chunks)
        )
        path_block = ""
        if learning_path:
            path_block = "\nLEARNING PATH:\n" + "\n".join(
                f"  {p['from']} → {' → '.join(p['steps'][1:-1]+[p['to']])} ({p['relationship']})"
                for p in learning_path
            )
        already_block = (
            f"\nCONCEPTS ALREADY COVERED THIS SESSION (do not re-explain): "
            f"{', '.join(skipped)}"
            if skipped else ""
        )
        new_block = (
            f"\nNEW CONCEPTS INTRODUCED THIS TURN: {', '.join(new_concepts)}"
            if new_concepts else ""
        )
        return (
            f"You are a patient study tutor. Teach using ONLY the provided sources. "
            f"Follow the learning path when it exists.{already_block}{new_block}\n"
            f"{path_block}\n\n"
            f"SOURCES:\n{context_block}\n\n"
            f"STUDENT QUESTION: {query}\n\n"
            f"TUTOR ANSWER:"
        )
