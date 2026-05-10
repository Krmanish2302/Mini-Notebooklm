"""
GraphRetriever  —  Study Mode Path B retrieval.

Traverses the KnowledgeGraph to find concept relationships,
prerequisite chains, and learning paths for a given query.

Retrieval flow
--------------
1. NER  : extract entities using spaCy en_core_web_sm; falls back to
          noun-chunk heuristic, then plain token filtering.
2. Seed : find KG nodes matching entities via exact substring + cosine sim.
3. BFS  : traverse edges sorted by priority:
          prerequisite_of > causes > is_a_type_of > semantic > related
4. Hydrate: resolve chunk_ids → content via StorageManager.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_RELATION_PRIORITY = {
    "prerequisite_of": 0, "causes": 1, "is_a_type_of": 2,
    "raptor_parent_of": 3, "semantic": 4, "led_to": 5, "related": 6,
    "followed_by": 99, "mentions": 99,
}


class GraphRetriever:
    """
    Retrieve concept-relationship context for Study Mode.

    Parameters
    ----------
    knowledge_graph  : KnowledgeGraph
    storage_manager  : StorageManager
    embedder         : object with .encode(text) -> np.ndarray
    top_k            : int  (default 5)
    traversal_depth  : int  (default 2)
    sim_threshold    : float  (default 0.45)
    """

    def __init__(
        self, knowledge_graph, storage_manager, embedder,
        top_k: int = 5, traversal_depth: int = 2, sim_threshold: float = 0.45,
    ):
        self.kg = knowledge_graph
        self.storage_manager = storage_manager
        self.embedder = embedder
        self.top_k = top_k
        self.traversal_depth = traversal_depth
        self.sim_threshold = sim_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_related_concepts(self, query: str) -> List[Dict[str, Any]]:
        if not self.kg.graph.number_of_nodes():
            return []
        entities = self._extract_entities(query) or query.split()[:3]
        seed_nodes = self._find_seed_nodes(entities)
        if not seed_nodes:
            return []

        results: List[Dict[str, Any]] = []
        seen_ids: set = set()
        for seed_id, concept_label in seed_nodes:
            for nbr_id, path, relation in self._traverse(seed_id):
                if nbr_id in seen_ids:
                    continue
                seen_ids.add(nbr_id)
                chunk = self._hydrate(nbr_id)
                if chunk:
                    chunk["concept"] = concept_label
                    chunk["path"] = path
                    chunk["relation"] = relation
                    results.append(chunk)
        return results[: self.top_k]

    def get_learning_sequence(
        self, start_concept: str, end_concept: str
    ) -> Optional[List[str]]:
        start = self._find_seed_nodes([start_concept])
        end = self._find_seed_nodes([end_concept])
        if not start or not end:
            return None
        return self.kg.find_path(start[0][0], end[0][0])

    # ------------------------------------------------------------------
    # NER
    # ------------------------------------------------------------------

    def _extract_entities(self, query: str) -> List[str]:
        try:
            import spacy  # type: ignore
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                nlp = spacy.load("en_core_web_md")
            doc = nlp(query)
            entities = [e.text for e in doc.ents] or [c.text for c in doc.noun_chunks]
            return entities[:5]
        except Exception:
            pass
        stopwords = {
            "what","how","why","when","where","is","are","the","a","an",
            "of","in","to","and","or","for","with","does","do","explain",
            "describe","tell","me","about",
        }
        return [t.lower() for t in query.split() if t.lower() not in stopwords][:5]

    # ------------------------------------------------------------------
    # Seed node matching
    # ------------------------------------------------------------------

    def _find_seed_nodes(self, entities: List[str]) -> List[tuple]:
        matches: List[tuple] = []
        seen: set = set()
        for entity in entities:
            el = entity.lower()
            for node_id, data in self.kg.graph.nodes(data=True):
                if el in data.get("content", "").lower() and node_id not in seen:
                    seen.add(node_id)
                    matches.append((node_id, entity))
            if not any(e == entity for _, e in matches):
                try:
                    emb = self.embedder.encode(entity, normalize_embeddings=True)
                    best_id, best_sim = None, -1.0
                    for node_id, data in self.kg.graph.nodes(data=True):
                        if node_id in seen:
                            continue
                        node_emb = data.get("embedding")
                        if node_emb is None:
                            continue
                        sim = float(np.dot(emb, np.array(node_emb, dtype="float32")))
                        if sim > best_sim:
                            best_sim, best_id = sim, node_id
                    if best_id and best_sim >= self.sim_threshold:
                        seen.add(best_id)
                        matches.append((best_id, entity))
                except Exception as exc:
                    logger.debug("GraphRetriever seed semantic search failed: %s", exc)
        return matches

    # ------------------------------------------------------------------
    # BFS traversal
    # ------------------------------------------------------------------

    def _traverse(self, seed_id: str) -> List[tuple]:
        visited = {seed_id}
        frontier = [(seed_id, [seed_id], "seed")]
        results: List[tuple] = []
        for _ in range(self.traversal_depth):
            next_frontier = []
            for node_id, path, rel in frontier:
                for nbr, edata in self.kg.graph[node_id].items():
                    if nbr in visited:
                        continue
                    edge_rel = edata.get("relation", "related")
                    if _RELATION_PRIORITY.get(edge_rel, 50) >= 99:
                        continue
                    visited.add(nbr)
                    new_path = path + [nbr]
                    results.append((nbr, new_path, edge_rel))
                    next_frontier.append((nbr, new_path, edge_rel))
            frontier = next_frontier
            if not frontier:
                break
        results.sort(key=lambda x: _RELATION_PRIORITY.get(x[2], 50))
        return results

    # ------------------------------------------------------------------
    # Hydration
    # ------------------------------------------------------------------

    def _hydrate(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        try:
            docs = self.storage_manager.get_chunks_as_documents([chunk_id])
            if docs:
                return {"id": chunk_id, "content": docs[0].page_content, **docs[0].metadata}
        except Exception as exc:
            logger.debug("GraphRetriever._hydrate %s failed: %s", chunk_id, exc)
        return None
