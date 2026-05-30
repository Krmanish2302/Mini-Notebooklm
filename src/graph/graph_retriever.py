"""
graph_retriever.py — LangChain BaseRetriever subclass for Study Mode.

Traverses GraphStore to find concept relationships, prerequisite chains,
and learning paths for a given query.

Retrieval flow
--------------
1. NER       : extract entities (spaCy → noun-chunk → token fallback)
2. Seed      : find KG nodes matching entities via substring + cosine sim
3. BFS       : traverse edges sorted by _RELATION_PRIORITY
4. Hydrate   : resolve chunk_ids → List[Document] via GraphStore.get_documents()

LangChain integration:
    - Subclasses BaseRetriever — can be used directly in LCEL chains:
          chain = GraphRetriever(...) | format_docs | llm | StrOutputParser()
    - _get_relevant_documents() is the required abstract method.
    - get_learning_sequence() is a Study-Mode-specific bonus method.
    - get_related_concepts() returns enriched Documents with graph metadata.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .graph_store import GraphStore, _RELATION_PRIORITY

logger = logging.getLogger(__name__)


class GraphRetriever(BaseRetriever):
    """
    LangChain BaseRetriever over a GraphStore.

    Parameters
    ----------
    graph_store      : GraphStore
    embedder         : object with .encode(texts, normalize_embeddings=True) -> np.ndarray
    top_k            : max docs to return  (default 5)
    traversal_depth  : BFS hops            (default 2)
    sim_threshold    : cosine sim floor for seed node matching (default 0.45)
    """

    graph_store:     Any   # GraphStore — typed as Any to satisfy Pydantic v1 compat
    embedder:        Any
    top_k:           int   = 5
    traversal_depth: int   = 2
    sim_threshold:   float = 0.45

    class Config:
        arbitrary_types_allowed = True

    # ── LangChain protocol ────────────────────────────────────────────────────

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Required by BaseRetriever. Called by .invoke() and LCEL chains.
        Returns enriched Documents with concept/path/relation in metadata.
        """
        return self.get_related_concepts(query)

    # ── Public Study-Mode API ─────────────────────────────────────────────────

    def get_related_concepts(self, query: str) -> List[Document]:
        """
        Full graph retrieval: NER → seed → BFS → hydrate.
        Returns List[Document] with metadata: concept, path, relation, chunk_id.
        """
        g = self.graph_store.graph
        if g.number_of_nodes() == 0:
            return []

        entities   = self._extract_entities(query) or query.split()[:3]
        seed_nodes = self._find_seed_nodes(entities)
        if not seed_nodes:
            return []

        results: List[Document] = []
        seen_ids: set            = set()

        for seed_id, concept_label in seed_nodes:
            for nbr_id, path, relation in self._traverse(seed_id):
                if nbr_id in seen_ids:
                    continue
                seen_ids.add(nbr_id)
                docs = self.graph_store.get_documents([nbr_id])
                if docs:
                    doc = docs[0]
                    doc.metadata.update({
                        "concept":  concept_label,
                        "path":     path,
                        "relation": relation,
                    })
                    results.append(doc)

        return results[: self.top_k]

    def get_learning_sequence(
        self,
        start_concept: str,
        end_concept:   str,
    ) -> Optional[List[str]]:
        """
        Find shortest concept path between start and end.
        Returns list of chunk_id node names, or None if no path exists.
        """
        start_seeds = self._find_seed_nodes([start_concept])
        end_seeds   = self._find_seed_nodes([end_concept])
        if not start_seeds or not end_seeds:
            return None
        return self.graph_store.find_path(start_seeds[0][0], end_seeds[0][0])

    def get_learning_path_documents(
        self,
        start_concept: str,
        end_concept:   str,
    ) -> List[Document]:
        """
        Like get_learning_sequence() but returns hydrated Documents for the path.
        """
        path = self.get_learning_sequence(start_concept, end_concept)
        if not path:
            return []
        return self.graph_store.get_documents(path)

    # ── NER ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_entities(query: str) -> List[str]:
        try:
            import spacy  # type: ignore
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                nlp = spacy.load("en_core_web_md")
            doc      = nlp(query)
            entities = [e.text for e in doc.ents] or [c.text for c in doc.noun_chunks]
            return entities[:5]
        except Exception:
            pass
        _STOP = {
            "what","how","why","when","where","is","are","the","a","an",
            "of","in","to","and","or","for","with","does","do","explain",
            "describe","tell","me","about",
        }
        return [t.lower() for t in query.split() if t.lower() not in _STOP][:5]

    # ── Seed node matching ────────────────────────────────────────────────────

    def _find_seed_nodes(self, entities: List[str]) -> List[tuple]:
        g       = self.graph_store.graph
        matches = []
        seen: set = set()

        for entity in entities:
            el = entity.lower()
            # Pass 1 — exact substring match
            for node_id, data in g.nodes(data=True):
                if el in data.get("content", "").lower() and node_id not in seen:
                    seen.add(node_id)
                    matches.append((node_id, entity))

            # Pass 2 — cosine similarity on stored embeddings
            if not any(e == entity for _, e in matches):
                try:
                    emb      = np.array(
                        self.embedder.encode(entity, normalize_embeddings=True),
                        dtype="float32",
                    )
                    best_id, best_sim = None, -1.0
                    for node_id, data in g.nodes(data=True):
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
                    logger.debug("[GraphRetriever] Semantic seed search failed: %s", exc)

        return matches

    # ── BFS traversal ─────────────────────────────────────────────────────────

    def _traverse(self, seed_id: str) -> List[tuple]:
        """
        BFS from seed_id up to traversal_depth hops.
        Returns [(nbr_id, path, relation), …] sorted by _RELATION_PRIORITY.
        Skips low-priority relations (priority ≥ 99).
        """
        g        = self.graph_store.graph
        visited  = {seed_id}
        frontier = [(seed_id, [seed_id], "seed")]
        results: List[tuple] = []

        for _ in range(self.traversal_depth):
            next_frontier = []
            for node_id, path, _ in frontier:
                if node_id not in g:
                    continue
                for nbr, edata in g[node_id].items():
                    if nbr in visited:
                        continue
                    rel = edata.get("relation", "related")
                    if _RELATION_PRIORITY.get(rel, 50) >= 99:
                        continue
                    visited.add(nbr)
                    new_path = path + [nbr]
                    results.append((nbr, new_path, rel))
                    next_frontier.append((nbr, new_path, rel))
            frontier = next_frontier
            if not frontier:
                break

        results.sort(key=lambda x: _RELATION_PRIORITY.get(x[2], 50))
        return results