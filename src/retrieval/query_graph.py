"""
QueryGraphRetriever  —  thin wrapper for StudyPipeline.

Exposes the standard retrieve(query) interface so StudyPipeline
can swap it with HybridRetriever without any signature changes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class QueryGraphRetriever:
    """
    Parameters
    ----------
    graph_retriever : GraphRetriever
    """

    def __init__(self, graph_retriever):
        self.graph_retriever = graph_retriever

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Find related concepts + prerequisite paths for *query*."""
        return self.graph_retriever.find_related_concepts(query)

    def get_learning_sequence(
        self, start_concept: str, end_concept: str
    ) -> Optional[List[str]]:
        return self.graph_retriever.get_learning_sequence(start_concept, end_concept)
