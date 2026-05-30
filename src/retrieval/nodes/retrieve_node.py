"""
retrieve_node.py — LangGraph node: hybrid dense+sparse retrieval.
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def retrieve_docs(state: dict) -> dict:
    try:
        queries          = state.get("expanded_queries", [state.get("query", "")])
        vectorstore_path = state.get("vectorstore_path", "")
        top_k            = state.get("top_k", 5)

        if not vectorstore_path:
            return {"error": "No vectorstore_path in state", "failed_node": "retrieve_docs"}

        from src.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(vectorstore_path, top_k=top_k * 3)

        seen, docs = set(), []
        for q in queries:
            for doc in retriever.retrieve(q, top_k=top_k * 2):
                key = hash(doc.page_content[:200])
                if key not in seen:
                    seen.add(key)
                    docs.append(doc)

        logger.info("[retrieve_docs] Retrieved %d unique docs", len(docs))
        return {"documents": docs}
    except Exception as exc:
        logger.exception("[retrieve_docs] Failed: %s", exc)
        return {"error": str(exc), "failed_node": "retrieve_docs"}