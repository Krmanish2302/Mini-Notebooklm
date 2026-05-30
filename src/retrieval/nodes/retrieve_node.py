"""
retrieve_node.py — LangGraph node: hybrid dense+sparse retrieval.

Fixes applied
-------------
* FIX #2 : source_ids read from state and forwarded to HybridRetriever.retrieve()
* FIX #5 : cap returned docs to top_k*2 to prevent 45-doc blowup without reranking
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def retrieve_docs(state: dict) -> dict:
    try:
        queries          = state.get("expanded_queries", [state.get("query", "")])
        vectorstore_path = state.get("vectorstore_path", "")
        top_k            = state.get("top_k", 5)
        # FIX #2: read source_ids from state so API source filtering works end-to-end
        source_ids       = state.get("source_ids") or None

        if not vectorstore_path:
            return {"error": "No vectorstore_path in state", "failed_node": "retrieve_docs"}

        from src.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(vectorstore_path, top_k=top_k * 3)

        seen, docs = set(), []
        for q in queries:
            # FIX #2: pass source_ids through to the retriever
            for doc in retriever.retrieve(q, top_k=top_k * 2, source_ids=source_ids):
                key = hash(doc.page_content[:200])
                if key not in seen:
                    seen.add(key)
                    docs.append(doc)

        # FIX #5: cap at top_k*2 so rerank=False path doesn't pass 45 docs to context
        docs = docs[: top_k * 2]

        logger.info("[retrieve_docs] Retrieved %d unique docs (source_ids=%s)", len(docs), source_ids)
        return {"documents": docs}
    except Exception as exc:
        logger.exception("[retrieve_docs] Failed: %s", exc)
        return {"error": str(exc), "failed_node": "retrieve_docs"}
