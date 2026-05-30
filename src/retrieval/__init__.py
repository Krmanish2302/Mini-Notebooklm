"""
src/retrieval/__init__.py

Public API for the retrieval package.

Usage:
    from src.retrieval import retrieve

    result = retrieve(
        query="What are the main findings?",
        vectorstore_path="data/vectorstores/report_001",
    )
    print(result["context"])   # formatted context string for LLM
    print(result["documents"]) # List[Document] of parent chunks
"""
from .retrieval_graph  import retrieval_app    # noqa: F401
from .state            import RetrievalState   # noqa: F401
from .hybrid_retriever import HybridRetriever  # noqa: F401


def retrieve(
    query:            str,
    vectorstore_path: str,
    top_k:            int  = 5,
    use_rerank:       bool = True,
    use_compression:  bool = False,
    do_expand:        bool = True,   # BUG-RET-06: renamed from expand_query
) -> dict:
    """One-line entry point. Returns final RetrievalState dict."""
    return retrieval_app.invoke(RetrievalState(
        query=query,
        vectorstore_path=vectorstore_path,
        top_k=top_k,
        use_rerank=use_rerank,
        use_compression=use_compression,
        do_expand=do_expand,
    ))


__all__ = ["retrieve", "retrieval_app", "RetrievalState", "HybridRetriever"]
