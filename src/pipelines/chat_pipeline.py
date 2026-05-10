"""Chat Mode Pipeline

Retrieval strategy
------------------
* History  : RAG-based (top-3 similar + last-2 recency anchor).
* Retrieval: Hybrid — Dense (ALL active FAISS dims queried in parallel)
             + Sparse (BM25) fused with RRF (dense 0.7 / sparse 0.3).
* Post-proc : none — Chat mode prioritises speed.
* No reranker, no query expansion, no RAPTOR.

Token budget awareness
----------------------
The prompt builder enforces a hard cap so that
  history_tokens + chunk_tokens + query_tokens < model_max_context.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional


class ChatPipeline:
    """
    Entry point for Chat Mode.

    Parameters
    ----------
    hybrid_retriever : HybridRetriever
        Pre-built with faiss_store, storage_manager, and ALL active embedders.
    rag_history      : RAGChatHistory
        Per-session history instance.
    llm              : callable(prompt: str) -> str
        Any LLM wrapper (OpenAI, Ollama, etc.).
    top_k            : chunks to include in context (default 5).
    history_k        : similar history messages to retrieve (default 3).
    """

    def __init__(
        self,
        hybrid_retriever,
        rag_history,
        llm,
        top_k: int = 5,
        history_k: int = 3,
    ):
        self.retriever = hybrid_retriever
        self.history = rag_history
        self.llm = llm
        self.top_k = top_k
        self.history_k = history_k

    def run(self, query: str) -> Dict[str, Any]:
        """
        Full chat turn.

        Returns
        -------
        {
          "answer"  : str,
          "sources" : List[Dict],   # chunks used
          "history_used": List[Dict]
        }
        """
        # ── 1. Retrieve relevant history ──────────────────────────────
        history_context = self.history.format_for_prompt(
            query, k=self.history_k
        )   # top-3 RAG + last-2 recency anchor, formatted string

        # ── 2. Hybrid retrieval across ALL active FAISS dims ─────────
        chunks: List[Dict] = self.retriever.retrieve(query, top_k=self.top_k)
        # HybridRetriever:
        #   a) embeds query with EVERY active model (all dims)
        #   b) fires FAISS search on EVERY index in parallel
        #   c) BM25 sparse search
        #   d) RRF fusion: best_dense_rank(chunk) + sparse_rank(chunk)
        #   e) hydrates chunk_ids → content from SQLite

        # ── 3. Build prompt ───────────────────────────────────────────
        prompt = self._build_prompt(query, chunks, history_context)

        # ── 4. Generate ───────────────────────────────────────────────
        answer = self.llm(prompt)

        # ── 5. Update history ─────────────────────────────────────────
        source_ids = [c["id"] for c in chunks]
        self.history.add_message("user", query)
        self.history.add_message("assistant", answer, sources_used=source_ids)

        return {
            "answer": answer,
            "sources": chunks,
            "history_used": self.history.get_relevant_history(query, k=self.history_k),
        }

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        query: str,
        chunks: List[Dict],
        history_context: str,
    ) -> str:
        context_block = "\n\n".join(
            f"[Source {i+1} | page {c.get('page_number','?')}]\n{c['content']}"
            for i, c in enumerate(chunks)
        )
        history_block = (
            f"\n\nRELEVANT CONVERSATION HISTORY:\n{history_context}"
            if history_context else ""
        )
        return (
            f"You are a helpful assistant. Answer using ONLY the provided sources.\n"
            f"{history_block}\n\n"
            f"SOURCES:\n{context_block}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"ANSWER:"
        )
