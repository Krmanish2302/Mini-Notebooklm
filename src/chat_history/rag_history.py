"""
rag_history.py — Vector-RAG chat history for Chat Mode and Deep Research Mode.

LangChain upgrade notes
-----------------------
* Storage   : LangChain InMemoryVectorStore + HuggingFaceEmbeddings
              (drop-in replaceable with Chroma / FAISS / Redis via same API)
* Retrieval : vectorstore.similarity_search_with_score()  — no manual np.dot
* Messages  : LangChain ChatMessageHistory (HumanMessage / AIMessage objects)
              exposed alongside the legacy dict API for backward compat.

Recency-anchor strategy (unchanged from original)
--------------------------------------------------
  get_relevant_history(query, k) returns:
    • top-k semantically similar past messages  (RAG pool)
    • last RECENCY_ANCHOR messages unconditionally
  Deduplicated, chronologically ordered.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

logger = logging.getLogger(__name__)

RECENCY_ANCHOR   = 2
DEFAULT_EMBED    = "all-MiniLM-L6-v2"


class RAGChatHistory:
    """
    Vector-based chat history for Chat Mode and Deep Research Mode.

    Parameters
    ----------
    session_id      : str
    embedding_model : HuggingFace model name (default: all-MiniLM-L6-v2)
    """

    def __init__(
        self,
        session_id:      str,
        embedding_model: str = DEFAULT_EMBED,
    ) -> None:
        self.session_id = session_id
        self._embedder  = HuggingFaceEmbeddings(model_name=embedding_model)
        self._store     = InMemoryVectorStore(embedding=self._embedder)
        self._lc_history = ChatMessageHistory()   # LangChain message objects
        self._messages:  List[Dict[str, Any]] = []  # legacy dict store
        self._doc_ids:   List[str]            = []  # parallel ids in vectorstore

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_message(
        self,
        role:         str,
        content:      str,
        sources_used: Optional[List[str]] = None,
    ) -> None:
        """Append message to both LangChain history and vector store."""
        idx = len(self._messages)
        msg: Dict[str, Any] = {
            "id":           f"{self.session_id}_{idx}",
            "session_id":   self.session_id,
            "role":         role,
            "content":      content,
            "sources_used": sources_used or [],
            "index":        idx,
        }
        self._messages.append(msg)

        # LangChain message history
        if role.lower() in ("user", "human"):
            self._lc_history.add_user_message(content)
        else:
            self._lc_history.add_ai_message(content)

        # Embed + store — metadata enables filtering
        doc_id = msg["id"]
        self._store.add_texts(
            texts=[content],
            metadatas=[{"role": role, "index": idx, "session_id": self.session_id}],
            ids=[doc_id],
        )
        self._doc_ids.append(doc_id)
        logger.debug("[RAGChatHistory] Added msg idx=%d role=%s", idx, role)

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_relevant_history(
        self,
        query: str,
        k:     int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid RAG + recency-anchor retrieval.

        Returns messages tagged with '_history_source': 'rag' | 'recency_anchor'.
        """
        if not self._messages:
            return []

        n            = len(self._messages)
        anchor_start = max(0, n - RECENCY_ANCHOR)
        anchor_idxs  = set(range(anchor_start, n))

        # RAG pool = everything except the recency anchor window
        rag_pool_size = anchor_start
        rag_idxs: List[int] = []
        if rag_pool_size > 0:
            effective_k = min(k, rag_pool_size)
            results = self._store.similarity_search_with_score(
                query, k=effective_k + RECENCY_ANCHOR
            )
            for doc, _score in results:
                idx = doc.metadata.get("index", -1)
                if idx != -1 and idx < anchor_start:
                    rag_idxs.append(idx)
                if len(rag_idxs) >= effective_k:
                    break

        seen: set  = set()
        final: List[Dict[str, Any]] = []
        for idx in rag_idxs + list(anchor_idxs):
            if idx not in seen and 0 <= idx < n:
                seen.add(idx)
                msg  = dict(self._messages[idx])
                msg["_history_source"] = (
                    "rag" if idx in set(rag_idxs) else "recency_anchor"
                )
                final.append(msg)

        # Chronological order
        final.sort(key=lambda m: m["index"])
        return final

    def format_for_prompt(self, query: str, k: int = 3) -> str:
        """Ready-to-inject string for the LLM prompt."""
        relevant = self.get_relevant_history(query, k=k)
        if not relevant:
            return ""
        lines = []
        for m in relevant:
            src = m.get("_history_source", "recency_anchor")
            tag = "[context]" if src == "rag" else "[recent]"
            lines.append(f"{tag} {m['role'].upper()}: {m['content']}")
        return "\n".join(lines)

    def get_recent_messages(self, n: int = 10) -> List[Dict[str, Any]]:
        return self.messages[-n:]

    # ── LangChain interop ─────────────────────────────────────────────────────

    @property
    def lc_messages(self) -> List[BaseMessage]:
        """LangChain BaseMessage list — for RunnableWithMessageHistory."""
        return self._lc_history.messages

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """Legacy dict list — backward compat with ChatHistoryManager."""
        return self._messages

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def clear(self) -> None:
        self._messages  = []
        self._doc_ids   = []
        self._store     = InMemoryVectorStore(embedding=self._embedder)
        self._lc_history.clear()
        logger.debug("[RAGChatHistory] Cleared session=%s", self.session_id)