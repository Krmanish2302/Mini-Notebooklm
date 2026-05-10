from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGChatHistory:
    """
    Vector-based chat history for Chat Mode and Deep Research Mode.

    Strategy (hybrid: RAG + recency anchor)
    ----------------------------------------
    * Every message is embedded and stored in-memory.
    * get_relevant_history() returns:
        - top-K semantically similar past messages  (RAG)
        - last-2 messages unconditionally           (recency anchor)
      This keeps context tight (never > K+2 messages) while ensuring
      the model is never blind to the most recent turn.

    Why in-memory (not a separate FAISS index)?
    --------------------------------------------
    Sessions are ephemeral.  Cosine-sim over 50-200 messages is < 1 ms.
    A per-session FAISS index would add disk I/O for no measurable gain.
    For cross-session long-term memory that's a separate feature (MemoryStore).
    """

    RECENCY_ANCHOR = 2   # always include last N messages regardless of similarity

    def __init__(
        self,
        session_id: str,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.session_id = session_id
        self.messages: List[Dict[str, Any]] = []
        self.embedder = SentenceTransformer(embedding_model)
        self._embeddings: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_message(
        self,
        role: str,           # "user" | "assistant"
        content: str,
        sources_used: Optional[List[str]] = None,
    ):
        """Append message and cache its embedding."""
        msg: Dict[str, Any] = {
            "id": f"{self.session_id}_{len(self.messages)}",
            "session_id": self.session_id,
            "role": role,
            "content": content,
            "sources_used": sources_used or [],
            "index": len(self.messages),
        }
        self.messages.append(msg)
        self._embeddings.append(self.embedder.encode(content, normalize_embeddings=True))

    # ------------------------------------------------------------------
    # Read — used by all three modes before building LLM prompt
    # ------------------------------------------------------------------

    def get_relevant_history(
        self,
        query: str,
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Returns up to k+RECENCY_ANCHOR messages, deduplicated.

        Order: [RAG-retrieved (most-similar first)] + [recent anchor]
        The caller formats these into the prompt.
        """
        if not self.messages:
            return []

        n = len(self.messages)
        query_emb = self.embedder.encode(query, normalize_embeddings=True)

        # Cosine similarity (vectors are L2-normalised → just dot product)
        sims = np.array([float(np.dot(query_emb, e)) for e in self._embeddings])

        # Exclude the very last RECENCY_ANCHOR messages from RAG pool
        # (they are added unconditionally below)
        anchor_start = max(0, n - self.RECENCY_ANCHOR)
        rag_pool_mask = np.ones(n, dtype=bool)
        rag_pool_mask[anchor_start:] = False

        rag_indices: List[int] = []
        if rag_pool_mask.any():
            pool_sims = sims.copy()
            pool_sims[~rag_pool_mask] = -999
            top_k = min(k, rag_pool_mask.sum())
            rag_indices = list(np.argsort(pool_sims)[-top_k:][::-1])

        anchor_indices = list(range(anchor_start, n))

        # Merge, deduplicate, preserve order (RAG then recency)
        seen = set()
        final: List[Dict[str, Any]] = []
        for idx in rag_indices + anchor_indices:
            if idx not in seen:
                seen.add(idx)
                msg = dict(self.messages[idx])
                msg["_history_source"] = (
                    "rag" if idx in rag_indices else "recency_anchor"
                )
                final.append(msg)

        return final

    def format_for_prompt(self, query: str, k: int = 3) -> str:
        """Ready-to-inject string for the LLM system/user prompt."""
        relevant = self.get_relevant_history(query, k=k)
        if not relevant:
            return ""
        lines = []
        for m in relevant:
            tag = "[context]" if m["_history_source"] == "rag" else "[recent]"
            lines.append(f"{tag} {m['role'].upper()}: {m['content']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_recent_messages(self, n: int = 10) -> List[Dict[str, Any]]:
        return self.messages[-n:]

    def clear(self):
        self.messages = []
        self._embeddings = []
