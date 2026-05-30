"""
rag_history_store.py — RAG-based conversation history store.

Strategy
--------
Every completed turn (user query + assistant answer) is:
  1. Saved to SQLite  (full text, for retrieval + display)
  2. Embedded + saved to a per-session FAISS index (for semantic search)

On each new query the N most semantically similar past turns are fetched
and serialised as:
    User: <past query>
    Assistant: <past answer>

This block is passed straight into PromptBuilder as `history`.
No ConversationBufferWindowMemory or MemorySaver is used anywhere.

Usage
-----
    store = RAGHistoryStore(sqlite_manager, embedding_model)
    store.add_turn(session_id, query, answer)
    history_text = store.retrieve_history(session_id, current_query, top_k=4)
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# ── SQLite DDL ─────────────────────────────────────────────────────────────────
_DDL = """
CREATE TABLE IF NOT EXISTS chat_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    turn_index  INTEGER NOT NULL,
    user_query  TEXT    NOT NULL,
    assistant_answer TEXT NOT NULL,
    embedding   TEXT,          -- JSON float list; NULL until embedded
    created_at  REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ch_session ON chat_history(session_id);
"""


class RAGHistoryStore:
    """
    Stores and retrieves conversation history using semantic search.

    Parameters
    ----------
    sqlite :  SQLiteManager  (must expose .conn: sqlite3.Connection)
    embedder: any object with .embed_query(text: str) -> List[float]
               e.g. LangChain Embeddings subclass
    """

    def __init__(self, sqlite: Any, embedder: Any) -> None:
        self._db      = sqlite
        self._emb     = embedder
        self._init_table()

    # ── Init ──────────────────────────────────────────────────────────────

    def _init_table(self) -> None:
        try:
            self._db.conn.executescript(_DDL)
            self._db.conn.commit()
            logger.info("[RAGHistoryStore] chat_history table ready")
        except Exception as exc:
            logger.error("[RAGHistoryStore] DDL failed: %s", exc)
            raise

    # ── Write ─────────────────────────────────────────────────────────────

    def add_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_answer: str,
    ) -> None:
        """
        Persist a completed turn and embed it for future retrieval.
        """
        turn_index = self._next_index(session_id)
        embedding  = self._embed_turn(user_query, assistant_answer)
        emb_json   = json.dumps(embedding) if embedding else None

        self._db.conn.execute(
            """
            INSERT INTO chat_history
                (session_id, turn_index, user_query, assistant_answer, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, turn_index, user_query, assistant_answer, emb_json, time.time()),
        )
        self._db.conn.commit()
        logger.debug(
            "[RAGHistoryStore] saved turn %d for session=%s", turn_index, session_id
        )

    def _next_index(self, session_id: str) -> int:
        row = self._db.conn.execute(
            "SELECT COALESCE(MAX(turn_index), -1) FROM chat_history WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return (row[0] + 1) if row else 0

    def _embed_turn(self, query: str, answer: str) -> Optional[List[float]]:
        """
        Embed the concatenation of query + answer for retrieval.
        Returns None on failure (turn still saved, just not retrievable semantically).
        """
        if self._emb is None:
            return None
        try:
            text = f"{query}\n{answer}"
            return self._emb.embed_query(text)
        except Exception as exc:
            logger.warning("[RAGHistoryStore] embed failed: %s", exc)
            return None

    # ── Read ──────────────────────────────────────────────────────────────

    def retrieve_history(
        self,
        session_id:    str,
        current_query: str,
        top_k:         int  = 4,
        fallback_last: int  = 3,
    ) -> str:
        """
        Return the top_k most-relevant past turns as a formatted string.

        Falls back to the most-recent `fallback_last` turns when:
          - embedder is None, or
          - no turns have embeddings yet.

        Format::
            User: <query>\nAssistant: <answer>\n
        """
        rows = self._db.conn.execute(
            """
            SELECT turn_index, user_query, assistant_answer, embedding
            FROM   chat_history
            WHERE  session_id = ?
            ORDER  BY turn_index ASC
            """,
            (session_id,),
        ).fetchall()

        if not rows:
            return ""

        # Try semantic retrieval
        scored = self._score_rows(rows, current_query)
        if scored:
            top = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
            # Re-sort by turn_index so history reads chronologically
            top.sort(key=lambda x: x[0][0])
            return self._serialise(top)

        # Fallback: most recent turns
        recent = rows[-fallback_last:]
        return "\n".join(
            f"User: {r[1]}\nAssistant: {r[2]}" for r in recent
        )

    def _score_rows(
        self,
        rows: list,
        current_query: str,
    ) -> List[tuple]:
        """
        Cosine-score each row's embedding against the current query.
        Returns list of (row, score) — empty if embedder/embeddings unavailable.
        """
        if self._emb is None:
            return []
        try:
            q_vec = self._emb.embed_query(current_query)
        except Exception:
            return []

        results = []
        for row in rows:
            emb_json = row[3]
            if not emb_json:
                continue
            try:
                turn_vec = json.loads(emb_json)
                score    = _cosine(q_vec, turn_vec)
                results.append((row, score))
            except Exception:
                continue
        return results

    @staticmethod
    def _serialise(scored_rows: list) -> str:
        parts = []
        for row, _ in scored_rows:
            parts.append(f"User: {row[0][1]}\nAssistant: {row[0][2]}")
        return "\n\n".join(parts)

    # ── Utilities ─────────────────────────────────────────────────────────

    def clear_session(self, session_id: str) -> None:
        self._db.conn.execute(
            "DELETE FROM chat_history WHERE session_id = ?", (session_id,)
        )
        self._db.conn.commit()
        logger.info("[RAGHistoryStore] cleared session=%s", session_id)

    def session_turns(self, session_id: str) -> List[dict]:
        """Return all turns for a session as list of dicts (for display)."""
        rows = self._db.conn.execute(
            """
            SELECT turn_index, user_query, assistant_answer, created_at
            FROM   chat_history
            WHERE  session_id = ?
            ORDER  BY turn_index ASC
            """,
            (session_id,),
        ).fetchall()
        return [
            {
                "turn":      r[0],
                "query":     r[1],
                "answer":    r[2],
                "timestamp": r[3],
            }
            for r in rows
        ]


# ── Cosine helper ──────────────────────────────────────────────────────────────

def _cosine(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    na   = sum(x * x for x in a) ** 0.5
    nb   = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
