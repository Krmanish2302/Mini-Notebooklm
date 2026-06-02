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
    query_embedding TEXT,      -- JSON float list; User query embedding
    response_embedding TEXT,   -- JSON float list; Assistant response embedding
    created_at  REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ch_session ON chat_history(session_id);

CREATE TABLE IF NOT EXISTS chat_history_chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    turn_index  INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text  TEXT    NOT NULL,
    chunk_embedding TEXT NOT NULL,
    created_at  REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chc_session ON chat_history_chunks(session_id);
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
            with self._db._conn() as conn:
                conn.executescript(_DDL)
                # Run migration to add columns if they do not exist in pre-existing DBs
                try:
                    conn.execute("ALTER TABLE chat_history ADD COLUMN query_embedding TEXT")
                except Exception:
                    pass
                try:
                    conn.execute("ALTER TABLE chat_history ADD COLUMN response_embedding TEXT")
                except Exception:
                    pass
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

        q_emb      = self._embed_text(user_query)
        q_emb_json = json.dumps(q_emb) if q_emb else None

        r_emb      = self._embed_text(assistant_answer)
        r_emb_json = json.dumps(r_emb) if r_emb else None

        with self._db._conn() as conn:
            conn.execute(
                """
                INSERT INTO chat_history
                    (session_id, turn_index, user_query, assistant_answer, embedding, query_embedding, response_embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, turn_index, user_query, assistant_answer, emb_json, q_emb_json, r_emb_json, time.time()),
            )
            
        # Chunk assistant answer by sentences
        import re
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", assistant_answer) if s.strip()]
        if sentences:
            with self._db._conn() as conn:
                for idx, sentence in enumerate(sentences):
                    s_emb = self._embed_text(sentence)
                    s_emb_json = json.dumps(s_emb) if s_emb else None
                    if s_emb_json:
                        conn.execute(
                            """
                            INSERT INTO chat_history_chunks
                                (session_id, turn_index, chunk_index, chunk_text, chunk_embedding, created_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (session_id, turn_index, idx, sentence, s_emb_json, time.time()),
                        )
                        
        logger.debug(
            "[RAGHistoryStore] saved turn %d for session=%s and split into %d chunks", turn_index, session_id, len(sentences)
        )


    def _embed_text(self, text: str) -> Optional[List[float]]:
        if self._emb is None:
            return None
        try:
            return self._emb.embed_query(text)
        except Exception as exc:
            logger.warning("[RAGHistoryStore] text embed failed: %s", exc)
            return None

    def _next_index(self, session_id: str) -> int:
        with self._db._conn() as conn:
            row = conn.execute(
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
        with self._db._conn() as conn:
            rows = conn.execute(
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

    def retrieve_history_docs(
        self,
        session_id:    str,
        current_query: str,
        top_k:         int  = 2,
        fallback_last: int  = 2,
    ) -> List[Document]:
        """
        Retrieve top_k past turns formatted as Document objects containing only responses.
        Matched against response chunk embeddings in chat_history_chunks.
        """
        from langchain_core.documents import Document

        # Try to retrieve from chat_history_chunks table first
        chunk_rows = []
        try:
            with self._db._conn() as conn:
                chunk_rows = conn.execute(
                    """
                    SELECT turn_index, chunk_index, chunk_text, chunk_embedding
                    FROM   chat_history_chunks
                    WHERE  session_id = ?
                    ORDER  BY turn_index ASC, chunk_index ASC
                    """,
                    (session_id,),
                ).fetchall()
        except Exception as e:
            logger.warning("[RAGHistoryStore] Failed to query chat_history_chunks: %s", e)

        # If chunk_rows exist, score and return top chunks
        if chunk_rows and self._emb is not None:
            scored_chunks = []
            try:
                q_vec = self._emb.embed_query(current_query)
                for row in chunk_rows:
                    emb_json = row[3] # chunk_embedding
                    if not emb_json:
                        continue
                    try:
                        chunk_vec = json.loads(emb_json)
                        score = _cosine(q_vec, chunk_vec)
                        scored_chunks.append((row, score))
                    except Exception:
                        continue
            except Exception as exc:
                logger.warning("[RAGHistoryStore] retrieve_history_docs chunk scoring failed: %s", exc)

            if scored_chunks:
                top = sorted(scored_chunks, key=lambda x: x[1], reverse=True)[:top_k]
                # Re-sort chronologically by turn_index and chunk_index
                top.sort(key=lambda x: (x[0][0], x[0][1]))
                
                docs = []
                for row, score in top:
                    docs.append(Document(
                        page_content=row[2], # chunk_text
                        metadata={
                            "source_id": "history",
                            "source_name": "Chat History",
                            "is_history": True,
                            "turn_index": row[0],
                            "chunk_index": row[1],
                            "relevance_score": score
                        }
                    ))
                return docs

        # Fallback if chat_history_chunks is empty: load raw turns from chat_history and chunk on-the-fly
        try:
            with self._db._conn() as conn:
                rows = conn.execute(
                    """
                    SELECT turn_index, user_query, assistant_answer, response_embedding
                    FROM   chat_history
                    WHERE  session_id = ?
                    ORDER  BY turn_index ASC
                    """,
                    (session_id,),
                ).fetchall()
        except Exception:
            rows = []

        if not rows:
            return []

        # If we have turns, we split them into chunks on-the-fly and score them
        scored_chunks = []
        if self._emb is not None:
            try:
                q_vec = self._emb.embed_query(current_query)
                import re
                for row in rows:
                    turn_idx = row[0]
                    assistant_answer = row[2]
                    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", assistant_answer) if s.strip()]
                    for chunk_idx, sentence in enumerate(sentences):
                        s_emb = self._embed_text(sentence)
                        if s_emb:
                            score = _cosine(q_vec, s_emb)
                            scored_chunks.append((turn_idx, chunk_idx, sentence, score))
            except Exception as exc:
                logger.warning("[RAGHistoryStore] retrieve_history_docs fallback chunking/scoring failed: %s", exc)

        if scored_chunks:
            top = sorted(scored_chunks, key=lambda x: x[3], reverse=True)[:top_k]
            top.sort(key=lambda x: (x[0], x[1]))
            
            docs = []
            for turn_idx, chunk_idx, text, score in top:
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source_id": "history",
                        "source_name": "Chat History",
                        "is_history": True,
                        "turn_index": turn_idx,
                        "chunk_index": chunk_idx,
                        "relevance_score": score
                    }
                ))
            return docs

        # Ultimate fallback: return last turns as whole documents
        turns = rows[-fallback_last:]
        docs = []
        for row in turns:
            docs.append(Document(
                page_content=row[2], # assistant_answer
                metadata={
                    "source_id": "history",
                    "source_name": "Chat History",
                    "is_history": True,
                    "turn_index": row[0],
                }
            ))
        return docs


    def check_semantic_cache(
        self,
        session_id:    str,
        current_query: str,
        threshold:     float = 0.90,
    ) -> Optional[dict]:
        """
        Compare current_query against query_embedding of past turns.
        Returns mapped response + similarity score if matching >= threshold.
        """
        if self._emb is None:
            return None

        with self._db._conn() as conn:
            rows = conn.execute(
                """
                SELECT turn_index, user_query, assistant_answer, query_embedding
                FROM   chat_history
                WHERE  session_id = ?
                ORDER  BY turn_index ASC
                """,
                (session_id,),
            ).fetchall()

        if not rows:
            return None

        try:
            q_vec = self._emb.embed_query(current_query)
        except Exception:
            return None

        best_row = None
        best_score = -1.0

        for row in rows:
            emb_json = row[3] # query_embedding
            if not emb_json:
                continue
            try:
                past_q_vec = json.loads(emb_json)
                score = _cosine(q_vec, past_q_vec)
                if score > best_score:
                    best_score = score
                    best_row = row
            except Exception:
                continue

        if best_score >= threshold and best_row is not None:
            logger.info(
                "[RAGHistoryStore] Semantic cache hit! Similarity: %.4f (threshold: %.4f)",
                best_score, threshold
            )
            return {
                "answer": best_row[2], # assistant_answer
                "similarity": best_score,
                "user_query": best_row[1],
            }

        return None


    # ── Utilities ─────────────────────────────────────────────────────────

    def clear_session(self, session_id: str) -> None:
        with self._db._conn() as conn:
            conn.execute(
                "DELETE FROM chat_history WHERE session_id = ?", (session_id,)
            )
        logger.info("[RAGHistoryStore] cleared session=%s", session_id)

    def session_turns(self, session_id: str) -> List[dict]:
        """Return all turns for a session as list of dicts (for display)."""
        with self._db._conn() as conn:
            rows = conn.execute(
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
