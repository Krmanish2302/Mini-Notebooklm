"""
sqlite_manager.py — SQLite store for chunks, sources, and chat sessions.

Stores:
    chunks   — chunk_id, source_id, content, metadata (JSON), embedding_dim
    sources  — source_id, name, type, metadata, created_at, active flag
    sessions — session_id, created_at
    messages — message_id, session_id, role, content, created_at

LangChain integration:
    - Chunks are returned as List[Document] via get_documents_by_source().
    - get_chunk_as_document() returns a single Document for citation assembly.
    - Chat message history stored here; retrieved as List[BaseMessage] via
      get_session_messages_as_lc() for ConversationBufferWindowMemory hydration.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_DEFAULT_DB = "./data/metadata.db"

_DDL = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    source_id     TEXT NOT NULL,
    content       TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    embedding_dim INTEGER,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    source_id     TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    source_type   TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    active        INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_source    ON chunks(source_id);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteManager:
    """
    SQLite-backed store for chunks, sources, and chat sessions.

    All chunk retrieval methods return LangChain Documents.
    All message retrieval methods return List[BaseMessage].
    """

    def __init__(self, db_path: str = _DEFAULT_DB):
        self.db_path = db_path
        self._init_db()

    # ── Connection context manager ────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)
        logger.info("[SQLiteManager] Initialized db: %s", self.db_path)

    # ── Chunk API ─────────────────────────────────────────────────────────────

    def save_chunk(
        self,
        chunk_id:      str,
        source_id:     str,
        content:       str,
        metadata:      Optional[Dict[str, Any]] = None,
        embedding_dim: Optional[int]            = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO chunks (chunk_id, source_id, content, metadata_json, embedding_dim, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    content       = excluded.content,
                    metadata_json = excluded.metadata_json,
                    embedding_dim = excluded.embedding_dim
                """,
                (
                    chunk_id, source_id, content,
                    json.dumps(metadata or {}),
                    embedding_dim,
                    _now(),
                ),
            )

    def save_chunks_batch(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Bulk-insert chunks. Each dict: {chunk_id, source_id, content, metadata?, embedding_dim?}
        Returns number of rows inserted/updated.
        """
        rows = [
            (
                c["chunk_id"], c["source_id"], c["content"],
                json.dumps(c.get("metadata") or {}),
                c.get("embedding_dim"),
                _now(),
            )
            for c in chunks
        ]
        with self._conn() as conn:
            conn.executemany(
                """
                INSERT INTO chunks (chunk_id, source_id, content, metadata_json, embedding_dim, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    content       = excluded.content,
                    metadata_json = excluded.metadata_json,
                    embedding_dim = excluded.embedding_dim
                """,
                rows,
            )
        return len(rows)

    def get_chunk_content(self, chunk_id: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT content FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        return row["content"] if row else None

    def get_chunk_as_document(self, chunk_id: str) -> Optional[Document]:
        """Return a LangChain Document for a single chunk."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        if not row:
            return None
        meta = json.loads(row["metadata_json"] or "{}")
        meta.update({
            "chunk_id":     row["chunk_id"],
            "source_id":    row["source_id"],
            "embedding_dim": row["embedding_dim"],
        })
        return Document(page_content=row["content"], metadata=meta)

    def get_chunk_with_source(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve chunk details joined with source details (including source name)."""
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT c.chunk_id, c.source_id, c.content, c.metadata_json, s.name AS source_name
                FROM chunks c
                LEFT JOIN sources s ON c.source_id = s.source_id
                WHERE c.chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()
        if not row:
            return None
        meta = json.loads(row["metadata_json"] or "{}")
        return {
            "chunk_id":    row["chunk_id"],
            "source_id":   row["source_id"],
            "source_name": row["source_name"] or row["source_id"],
            "content":     row["content"],
            "metadata":    meta,
        }


    def get_documents_by_source(self, source_id: str) -> List[Document]:
        """Return all chunks for a source as LangChain Documents."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE source_id = ? ORDER BY created_at",
                (source_id,),
            ).fetchall()
        docs = []
        for row in rows:
            meta = json.loads(row["metadata_json"] or "{}")
            meta.update({
                "chunk_id":      row["chunk_id"],
                "source_id":     row["source_id"],
                "embedding_dim": row["embedding_dim"],
            })
            docs.append(Document(page_content=row["content"], metadata=meta))
        return docs

    def get_chunk_ids_by_source(self, source_id: str) -> List[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM chunks WHERE source_id = ?", (source_id,)
            ).fetchall()
        return [r["chunk_id"] for r in rows]

    def delete_chunks_by_source(self, source_id: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM chunks WHERE source_id = ?", (source_id,)
            )
        return cur.rowcount

    # ── Source API ────────────────────────────────────────────────────────────

    def save_source(
        self,
        source_id:   str,
        name:        str,
        source_type: str,
        metadata:    Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO sources (source_id, name, source_type, metadata_json, created_at, active)
                VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(source_id) DO UPDATE SET
                    name          = excluded.name,
                    source_type   = excluded.source_type,
                    metadata_json = excluded.metadata_json
                """,
                (source_id, name, source_type, json.dumps(metadata or {}), _now()),
            )

    def get_source(self, source_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sources WHERE source_id = ?", (source_id,)
            ).fetchone()
        if not row:
            return None
        meta = json.loads(row["metadata_json"] or "{}")
        return {
            "source_id":   row["source_id"],
            "name":        row["name"],
            "source_type": row["source_type"],
            "metadata":    meta,
            "created_at":  row["created_at"],
            "active":      bool(row["active"]),
        }

    def list_sources(self, active_only: bool = False) -> List[Dict[str, Any]]:
        query = "SELECT * FROM sources"
        if active_only:
            query += " WHERE active = 1"
        query += " ORDER BY created_at DESC"
        with self._conn() as conn:
            rows = conn.execute(query).fetchall()
        return [
            {
                "source_id":   r["source_id"],
                "name":        r["name"],
                "source_type": r["source_type"],
                "metadata":    json.loads(r["metadata_json"] or "{}"),
                "created_at":  r["created_at"],
                "active":      bool(r["active"]),
            }
            for r in rows
        ]

    def set_source_active(self, source_id: str, active: bool) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE sources SET active = ? WHERE source_id = ?",
                (int(active), source_id),
            )

    def delete_source(self, source_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM sources WHERE source_id = ?", (source_id,))

    # ── Session / Message API (LangChain message types) ───────────────────────

    def create_session(self, session_id: Optional[str] = None) -> str:
        sid = session_id or str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, created_at) VALUES (?, ?)",
                (sid, _now()),
            )
        return sid

    def save_message(
        self,
        session_id: str,
        role:       str,
        content:    str,
    ) -> str:
        """role: 'human' | 'ai' | 'system'"""
        mid = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages (message_id, session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (mid, session_id, role, content, _now()),
            )
        return mid

    def get_session_messages(self, session_id: str) -> List[Dict[str, str]]:
        """Return raw dicts: [{role, content, created_at}]."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM messages "
                "WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"], "created_at": r["created_at"]}
                for r in rows]

    def get_session_messages_as_lc(self, session_id: str) -> List[BaseMessage]:
        """
        Return session messages as LangChain BaseMessage objects.
        Use this to hydrate ConversationBufferWindowMemory.
        """
        rows = self.get_session_messages(session_id)
        messages: List[BaseMessage] = []
        for r in rows:
            role = r["role"].lower()
            if role == "human":
                messages.append(HumanMessage(content=r["content"]))
            elif role == "ai":
                messages.append(AIMessage(content=r["content"]))
            elif role == "system":
                messages.append(SystemMessage(content=r["content"]))
        return messages

    def delete_session(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        with self._conn() as conn:
            chunks   = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            sources  = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return {
            "chunks":   chunks,
            "sources":  sources,
            "sessions": sessions,
            "messages": messages,
        }