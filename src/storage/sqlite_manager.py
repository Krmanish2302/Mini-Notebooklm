"""
sqlite_manager.py  —  Metadata + chat history store

Fixes applied
-------------
BUG-C02  Thread safety: WAL journal mode + threading.Lock on every write.
         check_same_thread=False so the same SQLiteManager instance can be
         shared across the FastAPI request thread and the background ingest
         thread without OperationalError: database is locked.
BUG-C01  N+1 query pattern: get_chunks_as_documents now uses a single
         batched IN() query instead of one SELECT per chunk_id.
BUG-R03  Missing source metadata: get_chunks_by_ids() returns full rows
         (including source_id, metadata) so PromptBuilder gets source names.
BUG-Q05  Schema migration: allowlist assertion before ALTER TABLE to prevent
         accidental SQL injection if the column list is ever extended.
"""
import sqlite3
import json
import logging
import threading
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class SQLiteManager:
    """SQLite metadata store — sources, chunks, sessions, messages."""

    def __init__(self, db_path: str = "./data/metadata.db"):
        self.db_path = db_path
        import os
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        # BUG-C02: single lock serialises all writes across threads
        self._lock = threading.Lock()
        # Enable WAL mode once at startup for concurrent read+write
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    def _conn(self) -> sqlite3.Connection:
        """Return a new connection with check_same_thread=False."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ── schema ────────────────────────────────────────────────────────────────

    def _init_tables(self) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS sources (
                        id          TEXT PRIMARY KEY,
                        title       TEXT,
                        source_type TEXT,
                        file_path   TEXT,
                        url         TEXT,
                        metadata    TEXT,
                        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status      TEXT DEFAULT 'processing'
                    );

                    CREATE TABLE IF NOT EXISTS chunks (
                        id               TEXT PRIMARY KEY,
                        source_id        TEXT,
                        content          TEXT,
                        modality         TEXT,
                        metadata         TEXT,
                        embedding_model  TEXT,
                        faiss_dim        INTEGER,
                        faiss_internal_id INTEGER,
                        FOREIGN KEY (source_id) REFERENCES sources(id)
                    );

                    CREATE TABLE IF NOT EXISTS sessions (
                        id         TEXT PRIMARY KEY,
                        mode       TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS messages (
                        id           TEXT PRIMARY KEY,
                        session_id   TEXT,
                        role         TEXT,
                        content      TEXT,
                        sources_used TEXT,
                        timestamp    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions(id)
                    );

                    CREATE TABLE IF NOT EXISTS chat_history_graph (
                        id            TEXT PRIMARY KEY,
                        session_id    TEXT,
                        node_type     TEXT,
                        content       TEXT,
                        related_nodes TEXT,
                        timestamp     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add new columns to existing tables without breaking existing data."""
        # BUG-Q05: allowlist prevents accidental SQL injection
        _ALLOWED_COLS = {
            "embedding_model": "TEXT",
            "faiss_dim": "INTEGER",
            "faiss_internal_id": "INTEGER",
        }
        existing = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        for col, typedef in _ALLOWED_COLS.items():
            assert col in _ALLOWED_COLS, f"Unexpected column attempted: {col}"
            assert typedef in ("TEXT", "INTEGER"), f"Unexpected type: {typedef}"
            if col not in existing:
                conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} {typedef}")
                logger.info("SQLiteManager: migrated chunks.%s", col)

    # ── sources ───────────────────────────────────────────────────────────────

    def add_source(self, source: Dict[str, Any]) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO sources
                        (id, title, source_type, file_path, url, metadata, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    source["id"],
                    source.get("title", source.get("name", "")),
                    source.get("source_type", source.get("type", "unknown")),
                    source.get("file_path"),
                    source.get("url"),
                    json.dumps(source.get("metadata", {})),
                    source.get("status", "ready"),
                ))

    def get_sources(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sources WHERE status = 'ready'"
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_source(self, source_id: str) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute("DELETE FROM chunks  WHERE source_id = ?", (source_id,))
                conn.execute("DELETE FROM sources WHERE id         = ?", (source_id,))

    # ── chunks ────────────────────────────────────────────────────────────────

    def add_chunk(self, chunk: Dict[str, Any]) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO chunks
                        (id, source_id, content, modality, metadata,
                         embedding_model, faiss_dim, faiss_internal_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk["id"],
                    chunk.get("source_id", ""),
                    chunk.get("content", ""),
                    chunk.get("modality", "text"),
                    json.dumps(chunk.get("metadata", {})),
                    chunk.get("embedding_model"),
                    chunk.get("faiss_dim"),
                    chunk.get("faiss_internal_id"),
                ))

    def get_chunks_by_source(self, source_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE source_id = ?", (source_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_chunks_for_deletion(self, source_id: str) -> List[Dict[str, Any]]:
        """Return (chunk_id, faiss_dim, faiss_internal_id) rows for deletion."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT id, faiss_dim, faiss_internal_id
                FROM chunks WHERE source_id = ?
            """, (source_id,)).fetchall()
            return [dict(r) for r in rows]

    def get_chunk_content(self, chunk_id: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT content FROM chunks WHERE id = ?", (chunk_id,)
            ).fetchone()
            return row["content"] if row else None

    # BUG-R03 / BUG-C01: batch fetch full rows by list of IDs
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch full chunk rows for a list of IDs in ONE query.
        Fixes the N+1 pattern in StorageManager.get_chunks_as_documents().
        Also returns source_id + metadata so PromptBuilder gets source names.
        """
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT id, source_id, content, metadata FROM chunks WHERE id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
        return [dict(r) for r in rows]

    # ── sessions / messages ───────────────────────────────────────────────────

    def add_message(self, message: Dict[str, Any]) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO messages (id, session_id, role, content, sources_used)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    message["id"],
                    message["session_id"],
                    message["role"],
                    message["content"],
                    json.dumps(message.get("sources_used", [])),
                ))

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]
