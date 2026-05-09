"""
sqlite_manager.py  —  Metadata + chat history store

Schema additions vs original:
    chunks.embedding_model   TEXT  — which model produced the embedding
    chunks.faiss_dim         INT   — which FAISS index holds this chunk
    chunks.faiss_internal_id INT   — FAISS int64 ID for O(1) deletion
    sources.status           hydrated on startup (fixes empty sources bug)
"""
import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class SQLiteManager:
    """SQLite metadata store — sources, chunks, sessions, messages."""

    def __init__(self, db_path: str = "./data/metadata.db"):
        self.db_path = db_path
        import os; os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_tables()

    # ── schema ────────────────────────────────────────────────────────────────

    def _init_tables(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
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
            # Migration: add new columns if upgrading from old schema
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add new columns to existing tables without breaking existing data."""
        existing = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        for col, typedef in [
            ("embedding_model",   "TEXT"),
            ("faiss_dim",         "INTEGER"),
            ("faiss_internal_id", "INTEGER"),
        ]:
            if col not in existing:
                conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} {typedef}")
                logger.info("SQLiteManager: migrated chunks.%s", col)

    # ── sources ───────────────────────────────────────────────────────────────

    def add_source(self, source: Dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sources
                    (id, title, source_type, file_path, url, metadata, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                source["id"],
                source.get("title", ""),
                source["source_type"],
                source.get("file_path"),
                source.get("url"),
                json.dumps(source.get("metadata", {})),
                source.get("status", "ready"),
            ))

    def get_sources(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM sources WHERE status = 'ready'"
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_source(self, source_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chunks  WHERE source_id = ?", (source_id,))
            conn.execute("DELETE FROM sources WHERE id         = ?", (source_id,))

    # ── chunks ────────────────────────────────────────────────────────────────

    def add_chunk(self, chunk: Dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM chunks WHERE source_id = ?", (source_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_chunks_for_deletion(self, source_id: str) -> List[Dict[str, Any]]:
        """Return (chunk_id, faiss_dim, faiss_internal_id) rows for deletion."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT id, faiss_dim, faiss_internal_id
                FROM chunks WHERE source_id = ?
            """, (source_id,)).fetchall()
            return [dict(r) for r in rows]

    def get_chunk_content(self, chunk_id: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT content FROM chunks WHERE id = ?", (chunk_id,)
            ).fetchone()
            return row[0] if row else None

    # ── sessions / messages ───────────────────────────────────────────────────

    def add_message(self, message: Dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]
