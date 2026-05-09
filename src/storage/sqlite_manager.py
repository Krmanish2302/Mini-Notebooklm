import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

class SQLiteManager:
    """SQLite for metadata, sources, and chat history."""
    
    def __init__(self, db_path: str = "./data/metadata.db"):
        self.db_path = db_path
        self._init_tables()
    
    def _init_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    source_type TEXT,
                    file_path TEXT,
                    url TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'processing'
                );
                
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    source_id TEXT,
                    content TEXT,
                    modality TEXT,
                    metadata TEXT,
                    FOREIGN KEY (source_id) REFERENCES sources(id)
                );
                
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    mode TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    sources_used TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );
                
                CREATE TABLE IF NOT EXISTS chat_history_graph (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    node_type TEXT,
                    content TEXT,
                    related_nodes TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
    
    def add_source(self, source: Dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sources (id, title, source_type, file_path, url, metadata, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                source["id"], source.get("title", ""), source["source_type"],
                source.get("file_path"), source.get("url"),
                json.dumps(source.get("metadata", {})), source.get("status", "ready")
            ))
    
    def add_chunk(self, chunk: Dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO chunks (id, source_id, content, modality, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                chunk["id"], chunk.get("source_id", ""),
                chunk.get("content", ""), chunk.get("modality", "text"),
                json.dumps(chunk.get("metadata", {}))
            ))
    
    def get_chunks_by_source(self, source_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM chunks WHERE source_id = ?", (source_id,)
            ).fetchall()
            return [dict(r) for r in rows]
    
    def get_sources(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM sources WHERE status = 'ready'").fetchall()
            return [dict(r) for r in rows]
    
    def add_message(self, message: Dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO messages (id, session_id, role, content, sources_used)
                VALUES (?, ?, ?, ?, ?)
            """, (
                message["id"], message["session_id"], message["role"],
                message["content"], json.dumps(message.get("sources_used", []))
            ))
    
    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp",
                (session_id,)
            ).fetchall()
            return [dict(r) for r in rows]