from typing import List, Dict, Any, Optional
import uuid
from .faiss_store import FAISSStore
from .sqlite_manager import SQLiteManager
from src.graph.graph_storage import GraphStorage

class SourceManager:
    """
    Manages sources with full lifecycle:
    - Add sources (upload or web search)
    - Remove sources (with storage cleanup)
    - Batch operations
    - Duplicate detection
    """
    
    def __init__(self, faiss_store: FAISSStore, sqlite_manager: SQLiteManager, 
                 graph_storage: GraphStorage = None):
        self.faiss = faiss_store
        self.sqlite = sqlite_manager
        self.graph = graph_storage
        self.sources: Dict[str, Dict[str, Any]] = {}
    
    def add_source(self, source: Dict[str, Any], chunks: List[Dict[str, Any]]) -> str:
        """Add source with chunks to all storage."""
        source_id = source.get("id", str(uuid.uuid4()))
        source["id"] = source_id
        
        # Store metadata
        self.sqlite.add_source(source)
        self.sources[source_id] = source
        
        # Store chunks in FAISS
        if chunks:
            self.faiss.add(chunks)
        
        # Persist chunks to SQLite for durability
        for chunk in chunks:
            chunk_record = {
                "id": chunk.get("id"),
                "source_id": source_id,
                "content": chunk.get("content", ""),
                "modality": chunk.get("modality", "text"),
                "metadata": chunk.get("metadata", {})
            }
            self.sqlite.add_chunk(chunk_record)
        
        # Add to graph
        if self.graph:
            for chunk in chunks:
                self.graph.add_chunk(chunk)
                # Link chunks from same source
                for other_chunk in chunks:
                    if chunk["id"] != other_chunk["id"]:
                        self.graph.add_relationship(
                            chunk["id"], other_chunk["id"], "same_source", 0.5
                        )
        
        return source_id
    
    def remove_source(self, source_id: str) -> bool:
        """Remove source and all associated data."""
        if source_id not in self.sources:
            return False
        
        # Get chunk IDs for this source
        chunk_ids = [
            cid for cid, chunk in self.faiss.metadata.items()
            if chunk.get("source_id") == source_id
        ]
        
        # Remove from FAISS
        if chunk_ids:
            self.faiss.delete(chunk_ids)
        
        # Remove from graph
        if self.graph:
            for cid in chunk_ids:
                if cid in self.graph.graph:
                    self.graph.graph.remove_node(cid)
        
        # Remove from SQLite
        import sqlite3
        with sqlite3.connect(self.sqlite.db_path) as conn:
            conn.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
            conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        
        # Remove from memory
        del self.sources[source_id]
        return True
    
    def batch_remove_sources(self, source_ids: List[str]) -> Dict[str, bool]:
        """Remove multiple sources."""
        results = {}
        for sid in source_ids:
            results[sid] = self.remove_source(sid)
        return results
    
    def remove_duplicates(self) -> int:
        """Remove duplicate chunks based on content hash."""
        # Calculate content hashes
        content_hashes = {}
        duplicates = []
        
        for cid, chunk in self.faiss.metadata.items():
            content_hash = hash(chunk["content"])
            if content_hash in content_hashes:
                duplicates.append(cid)
            else:
                content_hashes[content_hash] = cid
        
        # Remove duplicates
        if duplicates:
            self.faiss.delete(duplicates)
        
        return len(duplicates)
    
    def get_all_sources(self) -> List[Dict[str, Any]]:
        """Get all active sources."""
        return list(self.sources.values())
    
    def get_source_stats(self, source_id: str) -> Dict[str, int]:
        """Get statistics for a source."""
        chunk_count = sum(
            1 for c in self.faiss.metadata.values()
            if c.get("source_id") == source_id
        )
        return {
            "chunk_count": chunk_count,
            "source_type": self.sources.get(source_id, {}).get("source_type", "unknown")
        }