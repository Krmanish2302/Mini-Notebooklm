"""
src/storage  —  public API

Import surface for the storage layer:
    MultiFAISSStore  — multi-dimensional FAISS index (one per embedding dim)
    SQLiteManager    — metadata + chunk + session/message store
    KnowledgeGraph   — NetworkX chunk-level semantic graph
    StorageManager   — orchestrator: keeps all three stores in sync

Typical usage (from MasterPipeline.__init__):
    from src.storage import MultiFAISSStore, SQLiteManager, KnowledgeGraph, StorageManager

    faiss   = MultiFAISSStore(base_dir="./data/vector_store")
    sqlite  = SQLiteManager(db_path="./data/metadata.db")
    graph   = KnowledgeGraph(edge_threshold=0.75)
    storage = StorageManager(faiss, sqlite, graph)
"""
from .faiss_store      import MultiFAISSStore
from .sqlite_manager   import SQLiteManager
from .knowledge_graph  import KnowledgeGraph
from .storage_manager  import StorageManager

__all__ = [
    "MultiFAISSStore",
    "SQLiteManager",
    "KnowledgeGraph",
    "StorageManager",
]
