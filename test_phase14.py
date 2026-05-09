import sys
import os
import shutil
sys.path.append(os.path.abspath('.'))

from src.storage.faiss_store import FAISSStore
from src.storage.sqlite_manager import SQLiteManager
from src.graph.graph_storage import GraphStorage
from src.storage.source_manager import SourceManager

def test_source_manager():
    print("Testing Source Manager...")
    
    # Cleanup past tests
    for path in ["test_src_faiss", "test_src_meta.db", "test_src_graph"]:
        if os.path.exists(path):
            if os.path.isdir(path): shutil.rmtree(path)
            else: os.remove(path)
            
    faiss = FAISSStore(dimension=3, index_path="test_src_faiss/index.faiss")
    sqlite = SQLiteManager(db_path="test_src_meta.db")
    graph = GraphStorage(graph_path="test_src_graph/graph.pkl")
    manager = SourceManager(faiss_store=faiss, sqlite_manager=sqlite, graph_storage=graph)
    
    # 1. Add Source
    print("  Testing add_source...")
    source_data = {"title": "Test PDF", "source_type": "pdf"}
    chunks = [
        {"id": "c1", "content": "Hello world", "embedding": [1.0, 0.0, 0.0], "source_id": "test_src_id"},
        {"id": "c2", "content": "Testing", "embedding": [0.0, 1.0, 0.0], "source_id": "test_src_id"}
    ]
    
    # Mocking id creation so we know source_id for chunks
    source_data["id"] = "test_src_id"
    source_id = manager.add_source(source_data, chunks)
    assert len(manager.get_all_sources()) == 1
    assert faiss.get_stats()["total_chunks"] == 2
    print("    Add Source OK")
    
    # 2. Test Deduplication
    print("  Testing deduplication...")
    chunks_dup = [
        {"id": "c3", "content": "Hello world", "embedding": [1.0, 0.0, 0.0]} # Duplicate content
    ]
    manager.add_source({"title": "Dup PDF", "source_type": "pdf"}, chunks_dup)
    removed = manager.remove_duplicates()
    assert removed == 1
    print("    Deduplication OK")
    
    # 3. Remove Source
    print("  Testing remove_source...")
    success = manager.remove_source(source_id)
    assert success == True
    assert len(manager.get_all_sources()) == 1 # Dup PDF remains
    print("    Remove Source OK")
    
    # Cleanup
    for path in ["test_src_faiss", "test_src_meta.db", "test_src_graph"]:
        if os.path.exists(path):
            if os.path.isdir(path): shutil.rmtree(path)
            else: os.remove(path)
            
    print("SUCCESS: Source Manager verified")

if __name__ == "__main__":
    try:
        test_source_manager()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
