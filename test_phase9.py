import sys
import os
import shutil
import numpy as np
sys.path.append(os.path.abspath('.'))

from src.storage.faiss_store import FAISSStore
from src.storage.sqlite_manager import SQLiteManager
from src.graph.graph_storage import GraphStorage

def test_sqlite():
    print("Testing SQLiteManager...")
    db_path = "test_meta.db"
    if os.path.exists(db_path): os.remove(db_path)
    
    sql = SQLiteManager(db_path=db_path)
    sql.add_source({
        "id": "src1",
        "title": "Test Document",
        "source_type": "pdf",
        "status": "ready"
    })
    
    sources = sql.get_sources()
    print(f"  Sources retrieved: {len(sources)}")
    assert len(sources) == 1
    assert sources[0]["id"] == "src1"
    
    os.remove(db_path)
    print("  SQLite test passed!")

def test_faiss():
    print("Testing FAISSStore...")
    index_path = "test_faiss/index.faiss"
    if os.path.exists("test_faiss"): shutil.rmtree("test_faiss")
    
    faiss_store = FAISSStore(dimension=3, index_path=index_path)
    
    chunks = [
        {"id": "c1", "content": "doc1", "embedding": [1.0, 0.0, 0.0]},
        {"id": "c2", "content": "doc2", "embedding": [0.0, 1.0, 0.0]}
    ]
    
    faiss_store.add(chunks)
    print(f"  FAISS stats after add: {faiss_store.get_stats()}")
    assert faiss_store.get_stats()["total_chunks"] == 2
    
    results = faiss_store.search(np.array([0.9, 0.1, 0.0]), k=1)
    print(f"  Search result: {results[0]['id']} (Score: {results[0]['score']:.4f})")
    assert results[0]['id'] == "c1"
    
    shutil.rmtree("test_faiss")
    print("  FAISS test passed!")

def test_graph():
    print("Testing GraphStorage...")
    graph_path = "test_graph/graph.pkl"
    if os.path.exists("test_graph"): shutil.rmtree("test_graph")
    
    graph = GraphStorage(graph_path=graph_path)
    
    graph.add_chunk({"id": "n1", "content": "Node 1"})
    graph.add_chunk({"id": "n2", "content": "Node 2"})
    graph.add_relationship("n1", "n2", "mentions")
    
    related = graph.get_related("n1", depth=1)
    print(f"  Related to n1: {[r['chunk_id'] for r in related]}")
    assert len(related) == 1
    assert related[0]['chunk_id'] == "n2"
    
    shutil.rmtree("test_graph")
    print("  Graph test passed!")

if __name__ == "__main__":
    try:
        test_sqlite()
        test_faiss()
        test_graph()
        print("SUCCESS: Storage layer verified")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
