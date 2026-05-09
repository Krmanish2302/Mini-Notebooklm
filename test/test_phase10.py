import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.contextual_compressor import ContextualCompressor
from src.retrieval.reranker import Reranker
from src.retrieval.advanced_retriever import AdvancedRetriever
from src.retrieval.study_mode import StudyModeRetriever
from src.graph.graph_retriever import GraphRetriever
from src.storage.faiss_store import FAISSStore
from src.graph.graph_storage import GraphStorage

def test_retrieval():
    print("Testing Retrieval System...")
    
    # 1. Setup Mock FAISS Store
    if os.path.exists("test_retrieval_faiss"):
        import shutil
        shutil.rmtree("test_retrieval_faiss")
    faiss_store = FAISSStore(dimension=3, index_path="test_retrieval_faiss/index.faiss")
    
    chunks = [
        {"id": "c1", "content": "The quick brown fox jumps over the lazy dog.", "embedding": [0.9, 0.1, 0.0]},
        {"id": "c2", "content": "Machine learning is fascinating. It allows computers to learn from data.", "embedding": [0.1, 0.9, 0.0]},
        {"id": "c3", "content": "A dog is a man's best friend.", "embedding": [0.5, 0.5, 0.0]}
    ]
    faiss_store.add(chunks)
    
    # 2. Test Hybrid Retriever
    print("  Testing HybridRetriever...")
    hybrid = HybridRetriever(faiss_store, top_k=2)
    hybrid.build_sparse_index(chunks)
    query = "dog"
    q_emb = np.array([0.8, 0.2, 0.0])
    results = hybrid.retrieve(query, q_emb)
    assert len(results) > 0
    assert any("dog" in r["content"] for r in results)
    print("    Hybrid OK")
    
    # 3. Test Contextual Compressor
    print("  Testing ContextualCompressor...")
    compressor = ContextualCompressor("all-MiniLM-L6-v2", relevance_threshold=0.1)
    # Give it a chunk with multiple sentences
    comp_chunks = [{"id": "c2", "content": "Machine learning is fascinating. It allows computers to learn from data. Bananas are yellow."}]
    compressed = compressor.compress(comp_chunks, "Tell me about computers learning.")
    assert len(compressed) == 1
    assert "Bananas are yellow" not in compressed[0]["content"]
    print("    Compressor OK")
    
    # 4. Test Reranker
    print("  Testing Reranker...")
    reranker = Reranker("cross-encoder/ms-marco-TinyBERT-L-2-v2") # Using a tiny model for fast testing
    reranked = reranker.rerank("dog", chunks)
    assert len(reranked) > 0
    assert "rerank_score" in reranked[0]
    print("    Reranker OK")
    
    # Clean up
    import shutil
    shutil.rmtree("test_retrieval_faiss")
    print("SUCCESS: Retrieval System verified")

if __name__ == "__main__":
    try:
        test_retrieval()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
