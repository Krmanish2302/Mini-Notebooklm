import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.embedding.embedding_pipeline import EmbeddingPipeline
from src.ingestion.embedding.text_embedder import TextEmbedder

def test_embedding():
    print("Testing TextEmbedder...")
    embedder = TextEmbedder("all-MiniLM-L6-v2")
    dim = embedder.get_dimension()
    print(f"  Expected Dimension: {dim}")
    assert dim == 384
    
    sample_emb = embedder.embed_single("Test text")
    print(f"  Actual Dimension: {sample_emb.shape[0]}")
    assert sample_emb.shape[0] == 384
    
    print("Testing EmbeddingPipeline...")
    pipeline = EmbeddingPipeline("all-MiniLM-L6-v2", use_cache=True)
    texts = ["This is sentence one.", "This is sentence two.", "Another different text."] * 5
    
    start = time.time()
    embs_1 = pipeline.embed_batch(texts)
    time_1 = time.time() - start
    print(f"  First run time: {time_1:.4f} seconds")
    
    start = time.time()
    embs_2 = pipeline.embed_batch(texts)
    time_2 = time.time() - start
    print(f"  Second run time: {time_2:.4f} seconds")
    
    assert time_2 < time_1, "Cached run should be faster!"
    print("SUCCESS: Embedding verified")

if __name__ == "__main__":
    try:
        test_embedding()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
