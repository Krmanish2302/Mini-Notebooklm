import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.chunking.chunker_registry import ChunkerRegistry

def test_chunking():
    print("Testing ChunkerRegistry...")
    strategies = ChunkerRegistry.list_strategies()
    print(f"  Strategies: {strategies}")
    
    # Recursive Test
    chunker = ChunkerRegistry.get_chunker("recursive", chunk_size=50, chunk_overlap=10)
    text = "This is a test of the recursive chunking system. It should split this text into multiple parts."
    chunks = chunker.chunk(text, {"source_id": "test1"})
    print(f"  Recursive chunks: {len(chunks)}")
    assert len(chunks) > 0
    
    # Hierarchical Test
    h_chunker = ChunkerRegistry.get_chunker("hierarchical", parent_size=20, child_size=5)
    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
    h_chunks = h_chunker.chunk(text, {"source_id": "test2"})
    print(f"  Hierarchical chunks: {len(h_chunks)}")
    assert "parent_id" in h_chunks[0]['metadata']
    
    print("SUCCESS: Chunking verified")

if __name__ == "__main__":
    try:
        test_chunking()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
