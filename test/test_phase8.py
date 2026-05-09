import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.merging.cross_modal_merger import CrossModalMerger
from src.ingestion.embedding.embedding_pipeline import EmbeddingPipeline

def test_cross_modal():
    print("Testing CrossModalMerger...")
    pipeline = EmbeddingPipeline("all-MiniLM-L6-v2")
    
    chunks = [
        {
            "id": "text_1",
            "content": "The new solar panel efficiency reached 25% this year, a record high.",
            "modality": "text"
        },
        {
            "id": "text_2",
            "content": "Baking a chocolate cake requires flour, sugar, and cocoa powder.",
            "modality": "text"
        },
        {
            "id": "img_1",
            "content": "A graph showing solar panel efficiency climbing over the years.",
            "modality": "image_caption"
        },
        {
            "id": "vid_1",
            "content": "Now we add the cocoa powder to the mixture.",
            "modality": "transcript"
        }
    ]
    
    merger = CrossModalMerger(similarity_threshold=0.3, embedder=pipeline.embedder)
    merged_chunks = merger.merge(chunks)
    
    img_chunk = next(c for c in merged_chunks if c["id"] == "img_1")
    vid_chunk = next(c for c in merged_chunks if c["id"] == "vid_1")
    
    print(f"  Image relationships: {img_chunk.get('related_chunks')}")
    print(f"  Video relationships: {vid_chunk.get('related_chunks')}")
    
    assert len(img_chunk["related_chunks"]) > 0, "No related chunks found for image"
    assert img_chunk["related_chunks"][0]["chunk_id"] == "text_1", "Incorrect relation for image"
    
    assert len(vid_chunk["related_chunks"]) > 0, "No related chunks found for video"
    assert vid_chunk["related_chunks"][0]["chunk_id"] == "text_2", "Incorrect relation for video"
    
    print("SUCCESS: Cross-Modal Merger verified")

if __name__ == "__main__":
    try:
        test_cross_modal()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
