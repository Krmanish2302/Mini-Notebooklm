from sentence_transformers import SentenceTransformer
import numpy as np
from .base_chunker import BaseChunker
from typing import List, Dict, Any

class LateChunker(BaseChunker):
    """
    Embed entire document first, then chunk.
    This preserves global context in each chunk.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 chunk_size: int = 384, overlap: int = 50):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, content: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        # Step 1: Embed entire document
        doc_embedding = self.model.encode(content)
        
        # Step 2: Chunk with overlap
        words = content.split()
        chunks = []
        step = self.chunk_size - self.overlap
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Step 3: Each chunk gets document-level context
            chunks.append({
                "id": f"{metadata.get('source_id', 'unknown')}_chunk_{i}",
                "content": chunk_text,
                "metadata": {
                    **(metadata or {}),
                    "chunk_index": i // step,
                    "doc_embedding": doc_embedding.tolist(),  # Global context
                    "late_chunking": True
                },
                "modality": metadata.get("modality", "text")
            })
        
        return chunks
    
    def get_strategy_name(self) -> str:
        return "late"