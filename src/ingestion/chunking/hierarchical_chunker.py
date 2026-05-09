from .base_chunker import BaseChunker
from typing import List, Dict, Any

class HierarchicalChunker(BaseChunker):
    """
    Creates parent-child chunk relationships.
    Parent = large section, Children = smaller chunks within.
    """
    
    def __init__(self, parent_size: int = 1000, child_size: int = 200, overlap: int = 50):
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap
    
    def chunk(self, content: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        words = content.split()
        chunks = []
        parent_idx = 0
        
        for i in range(0, len(words), self.parent_size):
            parent_words = words[i:i + self.parent_size]
            parent_text = " ".join(parent_words)
            parent_id = f"{metadata.get('source_id', 'unknown')}_parent_{parent_idx}"
            
            # Create child chunks within parent
            child_idx = 0
            for j in range(0, len(parent_words), self.child_size - self.overlap):
                child_words = parent_words[j:j + self.child_size]
                child_text = " ".join(child_words)
                
                chunks.append({
                    "id": f"{parent_id}_child_{child_idx}",
                    "content": child_text,
                    "metadata": {
                        **(metadata or {}),
                        "parent_id": parent_id,
                        "parent_content": parent_text,
                        "chunk_level": "child",
                        "chunk_index": child_idx
                    },
                    "modality": metadata.get("modality", "text")
                })
                child_idx += 1
            
            parent_idx += 1
        
        return chunks
    
    def get_strategy_name(self) -> str:
        return "hierarchical"