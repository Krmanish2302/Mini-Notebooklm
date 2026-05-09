from .base_chunker import BaseChunker
from typing import List, Dict, Any

class ParagraphChunker(BaseChunker):
    """Chunks based on double newlines."""
    
    def chunk(self, content: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        paragraphs = content.split('\n\n')
        chunks = []
        for i, p in enumerate(paragraphs):
            if not p.strip(): continue
            chunks.append({
                "id": f"{metadata.get('source_id', 'unknown')}_para_{i}",
                "content": p.strip(),
                "metadata": {**(metadata or {}), "para_index": i},
                "modality": metadata.get("modality", "text")
            })
        return chunks
    
    def get_strategy_name(self) -> str:
        return "paragraph"
