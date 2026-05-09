from .base_chunker import BaseChunker
from typing import List, Dict, Any
import re

class ChapterChunker(BaseChunker):
    """Chunks based on Chapter markers."""
    
    def chunk(self, content: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        chapters = re.split(r'(?i)^Chapter\s+\d+.*$', content, flags=re.MULTILINE)
        chunks = []
        for i, chapter in enumerate(chapters):
            if not chapter.strip(): continue
            chunks.append({
                "id": f"{metadata.get('source_id', 'unknown')}_chapter_{i}",
                "content": chapter.strip(),
                "metadata": {**(metadata or {}), "chapter_index": i},
                "modality": metadata.get("modality", "text")
            })
        return chunks
    
    def get_strategy_name(self) -> str:
        return "chapter"
