from .base_chunker import BaseChunker
from typing import List, Dict, Any
import re

class PageChunker(BaseChunker):
    """Chunks based on page markers [Page X]."""
    
    def chunk(self, content: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        # Split by [Page X] or ## Page X
        pages = re.split(r'\[Page\s*\d+\]|##\s*Page\s*\d+', content)
        chunks = []
        for i, page in enumerate(pages):
            if not page.strip(): continue
            chunks.append({
                "id": f"{metadata.get('source_id', 'unknown')}_page_{i}",
                "content": page.strip(),
                "metadata": {**(metadata or {}), "page_index": i},
                "modality": metadata.get("modality", "text")
            })
        return chunks
    
    def get_strategy_name(self) -> str:
        return "page"
