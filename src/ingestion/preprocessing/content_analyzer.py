import re
from typing import Dict, Any

class ContentAnalyzer:
    """Analyzes content to recommend chunking strategy."""
    
    def __init__(self, sample_pages: int = 15, embedding_model_max_tokens: int = 384):
        self.sample_pages = sample_pages
        self.embedding_model_max_tokens = embedding_model_max_tokens
    
    def analyze(self, content: str, source_type: str) -> Dict[str, Any]:
        """
        Analyze content and recommend chunking strategy.
        Returns analysis with token estimates and strategy recommendation.
        """
        # Sample first N pages/paragraphs
        sample = self._get_sample(content)
        
        # Token estimation (rough: 1 token ≈ 0.75 words)
        words = sample.split()
        estimated_tokens = int(len(words) / 0.75)
        paragraphs = [p for p in sample.split('\n\n') if p.strip()]
        
        avg_tokens_per_paragraph = estimated_tokens // max(len(paragraphs), 1)
        
        # Detect structure
        has_chapters = bool(re.search(r'^(Chapter|CHAPTER)\s+\d+', sample, re.MULTILINE))
        has_page_markers = '[Page' in content
        
        # Recommend strategy based on source type and structure
        recommendation = self._recommend_strategy(
            source_type, has_chapters, has_page_markers, avg_tokens_per_paragraph
        )
        
        return {
            "sampled_length": len(sample),
            "estimated_tokens": estimated_tokens,
            "avg_tokens_per_paragraph": avg_tokens_per_paragraph,
            "paragraph_count": len(paragraphs),
            "structure": {
                "has_chapters": has_chapters,
                "has_page_markers": has_page_markers,
                "has_headings": bool(re.search(r'^#{1,3}\s', sample, re.MULTILINE))
            },
            "recommendation": recommendation
        }
    
    def _get_sample(self, content: str) -> str:
        """Get representative sample of content."""
        paragraphs = content.split('\n\n')
        if len(paragraphs) <= self.sample_pages:
            return content
        return '\n\n'.join(paragraphs[:self.sample_pages])
    
    def _recommend_strategy(self, source_type: str, has_chapters: bool, 
                           has_page_markers: bool, avg_tokens: int) -> Dict[str, Any]:
        """Recommend chunking strategy."""
        if source_type == "website":
            return {
                "strategy": "recursive",
                "reason": "Website content benefits from recursive chunking",
                "chunk_size": f"{self.embedding_model_max_tokens} tokens"
            }
        elif source_type == "youtube":
            return {
                "strategy": "semantic",
                "reason": "YouTube transcripts are plain text",
                "chunk_size": f"~{self.embedding_model_max_tokens} tokens"
            }
        elif has_chapters:
            return {
                "strategy": "chapter",
                "reason": "Document has clear chapter structure",
                "chunk_size": "1 chapter per chunk"
            }
        elif has_page_markers and avg_tokens > 200:
            return {
                "strategy": "page",
                "reason": "Document has page markers with dense content",
                "chunk_size": "1 page per chunk"
            }
        else:
            return {
                "strategy": "recursive",
                "reason": "Default safe strategy",
                "chunk_size": f"{self.embedding_model_max_tokens} tokens with overlap"
            }