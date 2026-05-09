import re
from typing import List, Dict, Any

class ResponseParser:
    """Parse LLM response to extract citations and content."""
    
    @staticmethod
    def parse(response_text: str) -> Dict[str, Any]:
        """
        Parse response into content and citations.
        Returns: {"content": str, "citations": List[Dict]}
        """
        # Extract citations [SOURCE_X]
        citation_pattern = r'\[SOURCE_(\d+)\]'
        citations_found = re.findall(citation_pattern, response_text)
        
        citations = []
        for cid in set(citations_found):
            citations.append({
                "source_index": int(cid),
                "source_id": f"source_{cid}",
                "confidence": 1.0
            })
        
        # Clean response (remove citation markers for display)
        clean_content = re.sub(citation_pattern, '', response_text)
        clean_content = re.sub(r'\s+', ' ', clean_content).strip()
        
        return {
            "content": clean_content,
            "citations": citations,
            "has_citations": len(citations) > 0
        }
    
    @staticmethod
    def validate_grounding(response: str, context: str) -> bool:
        """
        Basic validation: check if response contains content from context.
        Returns True if grounded, False if potentially hallucinated.
        """
        # Extract key phrases from context (simple approach)
        context_words = set(context.lower().split())
        response_words = set(response.lower().split())
        
        # Check overlap
        overlap = len(context_words & response_words)
        total = len(response_words)
        
        if total == 0:
            return False
        
        ratio = overlap / total
        return ratio > 0.3  # At least 30% words from context