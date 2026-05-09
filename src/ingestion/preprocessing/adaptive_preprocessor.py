from typing import Dict, Any
from .content_analyzer import ContentAnalyzer
from .source_cleaners.pdf_cleaner import PDFCleaner
from .source_cleaners.website_cleaner import WebsiteCleaner
from .source_cleaners.youtube_cleaner import YouTubeCleaner

class AdaptivePreprocessor:
    """Routes content to appropriate cleaner and analyzer."""
    
    def __init__(self):
        self.analyzer = ContentAnalyzer()
        self.cleaners = {
            "pdf": PDFCleaner(),
            "website": WebsiteCleaner(),
            "youtube": YouTubeCleaner()
        }
    
    def process(self, content: str, source_type: str, metadata: Dict = None) -> Dict[str, Any]:
        """
        Main preprocessing entry point.
        Returns: {"cleaned_content": str, "analysis": dict, "recommendation": dict}
        """
        # Clean based on source type
        cleaner = self.cleaners.get(source_type)
        cleaned = cleaner.clean(content) if cleaner else content
        
        # Analyze
        analysis = self.analyzer.analyze(cleaned, source_type)
        
        return {
            "cleaned_content": cleaned,
            "analysis": analysis,
            "recommendation": analysis["recommendation"],
            "preprocessing_applied": [source_type + "_cleaning", "content_analysis"]
        }