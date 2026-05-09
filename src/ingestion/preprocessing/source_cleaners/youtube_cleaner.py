import re

class YouTubeCleaner:
    """Cleans transcript formatting."""
    
    @staticmethod
    def clean(text: str) -> str:
        # Keep timestamps but normalize
        text = re.sub(r'\[(\d+):?(\d*):?(\d*)\]', r'[\1:\2]', text)
        # Remove music notes
        text = re.sub(r'♪.*?♪', '', text)
        return text.strip()