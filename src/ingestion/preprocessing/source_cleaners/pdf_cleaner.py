import re

class PDFCleaner:
    """Cleans PDF text artifacts."""
    
    @staticmethod
    def clean(text: str) -> str:
        # Normalize page markers
        text = re.sub(r'\[Page\s*(\d+)\]', r'\n\n## Page \1\n', text)
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Fix hyphenation
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        return text.strip()