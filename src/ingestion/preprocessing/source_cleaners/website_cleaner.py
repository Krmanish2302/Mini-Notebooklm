import re

class WebsiteCleaner:
    """Removes boilerplate from websites."""
    
    PATTERNS = [
        r'(?i)(accept cookies|cookie policy|privacy policy).*',
        r'(?i)(subscribe|sign up|newsletter).*',
        r'(?i)(all rights reserved|copyright ©).*',
        r'(?i)(share this|follow us on|connect with us).*',
        r'(?i)(advertisement|sponsored content).*'
    ]
    
    @staticmethod
    def clean(text: str) -> str:
        for pattern in WebsiteCleaner.PATTERNS:
            text = re.sub(pattern, '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()