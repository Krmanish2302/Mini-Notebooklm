import fitz  # PyMuPDF
from typing import Dict, Any, List
from src.core.models import Source

class PDFPipeline:
    """Extracts text from PDF with page numbers."""
    
    @staticmethod
    def process(file_path: str, source_id: str) -> Dict[str, Any]:
        """
        Process PDF file.
        Returns: {"content": str, "pages": List[Dict], "metadata": Dict}
        """
        doc = fitz.open(file_path)
        pages = []
        full_text = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            pages.append({
                "page_number": page_num + 1,
                "text": text,
                "word_count": len(text.split())
            })
            full_text.append(f"[Page {page_num + 1}]\n{text}")
        
        metadata = {
            "total_pages": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "source_id": source_id
        }
        
        doc.close()
        
        return {
            "content": "\n\n".join(full_text),
            "pages": pages,
            "metadata": metadata,
            "modality": "text"
        }