"""
pdf_pipeline.py

Extracts text from PDF files using PyMuPDF (fitz).
Handles encrypted / corrupt PDFs gracefully.
"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class PDFPipeline:
    """Extracts text from PDF with per-page metadata."""

    @staticmethod
    def process(file_path: str, source_id: str) -> Dict[str, Any]:
        """
        Process a PDF file.

        Args:
            file_path: Absolute or relative path to the PDF.
            source_id: Unique identifier for this source.

        Returns:
            dict with keys: content, pages, metadata, modality

        Raises:
            ValueError: If the file cannot be opened or contains no text.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is required for PDF ingestion. "
                "Install it with: pip install pymupdf"
            ) from exc

        try:
            doc = fitz.open(file_path)
        except Exception as exc:
            raise ValueError(f"Cannot open PDF '{file_path}': {exc}") from exc

        pages: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []

        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                text = page.get_text().strip()
            except Exception as page_err:
                logger.warning(
                    "PDFPipeline: skipping page %d of '%s' — %s",
                    page_num + 1, file_path, page_err,
                )
                text = ""

            pages.append({
                "page_number": page_num + 1,
                "text": text,
                "word_count": len(text.split()),
            })
            if text:
                full_text_parts.append(f"[Page {page_num + 1}]\n{text}")

        doc.close()

        content = "\n\n".join(full_text_parts)
        if not content.strip():
            raise ValueError(
                f"No readable text found in '{file_path}'. "
                "The PDF may be scanned / image-only."
            )

        metadata = {
            "total_pages": len(pages),
            "title": "",
            "author": "",
            "source_id": source_id,
        }

        # Try to read embedded PDF metadata (may not exist)
        try:
            raw_meta = fitz.open(file_path).metadata
            metadata["title"] = raw_meta.get("title", "")
            metadata["author"] = raw_meta.get("author", "")
        except Exception:
            pass

        return {
            "content": content,
            "pages": pages,
            "metadata": metadata,
            "modality": "text",
        }
