"""website_cleaner.py — Clean web-scraped LangChain Documents."""
from __future__ import annotations
import re
from typing import List
from langchain_core.documents import Document


class WebsiteCleaner:
    def clean(self, docs: List[Document]) -> List[Document]:
        result = []
        for doc in docs:
            text = doc.page_content
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s{2,}", " ", text)
            text = re.sub(r"(Subscribe|Cookie Policy|Privacy Policy|Accept Cookies)[^\n]*", "", text, flags=re.IGNORECASE)
            text = text.strip()
            if len(text.split()) >= 10:
                result.append(Document(page_content=text, metadata=doc.metadata))
        return result