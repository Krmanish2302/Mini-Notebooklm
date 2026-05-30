"""website_pipeline.py — LangChain WebBaseLoader wrapper."""
from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader


class WebsitePipeline:
    @staticmethod
    def process(url: str, source_id: str) -> List[Document]:
        docs = WebBaseLoader(url).load()
        for doc in docs:
            doc.metadata["source_id"]   = source_id
            doc.metadata["source_type"] = "website"
        return docs