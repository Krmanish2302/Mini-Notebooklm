"""csv_pipeline.py — LangChain CSVLoader wrapper."""
from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders.csv_loader import CSVLoader


class CSVPipeline:
    @staticmethod
    def process(file_path: str, source_id: str) -> List[Document]:
        docs = CSVLoader(file_path=file_path).load()
        for doc in docs:
            doc.metadata["source_id"]   = source_id
            doc.metadata["source_type"] = "csv"
        return docs