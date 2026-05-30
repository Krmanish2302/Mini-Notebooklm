"""
parent_retriever.py

Persistent ParentDocumentRetriever — replaces RAPTOR.
Zero LLM calls at ingestion time.

Architecture:
  FAISS            ← child chunks (400 chars) — searched at query time
  LocalFileStore   ← parent chunks (2000 chars) — returned to LLM as context

Disk layout (under vectorstore_path/):
  index.faiss
  index.pkl
  docstore/
      <uuid>.json   ← one file per parent chunk

Build (ingestion time):
    from src.ingestion.parent_retriever import build_parent_retriever
    build_parent_retriever(chunks, vectorstore_path="data/vectorstores/rep_001")

Load (query time):
    from src.ingestion.parent_retriever import load_parent_retriever
    retriever = load_parent_retriever("data/vectorstores/rep_001")
    docs = retriever.invoke("What is the conclusion?")
    # docs → List[Document] of 2000-char parent chunks → feed to LLM
"""
from __future__ import annotations
import logging
import os
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

PARENT_CHUNK_SIZE  = int(os.getenv("PARENT_CHUNK_SIZE",  "2000"))
CHILD_CHUNK_SIZE   = int(os.getenv("CHILD_CHUNK_SIZE",   "400"))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP",      "50"))
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")


def _get_embeddings():
    if EMBEDDING_PROVIDER == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _make_docstore(vectorstore_path: str):
    docstore_dir = os.path.join(vectorstore_path, "docstore")
    os.makedirs(docstore_dir, exist_ok=True)
    return create_kv_docstore(LocalFileStore(docstore_dir)), docstore_dir


def _make_splitters():
    parent = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return parent, child


def build_parent_retriever(
    chunks: List[Document],
    vectorstore_path: str,
) -> ParentDocumentRetriever:
    """
    Build and persist a ParentDocumentRetriever. Zero LLM calls.

    Args:
        chunks:           Documents from the ingestion pipeline.
        vectorstore_path: Directory where FAISS index lives.

    Returns:
        Configured ParentDocumentRetriever (also persisted to disk).
    """
    embeddings  = _get_embeddings()
    vectorstore = FAISS.load_local(
        vectorstore_path, embeddings, allow_dangerous_deserialization=True,
    )
    docstore, docstore_dir = _make_docstore(vectorstore_path)
    parent_splitter, child_splitter = _make_splitters()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    retriever.add_documents(chunks, ids=None)
    vectorstore.save_local(vectorstore_path)

    logger.info(
        "[parent_retriever] Built: parent=%d child=%d docstore='%s'",
        PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE, docstore_dir,
    )
    return retriever


def load_parent_retriever(vectorstore_path: str) -> ParentDocumentRetriever:
    """
    Load a persisted ParentDocumentRetriever for query time.

    Args:
        vectorstore_path: Same path used in build_parent_retriever().

    Returns:
        ParentDocumentRetriever — call .invoke("your query") to retrieve docs.
    """
    embeddings  = _get_embeddings()
    vectorstore = FAISS.load_local(
        vectorstore_path, embeddings, allow_dangerous_deserialization=True,
    )
    docstore, _ = _make_docstore(vectorstore_path)
    parent_splitter, child_splitter = _make_splitters()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    logger.info("[parent_retriever] Loaded from '%s'", vectorstore_path)
    return retriever