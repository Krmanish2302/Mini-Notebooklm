"""
parent_retriever.py

Builds and loads a persistent ParentDocumentRetriever on top of the
FAISS index produced by embed_node.py.

Why ParentDocumentRetriever instead of RAPTOR
---------------------------------------------
RAPTOR fired 15-20 LLM API calls per document to build a recursive
summary tree.  ParentDocumentRetriever achieves the same goal
(fine-grained retrieval + full-context response) with ZERO LLM calls
by using two chunk sizes:

  Child chunks  (CHILD_CHUNK_SIZE  = 400 chars)
    - Small, precise, embedded into FAISS
    - What the vector search matches against at query time

  Parent chunks (PARENT_CHUNK_SIZE = 2000 chars)
    - Large, context-rich, stored in LocalFileStore on disk
    - What gets returned to the LLM as context

Disk layout (all under vectorstore_path/)
-----------------------------------------
  <vectorstore_path>/
      index.faiss          ← FAISS child-chunk index
      index.pkl            ← FAISS metadata
      docstore/            ← LocalFileStore parent chunks (persisted)
          <uuid>.json
          <uuid>.json
          ...

Usage
-----
At ingestion time (called automatically by ingestion_runner):
    from src.ingestion.parent_retriever import build_parent_retriever
    retriever = build_parent_retriever(
        chunks=state["chunks"],
        vectorstore_path=state["vectorstore_path"],
    )

At query/retrieval time:
    from src.ingestion.parent_retriever import load_parent_retriever
    retriever = load_parent_retriever(vectorstore_path="data/vectorstores/report_001")
    docs = retriever.invoke("What is the main finding?")
    # docs are the PARENT (large) chunks — full context for the LLM
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

# ── Config (override via env) ─────────────────────────────────────────────────
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "2000"))
CHILD_CHUNK_SIZE  = int(os.getenv("CHILD_CHUNK_SIZE",  "400"))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP",     "50"))
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")


def _get_embeddings():
    """Return configured LangChain Embeddings (matches embed_node.py)."""
    if EMBEDDING_PROVIDER == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _make_docstore(vectorstore_path: str) -> tuple:
    """
    Create a persistent LocalFileStore-backed docstore.
    Returns (docstore, docstore_dir) tuple.
    """
    docstore_dir = os.path.join(vectorstore_path, "docstore")
    os.makedirs(docstore_dir, exist_ok=True)
    fs       = LocalFileStore(docstore_dir)
    docstore = create_kv_docstore(fs)
    return docstore, docstore_dir


def build_parent_retriever(
    chunks: List[Document],
    vectorstore_path: str,
) -> ParentDocumentRetriever:
    """
    Build a ParentDocumentRetriever with persistent disk-backed docstore.

    Zero LLM API calls — pure splitter logic.

    Steps
    -----
    1. Load the FAISS index already built by embed_node.py.
    2. Create a persistent LocalFileStore docstore.
    3. Instantiate ParentDocumentRetriever with:
         parent_splitter → 2000-char chunks  (stored in docstore)
         child_splitter  →  400-char chunks  (stored in FAISS)
    4. Feed the incoming chunks as the "parent" source documents.
    5. Re-save FAISS (now contains child-chunk vectors).

    Args:
        chunks:           Documents from ingestion pipeline (cleaned, pre-chunked).
        vectorstore_path: Directory of the FAISS index.

    Returns:
        Configured ParentDocumentRetriever (also persisted to disk).
    """
    embeddings  = _get_embeddings()
    vectorstore = FAISS.load_local(
        vectorstore_path, embeddings, allow_dangerous_deserialization=True
    )

    docstore, docstore_dir = _make_docstore(vectorstore_path)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # add_documents: internally splits into child + parent, indexes both
    retriever.add_documents(chunks, ids=None)

    # Persist updated FAISS (child vectors added)
    vectorstore.save_local(vectorstore_path)

    logger.info(
        "[parent_retriever] Built ParentDocumentRetriever: "
        "parent=%d chars, child=%d chars, docstore='%s'",
        PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE, docstore_dir,
    )
    return retriever


def load_parent_retriever(vectorstore_path: str) -> ParentDocumentRetriever:
    """
    Load a previously built ParentDocumentRetriever from disk.

    Call this at query/retrieval time (not ingestion time).

    Args:
        vectorstore_path: Same directory used during build_parent_retriever().

    Returns:
        ParentDocumentRetriever ready to call .invoke(query).

    Example:
        retriever = load_parent_retriever("data/vectorstores/report_001")
        docs = retriever.invoke("What is the conclusion?")
        # docs = List[Document] of PARENT chunks (2000 chars) → feed to LLM
    """
    embeddings  = _get_embeddings()
    vectorstore = FAISS.load_local(
        vectorstore_path, embeddings, allow_dangerous_deserialization=True
    )

    docstore, _ = _make_docstore(vectorstore_path)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    logger.info(
        "[parent_retriever] Loaded ParentDocumentRetriever from '%s'",
        vectorstore_path,
    )
    return retriever
