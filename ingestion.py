import os
import shutil
import warnings
from pathlib import Path
from typing import Dict, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import streamlit as st

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="langchain_community"
)

CHROMA_DB_DIR = "./db/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path=file_path)
    return loader.load()


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    return chunks


def db_exists(db_dir: str = CHROMA_DB_DIR) -> bool:
    return os.path.exists(db_dir)

@st.cache_resource
def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


def load_existing_db(db_dir: str = CHROMA_DB_DIR) -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings,
    )


def ingest_documents(
    file_paths: Dict[str, str],
    db_dir: str = CHROMA_DB_DIR,
    reset_db: bool = False,) -> Chroma:

    if reset_db and os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    all_chunks = []

    for doc_type, path in file_paths.items():
        if path and os.path.exists(path):
            raw_docs = load_pdf(path)

            for doc in raw_docs:
                doc.metadata["doc_type"] = doc_type

            chunks = chunk_documents(raw_docs)
            all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No documents were loaded. Verify your file paths.")

    embeddings = get_embeddings()

    db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=db_dir,
    )

    return db
 
