from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import streamlit as st

@st.cache_resource()
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_context(
    db: Chroma,
    query: str,
    k: int = 5,
    doc_type_filter: Optional[str] = None,
    use_reranking: bool = True,) -> List[Document]:
    
    fetch_k = k * 2 if use_reranking else k

    filter_dict = {"doc_type": doc_type_filter} if doc_type_filter else None

    docs = db.similarity_search(query, k=fetch_k, filter=filter_dict)

    if use_reranking and len(docs) > 0:
        docs = rerank(query, docs, top_k=k)

    return docs


def rerank(query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    model = get_reranker()
    pairs = []
    for doc in docs:
        pair = (query, doc.page_content)
        pairs.append(pair)

    scores = model.predict(pairs)

    paired_results = []
    for i in range(len(docs)):
        paired_item = (scores[i], docs[i])
        paired_results.append(paired_item)

    def get_score(item):
        return item[0]

    paired_results.sort(key=get_score, reverse=True)

    top_docs = []
    for item in paired_results[:top_k]:
        doc = item[1]
        top_docs.append(doc)

    return top_docs

def build_context_string(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("doc_type", "document").upper()
        page = doc.metadata.get("page", "?") 
        parts.append(f"[{source} - Page {page}]\n{doc.page_content}")
    
    return "\n\n---\n\n".join(parts)