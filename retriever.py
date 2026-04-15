# retriever.py
# Searches uploaded documents using hybrid search.
# Reuses the same BM25 + vector approach from StudyRAG.

import os
import pickle
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from sentence_transformers import CrossEncoder

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHROMA_PATH = "./chroma_db"
BM25_PATH = "./bm25_index.pkl"

print("Loading cross-encoder model...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def embed_query(query: str) -> list:
    """Convert question to vector embedding."""
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def vector_search(query: str, top_k: int = 10) -> list:
    """Search by meaning using ChromaDB."""
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = chroma_client.get_or_create_collection("adaptiverag")
        embedding = embed_query(query)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for i, doc in enumerate(results["documents"][0]):
            chunks.append({
                "text": doc,
                "source": results["metadatas"][0][i].get("source", "unknown"),
                "score": 1 - results["distances"][0][i],
                "method": "vector",
            })
        return chunks
    except Exception:
        return []


def bm25_search(query: str, top_k: int = 10) -> list:
    """Search by keywords using BM25."""
    if not os.path.exists(BM25_PATH):
        return []
    try:
        with open(BM25_PATH, "rb") as f:
            data = pickle.load(f)
        bm25 = data["bm25"]
        texts = data["texts"]
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        chunks = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunks.append({
                    "text": texts[idx],
                    "source": "bm25_index",
                    "score": float(scores[idx]),
                    "method": "bm25",
                })
        return chunks
    except Exception:
        return []


def rerank(query: str, chunks: list, top_k: int = 5) -> list:
    """Rerank chunks using cross-encoder."""
    if not chunks:
        return []
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


def search_documents(query: str) -> list:
    """Full document search pipeline."""
    vector_results = vector_search(query)
    bm25_results = bm25_search(query)

    # Merge results
    all_chunks = vector_results + bm25_results
    seen = set()
    unique_chunks = []
    for chunk in all_chunks:
        key = chunk["text"][:100]
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)

    return rerank(query, unique_chunks)


def has_documents() -> bool:
    """Check if any documents have been indexed."""
    return os.path.exists(BM25_PATH) and os.path.exists(CHROMA_PATH)