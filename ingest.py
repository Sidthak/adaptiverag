# ingest.py
# Loads your documents into ChromaDB and BM25 index.
# Run this once after adding files to ./docs folder.

import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DOCS_PATH = "./docs"
BM25_PATH = "./bm25_index.pkl"
CHROMA_PATH = "./chroma_db"


def load_documents():
    print("📂 Loading documents...")
    documents = []
    for file_path in Path(DOCS_PATH).rglob("*"):
        if file_path.suffix.lower() == ".pdf":
            print(f"  Loading PDF: {file_path.name}")
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())
        elif file_path.suffix.lower() in [".txt", ".md"]:
            print(f"  Loading text: {file_path.name}")
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())
    print(f"✅ Loaded {len(documents)} pages\n")
    return documents


def chunk_documents(documents):
    print("✂️  Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks\n")
    return chunks


def embed_text(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def store_in_chroma(chunks):
    print("🔢 Storing in ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        chroma_client.delete_collection("adaptiverag")
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection("adaptiverag")
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        texts = [c.page_content for c in batch]
        metadatas = [{"source": str(c.metadata.get("source", "unknown"))} for c in batch]
        embeddings = [embed_text(t) for t in texts]
        collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        print(f"   Stored chunks {i + 1} → {i + len(batch)}")
    print("✅ ChromaDB ready!\n")


def build_bm25_index(chunks):
    print("🔍 Building BM25 index...")
    texts = [c.page_content for c in chunks]
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "texts": texts}, f)
    print("✅ BM25 index saved!\n")


def main():
    os.makedirs(DOCS_PATH, exist_ok=True)
    files = (list(Path(DOCS_PATH).rglob("*.pdf")) +
             list(Path(DOCS_PATH).rglob("*.txt")) +
             list(Path(DOCS_PATH).rglob("*.md")))
    if not files:
        print("⚠️  No documents found in ./docs!")
        print("   Add your PDF or .txt files there first.")
        return
    documents = load_documents()
    chunks = chunk_documents(documents)
    store_in_chroma(chunks)
    build_bm25_index(chunks)
    print("🎉 Done! Now run: streamlit run app.py")


if __name__ == "__main__":
    main()