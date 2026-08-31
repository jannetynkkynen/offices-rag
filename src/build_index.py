#!/usr/bin/env python3
"""
Build a hybrid retriever (BM25 + FAISS) from chunked documents and save to disk.
Supports both OpenAI and local embeddings (Ollama).
"""
import argparse
import pickle
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.embeddings import OllamaEmbeddings


def build_hybrid_retriever(
    chunks,
    use_local: bool = False,
    embedding_model: str = "snowflake-arctic-embed2",
    openai_embedding_model: str = "text-embedding-3-small",
    bm25_k: int = 4,
    vector_k: int = 4,
    weights: list = None,
):
    if weights is None:
        weights = [0.5, 0.5]

    print("Building FAISS vector index...")
    if use_local:
        embeddings = OllamaEmbeddings(model=embedding_model)
        print(f"🔍 Using local embeddings: {embedding_model}")
    else:
        embeddings = OpenAIEmbeddings(model=openai_embedding_model)
        print(f"🔍 Using OpenAI embeddings: {openai_embedding_model}")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": vector_k})

    print("Building BM25 retriever...")
    bm25_retriever = BM25Retriever.from_documents(chunks, k=bm25_k)

    print("Creating EnsembleRetriever...")
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=weights,
    )
    return ensemble, vectorstore


def save_index(ensemble, vectorstore, save_dir: str, use_local: bool, embedding_model: str):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(save_path / "faiss_index"))

    bm25_docs = ensemble.retrievers[0].docs
    with open(save_path / "bm25_docs.pkl", "wb") as f:
        pickle.dump(bm25_docs, f)

    config = {
        "weights": ensemble.weights,
        "bm25_k": ensemble.retrievers[0].k,
        "vector_k": ensemble.retrievers[1].search_kwargs["k"],
        "use_local": use_local,
        "embedding_model": embedding_model,
    }
    with open(save_path / "config.pkl", "wb") as f:
        pickle.dump(config, f)

    print(f"✅ Index saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build hybrid index from chunks.")
    parser.add_argument("--chunks", default="chunks.pkl", help="Pickle file with chunks")
    parser.add_argument("--save-dir", default="index", help="Directory to save index")
    parser.add_argument("--use-local", action="store_true", help="Use local embeddings (Ollama)")
    parser.add_argument("--embedding-model", default="nomic-embed-text", help="Local embedding model name")
    parser.add_argument("--openai-embedding-model", default="text-embedding-3-small")
    parser.add_argument("--bm25-k", type=int, default=4)
    parser.add_argument("--vector-k", type=int, default=4)
    parser.add_argument("--weights", nargs=2, type=float, default=[0.5, 0.5])
    args = parser.parse_args()

    with open(args.chunks, "rb") as f:
        chunks = pickle.load(f)
    print(f"📂 Loaded {len(chunks)} chunks")

    ensemble, vectorstore = build_hybrid_retriever(
        chunks,
        use_local=args.use_local,
        embedding_model=args.embedding_model,
        openai_embedding_model=args.openai_embedding_model,
        bm25_k=args.bm25_k,
        vector_k=args.vector_k,
        weights=args.weights,
    )
    save_index(ensemble, vectorstore, args.save_dir, args.use_local, args.embedding_model)