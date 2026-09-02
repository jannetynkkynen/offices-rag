#!/usr/bin/env python3
"""
Simple RAG chatbot for OP branch information.
Uses hybrid retriever + local LLM with optional reranking (BGE-Reranker-v2-M3).
"""
import argparse
import pickle
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Local reranker
from FlagEmbedding import FlagReranker


def load_index(
    index_dir: str,
    use_local_override: bool = None,
    embedding_model_override: str = None,
):
    index_path = Path(index_dir)

    with open(index_path / "config.pkl", "rb") as f:
        config = pickle.load(f)

    use_local = config.get("use_local", False)
    embedding_model = config.get("embedding_model", "snowflake-arctic-embed2")
    hf_model = config.get("hf_embedding_model", None)

    if use_local_override is not None:
        use_local = use_local_override
    if embedding_model_override:
        embedding_model = embedding_model_override

    if hf_model:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=hf_model,
            model_kwargs={"device": "mps"},
            encode_kwargs={"normalize_embeddings": True},
        )
    elif use_local:
        embeddings = OllamaEmbeddings(model=embedding_model)
    else:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.load_local(
        str(index_path / "faiss_index"),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": config["vector_k"]})

    with open(index_path / "bm25_docs.pkl", "rb") as f:
        bm25_docs = pickle.load(f)
    bm25_retriever = BM25Retriever.from_documents(bm25_docs, k=config["bm25_k"])

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=config["weights"],
    )
    return ensemble


def load_system_prompt(prompt_file: str = "system_prompt.md") -> str:
    if Path(prompt_file).exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        return (
            "Olet suomenkielinen OP-pankin asiakaspalvelija. "
            "VASTAA AINA SUOMEKSI. Älä koskaan käytä englantia. "
            "Vastaa vain annetun kontekstin perusteella."
        )


def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        url = doc.metadata.get("url", "")
        content = doc.page_content
        if url:
            formatted.append(f"--- LÄHDE: {source} ---\nURL: {url}\n{content}\n--- LÄHTEEN LOPPU ---")
        else:
            formatted.append(f"--- LÄHDE: {source} ---\n{content}\n--- LÄHTEEN LOPPU ---")
    return "\n\n".join(formatted)


def expand_query(query: str) -> str:
    expanded = query
    if "konttori" in query.lower():
        expanded += " pankki toimipiste branch"
    if "aukiolo" in query.lower() or "auki" in query.lower():
        expanded += " aukioloajat opening hours"
    expanded += " op pankki branch"
    return expanded


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True):
        """
        Local reranker using BGE-Reranker-v2-M3 (multilingual, excellent for Finnish).
        """
        try:
            self.model = FlagReranker(model_name, use_fp16=use_fp16)
            print(f"🔄 Loaded reranker: {model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load reranker: {e}. Install FlagEmbedding: pip install -U FlagEmbedding")

    def rerank(self, query: str, documents: list, top_n: int = 5) -> list:
        if not documents:
            return documents

        try:
            pairs = [[query, doc.page_content] for doc in documents]
            scores = self.model.compute_score(pairs, normalize=True)
            scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored[:top_n]]
        except Exception as e:
            print(f"⚠️ Reranking failed: {e}. Returning top {top_n} original documents.")
            return documents[:top_n]


def create_rag_chain(
    retriever,
    use_local: bool = False,
    local_model: str = "qwen2.5:7b",
    openai_model: str = "gpt-4o-mini",
    system_prompt: str = "",
    rerank: bool = False,
    top_k_rerank: int = 5,
):
    if use_local:
        llm = ChatOllama(model=local_model, temperature=0.1, num_predict=200)
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=openai_model, temperature=0)

    reranker = None
    if rerank:
        reranker = Reranker()  # uses default BGE model

    def get_context(question):
        expanded = expand_query(question)
        docs = retriever.invoke(expanded)

        # Rerank if enabled
        if reranker is not None:
            docs = reranker.rerank(question, docs, top_n=top_k_rerank)

        # Truncate document content to avoid token overflow
        MAX_DOCS = 5
        MAX_CONTENT_LEN = 500
        for doc in docs:
            if len(doc.page_content) > MAX_CONTENT_LEN:
                doc.page_content = doc.page_content[:MAX_CONTENT_LEN] + "..."
        if len(docs) > MAX_DOCS:
            docs = docs[:MAX_DOCS]

        return format_docs(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Konteksti:\n{context}\n\nKysymys: {question}"),
    ])

    chain = (
        {
            "context": lambda x: get_context(x["question"]),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def interactive_chat(chain):
    print("\n🤖 OP Branch Chatbot (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue
        try:
            response = chain.invoke({"question": user_input})
            print(f"🤖 Agent: {response}\n")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG chatbot with optional reranking.")
    parser.add_argument("--index-dir", default="index", help="Directory containing saved index")
    parser.add_argument("--use-local", action="store_true", help="Use local models (Ollama)")
    parser.add_argument("--local-model", default="qwen2.5:7b", help="Local LLM model name")
    parser.add_argument("--embedding-model", default=None, help="Override embedding model")
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--system-prompt", default="system_prompt.md", help="Path to system prompt file")
    parser.add_argument("--rerank", action="store_true", help="Enable reranking (uses BGE-Reranker-v2-M3 locally)")
    parser.add_argument("--top-k-rerank", type=int, default=5, help="Top documents after reranking")
    args = parser.parse_args()

    hybrid_retriever = load_index(
        args.index_dir,
        use_local_override=args.use_local,
        embedding_model_override=args.embedding_model,
    )
    system_prompt = load_system_prompt(args.system_prompt)

    chain = create_rag_chain(
        hybrid_retriever,
        use_local=args.use_local,
        local_model=args.local_model,
        openai_model=args.openai_model,
        system_prompt=system_prompt,
        rerank=args.rerank,
        top_k_rerank=args.top_k_rerank,
    )
    interactive_chat(chain)