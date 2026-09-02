# OP Branch RAG Chatbot

Agentic RAG chatbot for OP bank branch documentation using hybrid search (BM25 + FAISS) with Finnish language support.

## Features

- **Hybrid Retrieval**: Combines BM25 (keyword) and FAISS (vector) search for optimal results
- **Finnish Language Support**: Optimized for Finnish queries and responses
- **Local LLM Support**: Works with Ollama models (Qwen, Llama-Poro, etc.)
- **Reranking (Optional)**: Support for vLLM or HuggingFace rerankers
- **Markdown Ingestion**: Processes OP bank branch markdown files with frontmatter
- **Docling Chunking**: Structure-aware document chunking for better retrieval

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd op-offices-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Ollama

```bash
# Install Ollama (if not already)
# Visit https://ollama.com for installation instructions

# Pull a Finnish-optimized model
ollama pull qwen2.5:7b
# or
ollama pull mradermacher/Llama-Poro-2-8B-Instruct-GGUF

# Pull an embedding model
ollama pull snowflake-arctic-embed2
# or
ollama pull nomic-embed-text
```

### 3. Prepare Data

Place your OP branch markdown files in `data/branches/`:

```
data/branches/
├── mikkeli.md
├── raasepori.md
├── kempele-zeppelin.md
└── ...
```

### 4. Run the Pipeline

```bash
# 1. Enrich metadata (optional)
python -m src.enrich_metadata --md-dir data/branches

# 2. Ingest and chunk documents
python -m src.ingest --md-dir data/branches --output chunks.pkl --max-tokens 512 --follow-links                             
# 3. Build the hybrid index
python -m src.build_index --chunks chunks.pkl --save-dir index \
    --use-local --embedding-model snowflake-arctic-embed2 \
    --bm25-k 30 --vector-k 30 --weights 0.5 0.5

# 4. Start the chatbot
python -m src.chat --index-dir index --use-local --local-model Llama-Poro-2-8B-Instruct-GGUF
```

### 5. Chat

```
Sinä: Mikä on Mikkelin konttorin osoite?
🤖 Avustaja: Mikkelin konttorin osoite on Porrassalmenkatu 19, 50100 Mikkeli.
Lähde: mikkeli.md – https://www.op.fi/osuuspankit/op-suur-savo/konttorit/mikkelin-konttori/
```

## Project Structure

```
op-offices-rag/
├── data/
│   └── branches/          # Your markdown files
├── src/
│   ├── __init__.py
│   ├── utils.py           # Shared utilities
│   ├── enrich_metadata.py # Add metadata to MD files
│   ├── ingest.py          # Chunk documents with Docling
│   ├── build_index.py     # Build hybrid index (BM25 + FAISS)
│   └── chat.py            # Chatbot with reranking support
├── index/                 # Generated hybrid index
├── system_prompt.md       # Instructions for the LLM
├── requirements.txt
└── README.md
```

## Configuration

### Index Building Options

| Option | Description | Default |
|--------|-------------|---------|
| `--use-local` | Use local Ollama embeddings | `False` |
| `--embedding-model` | Ollama embedding model | `snowflake-arctic-embed2` |
| `--hf-embedding-model` | HuggingFace embedding model | `None` |
| `--bm25-k` | Number of BM25 results | `30` |
| `--vector-k` | Number of vector results | `30` |
| `--weights` | BM25/vector weights | `0.5 0.5` |

### Chat Options

| Option | Description | Default |
|--------|-------------|---------|
| `--use-local` | Use local LLM (Ollama) | `False` |
| `--local-model` | Ollama model name | `qwen2.5:7b` |
| `--rerank` | Enable reranking | `False` |
| `--rerank-model` | Reranker model | `nvidia/llama-nemotron-rerank-1b-v2` |
| `--top-k-rerank` | Documents after reranking | `5` |

## Advanced Features

### Query Expansion

The chatbot automatically expands queries with synonyms:
- `"konttori"` → `"konttori pankki toimipiste"`
- `"aukiolo"` → `"aukiolo aukioloajat"`

### Reranking (Experimental)

Note: Do not use reranking unless you can find a reranker suitable for Finnish.
For improved relevance, you can enable reranking with a cross-encoder:

```bash
# Start vLLM server (in separate terminal)
vllm serve nvidia/llama-nemotron-rerank-1b-v2 --trust-remote-code --port 8000

# Run chatbot with reranking
python -m src.chat --index-dir index --use-local --local-model qwen2.5:7b --rerank
```

### HuggingFace Embeddings

```bash
python -m src.build_index --chunks chunks.pkl --save-dir index \
    --hf-embedding-model intfloat/multilingual-e5-large \
    --bm25-k 30 --vector-k 30
```

## Troubleshooting

### Common Issues

**Q: The model answers in English instead of Finnish**  
A: Ensure `system_prompt.md` has strict Finnish instructions. Use `--local-model` with a Finnish-optimized model like `siloai/Llama-Poro-2-8B-Instruct-SFT`.

**Q: Retrieval returns wrong branch**  
A: Increase `bm25-k` and `vector-k` (e.g., `--bm25-k 50 --vector-k 50`). Rebuild the index with `--use-local --embedding-model snowflake-arctic-embed2`.

**Q: Ollama embedding error (EOF)**  
A: Restart Ollama (`ollama serve`) or try `nomic-embed-text` instead of `snowflake-arctic-embed2`.

**Q: Reranking server connection refused**  
A: Start vLLM server first or disable reranking (remove `--rerank` flag).

## License

MIT

## Support

For issues, please open an issue on the repository.