import pickle
from pathlib import Path
from src.chat import load_index

# Load the index
index_dir = "index"
retriever = load_index(index_dir, use_local_override=True)

# Test query
query = "Mikkelin konttori"
docs = retriever.invoke(query)

print(f"Found {len(docs)} documents for '{query}':")
for i, doc in enumerate(docs):
    src = doc.metadata.get('source_file', 'unknown')
    content_preview = doc.page_content[:200].replace('\n', ' ')
    print(f"{i+1}. {src}: {content_preview}...")