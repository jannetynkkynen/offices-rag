# test_index.py
import pickle
from pathlib import Path
from src.chat import load_index

index_dir = "index"
retriever = load_index(index_dir, use_local_override=True)

if retriever is None:
    print("❌ Retriever is None. Check errors above.")
else:
    print("✅ Retriever loaded successfully.")
    # try a simple query
    docs = retriever.invoke("test")
    print(f"Retrieved {len(docs)} documents.")