#!/usr/bin/env python3
"""
Ingest OP branch Markdown files and chunk them using Docling's HybridChunker.
Chunks are saved as a pickled list for later indexing.
"""
import argparse
import pickle
import tempfile
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from langchain_core.documents import Document as LCDocument

from src.utils import load_markdown_documents, split_documents


def chunk_with_docling(
    md_content: str,
    metadata: dict,
    max_tokens: int = 512,
    tokenizer: str = "BAAI/bge-small-en-v1.5",
) -> list[LCDocument]:
    """
    Convert markdown content to a DoclingDocument and chunk it using HybridChunker.
    Returns a list of LangChain Documents.
    """
    # Write markdown content to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
        tmp.write(md_content)
        tmp_path = tmp.name

    try:
        converter = DocumentConverter()
        result = converter.convert(tmp_path)
        doc = result.document

        chunker = HybridChunker(
            max_tokens=max_tokens,
            merge_peers=True,
            tokenizer=tokenizer,
        )
        chunks = list(chunker.chunk(doc))

        # Convert to LangChain Document format
        lc_chunks = []
        for chunk in chunks:
            # Access chunk metadata (Docling uses .meta, not .metadata)
            chunk_meta = getattr(chunk, "meta", {})
            if isinstance(chunk_meta, dict):
                combined_meta = {**metadata, **chunk_meta}
            else:
                # fallback if meta is something else
                combined_meta = metadata.copy()
                combined_meta["docling_meta"] = str(chunk_meta)

            lc_chunks.append(LCDocument(
                page_content=chunk.text,
                metadata=combined_meta,
            ))
        return lc_chunks

    except Exception as e:
        print(f"⚠️ Docling failed: {e}")
        print("↳ Falling back to simple splitting for this document.")
        # Fallback: simple split
        return split_documents(
            [LCDocument(page_content=md_content, metadata=metadata)],
            chunk_size=max_tokens * 2,
            chunk_overlap=50,
        )
    finally:
        # Clean up temp file
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass


def ingest_from_md(
    directory: str,
    max_tokens: int = 512,
    output: str = "chunks.pkl",
    use_docling: bool = True,
):
    """Load .md files, chunk with Docling (or fallback to simple split), save pickle."""
    docs = load_markdown_documents(directory)
    print(f"Loaded {len(docs)} markdown documents")

    all_chunks = []
    for doc in docs:
        if use_docling:
            chunks = chunk_with_docling(doc.page_content, doc.metadata, max_tokens)
        else:
            chunks = split_documents([doc], chunk_size=max_tokens * 2, chunk_overlap=50)
        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunks (Docling mode: {use_docling})")

    with open(output, "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"Saved chunks to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Markdown files with Docling chunking.")
    parser.add_argument("--md-dir", required=True, help="Directory containing .md files")
    parser.add_argument("--output", default="chunks.pkl", help="Output pickle file")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per chunk")
    parser.add_argument("--no-docling", action="store_true", help="Disable Docling (use simple split)")
    args = parser.parse_args()

    ingest_from_md(
        directory=args.md_dir,
        max_tokens=args.max_tokens,
        output=args.output,
        use_docling=not args.no_docling,
    )