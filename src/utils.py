"""Shared utilities for OP branch RAG chatbot."""
import re
import frontmatter
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from markdown content."""
    match = re.match(r"^---\n.*?\n---\n(.*)", content, re.DOTALL)
    return match.group(1).strip() if match else content.strip()


def load_markdown_documents(directory: str) -> List[Document]:
    """Load all .md files, strip frontmatter, return LangChain Documents."""
    docs = []
    for md_path in Path(directory).glob("*.md"):
        with open(md_path, "r", encoding="utf-8") as f:
            post = frontmatter.load(f)
        body = strip_frontmatter(post.content)
        # Preserve all frontmatter as metadata
        metadata = post.metadata.copy()
        metadata["source_file"] = md_path.name
        # ensure required fields exist
        metadata.setdefault("branch_name", "")
        metadata.setdefault("address", "")
        metadata.setdefault("url", "")
        docs.append(Document(page_content=body, metadata=metadata))
    return docs


def split_documents(docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)