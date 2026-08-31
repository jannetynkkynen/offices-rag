#!/usr/bin/env python3
"""
Ingest OP branch Markdown files into a single-file SQLite store.

Replaces the pickle output with SQLite (stdlib, safe to load, FTS5 keyword search
built in) and folds in the corpus-specific handling: frontmatter metadata, image/
link stripping, soft-wrap repair, staff-heading demotion, duplicate removal, and
structured extraction of hours / capabilities / staff.

Docling's HybridChunker is still available via --chunker docling, but note that
for this corpus it is the weaker option: ~95% of these pages fit in a single
512-token chunk, and section-level chunking produces 3,665 fragments of which 67%
are byte-identical across documents. See --chunker native (the default).

Usage:
    python ingest.py --md-dir out/ --db offices.db
    python ingest.py --md-dir out/ --db offices.db --chunker docling
"""
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from langchain_core.documents import Document as LCDocument

from src.chunk_offices import parse, clean, fix_hierarchy, context_header
from src.chunker import chunk_markdown, HeuristicTokenizer, HFTokenizer
from src.enrich_metadata import enrich, to_filter_text
from src.build_sqlite import SCHEMA

# A multilingual tokenizer is mandatory here. The corpus is Finnish; an English
# tokenizer under-counts Finnish subwords by roughly 2x, so a "512-token" budget
# measured with bge-small-en silently yields chunks nearer 1,000 real tokens.
DEFAULT_TOKENIZER = "BAAI/bge-m3"


# ---------------------------------------------------------------- loading
def load_markdown_documents(directory: str) -> list[LCDocument]:
    """Load .md files, parsing YAML frontmatter into metadata rather than leaving
    it in the body. Left in place, the frontmatter is embedded as content and the
    `---` fences are parsed by Docling as horizontal rules."""
    docs = []
    for path in sorted(Path(directory).glob("*.md")):
        meta, body = parse(str(path))
        meta["source"] = str(path)
        docs.append(LCDocument(page_content=body, metadata=meta))
    return docs


# ---------------------------------------------------------------- chunkers
def chunk_native(meta: dict, cleaned: str, tk, budget: int, overlap: int = 0):
    """Document-as-chunk with a context header; splits only pages that overflow.
    Takes ALREADY-CLEANED text -- cleaning happens once, in the caller."""
    return [
        LCDocument(
            page_content=c.text,
            metadata={**meta, "part": c.part, "n_parts": c.n_parts,
                      "heading_path": c.heading_path, "chunk_id": c.chunk_id,
                      "n_tokens": c.n_tokens},
        )
        for c in chunk_markdown(cleaned, meta["id"], header=context_header(meta),
                                budget=budget, tokenizer=tk, overlap=overlap)
    ]


def make_docling_chunker(tokenizer_name: str, max_tokens: int):
    """Build a HybridChunker correctly. Three things the obvious call gets wrong:

    1. `max_tokens` is NOT a HybridChunker field. Pydantic's default is
       extra='ignore', so HybridChunker(max_tokens=512) silently discards it and
       falls back to the tokenizer's model_max_length. That is 512 for
       bge-small-en (so it looks fine) but 8192 for bge-m3 -- switching to a
       multilingual model would silently stop splitting anything at all.
       The limit belongs on the tokenizer object.
    2. `tokenizer` expects a BaseTokenizer instance, not a string.
    3. Chunk text for embedding must come from chunker.contextualize(chunk),
       not chunk.text -- see below.
    """
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer

    tok = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_name),
        max_tokens=max_tokens,          # <- the limit lives HERE
    )
    return HybridChunker(tokenizer=tok, merge_peers=True)


def chunk_docling(chunker, converter, meta: dict, body: str) -> list[LCDocument]:
    """Chunk one document with Docling. The converter and chunker are built once
    by the caller -- constructing DocumentConverter() per document, as is easy to
    do inside a loop, re-initialises the whole pipeline on every file."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import DocumentStream
    import io

    stream = DocumentStream(name=f"{meta.get('id', 'doc')}.md",
                            stream=io.BytesIO(body.encode("utf-8")))
    result = converter.convert(stream)
    out = []
    for chunk in chunker.chunk(result.document):
        # chunk.meta is a DocMeta pydantic model, NOT a dict -- an isinstance
        # check against dict is always False and quietly drops the headings.
        dm = chunk.meta.export_json_dict() if hasattr(chunk.meta, "export_json_dict") else {}
        out.append(LCDocument(
            page_content=chunk.text,                       # what you display
            metadata={**meta,
                      "heading_path": dm.get("headings", []),
                      # contextualize() prepends the heading path. Embedding
                      # chunk.text instead is the single most common Docling
                      # mistake, and it is fatal on this corpus: the bare section
                      # bodies are 67% duplicated across documents.
                      "embed_text": chunker.contextualize(chunk)},
        ))
    return out


# ---------------------------------------------------------------- ingest
def ingest(md_dir: str, db_path: str, chunker_kind: str = "native",
           max_tokens: int = 512, tokenizer: str = DEFAULT_TOKENIZER,
           overlap: int = 0, exact_tokenizer: bool = False):
    # exact counts when the tokenizer is available, labelled heuristic otherwise
    try:
        tk = HFTokenizer(tokenizer) if exact_tokenizer else HeuristicTokenizer()
    except Exception as e:
        print(f"tokenizer unavailable ({type(e).__name__}); using heuristic estimate")
        tk = HeuristicTokenizer()
    docs = load_markdown_documents(md_dir)
    print(f"Loaded {len(docs)} markdown documents")

    dchunker = dconverter = None
    if chunker_kind == "docling":
        from docling.document_converter import DocumentConverter
        dconverter = DocumentConverter()                   # built ONCE
        dchunker = make_docling_chunker(tokenizer, max_tokens)

    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)

    seen, n_chunks, n_dupes, n_failed = set(), 0, 0, 0
    for doc in docs:
        meta, body = doc.metadata, doc.page_content
        cleaned = fix_hierarchy(clean(body))

        if cleaned in seen:            # 10 byte-identical pages in this corpus
            n_dupes += 1
            continue
        seen.add(cleaned)

        md = enrich(meta, body, cleaned)
        did = md["doc_id"]
        con.execute("INSERT OR IGNORE INTO docs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (did, md["url"], md["title"], md["language"], md["summary"],
                     md["page_type"], md["org"], md["org_label"], md["branch_slug"],
                     md.get("branch_no"), md.get("coop_no"), md.get("street"),
                     md.get("postcode"), md.get("city"),
                     json.dumps(md["breadcrumbs"], ensure_ascii=False),
                     json.dumps(md["linked_pdfs"], ensure_ascii=False)))
        con.executemany("INSERT OR IGNORE INTO capabilities VALUES (?,?)",
                        [(did, c) for c in md["capabilities"]])
        con.executemany("INSERT OR IGNORE INTO hours VALUES (?,?,?,?,?)",
                        [(did, s, d, o, c) for s, w in md["hours"].items()
                         for d, (o, c) in w.items()])
        con.executemany("INSERT INTO staff VALUES (?,?,?,?,?,?)",
                        [(did, p["name"], p["role"], p["phone"], p["email"],
                          p["department"]) for p in md["staff"]])
        con.executemany("INSERT INTO phones VALUES (?,?)",
                        [(did, p) for p in md["phones"]])

        ftext = to_filter_text(md, include_title=False)
        if chunker_kind == "docling":
            try:
                lc_chunks = chunk_docling(dchunker, dconverter, meta, cleaned)
            except Exception as e:                   # narrow: conversion only
                print(f"  ! docling failed on {did[:8]}: {type(e).__name__}: {e}")
                print("    falling back to native chunking for this document")
                n_failed += 1
                lc_chunks = chunk_native(meta, cleaned, tk, max_tokens, overlap)
        else:
            lc_chunks = chunk_native(meta, cleaned, tk, max_tokens, overlap)

        for i, c in enumerate(lc_chunks):
            base = c.metadata.get("embed_text", c.page_content)
            etext = f"{ftext}\n\n{base}"
            con.execute(
                "INSERT INTO chunks (chunk_id,doc_id,part,n_parts,heading_path,"
                "text,embed_text,n_tokens) VALUES (?,?,?,?,?,?,?,?)",
                (c.metadata.get("chunk_id", f"{did}#{i}"), did, c.metadata.get("part", i),
                 c.metadata.get("n_parts", len(lc_chunks)),
                 json.dumps(c.metadata.get("heading_path", []), ensure_ascii=False),
                 c.page_content, etext, tk.count(etext)))
            n_chunks += 1

    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.execute("VACUUM")

    print(f"\nchunker      : {chunker_kind}")
    print(f"duplicates   : {n_dupes} pages skipped")
    if n_failed:
        print(f"docling fails: {n_failed} (fell back to native)")
    for t in ("docs", "chunks", "capabilities", "hours", "staff", "phones"):
        print(f"  {t:14s} {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:5d}")
    print(f"\nwrote {db_path} ({os.path.getsize(db_path) / 1e6:.1f} MB)")
    con.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ingest Markdown into a SQLite store.")
    p.add_argument("--md-dir", required=True, help="Directory containing .md files")
    p.add_argument("--db", default="offices.db", help="Output SQLite file")
    p.add_argument("--chunker", choices=["native", "docling"], default="native",
                   help="native = document-as-chunk (recommended for this corpus)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--overlap", type=int, default=0, help="token overlap between parts")
    p.add_argument("--exact-tokens", action="store_true",
                   help="count with the real tokenizer instead of the heuristic")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                   help="must be multilingual; the corpus is Finnish")
    a = p.parse_args()

    if "-en" in a.tokenizer or a.tokenizer.endswith("en-v1.5"):
        print(f"WARNING: {a.tokenizer} is an English tokenizer and will "
              f"under-count Finnish subwords by roughly 2x.\n", file=sys.stderr)

    ingest(a.md_dir, a.db, a.chunker, a.max_tokens, a.tokenizer,
           a.overlap, a.exact_tokens)
