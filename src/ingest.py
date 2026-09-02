#!/usr/bin/env python3
"""
Ingest documents from local Markdown files and/or the web, chunk them with
Docling's HybridChunker, and save the chunks as a pickled list for later indexing.

Sources can be:
  * a directory of .md files            (--md-dir)
  * explicit URLs                       (--url / --url-file)
  * every link found inside the .md files (--follow-links)

Docling's DocumentConverter takes an http(s) URL directly, so remote PDFs, HTML
pages, DOCX and PPTX all go through the same parse -> chunk path as local files.

Examples
--------
    # local markdown only
    python ingest.py --md-dir data/branches

    # see which links the markdown contains, without fetching anything
    python ingest.py --md-dir data/branches --follow-links --list-links

    # follow them, but only within one domain
    python ingest.py --md-dir data/branches --follow-links --link-domain https://www.op.fi/henkiloasiakkaat/asiakaspalvelu/konttorit

    # follow links, and links found on those pages too (a real crawl)
    python ingest.py --md-dir data/branches --follow-links --link-depth 2 \
                     --link-domain example.com --max-links 200 --delay 0.5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from langchain_core.documents import Document as LCDocument

from src.utils import load_markdown_documents, split_documents

DEFAULT_TOKENIZER = "BAAI/bge-small-en-v1.5"
USER_AGENT = "docling-ingest"

# Extensions Docling can't turn into useful text — skip them when following links.
ASSET_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".bmp", ".tiff",
    ".css", ".js", ".mjs", ".map", ".json", ".xml", ".rss", ".atom",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".mp3", ".mp4", ".webm", ".mov", ".avi", ".wav",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".exe", ".dmg", ".deb", ".rpm", ".iso",
}


# --------------------------------------------------------------------------- #
# link extraction
# --------------------------------------------------------------------------- #
FENCED_CODE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")

INLINE_LINK_RE = re.compile(r"\[[^\]]*\]\(\s*<?([^()\s>]+)>?[^)]*\)")
REF_LINK_RE = re.compile(r"^[ \t]{0,3}\[[^\]]+\]:[ \t]*<?(\S+)>?", re.MULTILINE)
AUTOLINK_RE = re.compile(r"<(https?://[^>\s]+)>")
HTML_HREF_RE = re.compile(r"<a\b[^>]*?href\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
# The body stops at ")" so a URL inside (parens) still terminates correctly;
# markdown inline links are caught by INLINE_LINK_RE and de-duplicated later.
BARE_URL_RE = re.compile(r"""(?<![\w<\["'])https?://[^\s<>"'\]),]+""")

_LINK_PATTERNS = (
    INLINE_LINK_RE,
    REF_LINK_RE,
    AUTOLINK_RE,
    HTML_HREF_RE,
    BARE_URL_RE,
)


def is_url(value: str) -> bool:
    """True for http(s) URLs."""
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def normalize_url(raw: str, base: str | None = None) -> str | None:
    """
    Clean up one raw href. Returns a canonical absolute http(s) URL, or None if
    it isn't something worth fetching (anchor, mailto:, unresolvable relative
    link, ...). Fragments are dropped so #section links don't look like
    separate pages.
    """
    url = raw.strip().strip("<>").rstrip(".,;:!?\"'")
    if not url or url.startswith("#"):
        return None

    scheme = urlparse(url).scheme.lower()
    if scheme and scheme not in ("http", "https"):
        return None  # mailto:, tel:, javascript:, data:, ...
    if not scheme:
        if base is None:
            return None  # relative link with nothing to resolve against
        url = urljoin(base, url)

    url, _ = urldefrag(url)
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    return urlunparse(
        (parts.scheme, parts.netloc.lower(), parts.path or "/", parts.params, parts.query, "")
    )


def extract_links(text: str, base: str | None = None) -> list[str]:
    """
    Pull every link out of a chunk of Markdown: inline [x](url), reference
    definitions, <autolinks>, raw <a href> tags and bare URLs. Code blocks and
    inline code spans are stripped first so example URLs don't get ingested.
    Returns normalized absolute URLs, de-duplicated, in document order.
    """
    stripped = INLINE_CODE_RE.sub(" ", FENCED_CODE_RE.sub(" ", text))

    found: list[str] = []
    for pattern in _LINK_PATTERNS:
        for match in pattern.finditer(stripped):
            found.append(match.group(1) if pattern.groups else match.group(0))

    out: list[str] = []
    seen: set[str] = set()
    for raw in found:
        url = normalize_url(raw, base)
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out


def is_asset(url: str) -> bool:
    return Path(urlparse(url).path).suffix.lower() in ASSET_EXTENSIONS


def link_allowed(
    url: str,
    domains: list[str] | None,
    exclude: list[re.Pattern] | None,
    skip_assets: bool = True,
) -> bool:
    """Apply the --link-domain / --link-exclude filters to a candidate URL."""
    if skip_assets and is_asset(url):
        return False
    if domains:
        host = urlparse(url).netloc.lower()
        if not any(host == d or host.endswith("." + d) for d in domains):
            return False
    if exclude and any(rx.search(url) for rx in exclude):
        return False
    return True


class RobotsCache:
    """Minimal robots.txt check, one parser per host. Unreachable robots = allow."""

    def __init__(self, enabled: bool = True, user_agent: str = USER_AGENT):
        self.enabled = enabled
        self.user_agent = user_agent
        self._cache: dict[str, RobotFileParser | None] = {}

    def allowed(self, url: str) -> bool:
        if not self.enabled:
            return True
        parts = urlparse(url)
        root = f"{parts.scheme}://{parts.netloc}"
        if root not in self._cache:
            parser: RobotFileParser | None = RobotFileParser()
            parser.set_url(root + "/robots.txt")
            try:
                parser.read()
            except Exception:
                parser = None
            self._cache[root] = parser
        parser = self._cache[root]
        if parser is None:
            return True
        try:
            return parser.can_fetch(self.user_agent, url)
        except Exception:
            return True


def read_url_file(path: str) -> list[str]:
    """Read a newline-separated list of URLs, ignoring blanks and # comments."""
    urls = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not is_url(line):
            print(f"⚠️  Skipping non-URL line in {path}: {line}")
            continue
        urls.append(line)
    return urls


# --------------------------------------------------------------------------- #
# saving fetched pages as markdown
# --------------------------------------------------------------------------- #
def url_to_filename(url: str, suffix: str = ".md") -> str:
    """
    Readable, collision-free filename for a URL:
        https://docs.example.com/guide/intro.html
        -> docs.example.com-guide-intro.html-3f2a1b9c.md
    The hash covers the full URL, so query strings and long paths stay distinct
    even after the readable part is truncated.
    """
    parts = urlparse(url)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{parts.netloc}{parts.path}").strip("-._")
    slug = slug[:120] or "page"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}{suffix}"


def save_markdown(
    directory: Path,
    url: str,
    markdown: str,
    metadata: dict,
    front_matter: bool = True,
) -> Path:
    """Write one fetched page to <directory>/<slug>.md and return the path."""
    path = directory / url_to_filename(url)
    body = markdown
    if front_matter:
        header = {
            "source": url,
            "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **{k: v for k, v in metadata.items() if k in ("depth", "referrer")},
        }
        lines = "\n".join(f"{k}: {v}" for k, v in header.items())
        body = f"---\n{lines}\n---\n\n{markdown}"
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# chunking helpers
# --------------------------------------------------------------------------- #
def build_chunker(max_tokens: int, tokenizer: str) -> HybridChunker:
    """
    Newer docling-core wants a tokenizer object and reads max_tokens off it;
    older versions take the model name plus max_tokens.
    """
    try:
        from docling_core.transforms.chunker.tokenizer.huggingface import (
            HuggingFaceTokenizer,
        )

        tok = HuggingFaceTokenizer.from_pretrained(
            model_name=tokenizer,
            max_tokens=max_tokens,
        )
        return HybridChunker(tokenizer=tok, merge_peers=True)
    except Exception:
        return HybridChunker(tokenizer=tokenizer, max_tokens=max_tokens, merge_peers=True)


def _meta_as_dict(chunk) -> dict:
    """Docling chunk .meta is a pydantic model, not a dict. Normalise it."""
    meta = getattr(chunk, "meta", None)
    if meta is None:
        return {}
    if isinstance(meta, dict):
        return meta
    for attr, kwargs in (("export_json_dict", {}), ("model_dump", {"mode": "json"})):
        fn = getattr(meta, attr, None)
        if callable(fn):
            try:
                return fn(**kwargs)
            except Exception:
                continue
    return {"docling_meta": str(meta)}


def chunk_metadata(chunk, base: dict, full_meta: bool = False) -> dict:
    """Merge the source metadata with the useful bits of Docling's chunk meta."""
    combined = dict(base)
    raw = _meta_as_dict(chunk)
    if not raw:
        return combined

    headings = raw.get("headings") or []
    captions = raw.get("captions") or []
    if headings:
        combined["headings"] = headings
    if captions:
        combined["captions"] = captions

    pages = sorted(
        {
            prov["page_no"]
            for item in raw.get("doc_items", []) or []
            for prov in (item.get("prov") or [])
            if isinstance(prov, dict) and "page_no" in prov
        }
    )
    if pages:
        combined["pages"] = pages

    origin = raw.get("origin") or {}
    if isinstance(origin, dict) and origin.get("filename"):
        combined.setdefault("docling_filename", origin["filename"])

    if full_meta:
        combined["docling_meta"] = raw
    return combined


def chunk_text(chunker: HybridChunker, chunk) -> str:
    """
    contextualize() prepends the heading trail to the chunk body, which helps
    retrieval. Fall back to raw text on older versions.
    """
    try:
        return chunker.contextualize(chunk=chunk)
    except Exception:
        return chunk.text


def chunks_from_doc(
    doc,
    metadata: dict,
    chunker: HybridChunker | None,
    max_tokens: int = 512,
    full_meta: bool = False,
    markdown: str | None = None,
) -> list[LCDocument]:
    """
    Chunk an already-converted DoclingDocument into LangChain Documents.
    Pass `markdown` if it has already been exported, to avoid doing it twice.
    """
    if chunker is None:
        text = markdown if markdown is not None else doc.export_to_markdown()
        return split_documents(
            [LCDocument(page_content=text, metadata=metadata)],
            chunk_size=max_tokens * 2,
            chunk_overlap=50,
        )
    return [
        LCDocument(
            page_content=chunk_text(chunker, chunk),
            metadata=chunk_metadata(chunk, metadata, full_meta),
        )
        for chunk in chunker.chunk(doc)
    ]


def chunk_markdown_doc(
    doc: LCDocument,
    converter: DocumentConverter,
    chunker: HybridChunker | None,
    max_tokens: int = 512,
    full_meta: bool = False,
) -> list[LCDocument]:
    """
    Chunk an already-loaded Markdown document. Converts the file on disk when
    the metadata points at one, otherwise writes the content to a temp file so
    Docling has something to convert.
    """
    if chunker is None:
        return split_documents([doc], chunk_size=max_tokens * 2, chunk_overlap=50)

    source, tmp_path = None, None
    for key in ("source", "file_path", "path"):
        value = doc.metadata.get(key)
        if isinstance(value, str) and value and not is_url(value) and Path(value).is_file():
            source = value
            break

    if source is None:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        )
        tmp.write(doc.page_content)
        tmp.close()
        source, tmp_path = tmp.name, Path(tmp.name)

    try:
        result = converter.convert(source)
        return chunks_from_doc(result.document, doc.metadata, chunker, max_tokens, full_meta)
    except Exception as e:
        print(f"⚠️  Docling failed on {doc.metadata.get('source', '<memory>')}: {e}")
        print("↳ Falling back to simple splitting for this document.")
        return split_documents([doc], chunk_size=max_tokens * 2, chunk_overlap=50)
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
def ingest(
    md_dir: str | None = None,
    urls: list[str] | None = None,
    max_tokens: int = 512,
    output: str = "chunks.pkl",
    tokenizer: str = DEFAULT_TOKENIZER,
    use_docling: bool = True,
    full_meta: bool = False,
    follow_links: bool = False,
    link_depth: int = 1,
    link_domains: list[str] | None = None,
    link_exclude: list[str] | None = None,
    link_base: str | None = None,
    max_links: int | None = None,
    delay: float = 0.0,
    respect_robots: bool = True,
    list_links: bool = False,
    save_md: str | None = None,
    front_matter: bool = True,
    skip_saved: bool = False,
):
    """
    Chunk local .md files, then walk a frontier of URLs: the ones given on the
    command line plus, with --follow-links, every link found in the markdown.
    With link_depth > 1, links found on fetched pages are followed too.
    """
    urls = list(dict.fromkeys(urls or []))
    exclude = [re.compile(p) for p in (link_exclude or [])]
    domains = [d.lower().lstrip(".") for d in (link_domains or [])]
    robots = RobotsCache(enabled=respect_robots)

    # One converter and one chunker for the whole run — both are expensive to
    # build (the chunker downloads a tokenizer on first use).
    converter = DocumentConverter()
    chunker = build_chunker(max_tokens, tokenizer) if use_docling else None

    save_dir = Path(save_md).expanduser() if save_md else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    all_chunks: list[LCDocument] = []
    failures: list[tuple[str, str]] = []
    queued: set[str] = set()
    frontier: list[tuple[str, int, str | None]] = []  # (url, depth, referrer)

    def enqueue(url: str, depth: int, referrer: str | None) -> bool:
        canonical = normalize_url(url)
        if not canonical or canonical in queued:
            return False
        if not link_allowed(canonical, domains, exclude):
            return False
        queued.add(canonical)
        frontier.append((canonical, depth, referrer))
        return True

    for url in urls:
        enqueue(url, 1, None)

    # ---- local markdown -------------------------------------------------- #
    if md_dir:
        docs = load_markdown_documents(md_dir)
        print(f"Loaded {len(docs)} markdown documents from {md_dir}")
        for doc in docs:
            if not list_links:
                all_chunks.extend(
                    chunk_markdown_doc(doc, converter, chunker, max_tokens, full_meta)
                )
            if follow_links:
                origin = doc.metadata.get("source")
                base = link_base or (origin if isinstance(origin, str) and is_url(origin) else None)
                for link in extract_links(doc.page_content, base):
                    enqueue(link, 1, origin if isinstance(origin, str) else md_dir)
        if follow_links:
            print(f"Found {len(frontier)} link(s) in the markdown to follow")

    # ---- dry run --------------------------------------------------------- #
    if list_links:
        print(f"\n{len(frontier)} URL(s) queued (nothing fetched):")
        for url, _, referrer in frontier:
            print(f"  {url}" + (f"   <- {referrer}" if referrer else ""))
        return []

    # ---- walk the frontier ----------------------------------------------- #
    fetched = 0
    while frontier:
        if max_links is not None and fetched >= max_links:
            print(f"Reached --max-links ({max_links}); {len(frontier)} URL(s) left unvisited.")
            break

        url, depth, referrer = frontier.pop(0)

        if not robots.allowed(url):
            print(f"⏭️  robots.txt disallows {url}")
            continue

        fetched += 1
        total = fetched + len(frontier)
        print(f"[{fetched}/{total}] (depth {depth}) {url}")

        if save_dir is not None and skip_saved:
            existing = save_dir / url_to_filename(url)
            if existing.exists():
                print(f"    ⏭️  already saved at {existing.name}")
                continue

        try:
            result = converter.convert(url)
        except Exception as e:
            print(f"⚠️  Failed to fetch/convert {url}: {e}")
            failures.append((url, str(e)))
            continue

        doc = result.document
        metadata = {"source": url, "source_type": "url", "depth": depth}
        if referrer:
            metadata["referrer"] = referrer

        # Export once — needed for saving, for link discovery, and for the
        # simple-splitter path.
        md_text = None
        if save_dir is not None or depth < link_depth or chunker is None:
            try:
                md_text = doc.export_to_markdown()
            except Exception as e:
                print(f"    ⚠️  Markdown export failed for {url}: {e}")

        if save_dir is not None and md_text is not None:
            try:
                saved = save_markdown(save_dir, url, md_text, metadata, front_matter)
                metadata["saved_md"] = str(saved)
                manifest.append(
                    {
                        "url": url,
                        "file": saved.name,
                        "depth": depth,
                        "referrer": referrer,
                        "chars": len(md_text),
                    }
                )
                print(f"    💾 {saved.name}")
            except OSError as e:
                print(f"    ⚠️  Could not save markdown for {url}: {e}")

        chunks = chunks_from_doc(
            doc, metadata, chunker, max_tokens, full_meta, markdown=md_text
        )
        all_chunks.extend(chunks)
        print(f"    → {len(chunks)} chunks")

        if depth < link_depth:
            discovered = extract_links(md_text, base=url) if md_text else []
            added = sum(enqueue(link, depth + 1, url) for link in discovered)
            if added:
                print(f"    + {added} new link(s) queued")

        if delay:
            time.sleep(delay)

    # ---- save ------------------------------------------------------------ #
    print(f"\nCreated {len(all_chunks)} chunks (Docling chunking: {use_docling})")

    if save_dir is not None and manifest:
        manifest_path = save_dir / "manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for row in manifest:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved {len(manifest)} markdown file(s) to {save_dir} (index: {manifest_path.name})")

    if failures:
        print(f"{len(failures)} source(s) failed:")
        for url, err in failures:
            print(f"  - {url}: {err}")

    if not all_chunks:
        print("Nothing to save — no chunks were produced.")
        return []

    with open(output, "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"Saved chunks to {output}")
    return all_chunks


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ingest Markdown files and/or web URLs with Docling chunking."
    )
    parser.add_argument("--md-dir", help="Directory containing .md files")
    parser.add_argument(
        "--url", action="append", default=[], metavar="URL",
        help="URL to ingest (repeatable). PDF, HTML, DOCX, PPTX, ... all work.",
    )
    parser.add_argument("--url-file", help="File with one URL per line (# comments allowed)")
    parser.add_argument("--output", default="chunks.pkl", help="Output pickle file")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per chunk")
    parser.add_argument(
        "--tokenizer", default=DEFAULT_TOKENIZER,
        help="Tokenizer the chunker sizes against (match your embedding model)",
    )
    parser.add_argument(
        "--no-docling", action="store_true",
        help="Chunk with the simple splitter instead of HybridChunker "
             "(URLs are still parsed by Docling)",
    )
    parser.add_argument(
        "--full-meta", action="store_true",
        help="Keep Docling's complete chunk metadata (bigger pickle)",
    )

    links = parser.add_argument_group("link following")
    links.add_argument(
        "--follow-links", action="store_true",
        help="Also ingest every URL found inside the .md files",
    )
    links.add_argument(
        "--link-depth", type=int, default=1, metavar="N",
        help="1 = only the links in the markdown (default); "
             "2+ = also follow links found on those pages",
    )
    links.add_argument(
        "--link-domain", action="append", default=[], metavar="DOMAIN",
        help="Only follow links on this domain or its subdomains (repeatable)",
    )
    links.add_argument(
        "--link-exclude", action="append", default=[], metavar="REGEX",
        help="Skip URLs matching this pattern (repeatable)",
    )
    links.add_argument(
        "--link-base", metavar="URL",
        help="Base URL used to resolve relative links in the markdown",
    )
    links.add_argument("--max-links", type=int, help="Stop after fetching this many URLs")
    links.add_argument("--delay", type=float, default=0.0, help="Seconds to wait between fetches")
    links.add_argument(
        "--ignore-robots", action="store_true", help="Don't check robots.txt before fetching"
    )
    links.add_argument(
        "--list-links", action="store_true",
        help="Print the URLs that would be fetched and exit without fetching",
    )

    saving = parser.add_argument_group("saving fetched pages")
    saving.add_argument(
        "--save-md", metavar="DIR",
        help="Write each fetched page to DIR as a .md file, plus a manifest.jsonl index",
    )
    saving.add_argument(
        "--no-front-matter", action="store_true",
        help="Write the markdown without the source/fetched YAML header",
    )
    saving.add_argument(
        "--skip-saved", action="store_true",
        help="Don't re-fetch a URL whose .md file already exists in --save-md "
             "(cheap cache for re-runs)",
    )

    args = parser.parse_args(argv)

    urls = list(args.url)
    if args.url_file:
        urls.extend(read_url_file(args.url_file))

    bad = [u for u in urls if not is_url(u)]
    if bad:
        parser.error("not http(s) URLs: " + ", ".join(bad))
    if not args.md_dir and not urls:
        parser.error("give at least one of --md-dir, --url, --url-file")
    if args.follow_links and not args.md_dir:
        parser.error("--follow-links needs --md-dir")
    if args.link_depth < 1:
        parser.error("--link-depth must be >= 1")
    if args.skip_saved and not args.save_md:
        parser.error("--skip-saved needs --save-md")
    if args.save_md and args.md_dir:
        save_dir = Path(args.save_md).expanduser().resolve()
        md_dir = Path(args.md_dir).expanduser().resolve()
        if save_dir == md_dir or md_dir in save_dir.parents:
            parser.error(
                "--save-md must not write into --md-dir; the saved pages would be "
                "picked up as source documents on the next run"
            )

    ingest(
        md_dir=args.md_dir,
        urls=urls,
        max_tokens=args.max_tokens,
        output=args.output,
        tokenizer=args.tokenizer,
        use_docling=not args.no_docling,
        full_meta=args.full_meta,
        follow_links=args.follow_links,
        link_depth=args.link_depth,
        link_domains=args.link_domain,
        link_exclude=args.link_exclude,
        link_base=args.link_base,
        max_links=args.max_links,
        delay=args.delay,
        respect_robots=not args.ignore_robots,
        list_links=args.list_links,
        save_md=args.save_md,
        front_matter=not args.no_front_matter,
        skip_saved=args.skip_saved,
    )


if __name__ == "__main__":
    sys.exit(main())
