#!/usr/bin/env python3
"""
Add structured metadata (address, opening hours, etc.) to frontmatter of each .md file.
Robust to malformed YAML frontmatter.
"""
import re
import argparse
from pathlib import Path
import frontmatter
import yaml


def load_post_with_fallback(filepath: Path):
    """
    Load a Markdown file with frontmatter. If YAML is malformed, strip the
    frontmatter manually and create a Post with empty metadata.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Try the normal way
    try:
        return frontmatter.loads(content)
    except (yaml.scanner.ScannerError, yaml.parser.ParserError, yaml.constructor.ConstructorError) as e:
        print(f"⚠️  YAML error in {filepath.name}: {e}. Falling back to manual parsing.")
        # Manual strip: remove first '--- ... ---' block
        pattern = r'^---\n.*?\n---\n'
        match = re.match(pattern, content, re.DOTALL)
        if match:
            body = content[match.end():]
        else:
            body = content
        # Create a new Post with no metadata
        return frontmatter.Post(body, **{})


def extract_metadata_from_body(body: str) -> dict:
    """Extract branch info from markdown body using regex patterns."""
    metadata = {}

    # Branch name
    match = re.search(r'^#\s*(.*?)$', body, re.MULTILINE)
    if match:
        metadata["branch_name"] = match.group(1).strip()

    # Address
    match = re.search(r'Osoite\s+(.*?)(?=\n|$)', body, re.IGNORECASE)
    if match:
        metadata["address"] = match.group(1).strip()

    # Accessibility
    match = re.search(r'(Esteetön[^.]*\.)', body, re.IGNORECASE)
    if match:
        metadata["accessibility"] = match.group(1).strip()

    # Opening hours
    match = re.search(r'Infopalveluiden aukioloajat\s+(.*?)(?=\n|$)', body, re.IGNORECASE)
    if match:
        metadata["info_opening_hours"] = match.group(1).strip()
    match = re.search(r'Kassapalveluiden aukioloajat\s+(.*?)(?=\n|$)', body, re.IGNORECASE)
    if match:
        metadata["cash_opening_hours"] = match.group(1).strip()

    # Services without appointment
    services_wo = []
    wo_section = re.search(r'## Palvelut ilman ajanvarausta(.*?)(?=##|$)', body, re.DOTALL | re.IGNORECASE)
    if wo_section:
        wo_text = wo_section.group(1)
        items = re.findall(r'####\s*(.*?)\s*\n(.*?)(?=####|\n\n|$)', wo_text, re.DOTALL)
        for name, desc in items:
            services_wo.append({"name": name.strip(), "description": desc.strip()})
    if services_wo:
        metadata["services_without_appointment"] = services_wo

    # Services with appointment
    services_w = []
    w_section = re.search(r'## Palvelut ajanvarauksella(.*?)(?=##|$)', body, re.DOTALL | re.IGNORECASE)
    if w_section:
        w_text = w_section.group(1)
        for line in w_text.splitlines():
            line = line.strip()
            if line and not line.startswith('[') and not line.startswith('Varaa'):
                services_w.append(line)
    if services_w:
        metadata["services_with_appointment"] = services_w

    # Contact numbers
    contact_numbers = {}
    contact_section = re.search(r'## Palvelunumerot(.*?)(?=##|$)', body, re.DOTALL | re.IGNORECASE)
    if contact_section:
        contact_text = contact_section.group(1)
        num_matches = re.findall(r'([A-ZÅÄÖa-zåäö\s]+?)\s+([0-9\s]+)', contact_text)
        for name, num in num_matches:
            name = name.strip()
            if name and num.strip():
                contact_numbers[name] = re.sub(r'\s+', ' ', num).strip()
    if contact_numbers:
        metadata["contact_numbers"] = contact_numbers

    # Other notes (short paragraph after address)
    after_addr = re.search(r'Osoite\s+.*?\n(.*?)(?=##|$)', body, re.DOTALL | re.IGNORECASE)
    if after_addr:
        note = after_addr.group(1).strip()
        if note and len(note) < 300:
            metadata["other_notes"] = note

    return metadata


def enrich_files(directory: str, overwrite: bool = False):
    """Add extracted metadata to frontmatter of each .md file."""
    for md_path in Path(directory).glob("*.md"):
        print(f"Processing {md_path.name}...")
        post = load_post_with_fallback(md_path)

        # Extract fresh metadata from body
        new_meta = extract_metadata_from_body(post.content)

        # Merge: only add keys not already present, or if overwrite is True
        if overwrite:
            post.metadata.update(new_meta)
        else:
            for key, value in new_meta.items():
                if key not in post.metadata:
                    post.metadata[key] = value

        # Write back (this will produce valid YAML)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))

        print(f"✅ Updated {md_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add extracted metadata to MD frontmatter.")
    parser.add_argument("--md-dir", required=True, help="Directory containing .md files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing metadata keys")
    args = parser.parse_args()
    enrich_files(args.md_dir, args.overwrite)