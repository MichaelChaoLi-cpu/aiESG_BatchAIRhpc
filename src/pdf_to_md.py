#!/usr/bin/env python3
"""
pdf_to_md.py — Extract text from a PDF and save it as a Markdown file.

Paragraph structure follows the visual block layout of the PDF:
each text block in the PDF becomes one paragraph (separated by a blank line).
No heading detection is applied — all text is treated as body text.

Skip rule: if the output folder (data/processed/{stem}/) already contains
any .md file, the PDF is skipped without processing.

Output: data/processed/{stem}/{stem}.md

Usage:
    python src/pdf_to_md.py                     # all PDFs in data/pdf/
    python src/pdf_to_md.py path/to/file.pdf    # specific file(s)
    python src/pdf_to_md.py --overwrite         # ignore existing .md files
"""

import re
import sys
import argparse
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Page extraction
# ---------------------------------------------------------------------------

def page_to_paragraphs(page: fitz.Page) -> list:
    """
    Return a list of paragraph strings for *page*.
    Each element corresponds to one text block from the PDF.
    Lines within a block are preserved with single newlines;
    blank and whitespace-only lines are dropped.
    """
    paragraphs = []
    # get_text("blocks") returns tuples:
    # (x0, y0, x1, y1, text, block_no, block_type)
    # block_type 0 = text, 1 = image
    for block in page.get_text("blocks", sort=True):
        if block[6] != 0:           # skip non-text blocks
            continue
        raw = block[4]              # raw text with \n between lines

        # Normalise: collapse runs of whitespace-only lines,
        # strip each line, drop empties
        lines = [ln.strip() for ln in raw.splitlines()]
        lines = [ln for ln in lines if ln]
        if not lines:
            continue

        paragraphs.append("\n".join(lines))

    return paragraphs


def doc_to_md(doc: fitz.Document) -> str:
    """
    Convert all pages of *doc* to a single Markdown string.
    Pages are separated by a horizontal rule; empty pages are omitted.
    """
    page_sections = []
    for page_num, page in enumerate(doc, start=1):
        paragraphs = page_to_paragraphs(page)
        if not paragraphs:
            continue
        body = "\n\n".join(paragraphs)
        page_sections.append(f"<!-- page {page_num} -->\n\n{body}")

    return "\n\n---\n\n".join(page_sections)


# ---------------------------------------------------------------------------
# File-level logic
# ---------------------------------------------------------------------------

def has_existing_md(output_dir: Path) -> bool:
    """Return True if *output_dir* exists and already contains a .md file."""
    return output_dir.is_dir() and any(output_dir.glob("*.md"))


def convert(pdf_path: Path, processed_root: Path, overwrite: bool = False) -> Optional[Path]:
    """
    Convert *pdf_path* to Markdown and write it to processed_root/{stem}/{stem}.md.
    Returns the output path on success, or None if skipped.
    """
    output_dir  = processed_root / pdf_path.stem
    output_path = output_dir / (pdf_path.stem + ".md")

    # Skip check: folder already contains a .md file
    if not overwrite and has_existing_md(output_dir):
        print(f"  [SKIP]  {pdf_path.name}  "
              f"(folder '{output_dir.name}' already contains a .md file)")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    content = doc_to_md(doc)
    page_count = doc.page_count
    doc.close()

    output_path.write_text(content, encoding="utf-8")
    print(f"  [DONE]  {output_path.relative_to(processed_root.parent)}"
          f"  ({page_count} page(s))")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract PDF text to Markdown files in data/processed/."
    )
    parser.add_argument(
        "pdfs",
        nargs="*",
        help="PDF files to process. Defaults to all PDFs in data/pdf/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate .md files even when they already exist.",
    )
    args = parser.parse_args()

    repo_root      = Path(__file__).parent.parent
    processed_root = repo_root / "data" / "processed"

    if args.pdfs:
        pdf_paths = [Path(p) for p in args.pdfs]
    else:
        pdf_dir = repo_root / "data" / "pdf"
        if not pdf_dir.exists():
            print(f"Directory not found: {pdf_dir}", file=sys.stderr)
            sys.exit(1)
        pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_paths:
        print("No PDF files found.")
        sys.exit(0)

    print(f"\nProcessing {len(pdf_paths)} PDF(s)...\n")
    done = skipped = errors = 0

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            print(f"  [ERROR]  {pdf_path}: file not found", file=sys.stderr)
            errors += 1
            continue
        try:
            result = convert(pdf_path, processed_root, overwrite=args.overwrite)
            if result is not None:
                done += 1
            else:
                skipped += 1
        except Exception as exc:
            print(f"  [ERROR]  {pdf_path.name}: {exc}", file=sys.stderr)
            errors += 1

    print(f"\nDone: {done} converted, {skipped} skipped, {errors} error(s).\n")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
