#!/usr/bin/env python3
"""
check_filenames.py — Validate PDF filenames against the aiESG4IR naming convention.

Expected format: {ticker}-{report_type}-{language}-{year}.pdf
    ticker      : stock symbol, may contain letters, digits, dots, underscores
                  (hyphens are NOT allowed — use '_' instead, e.g. BRK_B)
    report_type : one of IR, SR, AR, SecR  (see config.VALID_REPORT_TYPES)
    language    : two-letter code          (see config.VALID_LANGUAGES)
    year        : four-digit integer       (YEAR_MIN <= year <= YEAR_MAX)

For every PDF in data/pdf/ that passes validation, a same-named folder is
created under data/processed/ (e.g. AAPL-AR-EN-2023.pdf → data/processed/AAPL-AR-EN-2023/).

Exit code 0 if all files pass; non-zero if any file fails.
"""

import re
import sys
import argparse
from pathlib import Path

# Allow running the script from any working directory
sys.path.insert(0, str(Path(__file__).parent))
from config import VALID_REPORT_TYPES, VALID_LANGUAGES, YEAR_MIN, YEAR_MAX


# Pre-build the regex from config values so it stays in sync automatically.
# Ticker must NOT contain '-' (the field delimiter); use '_' for compound tickers.
_REPORT_TYPES_PATTERN = "|".join(re.escape(k) for k in VALID_REPORT_TYPES)
_LANGUAGES_PATTERN    = "|".join(re.escape(k) for k in VALID_LANGUAGES)

FILENAME_RE = re.compile(
    rf"^(?P<ticker>[^-]+)"
    rf"-(?P<report_type>{_REPORT_TYPES_PATTERN})"
    rf"-(?P<language>{_LANGUAGES_PATTERN})"
    rf"-(?P<year>\d{{4}})"
    rf"\.pdf$",
    re.IGNORECASE,
)


def validate_filename(filename: str) -> list[str]:
    """
    Validate a single filename string (basename only).
    Returns a list of error messages; empty list means the name is valid.
    """
    errors = []

    # Extension must be lowercase '.pdf'
    if not filename.endswith(".pdf"):
        errors.append("extension must be lowercase '.pdf'")
        return errors  # further checks are meaningless without a match

    # Reject tickers that contain '-' before attempting the full match
    stem = filename[:-4]  # strip '.pdf'
    parts = stem.split("-")
    # A well-formed name has exactly 4 dash-separated parts
    # (ticker may not contain '-'), so we can give a precise error
    if len(parts) != 4:
        errors.append(
            f"expected exactly 4 dash-separated fields "
            f"<ticker>-<report_type>-<language>-<YYYY>  "
            f"(got {len(parts)} field(s)); "
            f"note: hyphens in tickers must be replaced with '_'"
        )
        return errors

    m = FILENAME_RE.match(filename)
    if not m:
        errors.append(
            f"does not match pattern  <ticker>-<report_type>-<language>-<YYYY>.pdf  "
            f"(valid report types: {list(VALID_REPORT_TYPES)}, "
            f"valid languages: {list(VALID_LANGUAGES)})"
        )
        return errors

    ticker      = m.group("ticker")
    report_type = m.group("report_type")
    language    = m.group("language").upper()
    year        = int(m.group("year"))

    if not ticker:
        errors.append("ticker must not be empty")

    if "-" in ticker:
        errors.append(
            f"ticker '{ticker}' contains a hyphen; use '_' instead (e.g. BRK_B)"
        )

    if report_type not in VALID_REPORT_TYPES:
        errors.append(
            f"unknown report type '{report_type}' "
            f"(expected one of {list(VALID_REPORT_TYPES)})"
        )

    if language not in VALID_LANGUAGES:
        errors.append(
            f"unknown language code '{language}' "
            f"(expected one of {list(VALID_LANGUAGES)})"
        )

    if not (YEAR_MIN <= year <= YEAR_MAX):
        errors.append(f"year {year} is out of range [{YEAR_MIN}, {YEAR_MAX}]")

    return errors


def create_processed_folder(pdf_path: Path, processed_root: Path) -> tuple[Path, bool]:
    """
    Create a folder in *processed_root* named after the PDF stem.
    Returns (folder_path, created) where created=False means it already existed.
    """
    folder = processed_root / pdf_path.stem
    if folder.exists():
        return folder, False
    folder.mkdir(parents=True)
    return folder, True


def check_directory(directory: Path, recursive: bool = False) -> dict[str, list[str]]:
    """
    Scan *directory* for .pdf files and validate each filename.
    Returns {relative_path_str: [errors]} for every file found.
    """
    pattern = "**/*.pdf" if recursive else "*.pdf"
    results = {}
    for pdf_path in sorted(directory.glob(pattern)):
        rel = str(pdf_path.relative_to(directory))
        results[rel] = validate_filename(pdf_path.name)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate PDF filenames against the aiESG4IR naming convention."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="Files or directories to check. Defaults to data/pdf/ relative to repo root.",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recurse into subdirectories when a directory is given.",
    )
    parser.add_argument(
        "--no-create-folders",
        action="store_true",
        help="Skip creating output folders in data/processed/ for passing files.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    processed_root = repo_root / "data" / "processed"

    # Default target: data/pdf/
    using_default_pdf_dir = not args.targets
    if using_default_pdf_dir:
        default_dir = repo_root / "data" / "pdf"
        if not default_dir.exists():
            print(f"Default directory not found: {default_dir}", file=sys.stderr)
            sys.exit(1)
        args.targets = [str(default_dir)]

    # Collect all pdf paths alongside their validation results
    all_results: dict[str, list[str]] = {}   # relative key → errors
    pdf_paths:   dict[str, Path]      = {}   # same key → absolute Path

    for target_str in args.targets:
        target = Path(target_str)
        if target.is_dir():
            pattern = "**/*.pdf" if args.recursive else "*.pdf"
            for pdf_path in sorted(target.glob(pattern)):
                rel = str(pdf_path.relative_to(target))
                all_results[rel] = validate_filename(pdf_path.name)
                pdf_paths[rel]   = pdf_path
        elif target.is_file():
            key = str(target)
            all_results[key] = validate_filename(target.name)
            pdf_paths[key]   = target
        else:
            print(f"WARNING: '{target}' is not a file or directory, skipping.",
                  file=sys.stderr)

    if not all_results:
        print("No PDF files found.")
        sys.exit(0)

    passed = {f: e for f, e in all_results.items() if not e}
    failed = {f: e for f, e in all_results.items() if e}

    # Print results
    if failed:
        print(f"\n{'='*60}")
        print(f"  FAILED ({len(failed)} file(s))")
        print(f"{'='*60}")
        for filename, errors in failed.items():
            print(f"\n  [FAIL]  {filename}")
            for err in errors:
                print(f"          - {err}")

    if passed:
        print(f"\n{'='*60}")
        print(f"  PASSED ({len(passed)} file(s))")
        print(f"{'='*60}")
        for key in passed:
            suffix = ""
            if not args.no_create_folders:
                folder, created = create_processed_folder(pdf_paths[key], processed_root)
                rel = folder.relative_to(repo_root)
                suffix = f"  →  {rel} (created)" if created else f"  →  {rel} (already exists, skipped)"
            print(f"  [OK]    {key}{suffix}")

    print(f"\nTotal: {len(all_results)} file(s), {len(passed)} passed, {len(failed)} failed.\n")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
