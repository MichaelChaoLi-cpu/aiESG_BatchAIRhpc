#!/usr/bin/env python3
"""
check_experiment.py — Self-check an experiment before running the pipeline.

Verifies that all inputs are in place and prints a time estimate.

Usage:
    python src/check_experiment.py --exp exp/test

Checks performed:
  1. Experiment directory exists and contains indicator file
  2. aim file (if present) lists valid, known folder names
  3. For each target folder:
       - PDF exists in data/pdf/
       - .md file exists in data/processed/{folder}/
       - TokenId.joblib exists
       - IndexedFragment.npy exists
       - Per-keyword score .npy files exist
  4. Time estimate for remaining work
       - Embedding  : ~4 min per report
       - Match score: ~0.1 s per keyword per report
"""

import csv
import sys
import argparse
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import BERT_MODEL_NAME   # just to confirm config loads cleanly


# ---------------------------------------------------------------------------
# Time constants
# ---------------------------------------------------------------------------

EMBED_TIME_PER_REPORT_MIN  = 4.0    # minutes
MATCH_TIME_PER_KW_SEC      = 0.1    # seconds per (keyword × report)


# ---------------------------------------------------------------------------
# Reuse indicator / aim loaders from compute_match_score
# ---------------------------------------------------------------------------

def _read_table(path: Path) -> list:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        import pandas as pd
        return pd.read_excel(path).to_dict("records")
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_indicators(exp_dir: Path) -> list:
    for name in ("indicator.csv", "indicator.xlsx"):
        p = exp_dir / name
        if p.exists():
            rows = _read_table(p)
            return [r["indicator"].strip() for r in rows if r.get("indicator")]
    return []


def load_aim(exp_dir: Path) -> Optional[list]:
    for name in ("aim.csv", "aim.xlsx"):
        p = exp_dir / name
        if p.exists():
            rows = _read_table(p)
            return [r["aim"].strip() for r in rows if r.get("aim")]
    return None


def keyword_to_slug(keyword: str) -> str:
    return keyword.strip().replace(" ", "_")


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

def check_experiment(exp_dir: Path, repo_root: Path) -> bool:
    """
    Run all checks and print a report.
    Returns True if everything is ready (no blockers found).
    """
    ok = True
    sep = "-" * 60

    # ── 1. Experiment directory ──────────────────────────────────────────
    print(sep)
    print(f"  Experiment : {exp_dir}")
    print(sep)

    if not exp_dir.exists():
        print(f"  [ERROR] Experiment directory not found: {exp_dir}")
        return False

    # ── 2. Indicators ───────────────────────────────────────────────────
    indicators = load_indicators(exp_dir)
    if not indicators:
        print("  [ERROR] No indicator file found (indicator.csv / indicator.xlsx) "
              "or it is empty.")
        ok = False
    else:
        print(f"  Indicators ({len(indicators)}): "
              + ", ".join(indicators))

    # ── 3. Aim ──────────────────────────────────────────────────────────
    aim = load_aim(exp_dir)
    if aim is None:
        print("  Aim        : (no aim file — will process ALL processed folders)")
    else:
        print(f"  Aim        ({len(aim)}): " + ", ".join(aim))

    print()

    # ── 4. Resolve target folders ────────────────────────────────────────
    processed_root = repo_root / "data" / "processed"
    pdf_root       = repo_root / "data" / "pdf"

    if aim is not None:
        folders = [processed_root / name for name in aim]
    else:
        folders = sorted(f for f in processed_root.iterdir() if f.is_dir())

    if not folders:
        print("  [WARNING] No target folders resolved.")
        return ok

    # ── 5. Per-folder checks ─────────────────────────────────────────────
    need_embed   = []   # folders missing IndexedFragment.npy
    need_scoring = []   # (folder_name, keyword) pairs missing score .npy

    print(f"  {'Folder':<35} {'PDF':^5} {'MD':^5} {'Token':^7} {'Embed':^7}  Keywords")
    print(f"  {'-'*35} {'-'*5} {'-'*5} {'-'*7} {'-'*7}  {'-'*30}")

    for folder in folders:
        name = folder.name

        # file existence flags
        pdf_ok   = (pdf_root   / f"{name}.pdf").exists()
        md_ok    = any(folder.glob("*.md")) if folder.is_dir() else False
        token_ok = (folder / "TokenId.joblib").exists()
        embed_ok = (folder / "IndexedFragment.npy").exists()

        if not folder.is_dir():
            print(f"  {name:<35} [folder missing in processed/]")
            ok = False
            continue

        # keyword score files
        done_kws    = []
        missing_kws = []
        for kw in indicators:
            slug = keyword_to_slug(kw)
            if (folder / f"{slug}.npy").exists():
                done_kws.append(kw)
            else:
                missing_kws.append(kw)
                need_scoring.append((name, kw))

        if not embed_ok:
            need_embed.append(name)

        mark = lambda b: "OK" if b else "--"
        kw_summary = (
            f"all {len(done_kws)} done"
            if not missing_kws
            else f"{len(done_kws)}/{len(indicators)} done, "
                 f"missing: {', '.join(missing_kws)}"
        )

        # flag rows with problems
        row_ok = pdf_ok and md_ok and token_ok and embed_ok and not missing_kws
        prefix = "  " if row_ok else "! "
        print(f"{prefix}{name:<35} {mark(pdf_ok):^5} {mark(md_ok):^5} "
              f"{mark(token_ok):^7} {mark(embed_ok):^7}  {kw_summary}")

        if not pdf_ok:
            ok = False

    # ── 6. Summary ──────────────────────────────────────────────────────
    print()
    print(sep)
    print("  Summary")
    print(sep)

    if need_embed:
        print(f"  Folders needing embedding  : {len(need_embed)}")
        for n in need_embed:
            print(f"    - {n}")
    else:
        print("  Embedding                  : all up to date")

    if need_scoring:
        unique_folders = len({f for f, _ in need_scoring})
        print(f"  (folder, keyword) pairs needing scoring: "
              f"{len(need_scoring)}  across {unique_folders} folder(s)")
    else:
        print("  Match scoring              : all up to date")

    # ── 7. Time estimate ────────────────────────────────────────────────
    print()
    print(sep)
    print("  Time estimate")
    print(sep)

    embed_min  = len(need_embed)  * EMBED_TIME_PER_REPORT_MIN
    score_sec  = len(need_scoring) * MATCH_TIME_PER_KW_SEC

    total_min  = embed_min + score_sec / 60.0

    print(f"  Embedding  : {len(need_embed)} report(s)  × {EMBED_TIME_PER_REPORT_MIN} min"
          f"  =  {embed_min:.1f} min")
    print(f"  Scoring    : {len(need_scoring)} pair(s)   × {MATCH_TIME_PER_KW_SEC} s"
          f"    =  {score_sec:.1f} s")
    print(f"  Total est. : ~{total_min:.1f} min")

    print()
    if ok and not need_embed and not need_scoring:
        print("  [READY] All outputs already exist. Nothing to do.")
    elif ok:
        print("  [READY] All required inputs present. Pipeline can run.")
    else:
        print("  [BLOCKED] Fix the issues above before running the pipeline.")

    print()
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Self-check an experiment before running the pipeline."
    )
    parser.add_argument(
        "--exp", required=True,
        help="Path to the experiment directory (e.g. exp/test).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    exp_dir   = repo_root / args.exp

    ready = check_experiment(exp_dir, repo_root)
    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()
