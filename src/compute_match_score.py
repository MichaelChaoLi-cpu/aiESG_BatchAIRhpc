#!/usr/bin/env python3
"""
compute_match_score.py — Compute keyword match scores against embedded reports.

Usage:
    python src/compute_match_score.py --exp exp/test
    python src/compute_match_score.py --exp exp/test --overwrite

Experiment directory layout:
    exp/{name}/indicator.csv  (or .xlsx)  — must have a column named "indicator"
    exp/{name}/aim.csv        (or .xlsx)  — optional; column "aim" lists target
                                            folder names; if absent, all processed
                                            folders are used

Pipeline for each target folder × each keyword:
  1. Load IndexedFragment.npy from data/processed/{folder}/
  2. Tokenise keyword (BERT, max 16 tokens, PAD-padded) → (1, 16) array
  3. KwExtractor model  →  indexed keyword  (1, embed_dim)
  4. Tile keyword to (n_fragments, embed_dim)
  5. Matcher model([indexed_fragments, tiled_kw])  →  raw scores
  6. Apply  score = max(0, raw * 2 − 1)
  7. Save score array → data/processed/{folder}/{keyword_slug}.npy
  8. Append results to exp/{name}/match_score_summary.csv

All TensorFlow inference runs on CPU (CUDA_VISIBLE_DEVICES=-1).
"""

import csv
import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict

# ── Force CPU before any TF import ──────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.insert(0, str(Path(__file__).parent))
from config import BERT_MODEL_NAME

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KW_MAX_LENGTH = 16          # keyword token budget
MATCHER_BATCH  = 2048       # batch size for matcher (CPU — large batches are fine)
SUMMARY_FILE   = "match_score_summary.csv"
SUMMARY_FIELDS = [
    "folder_name", "keyword", "n_fragments",
    "mean_score", "max_score",
]


# ---------------------------------------------------------------------------
# Model / tokenizer loading (lazy, loaded once)
# ---------------------------------------------------------------------------

_tokenizer    = None
_kw_model     = None
_matcher_model = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import BertTokenizer
        print(f"  Loading tokenizer: {BERT_MODEL_NAME}")
        _tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    return _tokenizer


def get_kw_model(repo_root: Path):
    global _kw_model
    if _kw_model is None:
        import tensorflow as tf
        from tensorflow import keras
        from transformers import TFBertModel
        path = str(repo_root / "models" / "TMPTv2_KwExtractor")
        print(f"  Loading KwExtractor: {path}")
        _kw_model = keras.models.load_model(
            path, custom_objects={"TFBertModel": TFBertModel}
        )
    return _kw_model


def get_matcher_model(repo_root: Path):
    global _matcher_model
    if _matcher_model is None:
        import tensorflow as tf
        from tensorflow import keras
        from transformers import TFBertModel
        path = str(repo_root / "models" / "TMPTv2_Matcher")
        print(f"  Loading Matcher: {path}")
        _matcher_model = keras.models.load_model(
            path, custom_objects={"TFBertModel": TFBertModel}
        )
    return _matcher_model


# ---------------------------------------------------------------------------
# Keyword helpers
# ---------------------------------------------------------------------------

def keyword_to_slug(keyword: str) -> str:
    """'Air Pollution' → 'Air_Pollution'  (safe filename)."""
    return keyword.strip().replace(" ", "_")


def tokenize_keyword(keyword: str) -> np.ndarray:
    """
    Tokenise *keyword* to a (1, KW_MAX_LENGTH) int32 array.
    Truncated or PAD-padded to exactly KW_MAX_LENGTH tokens.
    """
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(keyword)
    if len(tokens) >= KW_MAX_LENGTH:
        tokens = tokens[:KW_MAX_LENGTH]
    else:
        tokens += ["[PAD]"] * (KW_MAX_LENGTH - len(tokens))
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return np.array([ids], dtype=np.int32)          # (1, 16)


def index_keyword(keyword: str, repo_root: Path) -> np.ndarray:
    """Embed keyword → (1, embed_dim) via KwExtractor."""
    kw_input  = tokenize_keyword(keyword)
    kw_model  = get_kw_model(repo_root)
    return kw_model.predict(kw_input, verbose=0)    # (1, embed_dim)


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_scores(indexed_fragments: np.ndarray,
                   indexed_kw: np.ndarray,
                   repo_root: Path) -> np.ndarray:
    """
    Parameters
    ----------
    indexed_fragments : (n, embed_dim)
    indexed_kw        : (1, embed_dim)

    Returns
    -------
    scores : (n,)  float32, values in [0, 1]
    """
    matcher  = get_matcher_model(repo_root)
    n        = indexed_fragments.shape[0]
    tiled_kw = np.tile(indexed_kw[0], (n, 1))       # (n, embed_dim)

    raw = matcher.predict(
        [indexed_fragments, tiled_kw],
        batch_size=MATCHER_BATCH,
        verbose=0,
    )                                                # (n, 1) or (n,)
    raw = raw.squeeze()                              # (n,)
    return np.maximum(0.0, raw * 2.0 - 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_table(path: Path) -> List[Dict]:
    """Read a CSV or Excel file; return list of row dicts."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        import pandas as pd
        return pd.read_excel(path).to_dict("records")
    else:
        with open(path, newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))


def load_indicators(exp_dir: Path) -> List[str]:
    """Return list of indicator strings from indicator.csv / .xlsx."""
    for name in ("indicator.csv", "indicator.xlsx"):
        p = exp_dir / name
        if p.exists():
            rows = read_table(p)
            return [r["indicator"].strip() for r in rows if r.get("indicator")]
    raise FileNotFoundError(
        f"No indicator.csv or indicator.xlsx found in {exp_dir}"
    )


def load_aim(exp_dir: Path) -> Optional[List[str]]:
    """
    Return list of target folder names from aim.csv / .xlsx,
    or None if the file does not exist (meaning: process all).
    """
    for name in ("aim.csv", "aim.xlsx"):
        p = exp_dir / name
        if p.exists():
            rows = read_table(p)
            return [r["aim"].strip() for r in rows if r.get("aim")]
    return None


def load_summary_keys(summary_path: Path) -> set:
    """Return set of (folder_name, keyword) already recorded in the summary."""
    if not summary_path.exists():
        return set()
    with open(summary_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {(r["folder_name"], r["keyword"]) for r in reader}


def append_summary(summary_path: Path, rows: List[Dict]) -> None:
    """Append *rows* to the summary CSV; create with header if absent."""
    if not rows:
        return
    file_exists = summary_path.exists()
    if not file_exists:
        with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Summary created: {summary_path}  ({len(rows)} row(s))")
    else:
        with open(summary_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            writer.writerows(rows)
        print(f"  Summary updated: +{len(rows)} row(s)")


# ---------------------------------------------------------------------------
# Per-folder processing
# ---------------------------------------------------------------------------

def process_folder(folder: Path,
                   keywords: List[str],
                   repo_root: Path,
                   done_keys: set,
                   overwrite: bool) -> List[Dict]:
    """
    Compute match scores for *folder* × *keywords*.
    Returns list of summary-row dicts for newly computed results.
    """
    indexed_frag_path = folder / "IndexedFragment.npy"
    if not indexed_frag_path.exists():
        print(f"  [SKIP] IndexedFragment.npy not found in {folder.name}")
        return []

    indexed_fragments = np.load(indexed_frag_path)  # (n, embed_dim)
    n = indexed_fragments.shape[0]
    new_rows = []

    for kw in keywords:
        key  = (folder.name, kw)
        slug = keyword_to_slug(kw)
        npy_path = folder / f"{slug}.npy"

        if not overwrite and key in done_keys:
            print(f"    [skip] {kw}  (already in summary)")
            continue
        if not overwrite and npy_path.exists():
            print(f"    [skip] {kw}  ({npy_path.name} exists)")
            continue

        print(f"    [{kw}] indexing keyword …", end=" ", flush=True)
        indexed_kw = index_keyword(kw, repo_root)   # (1, embed_dim)

        print("computing scores …", end=" ", flush=True)
        scores = compute_scores(indexed_fragments, indexed_kw, repo_root)

        np.save(npy_path, scores)
        print(f"saved  mean={scores.mean():.4f}  max={scores.max():.4f}")

        new_rows.append({
            "folder_name": folder.name,
            "keyword":     kw,
            "n_fragments": n,
            "mean_score":  round(float(scores.mean()), 6),
            "max_score":   round(float(scores.max()),  6),
        })

    return new_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute keyword match scores for embedded ESG reports."
    )
    parser.add_argument(
        "--exp", required=True,
        help="Path to the experiment directory (e.g. exp/test).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute even if .npy outputs already exist.",
    )
    args = parser.parse_args()

    repo_root      = Path(__file__).parent.parent
    exp_dir        = repo_root / args.exp
    processed_root = repo_root / "data" / "processed"
    summary_path   = exp_dir / SUMMARY_FILE

    if not exp_dir.exists():
        print(f"Experiment directory not found: {exp_dir}", file=sys.stderr)
        sys.exit(1)
    if not processed_root.exists():
        print(f"Processed directory not found: {processed_root}", file=sys.stderr)
        sys.exit(1)

    # Load experiment config
    keywords = load_indicators(exp_dir)
    aim      = load_aim(exp_dir)          # None → all folders
    print(f"Experiment : {exp_dir}")
    print(f"Keywords   : {len(keywords)}")
    print(f"Aim        : {aim if aim else '(all processed folders)'}\n")

    # Resolve target folders
    if aim is not None:
        folders = [processed_root / name for name in aim]
        missing = [f for f in folders if not f.is_dir()]
        if missing:
            print("WARNING: aim folders not found:", [f.name for f in missing],
                  file=sys.stderr)
        folders = [f for f in folders if f.is_dir()]
    else:
        folders = sorted(f for f in processed_root.iterdir() if f.is_dir())

    if not folders:
        print("No target folders found. Exiting.")
        sys.exit(0)

    # Load already-recorded (folder, keyword) pairs
    done_keys = load_summary_keys(summary_path)

    # Process
    total_new = total_err = 0
    for folder in folders:
        print(f"[{folder.name}]")
        try:
            new_rows = process_folder(
                folder, keywords, repo_root, done_keys, args.overwrite
            )
            append_summary(summary_path, new_rows)
            total_new += len(new_rows)
        except Exception as exc:
            print(f"  [ERROR] {exc}", file=sys.stderr)
            total_err += 1
        print()

    print(f"Done: {total_new} score file(s) written, {total_err} error(s).")
    sys.exit(1 if total_err else 0)


if __name__ == "__main__":
    main()
