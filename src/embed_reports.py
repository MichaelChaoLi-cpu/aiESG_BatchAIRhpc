#!/usr/bin/env python3
"""
embed_reports.py — Tokenise and embed MD reports using the PFE model.

Pipeline for each eligible folder in data/processed/:
  1. Read the .md file, strip <!-- ... --> page-marker comments.
  2. Tokenise with bert-base-multilingual-cased → flat int list.
  3. Save token ID list  →  TokenId.joblib
  4. Slide a window over the token IDs to build padded 512-column fragments.
     Window size and stride are language-specific (see config).
  5. Run fragments through the PFE model  →  embedding array.
  6. Save embedding array  →  IndexedFragment.npy

Skip rule: a folder is skipped if it already contains BOTH
           TokenId.joblib  AND  IndexedFragment.npy.

Usage:
    python src/embed_reports.py               # process all eligible folders
    python src/embed_reports.py --overwrite   # re-embed even if outputs exist
"""

import csv
import os
import re
import sys
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from joblib import dump

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # suppress TF info noise

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    FRAGMENT_LENGTH, FRAGMENT_STRIDE,
    PFE_MODEL_PATH, BERT_MODEL_NAME,
    VALID_LANGUAGES,
)


# ---------------------------------------------------------------------------
# Lazy-loaded globals (model and tokenizer are heavy; load once)
# ---------------------------------------------------------------------------
_tokenizer = None
_pfe_model = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import BertTokenizer
        print(f"  Loading tokenizer: {BERT_MODEL_NAME}")
        _tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    return _tokenizer


def _get_pfe_model(repo_root: Path):
    global _pfe_model
    if _pfe_model is None:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from transformers import TFBertModel

        class TransformerBlock(layers.Layer):
            def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1,
                         name=None, **kwargs):
                super().__init__(name=name, **kwargs)
                self.embed_dim = embed_dim
                self.num_heads = num_heads
                self.ff_dim    = ff_dim
                self.rate      = rate
                self.att        = layers.MultiHeadAttention(num_heads=num_heads,
                                                            key_dim=embed_dim)
                self.ffn        = keras.Sequential([
                    layers.Dense(ff_dim, activation="relu"),
                    layers.Dense(embed_dim),
                ])
                self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
                self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
                self.dropout1   = layers.Dropout(rate)
                self.dropout2   = layers.Dropout(rate)

            def get_config(self):
                config = super().get_config()
                config.update({
                    "embed_dim": self.embed_dim,
                    "num_heads": self.num_heads,
                    "ff_dim":    self.ff_dim,
                    "rate":      self.rate,
                })
                return config

            def call(self, inputs, training):
                attn_output = self.att(inputs, inputs)
                attn_output = self.dropout1(attn_output, training=training)
                out1 = self.layernorm1(inputs + attn_output)
                ffn_output = self.ffn(out1)
                ffn_output = self.dropout2(ffn_output, training=training)
                return self.layernorm2(out1 + ffn_output)

        model_path = str(repo_root / PFE_MODEL_PATH)
        print(f"  Loading PFE model: {model_path}")

        strategy = tf.distribute.MirroredStrategy()
        custom_objects = {
            "TFBertModel":      TFBertModel,
            "TransformerBlock": TransformerBlock,
        }
        with strategy.scope():
            _pfe_model = keras.models.load_model(
                model_path, custom_objects=custom_objects
            )
    return _pfe_model


# ---------------------------------------------------------------------------
# Batch size — scale with available GPUs
# ---------------------------------------------------------------------------

def get_batch_size() -> int:
    try:
        import tensorflow as tf
        gpu_count = len(tf.config.list_physical_devices("GPU"))
    except Exception:
        gpu_count = 0
    return 128 * max(1, gpu_count)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def strip_md_comments(text: str) -> str:
    """Remove <!-- ... --> markers inserted by pdf_to_md.py."""
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def read_md(folder: Path) -> str:
    """Return cleaned text from the first .md file found in *folder*."""
    md_files = list(folder.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md file in {folder}")
    raw = md_files[0].read_text(encoding="utf-8")
    return strip_md_comments(raw)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenise(text: str) -> list:
    """
    Tokenise *text* with the BERT tokenizer.
    Returns a flat list of integer token IDs.
    """
    tokenizer = _get_tokenizer()
    tokens    = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return token_ids


# ---------------------------------------------------------------------------
# Fragmentation
# ---------------------------------------------------------------------------

def build_fragment_array(token_ids: list, fragment_length: int,
                         stride: int) -> np.ndarray:
    """
    Slide a window of *fragment_length* over *token_ids* with *stride* steps.
    Each row is padded with zeros (PAD token) to length 512.
    Returns a numpy array of shape (n_fragments, 512).
    """
    fragments = []
    for i in range(0, len(token_ids), stride):
        chunk   = token_ids[i : i + fragment_length]
        padded  = chunk + [0] * (512 - len(chunk))
        fragments.append(padded)
    return np.array(fragments, dtype=np.int32)


# ---------------------------------------------------------------------------
# Folder eligibility
# ---------------------------------------------------------------------------

def parse_language(folder_name: str) -> str:
    """
    Extract the language code from a folder name of the form
    {ticker}-{report_type}-{language}-{year}.
    Raises ValueError if the name does not match.
    """
    parts = folder_name.split("-")
    if len(parts) != 4:
        raise ValueError(
            f"Folder name '{folder_name}' does not have exactly 4 "
            f"dash-separated fields."
        )
    lang = parts[2].upper()
    if lang not in VALID_LANGUAGES:
        raise ValueError(
            f"Language '{lang}' from folder '{folder_name}' is not in "
            f"VALID_LANGUAGES: {list(VALID_LANGUAGES)}"
        )
    return lang


def needs_embedding(folder: Path, overwrite: bool) -> bool:
    """
    Return True if *folder* has a .md file but is missing TokenId.joblib
    or IndexedFragment.npy (or overwrite is requested).
    """
    has_md       = any(folder.glob("*.md"))
    has_token    = (folder / "TokenId.joblib").exists()
    has_index    = (folder / "IndexedFragment.npy").exists()
    both_present = has_token and has_index

    if not has_md:
        return False
    if both_present and not overwrite:
        return False
    return True


# ---------------------------------------------------------------------------
# Main embedding routine for a single folder
# ---------------------------------------------------------------------------

def embed_folder(folder: Path, repo_root: Path, batch_size: int) -> None:
    """Tokenise and embed the report in *folder*. Saves two output files."""
    lang = parse_language(folder.name)

    if lang not in FRAGMENT_LENGTH:
        raise ValueError(
            f"No FRAGMENT_LENGTH configured for language '{lang}'. "
            f"Add it to config.py."
        )
    if lang not in FRAGMENT_STRIDE:
        raise ValueError(
            f"No FRAGMENT_STRIDE configured for language '{lang}'. "
            f"Add it to config.py."
        )

    frag_len = FRAGMENT_LENGTH[lang]
    stride   = FRAGMENT_STRIDE[lang]

    # 1. Read and clean MD text
    text = read_md(folder)

    # 2. Tokenise
    print(f"    Tokenising …", end=" ", flush=True)
    token_ids = tokenise(text)
    print(f"{len(token_ids):,} tokens")

    # 3. Save TokenId.joblib
    dump(token_ids, folder / "TokenId.joblib")

    # 4. Build fragment array
    fragment_array = build_fragment_array(token_ids, frag_len, stride)
    print(f"    Fragments: {fragment_array.shape}  "
          f"(length={frag_len}, stride={stride})")

    # 5. Run PFE
    pfe = _get_pfe_model(repo_root)
    print(f"    Embedding with batch_size={batch_size} …", flush=True)
    indexed = pfe.predict(fragment_array, batch_size=batch_size, verbose=1)

    # 6. Save IndexedFragment.npy
    np.save(folder / "IndexedFragment.npy", indexed)
    print(f"    Saved IndexedFragment.npy  shape={indexed.shape}")


# ---------------------------------------------------------------------------
# Status CSV
# ---------------------------------------------------------------------------

def parse_folder_name(folder_name: str) -> Optional[dict]:
    """
    Parse a folder name of the form {ticker}-{report_type}-{language}-{year}.
    Returns a dict with keys ticker, report_type, language, year,
    or None if the name does not match the expected pattern.
    """
    parts = folder_name.split("-")
    if len(parts) != 4:
        return None
    ticker, report_type, language, year = parts
    return {
        "ticker":      ticker,
        "report_type": report_type,
        "language":    language,
        "year":        year,
    }


def update_status_csv(processed_root: Path) -> None:
    """
    Create data/processed/processing_status.csv if it does not exist,
    or append new rows (folders not already recorded) if it does.

    Columns:
        folder_name     — folder name
        ticker          — ticker
        report_type     — report type
        language        — language
        year            — year
        md_done         — True if a .md file exists in the folder
        TokenId         — True if TokenId.joblib exists
        IndexedFragment — True if IndexedFragment.npy exists
    """
    csv_path = processed_root / "processing_status.csv"
    fieldnames = [
        "folder_name", "ticker", "report_type", "language", "year",
        "md_done", "TokenId", "IndexedFragment",
    ]

    # Load already-recorded folder names so we never write duplicates
    file_exists = csv_path.exists()
    existing_folders = set()  # type: set
    if file_exists:
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_folders.add(row["folder_name"])

    new_rows = []
    for folder in sorted(processed_root.iterdir()):
        if not folder.is_dir():
            continue
        parsed = parse_folder_name(folder.name)
        if parsed is None:
            continue
        if folder.name in existing_folders:
            continue  # already recorded — skip
        new_rows.append({
            "folder_name":     folder.name,
            "ticker":          parsed["ticker"],
            "report_type":     parsed["report_type"],
            "language":        parsed["language"],
            "year":            parsed["year"],
            "md_done":         any(folder.glob("*.md")),
            "TokenId":         (folder / "TokenId.joblib").exists(),
            "IndexedFragment": (folder / "IndexedFragment.npy").exists(),
        })

    if not file_exists:
        # Create new file with BOM so Excel handles Chinese columns correctly
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new_rows)
        print(f"Status CSV created: {csv_path}  ({len(new_rows)} record(s))")
    else:
        # Append new rows without re-writing the header
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(new_rows)
        print(f"Status CSV updated: {csv_path}  "
              f"({len(new_rows)} new record(s), "
              f"{len(existing_folders)} existing)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tokenise and embed MD reports via the PFE model."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-embed folders that already have TokenId.joblib and IndexedFragment.npy.",
    )
    args = parser.parse_args()

    repo_root      = Path(__file__).parent.parent
    processed_root = repo_root / "data" / "processed"

    if not processed_root.exists():
        print(f"processed directory not found: {processed_root}", file=sys.stderr)
        sys.exit(1)

    # Collect eligible folders
    candidates = sorted(
        f for f in processed_root.iterdir()
        if f.is_dir() and needs_embedding(f, args.overwrite)
    )

    if not candidates:
        print("No folders require embedding.")
        update_status_csv(processed_root)
        sys.exit(0)

    print(f"\nFolders to embed: {len(candidates)}")
    batch_size = get_batch_size()
    print(f"Batch size: {batch_size}  (128 × {batch_size // 128} GPU(s))\n")

    done = errors = 0
    for folder in candidates:
        print(f"[{folder.name}]")
        try:
            embed_folder(folder, repo_root, batch_size)
            done += 1
        except Exception as exc:
            print(f"  [ERROR] {exc}", file=sys.stderr)
            errors += 1
        print()

    print(f"Done: {done} embedded, {errors} error(s).")
    update_status_csv(processed_root)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
