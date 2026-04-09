#!/usr/bin/env bash
# run_pipeline.sh — End-to-end pipeline: check → embed → score → check
#
# Usage:
#   bash run_pipeline.sh --exp exp/test
#   bash run_pipeline.sh --exp exp/test --overwrite

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
EXP=""
OVERWRITE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp)       EXP="$2";    shift 2 ;;
        --overwrite) OVERWRITE="--overwrite"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$EXP" ]]; then
    echo "Usage: bash run_pipeline.sh --exp <exp_dir> [--overwrite]" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
section() { echo; echo "══════════════════════════════════════════════════════════"; echo "  $1"; echo "══════════════════════════════════════════════════════════"; }

# ---------------------------------------------------------------------------
# 1. Pre-check
# ---------------------------------------------------------------------------
section "PRE-CHECK  [${EXP}]"
python src/check_experiment.py --exp "$EXP"
PRE_STATUS=$?
if [[ $PRE_STATUS -ne 0 ]]; then
    echo
    echo "[ABORT] Pre-check reported blockers. Fix them before running the pipeline." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Embed reports  (never overwrite — re-embedding is too expensive)
# ---------------------------------------------------------------------------
section "EMBED REPORTS"
python src/embed_reports.py

# ---------------------------------------------------------------------------
# 3. Compute match scores
# ---------------------------------------------------------------------------
section "COMPUTE MATCH SCORES  [${EXP}]"
python src/compute_match_score.py --exp "$EXP" $OVERWRITE

# ---------------------------------------------------------------------------
# 4. Post-check
# ---------------------------------------------------------------------------
section "POST-CHECK  [${EXP}]"
python src/check_experiment.py --exp "$EXP"

echo
echo "Pipeline complete."
