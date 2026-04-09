# aiESG4IR_BatchAIRhpc

This repository serves two purposes:

1. **Ease of use** — Allow AIR Team members to launch aiESGIR analysis with minimal setup.
2. **Centralised document management** — Maintain a single copy of all embedded reports so that costly re-embedding is never duplicated across experiments or team members.

---

Batch pipeline for embedding ESG/IR reports and computing keyword match scores using the TMPT v2 model family.

---

## Directory Structure

```
aiESG4IR_BatchAIRhpc/
├── data/
│   ├── pdf/                        # Input PDFs  {ticker}-{report_type}-{language}-{year}.pdf
│   └── processed/
│       ├── {folder_name}/          # One folder per report
│       │   ├── {folder_name}.md    # Converted markdown (from pdf_to_md)
│       │   ├── TokenId.joblib      # BERT token ID list
│       │   ├── IndexedFragment.npy # PFE embeddings  (n_fragments × embed_dim)
│       │   └── {Keyword_Slug}.npy  # Match score array per keyword  (n_fragments,)
│       └── processing_status.csv   # Embedding completion status
├── exp/
│   └── {exp_name}/                 # One directory per experiment
│       ├── indicator.csv / .xlsx   # Keywords — must have column: indicator
│       ├── aim.csv / .xlsx         # Target folders — column: aim  (optional)
│       └── match_score_summary.csv # Score summary produced by the pipeline
├── models/
│   ├── TMPTv2_ParaExtractor/       # PFE model — paragraph embedder
│   ├── TMPTv2_KwExtractor/         # Keyword embedder
│   └── TMPTv2_Matcher/             # Match scorer
├── src/
│   ├── config.py                   # Central configuration
│   ├── check_filenames.py          # Validate PDF naming convention
│   ├── pdf_to_md.py                # Convert PDF → markdown
│   ├── embed_reports.py            # Tokenise and embed markdown reports
│   ├── compute_match_score.py      # Compute keyword match scores
│   └── check_experiment.py         # Experiment self-check and time estimate
└── run_pipeline.sh                 # End-to-end pipeline runner
```

---

## Naming Convention

PDFs must follow the pattern:

```
{ticker}-{report_type}-{language}-{year}.pdf
```

| Field         | Values                                      |
|---------------|---------------------------------------------|
| `ticker`      | Stock symbol; use `_` instead of `-`        |
| `report_type` | `IR`, `SR`, `AR`, `SecR`                    |
| `language`    | `EN`, `CH`, `JP`, `KR`, `TH`               |
| `year`        | Four-digit year                             |

Example: `6702.T-IR-JP-2025.pdf`

Validate filenames with:

```bash
python src/check_filenames.py
```

---

## Experiment Setup

Each experiment lives in `exp/{name}/`.

**indicator.csv** — keywords to score against (required):

```
indicator
Air Pollution
Greenhouse Gas
...
```

**aim.csv** — target report folders (optional; omit to process all):

```
aim
6702.T-IR-JP-2025
```

---

## Pipeline Operation

### Full pipeline (recommended)

```bash
bash run_pipeline.sh --exp exp/{name}
```

Steps executed in order:

1. **Pre-check** — verifies all inputs are present; aborts if any PDF is missing
2. **Embed reports** — converts new reports to token IDs and PFE embeddings (skips already-embedded)
3. **Compute match scores** — scores each target folder against every keyword (skips already-scored)
4. **Post-check** — confirms all outputs were produced

### Re-score without re-embedding

```bash
bash run_pipeline.sh --exp exp/{name} --overwrite
```

`--overwrite` forces re-computation of match scores only. Embedding is always incremental regardless of this flag (re-embedding is expensive).

### Run individual steps

```bash
# 1. Validate PDF filenames and create processed/ folders
python src/check_filenames.py

# 2. Convert PDFs to markdown
python src/pdf_to_md.py

# 3. Embed reports (skips already-embedded)
python src/embed_reports.py

# 4. Check experiment before running
python src/check_experiment.py --exp exp/{name}

# 5. Compute match scores
python src/compute_match_score.py --exp exp/{name}

# 6. Check experiment after running
python src/check_experiment.py --exp exp/{name}
```

---

## Output

| File | Location | Description |
|------|----------|-------------|
| `TokenId.joblib` | `data/processed/{folder}/` | BERT token ID list |
| `IndexedFragment.npy` | `data/processed/{folder}/` | Fragment embeddings, shape `(n, embed_dim)` |
| `{Keyword_Slug}.npy` | `data/processed/{folder}/` | Match scores per fragment, shape `(n,)`, range `[0, 1]` |
| `processing_status.csv` | `data/processed/` | Embedding status for all folders |
| `match_score_summary.csv` | `exp/{name}/` | Per-(folder, keyword) stats: `mean_score`, `max_score` |
