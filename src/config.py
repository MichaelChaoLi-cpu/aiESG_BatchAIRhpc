# config.py — Central configuration for aiESG4IR pipeline

# ---------------------------------------------------------------------------
# Filename naming convention: {ticker}-{report_type}-{language}-{year}.pdf
# ---------------------------------------------------------------------------

VALID_REPORT_TYPES = {
    "IR":   "Integrated Report",
    "SR":   "Sustainability Report",
    "AR":   "Annual Report",
    "SecR": "Security Report",
}

VALID_LANGUAGES = {
    "EN": "English",
    "CH": "Chinese",
    "JP": "Japanese",
    "KR": "Korean",
    "TH": "Thai",
}

YEAR_MIN = 1990
YEAR_MAX = 2100

# ---------------------------------------------------------------------------
# Embedding (PFE) parameters — keyed by language code
# ---------------------------------------------------------------------------

# Number of real token positions per fragment (rest of the 512 input is PAD=0)
FRAGMENT_LENGTH = {
    "EN": 32,
    "CH": 32,
    "JP": 32,
    "KR": 32,
    "TH": 256,
}

# Sliding-window stride (tokens advanced per fragment)
FRAGMENT_STRIDE = {
    "EN": 1,
    "CH": 1,
    "JP": 1,
    "KR": 1,
    "TH": 8,
}

# Path to the PFE (Paragraph Feature Extractor) model, relative to repo root
PFE_MODEL_PATH = "models/TMPTv2_ParaExtractor"

# BERT tokenizer name used for all languages
BERT_MODEL_NAME = "bert-base-multilingual-cased"
