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
