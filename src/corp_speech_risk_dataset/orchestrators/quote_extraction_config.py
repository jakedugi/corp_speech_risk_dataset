# corp_speech_risk_dataset/orchestrators/quote_extraction_config.py

from pathlib import Path

# Project root (go up 3 levels from this file: src/corp_speech_risk_dataset/orchestrators/)
ROOT = Path(__file__).parents[3]

# Directory containing your test JSON files
JSON_DIR = ROOT / "runtime" / "tmp" / "tmp_json"

# Directory containing your test TXT files
TXT_DIR  = ROOT / "runtime" / "tmp" / "tmp_txt"
KEYWORDS        = [
    "Section 5", "15 U.S.C.", "tweet", "Facebook", "misleading", "deceptive"
]
COMPANY_ALIASES = [
    "whatsapp", "facebook", "koum", "zuckerberg"
]
SEED_QUOTES     = [
    "We will not sell your personal information to anyone.",
    "We do not use your data for advertising purposes without consent."
]
THRESHOLD       = 0.0