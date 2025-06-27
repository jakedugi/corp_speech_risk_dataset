# corp_speech_risk_dataset/orchestrators/quote_extraction_config.py

import csv
from pathlib import Path

# Project root (go up 3 levels from this file: src/corp_speech_risk_dataset/orchestrators/)
ROOT = Path(__file__).parents[3]

# Where to find your S&P 500 aliases CSV
SP500_CSV = ROOT / "data" / "sp500_officers_cleaned.csv"

# Function to load company aliases from CSV
# CSV columns: ticker,official_name,exec1,exec2,exec3,...
def load_company_aliases(path: Path = SP500_CSV) -> set[str]:
    aliases: set[str] = set()
    if not path.exists():
        raise FileNotFoundError(f"Expected aliases CSV at {path}")
    with path.open(newline="", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # ticker, official_name
            aliases.add(row["ticker"].lower())
            aliases.add(row["official_name"].lower())
            # optional exec/board columns
            for col in row:
                if col.startswith("exec") or col.startswith("board"):
                    name = row[col].strip()
                    if name:
                        aliases.add(name.lower())
    return aliases

# load once
COMPANY_ALIASES = load_company_aliases()

# add generic "role" words that—if mentioned in the context of a company—
# you'll assume they refer to that company's people
# generic “role” words + board & executive titles
ROLE_KEYWORDS = {
    # 1) generic staff roles
    "employee", "staff", "associate", "worker",
    "contractor", "agent", "representative", "team member",
    "intern", "volunteer", "analyst", "consultant",

    # 2) board & governance
    "director", "board member", "board chair", "board chairman",
    "chairperson", "board chairwoman", "trustee", "governor",
    "audit committee", "nominating committee",

    # 3) C-suite / executives
    "chief executive officer", "ceo",
    "chief operating officer", "coo",
    "chief financial officer", "cfo",
    "chief technology officer", "cto",
    "chief information officer", "cio",
    "chief marketing officer", "cmo",
    "chief legal officer", "clo", "general counsel",
    "chief compliance officer", "cco",
    "chief human resources officer", "chro",
    "chief strategy officer", "cso",
    "chief risk officer", "cro",
    "chief data officer", "cdo",
    "chief innovation officer", "cio",  # sometimes overlaps
    "chief sustainability officer", "cso",
    "chief product officer", "cpo",
    "chief customer officer", "cco",
    "chief growth officer", "cgo",

    # 4) senior & middle management
    "president", "vice president", "vp", "senior vice president", "svp",
    "executive vice president", "evp",
    "managing director", "md",
    "director of", "head of", "manager",
    "regional manager", "branch manager",
    "team leader", "project lead",

    # 5) ownership / investors
    "shareholder", "stockholder", "investor",
    "partner", "founder", "co-founder", "owner",
    "principal", "venture partner",
    "angel investor", "limited partner", "lp",

    # 6) other “people” headings you sometimes see
    "key people", "leadership", "management team",
    "executive team", "governing body", "senior management"
}

# Directory containing your test JSON files
JSON_DIR = ROOT / "runtime" / "tmp" / "tmp_json"

# Directory containing your test TXT files
TXT_DIR  = ROOT / "runtime" / "tmp" / "tmp_txt"

KEYWORDS = [
    "Section 5", "15 U.S.C.", "tweet", "Facebook", "misleading", "deceptive",
    # you can even dynamically add company names here if you want
]

SEED_QUOTES = [
    "We will not sell your personal information to anyone.",
    "We do not use your data for advertising purposes without consent."
]

THRESHOLD = 0.0