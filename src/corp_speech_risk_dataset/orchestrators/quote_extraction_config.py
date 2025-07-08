# corp_speech_risk_dataset/orchestrators/quote_extraction_config.py

import csv
from pathlib import Path
from typing import Set

# Project root (go up 3 levels from this file: src/corp_speech_risk_dataset/orchestrators/)
ROOT = Path(__file__).parents[3]

# point to your cleaned CSVs
SP500_OFFICERS_CSV  = Path(__file__).parents[3] / "data" / "sp500_officers_cleaned.csv"
SP500_OFFICIALS_CSV = Path(__file__).parents[3] / "data" / "sp500_official_names_cleaned.csv"

def load_company_aliases(
    officers_path: Path = SP500_OFFICERS_CSV,
    official_names_path: Path = SP500_OFFICIALS_CSV
) -> Set[str]:
    aliases: Set[str] = set()

    # 1) load officer names
    with officers_path.open(newline="", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name", "").strip()
            if name:
                aliases.add(name)

    # 2) load official company names
    with official_names_path.open(newline="", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cname = row.get("official_name", "").strip()
            if cname:
                aliases.add(cname)

    return aliases

# your single master alias set
COMPANY_ALIASES = {a.lower() for a in load_company_aliases()}

# add generic "role" words that—if mentioned in the context of a company—
# you'll assume they refer to that company's people
# generic "role" words + board & executive titles
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

    # 6) other "people" headings you sometimes see
    "key people", "leadership", "management team",
    "executive team", "governing body", "senior management"
}

# # # Directory containing your test JSON files
# JSON_DIR = ROOT / "runtime" / "tmp" / "tmp_json"

# # # Directory containing your test TXT files
# TXT_DIR = ROOT / "runtime" / "tmp" / "tmp_txt"

# JSON_DIR = ROOT / "results" / "courtlistener_v11" / "2:12-cv-00695_caed" / "entries" / "entry_1872715_documents" 

# TXT_DIR = ROOT / "results" / "courtlistener_v11" / "2:12-cv-00695_caed" / "entries" / "entry_1872715_documents"

# / "doc_13417093_text.txt"

# Single metadata JSON
JSON_DIR = (
    ROOT
    / "results"
    / "courtlistener_v11"
    / "2:12-cv-00695_caed"
    / "entries"
    / "entry_1872715_documents"

)

# Single text file
TXT_DIR = (
    ROOT
    / "results"
    / "courtlistener_v11"
    / "2:12-cv-00695_caed"
    / "entries"
    / "entry_1872715_documents"

)

# New top-level knobs for ETL I/O roots
DB_DIR = ROOT / "results" / "courtlistener_v11"  # or CLI override
MIRROR_OUT_DIR = ROOT / "extracted_mirror"        # or CLI override

KEYWORDS = [
    "Section 5", "15 U.S.C.", "tweet", "Facebook", "misleading", "deceptive",
    # you can even dynamically add company names here if you want
]

SEED_QUOTES = [
    "We will not sell your personal information to anyone.",
    "We do not use your data for advertising purposes without consent."
]

THRESHOLD = 0.0