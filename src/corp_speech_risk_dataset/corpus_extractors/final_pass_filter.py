import re
from typing import List, Dict
from langdetect import detect  # pip install langdetect

# 1) Precompile patterns and prepare exclusion sets
plaintiff_re = re.compile(r"\bplaintiff(s)?\b", re.I)
unicode_quotes_re = re.compile(r"[“”‘’]")
case_name_re = re.compile(r"\b\w+\s+v\.\s+\w+\b", re.I)
reporter_re = re.compile(r"\b\d+\s+F\.\d+[bd]\s+\d+\b", re.I)
email_re = re.compile(r"[\w\.-]+@[\w\.-]+")
phone_re = re.compile(r"\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}")
addr_re = re.compile(
    r"\b\d{1,5}\s+[A-Za-z0-9\s]+\s+(Road|Rd\.|Street|St\.|Suite|Blvd|Ave)\b", re.I
)

bad_tokens = [
    "court",
    "ftc",
    "fed",
    "plaintiff",
    "state",
    "commission",
    "congress",
    "circuit",
    "fda",
    "federal law",
    "statutory",
    "juncture",
    "litigation",
    "counsel",
    "llc",
    "supp",
    "judge",
    "litigant",
    "id",
    "id.",
    "v.",
    "class",
    "lawyers",
    "law",
    "subclass",
    "subclasses",
    "defendant",
    "defendants",
    "plaintiff",
    "plaintiffs",
]
bad_token_res = [re.compile(r"\b{}\b".format(re.escape(w)), re.I) for w in bad_tokens]

exclude_speakers = {
    w.lower()
    for w in [
        "Unknown",
        "Court",
        "FTC",
        "Fed",
        "Plaintiff",
        "State",
        "Commission",
        "Congress",
        "Circuit",
        "FDA",
        "LLC",
        "Supp.",
        "Judge",
        "Litigant",
        "F. Supp.",
        "F. Supp",
        "FAC",
        "Id",
        "Id.",
        "SAC",
        "UCL",
        "Cal",
        "N.D. Cal",
        "Lp",
        "Dkt",
        "JPX",
        "PX",
        "SEC",
        "DOJ",
        "Inc.",
        "MPS",
        "ICP",
        "AG",
        "USDA",
        "CLRA",
        "ALJ",
        "F.3d",
        "EPA",
        "Order",
        "Orders",
        "CDA",
        "Medicare",
        "Medicaid",
        "The Supreme Court",
        "C.D. Cal",
        "Plaintiff",
        "Plaintiffs",
        "Judge",
        "Litigant",
        "Litigants",
        "Judge",
        "Judges",
        "the Ninth Circuit",
    ]
}


def filter_speakers(entries: List[Dict], exclude: List[str]) -> List[Dict]:
    """Drop any entry whose 'speaker' matches exactly one in `exclude`."""
    exclude_set = set(exclude)
    return [e for e in entries if e.get("speaker") not in exclude_set]


def filter_heuristics(entries: List[Dict]) -> List[Dict]:
    """
    Apply optimized heuristics in order:
      1. Drop entries with fewer than 5 words.
      2. Drop entries containing §, (, ), or Unicode quotes.
      3. Drop entries whose speaker matches excluded names.
      4. Drop entries matching patterns: 'plaintiff', case names, citations, or other bad tokens.
    """
    filtered: List[Dict] = []
    for e in entries:
        text = e.get("text", "") or ""
        speaker = e.get("speaker", "") or ""

        # NEW #00: Boilerplate contact info / address
        if email_re.search(text) or phone_re.search(text) or addr_re.search(text):
            continue

        # NEW #000: Garbage speaker names
        sp = speaker.strip()
        if any(ch.isdigit() for ch in sp) or len(re.findall(r"[A-Za-z]", sp)) < 2:
            continue

        # 1) Too short?
        if len(text.split()) < 5:
            continue

        # 2) Symbol / Unicode-quote filter
        if "§" in text or "(" in text or ")" in text:
            continue
        if unicode_quotes_re.search(text):
            continue

        # 3) Speaker exact-match exclusion
        if speaker.lower() in exclude_speakers:
            continue

        # 4) Regex checks
        if (
            plaintiff_re.search(text)
            or case_name_re.search(text)
            or reporter_re.search(text)
            or any(rx.search(text) for rx in bad_token_res)
        ):
            continue

        # NEW #0: Non-English passages
        try:
            if detect(text) != "en":
                continue
        except Exception:
            pass

        filtered.append(e)

    return filtered
