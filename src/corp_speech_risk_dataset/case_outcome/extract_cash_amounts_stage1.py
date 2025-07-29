#!/usr/bin/env python3
"""
extract_amounts_stage1.py

Recursively scans `_text_stage1.jsonl` files under the specified directory,
extracts dollar amounts and "million/billion" phrases with surrounding context,
deduplicates via Simhash, groups by case (two levels up), and writes results to
a JSONL file.  At the end, prints file‐level and case‐level hit summaries, plus
counts of `_stage3.jsonl` entries in hit vs. no‐hit cases.
"""

import re
import json
import glob
import os
import hashlib
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Any, Dict


DEFAULT_INPUT_PATTERN = "data/extracted/**/*_text_stage1.jsonl"
DEFAULT_OUTPUT_FILE = "data/output_stage1_amounts.jsonl"
DEFAULT_CONTEXT_CHARS = 100
DEFAULT_BASE_DIR = "data/extracted"
DEFAULT_MIN_AMOUNT = 10000  # Lowered from $1M to $10k

CONTEXT_CHARS = DEFAULT_CONTEXT_CHARS
MIN_AMOUNT = DEFAULT_MIN_AMOUNT


# ------------------------------------------------------------------------------
# Voting Weights Configuration
# ------------------------------------------------------------------------------


class VotingWeights:
    """
    Configurable weights for different voting system components.
    Each weight defaults to 1.0 but can be customized for different scenarios.
    """

    def __init__(
        self,
        proximity_pattern_weight: float = 1.0,
        judgment_verbs_weight: float = 1.0,
        case_position_weight: float = 1.0,
        docket_position_weight: float = 1.0,
        all_caps_titles_weight: float = 1.0,
        document_titles_weight: float = 1.0,
    ):
        """
        Initialize voting weights for each component.

        Args:
            proximity_pattern_weight: Weight for monetary context words (settlement, judgment, etc.)
            judgment_verbs_weight: Weight for legal action verbs (awarded, ordered, etc.)
            case_position_weight: Weight for chronological position within case
            docket_position_weight: Weight for chronological position within docket
            all_caps_titles_weight: Weight for ALL CAPS section titles
            document_titles_weight: Weight for document titles
        """
        self.proximity_pattern_weight = proximity_pattern_weight
        self.judgment_verbs_weight = judgment_verbs_weight
        self.case_position_weight = case_position_weight
        self.docket_position_weight = docket_position_weight
        self.all_caps_titles_weight = all_caps_titles_weight
        self.document_titles_weight = document_titles_weight

    def to_dict(self) -> dict[str, float]:
        """Convert weights to dictionary for serialization."""
        return {
            "proximity_pattern_weight": self.proximity_pattern_weight,
            "judgment_verbs_weight": self.judgment_verbs_weight,
            "case_position_weight": self.case_position_weight,
            "docket_position_weight": self.docket_position_weight,
            "all_caps_titles_weight": self.all_caps_titles_weight,
            "document_titles_weight": self.document_titles_weight,
        }

    @classmethod
    def from_dict(cls, weights_dict: dict[str, float]) -> "VotingWeights":
        """Create VotingWeights from dictionary."""
        return cls(**weights_dict)


# Default voting weights
DEFAULT_VOTING_WEIGHTS = VotingWeights()

# Improved regexes for better cash amount detection - ordered by specificity
AMOUNT_REGEX = re.compile(
    r"\$[0-9]+(?:\.[0-9]+)?\s*(?:million|billion)\b|"  # $5.2 million, $92 million (most specific)
    r"\b[0-9]+(?:\.[0-9]+)?\s*(?:million|billion)\b|"  # 5.2 million, 92 million
    r"\$[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]{2})?|"  # $1,234,567.89 (comma-separated)
    r"\$[0-9]+(?:\.[0-9]{2})?",  # $123 or $123.45 (simple amounts - least specific)
    re.IGNORECASE,
)

# Enhanced spaCy support for better amount detection
try:
    import spacy
    from spacy.pipeline import EntityRuler

    spacy_available = True
except ImportError:
    spacy_available = False

# Enhanced judgment-verb context filter (new addition)
JUDGMENT_VERBS = re.compile(
    r"\b(?:award(?:ed)?|order(?:ed)?|grant(?:ed)?|enter(?:ed)?|assess(?:ed)?|recover(?:y|ed)?)\b",
    re.IGNORECASE,
)

# More permissive proximity pattern to catch monetary contexts
PROXIMITY_PATTERN = re.compile(
    r"\b(?:settlement|judgment|judgement|damages|award|penalty|fine|amount|paid|cost|price|fee|compensation|restitution|claim|relief|recover|value|sum)\b",
    re.IGNORECASE,
)

# Enhanced regex patterns for spelled-out amounts and USD prefixes (new additions)
SPELLED_OUT_AMOUNTS = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\s+(?:million|billion)\s+(?:dollars?|USD)\b",
    re.IGNORECASE,
)

USD_AMOUNTS = re.compile(
    r"\bUSD?\s?[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]{2})?\b", re.IGNORECASE
)


# Court name extraction patterns
COURT_NAME_PATTERNS = [
    re.compile(r"IN THE UNITED STATES DISTRICT COURT", re.IGNORECASE),
    re.compile(r"IN THE UNITED STATES BANKRUPTCY COURT", re.IGNORECASE),
    re.compile(r"UNITED STATES DISTRICT COURT", re.IGNORECASE),
    re.compile(r"UNITED STATES BANKRUPTCY COURT", re.IGNORECASE),
    re.compile(r"UNITED STATES COURT OF APPEALS", re.IGNORECASE),
    re.compile(r"UNITED STATES SUPREME COURT", re.IGNORECASE),
]

# Bankruptcy court identifiers
BANKRUPTCY_COURT_PATTERNS = [
    re.compile(r"BANKRUPTCY COURT", re.IGNORECASE),
    re.compile(r"BANKRUPTCY", re.IGNORECASE),
]

# ALL CAPS section titles that indicate important legal content
ALL_CAPS_SECTION_TITLES = [
    re.compile(r"\bORDERED\b"),
    re.compile(r"\bADJUDGED\b"),
    re.compile(r"\bGRANTED\b"),
    re.compile(r"\bDONE\b"),
    re.compile(r"\bSO ORDERED\b"),
    re.compile(r"\bIT IS FURTHER ORDERED\b"),
    re.compile(r"\bFURTHER ORDERED\b"),
    re.compile(r"\bDENIES\b"),
    re.compile(r"\bCONCLUSION\b"),
    re.compile(r"\bORDER DENYING PLAINTIFF\b"),
]

# Document titles that indicate important legal documents
DOCUMENT_TITLES = [
    re.compile(r"\bORDER GRANTING MOTION FOR FEES AND COSTS\b"),
    re.compile(r"\bLONG-FORM SETTLEMENT AGREEMENT\b"),
    re.compile(r"\bTHIS SETTLEMENT AGREEMENT\b"),
    re.compile(r"\bMEMORANDUM AMD PRETRIAL ORDER\b"),
    re.compile(r"\bMEMORANDUM OPINION AND ORDER\b"),
    re.compile(r"\bFINAL ORDER REGARDING ATTORNEYS' FEES AND EXPENSES\b"),
    re.compile(
        r"\bMEMORANDUM OF LAW IN SUPPORT OF PLAINTIFFS' MOTION FOR AN AWARD OF ATTORNEYS' FEES\b"
    ),
    re.compile(r"\bSETTLEMENT AGREEMENT\b"),
    re.compile(r"\bSTIPULATED ORDER FOR INJUNCTION AND MONETARY JUDGMENT\b"),
    re.compile(r"\bMONETARY JUDGMENT FOR CIVIL PENALTY\b"),
    re.compile(r"\bMEMORANDUM OPINION AND ORDER\b"),
    re.compile(r"\bMONETARY JUDGMENT\b"),
    re.compile(r"\bSTIPULATION AND AGREEMENT OF SETTLEMENT\b"),
    re.compile(r"\bFINAL ORDER REGARDING ATTORNEYS' FEES AND EXPENSES\b"),
    re.compile(r"\bFINAL JUDGMENT\b"),
    re.compile(
        r"\bSTIPULATED ORDER FOR CIVIL PENALTY, MONETARY JUDGMENT, AND INJUNCTIVE RELIEF\b"
    ),
    re.compile(r"\bORDER DENYING PLAINTIFF\b"),
]

# Dismissal patterns that indicate class action dismissals
DISMISSAL_PATTERNS = [
    re.compile(r"\bDefendants' motion to dismiss is granted\b", re.IGNORECASE),
    re.compile(r"\bDISMISSED\b", re.IGNORECASE),
    re.compile(r"\bAccordingly, the Court decertifies the classes\b", re.IGNORECASE),
    re.compile(
        r"\bDefendants' motion to decertify the classes is GRANTED\b", re.IGNORECASE
    ),
    re.compile(r"\bCourt hereby DENIES Plaintiff's Amended Motion\b", re.IGNORECASE),
    re.compile(
        r"\bCourt hereby DENIES Plaintiff's Amended Motion for Class Certification\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bTherefore, the Court declines Plaintiff's motion to certify a damages class\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bCourt hereby DENIES Plaintiff's Motion for Class\b", re.IGNORECASE),
    re.compile(r"\bORDER DENYING PLAINTIFF'S.*MOTION FOR CLASS\b", re.IGNORECASE),
    re.compile(r"\bCourt.*DENIES.*Class\b", re.IGNORECASE),
    re.compile(r"\bclass.*decertified\b", re.IGNORECASE),
    re.compile(r"\bclass.*dismissed\b", re.IGNORECASE),
]

# Fee-shifting patterns
FEE_SHIFTING_PATTERNS = [
    re.compile(r"\bfee.?shifting\b", re.IGNORECASE),
]

# Large patent amount patterns (typically $10M+ in patent cases)
LARGE_PATENT_PATTERNS = [
    re.compile(r"\bpatent\b", re.IGNORECASE),  # Just count "patent" occurrences
]


def extract_court_name(text: str) -> str | None:
    """
    Extract court name from document text.

    Args:
        text: Document text to search for court names

    Returns:
        str | None: Court name if found, None otherwise
    """
    # Look for court name patterns in the first 2000 characters (header area)
    header_text = text[:2000]

    for pattern in COURT_NAME_PATTERNS:
        match = pattern.search(header_text)
        if match:
            return match.group(0)

    return None


def is_bankruptcy_court(text: str) -> bool:
    """
    Check if the document is from a bankruptcy court.

    Args:
        text: Document text to check

    Returns:
        bool: True if bankruptcy court, False otherwise
    """
    court_name = extract_court_name(text)
    if not court_name:
        return False

    # Check if any bankruptcy patterns match
    for pattern in BANKRUPTCY_COURT_PATTERNS:
        if pattern.search(court_name):
            return True

    return False


def get_case_court_type(
    case_root: Path, bankruptcy_ratio_threshold: float = 0.5
) -> str | None:
    """
    Determine the court type for a case by examining stage1 files.

    Args:
        case_root: Path to the case directory
        bankruptcy_ratio_threshold: Minimum ratio of bankruptcy court documents to flag (default: 0.5)

    Returns:
        str | None: Court type if found, None otherwise
    """
    bankruptcy_count = 0
    total_documents = 0

    # Scan all stage1 files in the case
    for path in case_root.rglob("*_stage1.jsonl"):
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if not text:
                        continue

                    total_documents += 1
                    if is_bankruptcy_court(text):
                        bankruptcy_count += 1

                except json.JSONDecodeError:
                    continue

    # If ratio of bankruptcy documents meets threshold, return bankruptcy
    if (
        total_documents > 0
        and bankruptcy_count / total_documents >= bankruptcy_ratio_threshold
    ):
        return "BANKRUPTCY"

    return None


def count_all_caps_section_titles(text: str) -> int:
    """
    Count the number of ALL CAPS section titles in the text.

    Args:
        text: Document text to search

    Returns:
        int: Number of ALL CAPS section titles found
    """
    count = 0
    for pattern in ALL_CAPS_SECTION_TITLES:
        matches = pattern.findall(text)
        count += len(matches)
    return count


def count_document_titles(text: str, header_chars: int = 2000) -> int:
    """
    Count the number of important document titles in the header of the text.

    Args:
        text: Document text to search
        header_chars: Number of characters to consider as header (default: 2000)

    Returns:
        int: Number of document titles found in header
    """
    count = 0
    # Only search in the header portion of the document
    header_text = text[:header_chars]
    for pattern in DOCUMENT_TITLES:
        matches = pattern.findall(header_text)
        count += len(matches)
    return count


def count_dismissal_patterns(text: str) -> int:
    """
    Count the number of dismissal patterns in the text.

    Args:
        text: Document text to search

    Returns:
        int: Number of dismissal patterns found
    """
    count = 0
    for pattern in DISMISSAL_PATTERNS:
        matches = pattern.findall(text)
        count += len(matches)
    return count


def is_case_dismissed(case_root: Path, dismissal_ratio_threshold: float = 0.5) -> bool:
    """
    Determine if a case is dismissed by examining stage1 files for dismissal language.
    Uses a ratio-based approach where the threshold is the ratio of documents containing
    dismissal language to total documents.

    Args:
        case_root: Path to the case directory
        dismissal_ratio_threshold: Minimum ratio of documents with dismissal language (default: 0.5)

    Returns:
        bool: True if case appears to be dismissed, False otherwise
    """
    dismissal_documents = 0
    total_documents = 0

    # Scan all stage1 files in the case
    for path in case_root.rglob("*_stage1.jsonl"):
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if not text:
                        continue

                    total_documents += 1
                    dismissal_matches = count_dismissal_patterns(text)
                    if dismissal_matches > 0:
                        dismissal_documents += 1

                except json.JSONDecodeError:
                    continue

    # Calculate ratio and compare to threshold
    if total_documents == 0:
        return False

    dismissal_ratio = dismissal_documents / total_documents
    return dismissal_ratio >= dismissal_ratio_threshold


def has_fee_shifting(
    case_root: Path, fee_shifting_ratio_threshold: float = 1.0
) -> bool:
    """
    Determine if a case contains fee-shifting language above threshold ratio.

    Args:
        case_root: Path to the case directory
        fee_shifting_ratio_threshold: Minimum ratio of fee-shifting occurrences per document to flag (default: 1.0)

    Returns:
        bool: True if case contains fee-shifting language above threshold ratio, False otherwise
    """
    fee_shifting_count = 0
    total_documents = 0

    # Scan all stage1 files in the case
    for path in case_root.rglob("*_stage1.jsonl"):
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if not text:
                        continue

                    total_documents += 1
                    # Count fee-shifting patterns
                    for pattern in FEE_SHIFTING_PATTERNS:
                        matches = pattern.findall(text)
                        fee_shifting_count += len(matches)

                except json.JSONDecodeError:
                    continue

    # Calculate ratio and compare to threshold
    if total_documents == 0:
        return False

    fee_shifting_ratio = fee_shifting_count / total_documents
    return fee_shifting_ratio >= fee_shifting_ratio_threshold


def has_large_patent_amounts(
    case_root: Path, patent_ratio_threshold: float = 20.0
) -> bool:
    """
    Determine if a case contains many patent references above threshold ratio.

    Args:
        case_root: Path to the case directory
        patent_ratio_threshold: Minimum ratio of patent occurrences per document to flag (default: 20.0)

    Returns:
        bool: True if case contains many patent references above threshold ratio, False otherwise
    """
    patent_count = 0
    total_documents = 0

    # Scan all stage1 files in the case
    for path in case_root.rglob("*_stage1.jsonl"):
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if not text:
                        continue

                    total_documents += 1
                    # Count patent occurrences
                    for pattern in LARGE_PATENT_PATTERNS:
                        matches = pattern.findall(text)
                        patent_count += len(matches)

                except json.JSONDecodeError:
                    continue

    # Calculate ratio and compare to threshold
    if total_documents == 0:
        return False

    patent_ratio = patent_count / total_documents
    return patent_ratio >= patent_ratio_threshold


def get_case_flags(
    case_root: Path,
    fee_shifting_ratio_threshold: float = 1.0,
    patent_ratio_threshold: float = 20.0,
    dismissal_ratio_threshold: float = 0.5,
    bankruptcy_ratio_threshold: float = 0.5,
) -> dict[str, bool]:
    """
    Get all flags for a case (dismissal, fee-shifting, large patent amounts, bankruptcy court).

    Args:
        case_root: Path to the case directory
        fee_shifting_ratio_threshold: Minimum ratio of fee-shifting occurrences per document to flag (default: 1.0)
        patent_ratio_threshold: Minimum ratio of patent occurrences per document to flag (default: 20.0)
        dismissal_ratio_threshold: Minimum ratio of documents with dismissal language (default: 0.5)
        bankruptcy_ratio_threshold: Minimum ratio of bankruptcy court documents to flag (default: 0.5)

    Returns:
        dict[str, bool]: Dictionary with flag information
    """
    flags = {
        "is_dismissed": is_case_dismissed(case_root, dismissal_ratio_threshold),
        "has_fee_shifting": has_fee_shifting(case_root, fee_shifting_ratio_threshold),
        "has_large_patent_amounts": has_large_patent_amounts(
            case_root, patent_ratio_threshold
        ),
        "is_bankruptcy_court": get_case_court_type(
            case_root, bankruptcy_ratio_threshold
        )
        == "BANKRUPTCY",
    }

    return flags


def compute_simhash(text: str) -> int:
    """Inline 64-bit Simhash implementation."""
    v = [0] * 64
    for token in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(token.encode()).hexdigest()[-16:], 16)
        for i in range(64):
            v[i] += 1 if (h >> i) & 1 else -1
    fingerprint = 0
    for i, weight in enumerate(v):
        if weight > 0:
            fingerprint |= 1 << i
    return fingerprint


def extract_spelled_out_amount(text: str, match) -> float:
    """Extract monetary value from spelled-out amounts like 'five million dollars'."""
    amt_text = match.group(0).lower()
    words = amt_text.split()

    # Simple mapping for common spelled-out numbers
    number_map = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
        "hundred": 100,
    }

    multiplier = 1_000_000 if "million" in words else 1_000_000_000

    for word in words:
        if word in number_map:
            return number_map[word] * multiplier

    return 0.0


def extract_usd_amount(text: str, match) -> float:
    """Extract monetary value from USD amounts like 'USD 1,234,567.89'."""
    amt_text = match.group(0)
    # Extract numeric value from USD amounts
    numeric_match = re.search(r"[0-9,]+(?:\.[0-9]{2})?", amt_text)
    if numeric_match:
        try:
            return float(numeric_match.group().replace(",", ""))
        except ValueError:
            return 0.0
    return 0.0


def get_spacy_nlp() -> Optional[Any]:
    """Initialize spaCy pipeline with EntityRuler for enhanced amount detection."""
    if not spacy_available:
        return None

    try:
        import spacy

        nlp = spacy.blank("en")

        # Add entity ruler using the factory
        ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})

        patterns = [
            # $X million/billion
            {
                "label": "MONEY",
                "pattern": [
                    {"IS_CURRENCY": True},
                    {"LIKE_NUM": True},
                    {"LOWER": {"IN": ["million", "billion"]}},
                ],
            },
            # $X.X million/billion
            {
                "label": "MONEY",
                "pattern": [
                    {"IS_CURRENCY": True},
                    {"LIKE_NUM": True},
                    {"TEXT": "."},
                    {"LIKE_NUM": True},
                    {"LOWER": {"IN": ["million", "billion"]}},
                ],
            },
            # spelled-out numbers with million/billion
            {
                "label": "MONEY",
                "pattern": [
                    {
                        "LOWER": {
                            "IN": [
                                "one",
                                "two",
                                "three",
                                "four",
                                "five",
                                "six",
                                "seven",
                                "eight",
                                "nine",
                                "ten",
                            ]
                        }
                    },
                    {"LOWER": "million"},
                    {"LOWER": {"IN": ["dollars", "usd"]}},
                ],
            },
            # X million/billion dollars
            {
                "label": "MONEY",
                "pattern": [
                    {"LIKE_NUM": True},
                    {"LOWER": {"IN": ["million", "billion"]}},
                    {"LOWER": {"IN": ["dollars", "usd"]}},
                ],
            },
            # USD 12,000,000.00
            {
                "label": "MONEY",
                "pattern": [
                    {
                        "TEXT": {
                            "REGEX": r"USD?\s?[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]{2})?"
                        }
                    }
                ],
            },
            # $12,000,000.00
            {
                "label": "MONEY",
                "pattern": [
                    {"TEXT": {"REGEX": r"\$[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]{2})?"}}
                ],
            },
        ]

        # Add patterns to the ruler - use try/except to handle different spaCy versions
        try:
            # Use getattr to safely access the method
            add_patterns_method = getattr(ruler, "add_patterns", None)
            if add_patterns_method:
                add_patterns_method(patterns)
        except (AttributeError, TypeError):
            # Fallback for older spaCy versions or if method doesn't exist
            try:
                for pattern in patterns:
                    add_patterns_method = getattr(ruler, "add_patterns", None)
                    if add_patterns_method:
                        add_patterns_method([pattern])
            except (AttributeError, TypeError):
                # If all else fails, just continue without patterns
                pass

        return nlp
    except Exception:
        # Fallback to None if spaCy setup fails
        return None


def extract_spacy_amounts(
    text: str,
    nlp: Optional[Any],
    min_amount: float,
    context_chars: int,
    fast_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Extract amounts using spaCy EntityRuler for spelled-out and USD amounts."""
    # Skip spaCy processing entirely in fast mode for speed
    if fast_mode or not nlp:
        return []

    candidates = []
    try:
        doc = nlp(text)

        # Debug: Print all entities found by spaCy (only in non-fast mode)
        if doc.ents and not fast_mode:
            print(
                f"🔍 spaCy found {len(doc.ents)} entities: {[(ent.text, ent.label_) for ent in doc.ents[:3]]}"
            )

        for ent in doc.ents:
            if ent.label_ == "MONEY":
                amt_text = ent.text.lower()
                if not fast_mode:
                    print(f"💰 spaCy MONEY entity: '{ent.text}' -> '{amt_text}'")

                # Handle spelled-out numbers with million
                if "million" in amt_text and "dollars" in amt_text:
                    # Extract numeric part
                    number_map = {
                        "one": 1,
                        "two": 2,
                        "three": 3,
                        "four": 4,
                        "five": 5,
                        "six": 6,
                        "seven": 7,
                        "eight": 8,
                        "nine": 9,
                        "ten": 10,
                    }

                    words = amt_text.split()
                    for word in words:
                        if word in number_map:
                            val = number_map[word] * 1_000_000
                            if val >= min_amount:
                                start, end = ent.start_char, ent.end_char
                                ctx = text[
                                    max(0, start - context_chars) : end + context_chars
                                ].replace("\n", " ")
                                candidates.append(
                                    {
                                        "amount": ent.text,
                                        "value": val,
                                        "context": ctx,
                                        "type": "spacy_spelled_out",
                                    }
                                )
                            break

                # Handle USD amounts
                elif amt_text.startswith(("usd", "us")):
                    # Extract numeric value from USD amounts
                    numeric_match = re.search(r"[0-9,]+(?:\.[0-9]{2})?", amt_text)
                    if numeric_match:
                        try:
                            val = float(numeric_match.group().replace(",", ""))
                            if val >= min_amount:
                                start, end = ent.start_char, ent.end_char
                                ctx = text[
                                    max(0, start - context_chars) : end + context_chars
                                ].replace("\n", " ")
                                candidates.append(
                                    {
                                        "amount": ent.text,
                                        "value": val,
                                        "context": ctx,
                                        "type": "spacy_usd",
                                    }
                                )
                        except ValueError:
                            continue

                # Handle $X million/billion patterns
                elif "$" in amt_text and (
                    "million" in amt_text or "billion" in amt_text
                ):
                    # Extract numeric value from $X million/billion
                    numeric_match = re.search(r"\$?([0-9,]+(?:\.[0-9]+)?)", amt_text)
                    if numeric_match:
                        try:
                            val = float(numeric_match.group(1).replace(",", ""))
                            if "billion" in amt_text:
                                val *= 1_000_000_000
                            elif "million" in amt_text:
                                val *= 1_000_000

                            if val >= min_amount:
                                start, end = ent.start_char, ent.end_char
                                ctx = text[
                                    max(0, start - context_chars) : end + context_chars
                                ].replace("\n", " ")
                                candidates.append(
                                    {
                                        "amount": ent.text,
                                        "value": val,
                                        "context": ctx,
                                        "type": "spacy_dollar_million",
                                    }
                                )
                        except ValueError:
                            continue

                # Handle X million/billion dollars patterns
                elif (
                    "million" in amt_text or "billion" in amt_text
                ) and "dollars" in amt_text:
                    # Extract numeric value from X million/billion dollars
                    numeric_match = re.search(r"([0-9,]+(?:\.[0-9]+)?)", amt_text)
                    if numeric_match:
                        try:
                            val = float(numeric_match.group(1).replace(",", ""))
                            if "billion" in amt_text:
                                val *= 1_000_000_000
                            elif "million" in amt_text:
                                val *= 1_000_000

                            if val >= min_amount:
                                start, end = ent.start_char, ent.end_char
                                ctx = text[
                                    max(0, start - context_chars) : end + context_chars
                                ].replace("\n", " ")
                                candidates.append(
                                    {
                                        "amount": ent.text,
                                        "value": val,
                                        "context": ctx,
                                        "type": "spacy_million_dollars",
                                    }
                                )
                        except ValueError:
                            continue

                # Handle comma-separated dollar amounts
                elif "$" in amt_text and "," in amt_text:
                    numeric_match = re.search(r"\$?([0-9,]+(?:\.[0-9]{2})?)", amt_text)
                    if numeric_match:
                        try:
                            val = float(numeric_match.group(1).replace(",", ""))
                            if val >= min_amount:
                                start, end = ent.start_char, ent.end_char
                                ctx = text[
                                    max(0, start - context_chars) : end + context_chars
                                ].replace("\n", " ")
                                candidates.append(
                                    {
                                        "amount": ent.text,
                                        "value": val,
                                        "context": ctx,
                                        "type": "spacy_comma_amount",
                                    }
                                )
                        except ValueError:
                            continue
    except Exception as e:
        # Fallback to empty list if spaCy processing fails
        if not fast_mode:
            print(f"❌ spaCy processing error: {e}")
        pass

    return candidates


def compute_chronological_position_votes(
    file_path: str,
    case_position_threshold: float = 0.5,
    docket_position_threshold: float = 0.5,
) -> int:
    """
    Compute votes based on chronological position within case and docket.

    Args:
        file_path: Path to the current file (e.g., "data/tokenized/.../0:12-cv-62086_flsd/doc_125129554_text_stage8.jsonl")
        case_position_threshold: Threshold for case position (0.5 = latter half of case)
        docket_position_threshold: Threshold for docket position (0.5 = latter half of docket)

    Returns:
        int: Number of votes (0-2) based on position criteria
    """
    votes = 0

    try:
        # Parse the file path to get case directory and file info
        path_obj = Path(file_path)
        case_dir = path_obj.parent
        current_file = path_obj.name

        # Get all files in the case directory (sorted chronologically by filename)
        all_case_files = sorted(
            [f.name for f in case_dir.glob("*.jsonl") if f.is_file()]
        )

        if len(all_case_files) <= 1:
            return votes

        # Calculate case position
        case_position_index = all_case_files.index(current_file)
        case_position_ratio = case_position_index / (len(all_case_files) - 1)

        # Vote if in latter portion of case
        if case_position_ratio >= case_position_threshold:
            votes += 1

        # Calculate docket position (entries within this specific document)
        # For this, we need to examine the line number within the file
        # Since we're processing line by line, we'll need to track this in the calling context
        # For now, we'll implement a simpler version that gives a vote if the document
        # appears to be in the later chronological sequence based on doc_id numbers

        # Extract doc_id number for rough chronological ordering within docket
        doc_id_match = re.search(r"doc_(\d+)_", current_file)
        if doc_id_match:
            current_doc_id = int(doc_id_match.group(1))

            # Get all doc_ids in this case
            doc_ids = []
            for fname in all_case_files:
                doc_match = re.search(r"doc_(\d+)_", fname)
                if doc_match:
                    doc_ids.append(int(doc_match.group(1)))

            if len(doc_ids) > 1:
                doc_ids_sorted = sorted(doc_ids)
                doc_position_index = doc_ids_sorted.index(current_doc_id)
                doc_position_ratio = doc_position_index / (len(doc_ids_sorted) - 1)

                # Vote if in latter portion of docket sequence
                if doc_position_ratio >= docket_position_threshold:
                    votes += 1

    except (ValueError, IndexError, OSError) as e:
        # If we can't determine position, don't add votes
        pass

    return votes


def compute_feature_votes(context: str, weights: VotingWeights) -> int:
    """
    Compute the number of heuristic feature matches in the context.
    Each individual pattern match counts as +1 vote.
    Features:
    1. PROXIMITY_PATTERN - monetary context words (settlement, judgment, damages, etc.)
    2. JUDGMENT_VERBS - legal action verbs (awarded, ordered, granted, etc.)

    Returns:
        int: Total number of pattern matches found
    """
    feature_count = 0.0

    # Count all PROXIMITY_PATTERN matches
    proximity_matches = PROXIMITY_PATTERN.findall(context)
    feature_count += len(proximity_matches) * weights.proximity_pattern_weight

    # Count all JUDGMENT_VERBS matches
    judgment_matches = JUDGMENT_VERBS.findall(context)
    feature_count += len(judgment_matches) * weights.judgment_verbs_weight

    return int(feature_count)


def compute_enhanced_feature_votes(
    context: str,
    file_path: str,
    case_position_threshold: float = 0.5,
    docket_position_threshold: float = 0.5,
    weights: VotingWeights = DEFAULT_VOTING_WEIGHTS,
) -> int:
    """
    Enhanced feature voting that includes chronological position votes.

    Args:
        context: Text context around the amount
        file_path: Path to the current file for position calculation
        case_position_threshold: Threshold for case position voting
        docket_position_threshold: Threshold for docket position voting

    Returns:
        int: Total number of feature votes including position votes
    """
    # Get standard feature votes
    votes = float(compute_feature_votes(context, weights))

    # Add chronological position votes
    position_votes = (
        compute_chronological_position_votes(
            file_path, case_position_threshold, docket_position_threshold
        )
        * weights.case_position_weight
    )
    votes += position_votes

    return int(votes)


def passes_feature_filter(
    context: str, min_features: int, weights: VotingWeights
) -> bool:
    """
    Check if context passes the minimum feature requirement.

    Args:
        context: Text context around the amount
        min_features: Minimum number of heuristic features required (0-2)

    Returns:
        bool: True if context has at least min_features
    """
    return compute_feature_votes(context, weights) >= min_features


def passes_enhanced_feature_filter(
    context: str,
    file_path: str,
    min_features: int,
    case_position_threshold: float = 0.5,
    docket_position_threshold: float = 0.5,
    weights: VotingWeights = DEFAULT_VOTING_WEIGHTS,
) -> bool:
    """
    Enhanced feature filter that includes chronological position votes.

    Args:
        context: Text context around the amount
        file_path: Path to the current file for position calculation
        min_features: Minimum number of features required
        case_position_threshold: Threshold for case position voting
        docket_position_threshold: Threshold for docket position voting

    Returns:
        bool: True if context+position has at least min_features
    """
    votes = compute_enhanced_feature_votes(
        context, file_path, case_position_threshold, docket_position_threshold, weights
    )
    return votes >= min_features


def compute_enhanced_feature_votes_with_titles(
    context: str,
    file_path: str,
    case_position_threshold: float = 0.5,
    docket_position_threshold: float = 0.5,
    weights: VotingWeights = DEFAULT_VOTING_WEIGHTS,
    header_chars: int = 2000,
) -> int:
    """
    Enhanced feature voting that includes chronological position votes and title proximity votes.

    Args:
        context: Text context around the amount
        file_path: Path to the current file for position calculation
        case_position_threshold: Threshold for case position voting
        docket_position_threshold: Threshold for docket position voting
        header_chars: Number of characters to consider as header for document titles

    Returns:
        int: Total number of feature votes including position and title votes
    """
    # Get standard feature votes
    votes = float(
        compute_enhanced_feature_votes(
            context,
            file_path,
            case_position_threshold,
            docket_position_threshold,
            weights,
        )
    )

    # Add votes for ALL CAPS section titles in context
    title_votes = (
        count_all_caps_section_titles(context) * weights.all_caps_titles_weight
    )
    votes += title_votes

    # Add votes for document titles in header
    doc_title_votes = (
        count_document_titles(context, header_chars) * weights.document_titles_weight
    )
    votes += doc_title_votes

    return int(votes)


def passes_enhanced_feature_filter_with_titles(
    context: str,
    file_path: str,
    min_features: int,
    case_position_threshold: float = 0.5,
    docket_position_threshold: float = 0.5,
    weights: VotingWeights = DEFAULT_VOTING_WEIGHTS,
    header_chars: int = 2000,
) -> bool:
    """
    Enhanced feature filter that includes chronological position votes and title proximity votes.

    Args:
        context: Text context around the amount
        file_path: Path to the current file for position calculation
        min_features: Minimum number of features required
        case_position_threshold: Threshold for case position voting
        docket_position_threshold: Threshold for docket position voting
        header_chars: Number of characters to consider as header for document titles

    Returns:
        bool: True if context+position+titles has at least min_features
    """
    votes = compute_enhanced_feature_votes_with_titles(
        context,
        file_path,
        case_position_threshold,
        docket_position_threshold,
        weights,
        header_chars,
    )
    return votes >= min_features


def export_candidates_to_csv(candidates_data, csv_output_path, max_cases):
    """Export candidates to CSV for annotation following the Tier 2 schema."""
    csv_rows = []
    cases_seen = set()

    for candidate in candidates_data:
        # Limit number of cases
        if len(cases_seen) >= max_cases:
            break

        case_id = candidate.get("case_id", "")
        cases_seen.add(case_id)

        # Extract doc_id from file path
        file_name = candidate.get("file", "")
        doc_id = file_name.replace("_text_stage1.jsonl", "").replace("doc_", "")

        # Compute features
        context = candidate.get("context", "")
        has_verb = bool(JUDGMENT_VERBS.search(context))
        has_proximity = bool(PROXIMITY_PATTERN.search(context))

        # Parse amount value
        amount_text = candidate.get("amount", "")
        amount_val = 0.0
        try:
            if "million" in amount_text.lower():
                num_part = re.search(r"[\d.]+", amount_text)
                if num_part:
                    amount_val = float(num_part.group()) * 1_000_000
            elif "billion" in amount_text.lower():
                num_part = re.search(r"[\d.]+", amount_text)
                if num_part:
                    amount_val = float(num_part.group()) * 1_000_000_000
            else:
                # Remove currency symbols and parse
                clean_amount = re.sub(r"[,$USD]", "", amount_text).strip()
                if clean_amount:
                    amount_val = float(clean_amount)
        except (ValueError, AttributeError):
            amount_val = 0.0

        csv_rows.append(
            {
                "case_id": case_id,
                "doc_id": doc_id,
                "page": "",  # Not available in current data
                "candidate_text": context,
                "amount_val": amount_val,
                "has_verb": has_verb,
                "has_proximity": has_proximity,
                "rel_position": "",  # Not available in current data
                "label": "",  # To be filled by annotator
                "notes": "",  # To be filled by annotator
                "simhash": candidate.get("simhash", ""),  # For matching back
                "amount_text": amount_text,  # Original amount for reference
            }
        )

    # Sort by priority: has_verb desc, has_proximity desc, amount_val desc
    csv_rows.sort(
        key=lambda x: (x["has_verb"], x["has_proximity"], x["amount_val"]), reverse=True
    )

    # Write CSV
    with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "case_id",
            "doc_id",
            "page",
            "candidate_text",
            "amount_val",
            "has_verb",
            "has_proximity",
            "rel_position",
            "label",
            "notes",
            "simhash",
            "amount_text",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(
        f"✓ Exported {len(csv_rows)} candidates from {len(cases_seen)} cases to {csv_output_path}"
    )
    print("Annotation workflow:")
    print("  1. Open CSV in Excel/Google Sheets")
    print("  2. Sort by has_verb desc, has_proximity desc, amount_val desc")
    print("  3. Fill 'label' column with: TRUE_FINAL, FALSE_MISC, or PROC_DOC")
    print("  4. Add notes as needed")
    print("  5. Save and use --apply-csv-labels to apply back to JSONL")


def apply_csv_labels_to_jsonl(csv_path, jsonl_input_path, jsonl_output_path):
    """Apply CSV annotations back to JSONL output."""
    # Load CSV annotations
    labels_by_simhash = {}
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            simhash = row.get("simhash", "").strip()
            label = row.get("label", "").strip()
            notes = row.get("notes", "").strip()
            if simhash and label:
                labels_by_simhash[simhash] = {"label": label, "notes": notes}

    print(f"✓ Loaded {len(labels_by_simhash)} annotations from {csv_path}")

    # Apply labels to JSONL
    applied_count = 0
    with open(jsonl_input_path, "r", encoding="utf-8") as infile, open(
        jsonl_output_path, "w", encoding="utf-8"
    ) as outfile:

        for line in infile:
            data = json.loads(line)
            simhash = str(data.get("simhash", ""))

            if simhash in labels_by_simhash:
                annotation = labels_by_simhash[simhash]
                data["annotation_label"] = annotation["label"]
                data["annotation_notes"] = annotation["notes"]
                applied_count += 1

            outfile.write(json.dumps(data) + "\n")

    print(f"✓ Applied {applied_count} annotations to {jsonl_output_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract dollar amounts from stage1 JSONL files"
    )
    p.add_argument(
        "--input-pattern",
        default=DEFAULT_INPUT_PATTERN,
        help="Glob pattern for input `_text_stage1.jsonl` files",
    )
    p.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="Path to write extracted JSONL records",
    )
    p.add_argument(
        "--context-chars",
        type=int,
        default=DEFAULT_CONTEXT_CHARS,
        help="Number of chars of context before/after each match",
    )
    p.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
        help="Base directory under which to look for `_stage3.jsonl` files",
    )
    p.add_argument(
        "--min-amount",
        type=float,
        default=DEFAULT_MIN_AMOUNT,
        help="Skip numeric values below this threshold (unless 'million'/'billion')",
    )
    p.add_argument(
        "--min-features",
        type=int,
        default=2,
        help="Minimum number of heuristic feature votes needed to pass filter (each pattern match = +1)",
    )
    p.add_argument(
        "--csv-output",
        help="Export candidates to CSV for annotation (e.g., candidates.csv)",
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=50,
        help="Limit number of cases for CSV export (default: 50)",
    )
    p.add_argument(
        "--apply-csv-labels",
        help="Apply labels from annotated CSV back to JSONL output",
    )
    p.add_argument(
        "--case-position-threshold",
        type=float,
        default=0.5,
        help="Threshold for case position voting (0.5 = latter half of case gets vote)",
    )
    p.add_argument(
        "--docket-position-threshold",
        type=float,
        default=0.5,
        help="Threshold for docket position voting (0.5 = latter half of docket gets vote)",
    )
    return p.parse_args()


def main(
    input_pattern: str,
    output_file: str,
    context_chars: int,
    base_dir: str,
    min_amount: float,
    min_features: int = 2,
    csv_output: Optional[str] = None,
    max_cases: int = 50,
    apply_csv_labels: Optional[str] = None,
    case_position_threshold: float = 0.5,
    docket_position_threshold: float = 0.5,
):
    # Handle CSV label application mode
    if apply_csv_labels:
        apply_csv_labels_to_jsonl(
            apply_csv_labels,
            output_file,
            output_file.replace(".jsonl", "_annotated.jsonl"),
        )
        return

    seen = set()
    total_files = 0
    files_with_hits = 0
    case_dirs = set()
    case_with_hits = set()
    candidates_for_csv = []  # Store candidates for CSV export

    # Initialize spaCy pipeline once for reuse
    nlp = get_spacy_nlp()
    if nlp:
        print("✓ spaCy EntityRuler initialized successfully")
    else:
        print("⚠ spaCy not available, using regex-only extraction")

    with open(output_file, "w", encoding="utf-8") as out_f:
        for path in glob.glob(input_pattern, recursive=True):
            total_files += 1
            case_id = os.path.basename(
                os.path.dirname(os.path.dirname(os.path.dirname(path)))
            )
            case_dirs.add(case_id)
            file_hit = False

            with open(path, encoding="utf-8") as in_f:
                for line in in_f:
                    text = json.loads(line).get("text", "")

                    # Procedural-doc filter: Skip any Stage-1 file that contains no money pattern and no judgment verb
                    if not (AMOUNT_REGEX.search(text) or JUDGMENT_VERBS.search(text)):
                        continue  # drop as procedural

                    # Process spaCy EntityRuler amounts (new - highest priority)
                    spacy_candidates = extract_spacy_amounts(
                        text, nlp, min_amount, context_chars
                    )
                    for candidate in spacy_candidates:
                        ctx = candidate["context"]
                        # Enhanced filtering: require minimum feature votes
                        if passes_enhanced_feature_filter_with_titles(
                            ctx,
                            path,
                            min_features,
                            case_position_threshold,
                            docket_position_threshold,
                        ):
                            sim = compute_simhash(ctx)
                            if sim not in seen:
                                seen.add(sim)
                                file_hit = True
                                case_with_hits.add(case_id)
                                candidate_data = {
                                    "file": os.path.basename(path),
                                    "amount": candidate["amount"],
                                    "context": ctx,
                                    "simhash": sim,
                                    "type": candidate["type"],
                                    "case_id": case_id,
                                }
                                out_f.write(json.dumps(candidate_data) + "\n")
                                if csv_output:
                                    candidates_for_csv.append(candidate_data)

                    # Process enhanced spelled-out amounts (new)
                    for m in SPELLED_OUT_AMOUNTS.finditer(text):
                        val = extract_spelled_out_amount(text, m)
                        if val >= min_amount:
                            start, end = m.span()
                            pre = text[max(0, start - context_chars) : start]
                            post = text[end : end + context_chars]
                            ctx = (pre + m.group(0) + post).replace("\n", " ")

                            # Enhanced filtering: require minimum feature votes
                            if passes_enhanced_feature_filter_with_titles(
                                ctx,
                                path,
                                min_features,
                                case_position_threshold,
                                docket_position_threshold,
                            ):
                                sim = compute_simhash(ctx)
                                if sim not in seen:
                                    seen.add(sim)
                                    file_hit = True
                                    case_with_hits.add(case_id)
                                    candidate_data = {
                                        "file": os.path.basename(path),
                                        "amount": m.group(0),
                                        "context": ctx,
                                        "simhash": sim,
                                        "type": "spelled_out",
                                        "case_id": case_id,
                                    }
                                    out_f.write(json.dumps(candidate_data) + "\n")
                                    if csv_output:
                                        candidates_for_csv.append(candidate_data)

                    # Process enhanced USD amounts (new)
                    for m in USD_AMOUNTS.finditer(text):
                        val = extract_usd_amount(text, m)
                        if val >= min_amount:
                            start, end = m.span()
                            pre = text[max(0, start - context_chars) : start]
                            post = text[end : end + context_chars]
                            ctx = (pre + m.group(0) + post).replace("\n", " ")

                            # Enhanced filtering: require minimum feature votes
                            if passes_enhanced_feature_filter_with_titles(
                                ctx,
                                path,
                                min_features,
                                case_position_threshold,
                                docket_position_threshold,
                            ):
                                sim = compute_simhash(ctx)
                                if sim not in seen:
                                    seen.add(sim)
                                    file_hit = True
                                    case_with_hits.add(case_id)
                                    out_f.write(
                                        json.dumps(
                                            {
                                                "file": os.path.basename(path),
                                                "amount": m.group(0),
                                                "context": ctx,
                                                "simhash": sim,
                                                "type": "usd_prefix",
                                            }
                                        )
                                        + "\n"
                                    )

                    # Continue with existing regex extraction (enhanced with judgment-verb filtering)
                    for m in AMOUNT_REGEX.finditer(text):
                        amt = m.group(0)
                        norm = amt.lower().replace(",", "").replace("$", "").strip()

                        # numeric threshold
                        if "million" not in norm and "billion" not in norm:
                            try:
                                search_result = re.search(r"[0-9.]+", norm)
                                if search_result is None:
                                    continue
                                val = float(search_result.group(0))
                                if val < min_amount:
                                    continue
                            except Exception:
                                continue

                        start, end = m.span()
                        pre = text[max(0, start - context_chars) : start]
                        post = text[end : end + context_chars]
                        ctx = (pre + amt + post).replace("\n", " ")

                        # Enhanced filtering: require minimum feature votes
                        if not passes_enhanced_feature_filter_with_titles(
                            ctx,
                            path,
                            min_features,
                            case_position_threshold,
                            docket_position_threshold,
                        ):
                            continue

                        sim = compute_simhash(ctx)
                        if sim in seen:
                            continue

                        seen.add(sim)
                        file_hit = True
                        case_with_hits.add(case_id)

                        out_f.write(
                            json.dumps(
                                {
                                    "file": os.path.basename(path),
                                    "amount": amt,
                                    "context": ctx,
                                    "simhash": sim,
                                    "type": "standard",
                                }
                            )
                            + "\n"
                        )

            if file_hit:
                files_with_hits += 1

    # Summaries
    total_cases = len(case_dirs)
    hit_cases = len(case_with_hits)
    print(f"Extraction complete → {output_file}")
    print(
        f"Files scanned: {total_files}, hits: {files_with_hits}, no-hits: {total_files - files_with_hits}"
    )
    print(
        f"Cases total: {total_cases}, hit-cases: {hit_cases}, no-hit-cases: {total_cases - hit_cases}"
    )

    def count_stage3(case_set):
        cnt = 0
        for case in case_set:
            for fn in glob.glob(
                os.path.join(base_dir, case, "**", "*_stage3.jsonl"), recursive=True
            ):
                with open(fn, encoding="utf-8") as f3:
                    cnt += sum(1 for _ in f3)
        return cnt

    print(f"Stage-3 entries in hit-cases    : {count_stage3(case_with_hits)}")
    print(
        f"Stage-3 entries in no-hit-cases : {count_stage3(case_dirs - case_with_hits)}"
    )

    # Export to CSV if requested
    if csv_output and candidates_for_csv:
        export_candidates_to_csv(candidates_for_csv, csv_output, max_cases)


if __name__ == "__main__":
    args = parse_args()
    main(
        args.input_pattern,
        args.output_file,
        args.context_chars,
        args.base_dir,
        args.min_amount,
        args.min_features,
        args.csv_output,
        args.max_cases,
        args.apply_csv_labels,
        args.case_position_threshold,
        args.docket_position_threshold,
    )
