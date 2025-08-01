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
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Any, Dict
from fractions import Fraction

# High-performance JSON parser for Apple M1/ARM64 optimization
try:
    import orjson

    # Fast JSON loading function
    def fast_json_loads(data: str) -> dict:
        """Fast JSON parsing using orjson (optimized for ARM64/M1)."""
        return orjson.loads(data)

except ImportError:
    # Fallback to standard json if orjson not available
    def fast_json_loads(data: str) -> dict:
        """Fallback JSON parsing using standard library."""
        return json.loads(data)


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
        # New enhanced features
        financial_terms_weight: float = 1.0,
        settlement_terms_weight: float = 1.0,
        legal_proceedings_weight: float = 1.0,
        monetary_phrases_weight: float = 1.0,
        dependency_parsing_weight: float = 1.0,
        fraction_extraction_weight: float = 1.0,
        percentage_extraction_weight: float = 1.0,
        implied_totals_weight: float = 1.0,
        document_structure_weight: float = 1.0,
        table_detection_weight: float = 1.0,
        header_detection_weight: float = 1.0,
        section_boundaries_weight: float = 1.0,
        numeric_gazetteer_weight: float = 1.0,
        mixed_numbers_weight: float = 1.0,
        sentence_boundary_weight: float = 1.0,
        paragraph_boundary_weight: float = 1.0,
        # Confidence boosting features
        high_confidence_patterns_weight: float = 1.0,
        amount_adjacent_keywords_weight: float = 1.0,
        confidence_boost_weight: float = 1.0,
        # High/Low signal regex weights for fine-grained control
        high_signal_financial_weight: float = 1.0,
        low_signal_financial_weight: float = 0.5,
        high_signal_settlement_weight: float = 1.0,
        low_signal_settlement_weight: float = 0.5,
        calculation_boost_multiplier: float = 1.0,
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
            financial_terms_weight: Weight for financial terminology gazetteer
            settlement_terms_weight: Weight for settlement-specific terms
            legal_proceedings_weight: Weight for legal proceedings vocabulary
            monetary_phrases_weight: Weight for monetary phrase patterns
            dependency_parsing_weight: Weight for dependency parsing features
            fraction_extraction_weight: Weight for fraction extraction
            percentage_extraction_weight: Weight for percentage extraction
            implied_totals_weight: Weight for implied total calculations
            document_structure_weight: Weight for document structure features
            table_detection_weight: Weight for table detection
            header_detection_weight: Weight for header detection
            section_boundaries_weight: Weight for section boundary detection
            numeric_gazetteer_weight: Weight for numeric gazetteer matches
            mixed_numbers_weight: Weight for mixed number patterns
            sentence_boundary_weight: Weight for sentence boundary context
            paragraph_boundary_weight: Weight for paragraph boundary context
            high_confidence_patterns_weight: Weight for high-confidence monetary patterns
            amount_adjacent_keywords_weight: Weight for amount-adjacent keywords
            confidence_boost_weight: Weight for overall confidence boost score
        """
        self.proximity_pattern_weight = proximity_pattern_weight
        self.judgment_verbs_weight = judgment_verbs_weight
        self.case_position_weight = case_position_weight
        self.docket_position_weight = docket_position_weight
        self.all_caps_titles_weight = all_caps_titles_weight
        self.document_titles_weight = document_titles_weight
        # New enhanced features
        self.financial_terms_weight = financial_terms_weight
        self.settlement_terms_weight = settlement_terms_weight
        self.legal_proceedings_weight = legal_proceedings_weight
        self.monetary_phrases_weight = monetary_phrases_weight
        self.dependency_parsing_weight = dependency_parsing_weight
        self.fraction_extraction_weight = fraction_extraction_weight
        self.percentage_extraction_weight = percentage_extraction_weight
        self.implied_totals_weight = implied_totals_weight
        self.document_structure_weight = document_structure_weight
        self.table_detection_weight = table_detection_weight
        self.header_detection_weight = header_detection_weight
        self.section_boundaries_weight = section_boundaries_weight
        self.numeric_gazetteer_weight = numeric_gazetteer_weight
        self.mixed_numbers_weight = mixed_numbers_weight
        self.sentence_boundary_weight = sentence_boundary_weight
        self.paragraph_boundary_weight = paragraph_boundary_weight
        # Confidence boosting features
        self.high_confidence_patterns_weight = high_confidence_patterns_weight
        self.amount_adjacent_keywords_weight = amount_adjacent_keywords_weight
        self.confidence_boost_weight = confidence_boost_weight
        self.high_signal_financial_weight = high_signal_financial_weight
        self.low_signal_financial_weight = low_signal_financial_weight
        self.high_signal_settlement_weight = high_signal_settlement_weight
        self.low_signal_settlement_weight = low_signal_settlement_weight
        self.calculation_boost_multiplier = calculation_boost_multiplier

    def to_dict(self) -> dict[str, float]:
        """Convert weights to dictionary for serialization."""
        return {
            "proximity_pattern_weight": self.proximity_pattern_weight,
            "judgment_verbs_weight": self.judgment_verbs_weight,
            "case_position_weight": self.case_position_weight,
            "docket_position_weight": self.docket_position_weight,
            "all_caps_titles_weight": self.all_caps_titles_weight,
            "document_titles_weight": self.document_titles_weight,
            "financial_terms_weight": self.financial_terms_weight,
            "settlement_terms_weight": self.settlement_terms_weight,
            "legal_proceedings_weight": self.legal_proceedings_weight,
            "monetary_phrases_weight": self.monetary_phrases_weight,
            "dependency_parsing_weight": self.dependency_parsing_weight,
            "fraction_extraction_weight": self.fraction_extraction_weight,
            "percentage_extraction_weight": self.percentage_extraction_weight,
            "implied_totals_weight": self.implied_totals_weight,
            "document_structure_weight": self.document_structure_weight,
            "table_detection_weight": self.table_detection_weight,
            "header_detection_weight": self.header_detection_weight,
            "section_boundaries_weight": self.section_boundaries_weight,
            "numeric_gazetteer_weight": self.numeric_gazetteer_weight,
            "mixed_numbers_weight": self.mixed_numbers_weight,
            "sentence_boundary_weight": self.sentence_boundary_weight,
            "paragraph_boundary_weight": self.paragraph_boundary_weight,
            "high_confidence_patterns_weight": self.high_confidence_patterns_weight,
            "amount_adjacent_keywords_weight": self.amount_adjacent_keywords_weight,
            "confidence_boost_weight": self.confidence_boost_weight,
            "high_signal_financial_weight": self.high_signal_financial_weight,
            "low_signal_financial_weight": self.low_signal_financial_weight,
            "high_signal_settlement_weight": self.high_signal_settlement_weight,
            "low_signal_settlement_weight": self.low_signal_settlement_weight,
            "calculation_boost_multiplier": self.calculation_boost_multiplier,
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
    r"\b(?:award(?:ed)?|order(?:ed)?|grant(?:ed)?|enter(?:ed)?|assess(?:ed)?|recover(?:y|ed)?|release(?:d)?|dismiss(?:al|ed)?|preliminary\s+approval)\b",
    re.IGNORECASE,
)

# More permissive proximity pattern to catch monetary contexts
PROXIMITY_PATTERN = re.compile(
    r"\b(?:settlement|judgment|judgement|damages|award|penalty|fine|amount|paid|cost|price|fee|compensation|restitution|claim|relief|recover|value|sum|settlement\s+fund|released\s+claims|actions|escrow|payment\s+into)\b",
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

# ------------------------------------------------------------------------------
# Enhanced Pattern Inventory for Improved Coverage
# ------------------------------------------------------------------------------

# Financial terminology gazetteer - split into high and low signal
# High signal financial terms (direct monetary awards/judgments)
HIGH_SIGNAL_FINANCIAL_TERMS = re.compile(
    r"\b(?:settlement\s+fund|common\s+fund|escrow\s+account)\b",
    re.IGNORECASE,
)

# Low signal financial terms (general financial vocabulary)
LOW_SIGNAL_FINANCIAL_TERMS = re.compile(
    r"\b(?:net\s+present\s+value|trust\s+fund|reserve\s+fund|contingency\s+fund|"
    r"fair\s+market\s+value|book\s+value|asset\s+value|enterprise\s+value|equity\s+value|"
    r"contingent\s+value\s+right|working\s+capital|net\s+worth|shareholders?\s+equity|"
    r"retained\s+earnings|accounts\s+receivable|accounts\s+payable|notes\s+payable|"
    r"debt\s+service|interest\s+expense|dividend\s+payment|capital\s+expenditure|"
    r"operating\s+expense|gross\s+revenue|net\s+revenue|ebitda|net\s+income|"
    r"gross\s+profit|operating\s+income|total\s+consideration|"
    r"purchase\s+price|sale\s+price|transaction\s+value|deal\s+value|merger\s+consideration|"
    r"acquisition\s+cost|breakup\s+fee|termination\s+fee|earnout|liquidation\s+value)\b",
    re.IGNORECASE,
)

# Legacy pattern for backward compatibility
FINANCIAL_TERMS = re.compile(
    r"\b(?:net\s+present\s+value|total\s+consideration|common\s+fund|escrow\s+account|"
    r"settlement\s+fund|trust\s+fund|reserve\s+fund|contingency\s+fund|liquidation\s+value|"
    r"fair\s+market\s+value|book\s+value|asset\s+value|enterprise\s+value|equity\s+value|"
    r"purchase\s+price|sale\s+price|transaction\s+value|deal\s+value|merger\s+consideration|"
    r"acquisition\s+cost|breakup\s+fee|termination\s+fee|earnout|contingent\s+value\s+right|"
    r"working\s+capital|net\s+worth|shareholders?\s+equity|retained\s+earnings|"
    r"accounts\s+receivable|accounts\s+payable|notes\s+payable|debt\s+service|"
    r"interest\s+expense|dividend\s+payment|capital\s+expenditure|operating\s+expense|"
    r"gross\s+revenue|net\s+revenue|ebitda|net\s+income|gross\s+profit|operating\s+income)\b",
    re.IGNORECASE,
)

# Settlement-specific terminology - split into high and low signal
# High signal settlement terms (direct settlement amounts/awards)
HIGH_SIGNAL_SETTLEMENT_TERMS = re.compile(
    r"\b(?:settlement\s+agreement|final\s+approval|preliminary\s+approval|"
    r"class\s+action\s+settlement|mediation\s+settlement|arbitration\s+award|"
    r"consent\s+judgment|settlement\s+fund|qualified\s+settlement\s+fund|"
    r"attorney\s+fees\s+award|incentive\s+award|service\s+award)\b",
    re.IGNORECASE,
)

# Low signal settlement terms (procedural/administrative)
LOW_SIGNAL_SETTLEMENT_TERMS = re.compile(
    r"\b(?:consent\s+decree|stipulated\s+order|collective\s+action\s+settlement|"
    r"stipulation|release\s+agreement|distribution\s+plan|claims\s+administrator|"
    r"settlement\s+administrator|notice\s+to\s+class\s+members|opt-out\s+period|"
    r"fairness\s+hearing|objection\s+deadline|settlement\s+class|qsf|"
    r"cy\s+pres|residual\s+funds|unclaimed\s+funds|pro\s+rata\s+distribution|"
    r"claims\s+process|claim\s+form|settlement\s+website|settlement\s+notice|"
    r"representative\s+enhancement)\b",
    re.IGNORECASE,
)

# Legacy pattern for backward compatibility
SETTLEMENT_TERMS = re.compile(
    r"\b(?:settlement\s+agreement|consent\s+decree|stipulated\s+order|final\s+approval|"
    r"preliminary\s+approval|class\s+action\s+settlement|collective\s+action\s+settlement|"
    r"mediation\s+settlement|arbitration\s+award|consent\s+judgment|stipulation|"
    r"release\s+agreement|distribution\s+plan|claims\s+administrator|settlement\s+administrator|"
    r"notice\s+to\s+class\s+members|opt-out\s+period|fairness\s+hearing|objection\s+deadline|"
    r"settlement\s+class|settlement\s+fund|qualified\s+settlement\s+fund|qsf|"
    r"cy\s+pres|residual\s+funds|unclaimed\s+funds|pro\s+rata\s+distribution|"
    r"claims\s+process|claim\s+form|settlement\s+website|settlement\s+notice|"
    r"attorney\s+fees\s+award|incentive\s+award|service\s+award|representative\s+enhancement)\b",
    re.IGNORECASE,
)

# Legal proceedings vocabulary - split into high and low signal
# High signal legal proceedings (direct monetary outcomes)
HIGH_SIGNAL_LEGAL_PROCEEDINGS = re.compile(
    r"\b(?:default\s+judgment|summary\s+judgment|directed\s+verdict|"
    r"judgment\s+as\s+a\s+matter\s+of\s+law|jury\s+verdict|trial\s+verdict|"
    r"final\s+judgment|restitution|disgorgement)\b",
    re.IGNORECASE,
)

# Low signal legal proceedings (procedural/case management)
LOW_SIGNAL_LEGAL_PROCEEDINGS = re.compile(
    r"\b(?:complaint|counterclaim|cross-claim|third-party\s+complaint|amended\s+complaint|"
    r"motion\s+to\s+dismiss|motion\s+for\s+summary\s+judgment|motion\s+in\s+limine|"
    r"daubert\s+motion|class\s+certification|preliminary\s+injunction|"
    r"temporary\s+restraining\s+order|discovery\s+motion|motion\s+to\s+compel|"
    r"protective\s+order|sanctions\s+motion|bench\s+trial|interlocutory\s+appeal|"
    r"appeal|remand|reversal|affirmance|mandamus|certiorari|writ\s+of\s+error|"
    r"stipulation|consent\s+decree|injunctive\s+relief|declaratory\s+judgment|"
    r"constructive\s+trust|receive|accounting|receivership)\b",
    re.IGNORECASE,
)

# Legacy pattern for backward compatibility
LEGAL_PROCEEDINGS = re.compile(
    r"\b(?:complaint|counterclaim|cross-claim|third-party\s+complaint|amended\s+complaint|"
    r"motion\s+to\s+dismiss|motion\s+for\s+summary\s+judgment|motion\s+in\s+limine|"
    r"daubert\s+motion|class\s+certification|preliminary\s+injunction|temporary\s+restraining\s+order|"
    r"discovery\s+motion|motion\s+to\s+compel|protective\s+order|sanctions\s+motion|"
    r"default\s+judgment|summary\s+judgment|directed\s+verdict|judgment\s+as\s+a\s+matter\s+of\s+law|"
    r"jury\s+verdict|bench\s+trial|trial\s+verdict|final\s+judgment|interlocutory\s+appeal|"
    r"appeal|remand|reversal|affirmance|mandamus|certiorari|writ\s+of\s+error|"
    r"stipulation|consent\s+decree|injunctive\s+relief|declaratory\s+judgment|"
    r"restitution|disgorgement|constructive\s+trust|accounting|receiver|receivership)\b",
    re.IGNORECASE,
)

# Enhanced monetary phrases - split into high and low signal
# High signal monetary phrases (direct financial awards)
HIGH_SIGNAL_MONETARY_PHRASES = re.compile(
    r"\b(?:monetary\s+relief|compensatory\s+damages|punitive\s+damages|exemplary\s+damages|"
    r"liquidated\s+damages|actual\s+damages|direct\s+damages|lost\s+profits|"
    r"lost\s+revenue|lost\s+income|attorney\s+fees|attorneys?\s+fees\s+and\s+costs|"
    r"civil\s+penalty|civil\s+fine|restitution\s+order|disgorgement\s+order|"
    r"treble\s+damages|double\s+damages|enhanced\s+damages|statutory\s+damages)\b",
    re.IGNORECASE,
)

# Low signal monetary phrases (procedural/incidental)
LOW_SIGNAL_MONETARY_PHRASES = re.compile(
    r"\b(?:equitable\s+relief|injunctive\s+relief|declaratory\s+relief|"
    r"consequential\s+damages|incidental\s+damages|special\s+damages|general\s+damages|"
    r"out-of-pocket\s+expenses|reasonable\s+expenses|necessary\s+expenses|"
    r"litigation\s+costs|court\s+costs|expert\s+witness\s+fees|discovery\s+costs|"
    r"trial\s+costs|appeal\s+costs|pre-judgment\s+interest|post-judgment\s+interest|"
    r"statutory\s+interest|compound\s+interest|simple\s+interest|interest\s+rate|"
    r"discount\s+rate|criminal\s+fine|criminal\s+penalty|forfeiture|accounting\s+order|"
    r"minimum\s+damages|maximum\s+damages|damage\s+cap|liability\s+cap)\b",
    re.IGNORECASE,
)

# Legacy pattern for backward compatibility
MONETARY_PHRASES = re.compile(
    r"\b(?:monetary\s+relief|equitable\s+relief|injunctive\s+relief|declaratory\s+relief|"
    r"compensatory\s+damages|punitive\s+damages|exemplary\s+damages|liquidated\s+damages|"
    r"consequential\s+damages|incidental\s+damages|special\s+damages|general\s+damages|"
    r"actual\s+damages|direct\s+damages|lost\s+profits|lost\s+revenue|lost\s+income|"
    r"out-of-pocket\s+expenses|reasonable\s+expenses|necessary\s+expenses|"
    r"attorney\s+fees|attorneys?\s+fees\s+and\s+costs|litigation\s+costs|court\s+costs|"
    r"expert\s+witness\s+fees|discovery\s+costs|trial\s+costs|appeal\s+costs|"
    r"pre-judgment\s+interest|post-judgment\s+interest|statutory\s+interest|"
    r"compound\s+interest|simple\s+interest|interest\s+rate|discount\s+rate|"
    r"civil\s+penalty|civil\s+fine|criminal\s+fine|criminal\s+penalty|forfeiture|"
    r"restitution\s+order|disgorgement\s+order|accounting\s+order|"
    r"treble\s+damages|double\s+damages|enhanced\s+damages|statutory\s+damages|"
    r"minimum\s+damages|maximum\s+damages|damage\s+cap|liability\s+cap)\b",
    re.IGNORECASE,
)

# High-confidence monetary context patterns for score boosting
HIGH_CONFIDENCE_PATTERNS = re.compile(
    r"\b(?:judgment\s+in\s+the\s+amount\s+of|ordered\s+to\s+pay|award(?:ed|s)?\s+(?:to|in\s+favor\s+of)|"
    r"settled?\s+for|settlement\s+(?:amount|fund|value)\s+of|compensation\s+of|"
    r"damages?\s+(?:in\s+the\s+amount\s+)?of|penalty\s+of|fine\s+of|"
    r"total\s+(?:settlement|award|judgment|damages?|compensation)\s+(?:amount\s+)?(?:is|was|of)|"
    r"monetary\s+(?:award|judgment|settlement|relief)\s+of|"
    r"sum\s+of|amount\s+(?:awarded|granted|ordered)\s+(?:is|was)|"
    r"court\s+(?:awards?|orders?)\s+(?:plaintiff|defendant)|"
    r"(?:final|total)\s+judgment\s+(?:amount\s+)?of|"
    r"civil\s+penalty\s+(?:in\s+the\s+amount\s+)?of|"
    r"restitution\s+(?:in\s+the\s+amount\s+)?of|"
    r"disgorgement\s+(?:in\s+the\s+amount\s+)?of)\b",
    re.IGNORECASE,
)

# Amount-adjacent keywords that strongly suggest the amount is the main award
AMOUNT_ADJACENT_KEYWORDS = re.compile(
    r"(?:is\s+hereby\s+)?(?:awarded|granted|ordered|settled|paid|compensated|reimbursed|"
    r"entitled\s+to|shall\s+pay|must\s+pay|required\s+to\s+pay|judgment\s+for|"
    r"in\s+damages|as\s+damages|for\s+damages|total\s+of|sum\s+of|amount\s+of)\b",
    re.IGNORECASE,
)

# QUICK WIN: Calculation-based extraction patterns
CLASS_MEMBER_CALCULATION = re.compile(
    r"(\d+(?:,\d{3})*)\s+class\s+members?.{0,200}?(?:eligible|entitled)\s+(?:for|to).{0,100}?\\?\$(\d+(?:,\d{3})*(?:\.\d{2})?)",
    re.IGNORECASE | re.DOTALL,
)

ONE_THIRD_CALCULATION = re.compile(
    r"(?:fees|amount)\s+of\s+\\?\$?([\d,]+(?:\.\d+)?)\s+(?:million|thousand)?.{0,100}?(?:one[\-\s]third|1/3|33\.?3?\%|thirty[\-\s]three\s+percent)",
    re.IGNORECASE | re.DOTALL,
)

# Enhanced fund addition - broader pattern for settlement components
FUND_ADDITION_PATTERN = re.compile(
    r"\\?\$?([\d,]+(?:\.\d+)?)\s*(?:million|thousand)?.{0,300}?(?:additional|fund\s+B|plus|and|along\s+with).{0,100}?\\?\$?([\d,]+(?:\.\d+)?)\s*(?:million|thousand)?",
    re.IGNORECASE | re.DOTALL,
)

DAMAGE_COMPONENTS = re.compile(
    r"(?:compensatory|punitive|exemplary|statutory|actual)\s+damages.{0,50}?\\?\$?([\d,]+(?:\.\d+)?)\s*(?:million|thousand)?",
    re.IGNORECASE | re.DOTALL,
)

# NEW: Attorney fees + expenses patterns
ATTORNEY_FEES_EXPENSES = re.compile(
    r"\\?\$?([\d,]+(?:\.\d+)?)\s*(?:million|thousand)?\s+in\s+attorney.{0,50}?fees.{0,100}?\\?\$?([\d,]+(?:\.\d+)?)\s*(?:million|thousand)?\s+in\s+expenses",
    re.IGNORECASE | re.DOTALL,
)

# NEW: Multi-component settlement sums (3+ amounts) - simplified for broader matching
MULTI_COMPONENT_SETTLEMENT = re.compile(
    r"\\?\$?([\d,]+(?:\.\d+)?)\s*(?:million|thousand)?.{0,400}?\\?\$?([\d,]+(?:\.\d+)?)\s*(?:million|thousand)?.{0,400}?\\?\$?([\d,]+(?:\.\d+)?)\s*(?:million|thousand)?",
    re.IGNORECASE | re.DOTALL,
)

# NEW: Settlement benefit calculation pattern
SETTLEMENT_BENEFIT_TOTAL = re.compile(
    r"total\s+benefit.{0,100}?\\?\$?([\d,]+(?:\.\d+)?)\s*(?:million|thousand)?",
    re.IGNORECASE | re.DOTALL,
)

# Enhanced fraction and percentage patterns
FRACTION_PATTERNS = re.compile(
    r"\b(?:one\s+and\s+(?:a\s+)?half|two\s+and\s+(?:a\s+)?half|three\s+and\s+(?:a\s+)?half|"
    r"one\s+and\s+(?:a\s+|one\s+)?quarter|two\s+and\s+(?:a\s+|one\s+)?quarter|"
    r"one\s+and\s+three\s+quarters|two\s+and\s+three\s+quarters|"
    r"half\s+a|quarter\s+of\s+a|three\s+quarters\s+of\s+a|"
    r"\d+\s+and\s+\d+/\d+|\d+\s*/\s*\d+|"
    r"\d+\s+and\s+(?:a\s+)?half|\d+\s+and\s+(?:a\s+|one\s+)?quarter)\s+"
    r"(?:million|billion|thousand|hundred)\b",
    re.IGNORECASE,
)

PERCENTAGE_PATTERNS = re.compile(
    r"\b(?:\d+(?:\.\d+)?\s*%|"
    r"\d+(?:\.\d+)?\s*percent|"
    r"\d+(?:\.\d+)?\s*per\s+cent)\s+"
    r"(?:of\s+(?:the\s+)?(?:\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|thousand))?|"
    r"\d+(?:\.\d+)?\s*(?:million|billion|thousand)\s*(?:dollars?)?|"
    r"settlement\s+fund|total\s+fund|common\s+fund|escrow\s+account))\b",
    re.IGNORECASE,
)

# Enhanced numeric gazetteer - comprehensive spelled-out numbers
NUMERIC_GAZETTEER = {
    # Basic numbers
    "zero": 0,
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
    "thousand": 1000,
    "million": 1000000,
    "billion": 1000000000,
    "trillion": 1000000000000,
    # Fractions
    "half": 0.5,
    "quarter": 0.25,
    "third": 0.333,
    "fourth": 0.25,
    "fifth": 0.2,
    "sixth": 0.167,
    "seventh": 0.143,
    "eighth": 0.125,
    "ninth": 0.111,
    "tenth": 0.1,
}

# 🚀 PERFORMANCE: Pre-compile regex patterns for numeric gazetteer
_NUMERIC_GAZETTEER_COMPILED_PATTERNS = {
    word: re.compile(r"\b" + re.escape(word) + r"\b")
    for word in NUMERIC_GAZETTEER.keys()
}

# Document structure patterns
TABLE_PATTERNS = re.compile(
    r"(?:table\s+\d+|exhibit\s+\d+|schedule\s+\d+|appendix\s+[a-z]|attachment\s+[a-z]|"
    r"\|\s*[^|]+\s*\||"  # pipe-separated tables
    r"(?:\s{2,}[^\s]+){3,}|"  # space-separated columns
    r"^\s*\d+\.\s+|"  # numbered lists
    r"^\s*[a-z]\)\s+|"  # lettered lists
    r"^\s*[ivx]+\.\s+)"  # roman numeral lists
    r"",
    re.IGNORECASE | re.MULTILINE,
)

HEADER_PATTERNS = re.compile(
    r"(?:^[A-Z\s]{10,}$|"  # ALL CAPS headers
    r"^[\d\s]*[A-Z][A-Z\s]+$|"  # Mostly caps with numbers
    r"^\s*(?:SECTION|PART|CHAPTER|ARTICLE)\s+[IVXLCDM\d]+|"  # Section headers
    r"^\s*[A-Z]\.\s+[A-Z][A-Z\s]+|"  # Lettered section headers
    r"^\s*\d+\.\s+[A-Z][A-Z\s]+)"  # Numbered section headers
    r"",
    re.MULTILINE,
)

SECTION_BOUNDARIES = re.compile(
    r"(?:WHEREAS|NOW THEREFORE|IT IS HEREBY|ORDERED AND ADJUDGED|"
    r"FOR THE FOREGOING REASONS|IN CONCLUSION|ACCORDINGLY|"
    r"BACKGROUND|PROCEDURAL HISTORY|FACTUAL BACKGROUND|DISCUSSION|ANALYSIS|"
    r"CONCLUSION|RELIEF|DAMAGES|SETTLEMENT TERMS|FINAL JUDGMENT|"
    r"MONETARY JUDGMENT|INJUNCTIVE RELIEF|ATTORNEY FEES|COSTS)",
    re.IGNORECASE,
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

# Dismissal patterns that indicate class action dismissals (enhanced) - split into high and low signal
# High signal dismissal patterns (definitive case endings)
HIGH_SIGNAL_DISMISSAL_PATTERNS = [
    re.compile(r"\bDefendants' motion to dismiss is granted\b", re.IGNORECASE),
    re.compile(r"\bcase is hereby dismissed\b", re.IGNORECASE),
    re.compile(r"\bcase dismissed\b", re.IGNORECASE),
    re.compile(r"\bwith prejudice\b", re.IGNORECASE),
    re.compile(r"\bcase terminated\b", re.IGNORECASE),
    re.compile(r"\bAccordingly, the Court decertifies the classes\b", re.IGNORECASE),
    re.compile(
        r"\bDefendants' motion to decertify the classes is GRANTED\b", re.IGNORECASE
    ),
    re.compile(r"\bclass.*decertified\b", re.IGNORECASE),
    re.compile(r"\bclass.*dismissed\b", re.IGNORECASE),
    re.compile(r"\bFINAL.*DISMISS\b", re.IGNORECASE),
    re.compile(r"\bJUDGMENT.*DISMISS\b", re.IGNORECASE),
]

# Low signal dismissal patterns (procedural/potentially reversible)
LOW_SIGNAL_DISMISSAL_PATTERNS = [
    re.compile(r"\bDISMISSED\b", re.IGNORECASE),
    re.compile(r"\bwithout prejudice\b", re.IGNORECASE),
    re.compile(r"\bterminated\b", re.IGNORECASE),
    re.compile(r"\bclass certification.*denied\b", re.IGNORECASE),
    re.compile(r"\bclass certification.*dismissed\b", re.IGNORECASE),
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
    re.compile(r"\bMotion.*DENIED\b", re.IGNORECASE),
    re.compile(r"\bMotion.*DISMISSED\b", re.IGNORECASE),
    re.compile(r"\bcase closed\b", re.IGNORECASE),
    re.compile(r"\baction dismissed\b", re.IGNORECASE),
    re.compile(r"\bsuit dismissed\b", re.IGNORECASE),
    re.compile(r"\blawsuit dismissed\b", re.IGNORECASE),
    re.compile(r"\bcomplaint dismissed\b", re.IGNORECASE),
    re.compile(r"\bpetition dismissed\b", re.IGNORECASE),
    re.compile(r"\bappeal dismissed\b", re.IGNORECASE),
    re.compile(r"\bsettlement.*dismissed\b", re.IGNORECASE),
    re.compile(r"\bcase.*settled.*dismissed\b", re.IGNORECASE),
    re.compile(r"\bvoluntary dismissal\b", re.IGNORECASE),
    re.compile(r"\bstipulated dismissal\b", re.IGNORECASE),
    re.compile(r"\bORDER.*DISMISS\b", re.IGNORECASE),
    re.compile(r"\bORDER.*DENY\b", re.IGNORECASE),
]

# Legacy pattern for backward compatibility
DISMISSAL_PATTERNS = [
    # Core dismissal phrases
    re.compile(r"\bDefendants' motion to dismiss is granted\b", re.IGNORECASE),
    re.compile(r"\bDISMISSED\b", re.IGNORECASE),
    re.compile(r"\bcase is hereby dismissed\b", re.IGNORECASE),
    re.compile(r"\bcase dismissed\b", re.IGNORECASE),
    re.compile(r"\bwithout prejudice\b", re.IGNORECASE),
    re.compile(r"\bwith prejudice\b", re.IGNORECASE),
    re.compile(r"\bterminated\b", re.IGNORECASE),
    re.compile(r"\bcase terminated\b", re.IGNORECASE),
    # Class action specific dismissals
    re.compile(r"\bAccordingly, the Court decertifies the classes\b", re.IGNORECASE),
    re.compile(
        r"\bDefendants' motion to decertify the classes is GRANTED\b", re.IGNORECASE
    ),
    re.compile(r"\bclass.*decertified\b", re.IGNORECASE),
    re.compile(r"\bclass.*dismissed\b", re.IGNORECASE),
    re.compile(r"\bclass certification.*denied\b", re.IGNORECASE),
    re.compile(r"\bclass certification.*dismissed\b", re.IGNORECASE),
    # Motion denials
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
    re.compile(r"\bMotion.*DENIED\b", re.IGNORECASE),
    re.compile(r"\bMotion.*DISMISSED\b", re.IGNORECASE),
    # Additional dismissal variants
    re.compile(r"\bcase closed\b", re.IGNORECASE),
    re.compile(r"\baction dismissed\b", re.IGNORECASE),
    re.compile(r"\bsuit dismissed\b", re.IGNORECASE),
    re.compile(r"\blawsuit dismissed\b", re.IGNORECASE),
    re.compile(r"\bcomplaint dismissed\b", re.IGNORECASE),
    re.compile(r"\bpetition dismissed\b", re.IGNORECASE),
    re.compile(r"\bappeal dismissed\b", re.IGNORECASE),
    # Settlement-related dismissals
    re.compile(r"\bsettlement.*dismissed\b", re.IGNORECASE),
    re.compile(r"\bcase.*settled.*dismissed\b", re.IGNORECASE),
    re.compile(r"\bvoluntary dismissal\b", re.IGNORECASE),
    re.compile(r"\bstipulated dismissal\b", re.IGNORECASE),
    # Court order dismissals
    re.compile(r"\bORDER.*DISMISS\b", re.IGNORECASE),
    re.compile(r"\bORDER.*DENY\b", re.IGNORECASE),
    re.compile(r"\bJUDGMENT.*DISMISS\b", re.IGNORECASE),
    re.compile(r"\bFINAL.*DISMISS\b", re.IGNORECASE),
]

# Contextual sections where dismissal language is most relevant
DISMISSAL_SECTIONS = [
    re.compile(r"^\s*ORDER\b", re.IGNORECASE),
    re.compile(r"^\s*CONCLUSION\b", re.IGNORECASE),
    re.compile(r"^\s*DECISION\b", re.IGNORECASE),
    re.compile(r"^\s*JUDGMENT\b", re.IGNORECASE),
    re.compile(r"^\s*FINAL\b", re.IGNORECASE),
    re.compile(r"^\s*IT IS ORDERED\b", re.IGNORECASE),
    re.compile(r"^\s*IT IS FURTHER ORDERED\b", re.IGNORECASE),
    re.compile(r"^\s*SO ORDERED\b", re.IGNORECASE),
    re.compile(r"^\s*DONE AND ORDERED\b", re.IGNORECASE),
]

# Document types that are more indicative of dismissal outcomes
DISMISSAL_DOCUMENT_TYPES = [
    re.compile(r"\bORDER\b", re.IGNORECASE),
    re.compile(r"\bJUDGMENT\b", re.IGNORECASE),
    re.compile(r"\bMEMORANDUM OPINION\b", re.IGNORECASE),
    re.compile(r"\bFINAL ORDER\b", re.IGNORECASE),
    re.compile(r"\bDISMISSAL ORDER\b", re.IGNORECASE),
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
                    data = fast_json_loads(line)
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
    OPTIMIZED: Early termination after finding patterns for speed.

    Args:
        text: Document text to search

    Returns:
        int: Number of dismissal patterns found
    """
    count = 0
    # 🚀 PERFORMANCE OPTIMIZATION - Early termination after reasonable count
    for pattern in DISMISSAL_PATTERNS:
        matches = pattern.findall(text)
        count += len(matches)
        # Early termination: if we found dismissal patterns, we have enough signal
        if count >= 3:  # Stop after finding sufficient dismissal evidence
            break
    return count


def count_contextual_dismissal_patterns(text: str) -> int:
    """
    Count dismissal patterns only within relevant sections (ORDER, CONCLUSION, etc.).

    Args:
        text: Document text to search

    Returns:
        int: Number of contextual dismissal patterns found
    """
    count = 0
    lines = text.split("\n")

    for i, line in enumerate(lines):
        # Check if this line starts a relevant section
        in_relevant_section = any(pattern.match(line) for pattern in DISMISSAL_SECTIONS)

        if in_relevant_section:
            # Look for dismissal patterns in this section (next few lines)
            section_text = "\n".join(lines[i : i + 10])  # Check next 10 lines
            for pattern in DISMISSAL_PATTERNS:
                matches = pattern.findall(section_text)
                count += len(matches)

    return count


# ------------------------------------------------------------------------------
# Enhanced Feature Extraction Functions
# ------------------------------------------------------------------------------


def count_financial_terms(text: str) -> int:
    """Count financial terminology matches in text."""
    return len(FINANCIAL_TERMS.findall(text))


def count_settlement_terms(text: str) -> int:
    """Count settlement-specific terminology matches in text."""
    return len(SETTLEMENT_TERMS.findall(text))


def count_legal_proceedings(text: str) -> int:
    """Count legal proceedings vocabulary matches in text."""
    return len(LEGAL_PROCEEDINGS.findall(text))


def count_monetary_phrases(text: str) -> int:
    """Count enhanced monetary phrase matches in text."""
    return len(MONETARY_PHRASES.findall(text))


def count_high_signal_financial_terms(text: str) -> int:
    """Count high signal financial terminology matches in text."""
    return len(HIGH_SIGNAL_FINANCIAL_TERMS.findall(text))


def count_low_signal_financial_terms(text: str) -> int:
    """Count low signal financial terminology matches in text."""
    return len(LOW_SIGNAL_FINANCIAL_TERMS.findall(text))


def count_high_signal_settlement_terms(text: str) -> int:
    """Count high signal settlement terminology matches in text."""
    return len(HIGH_SIGNAL_SETTLEMENT_TERMS.findall(text))


def count_low_signal_settlement_terms(text: str) -> int:
    """Count low signal settlement terminology matches in text."""
    return len(LOW_SIGNAL_SETTLEMENT_TERMS.findall(text))


def count_high_signal_legal_proceedings(text: str) -> int:
    """Count high signal legal proceedings matches in text."""
    return len(HIGH_SIGNAL_LEGAL_PROCEEDINGS.findall(text))


def count_low_signal_legal_proceedings(text: str) -> int:
    """Count low signal legal proceedings matches in text."""
    return len(LOW_SIGNAL_LEGAL_PROCEEDINGS.findall(text))


def count_high_signal_monetary_phrases(text: str) -> int:
    """Count high signal monetary phrase matches in text."""
    return len(HIGH_SIGNAL_MONETARY_PHRASES.findall(text))


def count_low_signal_monetary_phrases(text: str) -> int:
    """Count low signal monetary phrase matches in text."""
    return len(LOW_SIGNAL_MONETARY_PHRASES.findall(text))


def count_high_signal_dismissal_patterns(text: str) -> int:
    """Count high signal dismissal pattern matches in text."""
    return sum(len(pattern.findall(text)) for pattern in HIGH_SIGNAL_DISMISSAL_PATTERNS)


def count_low_signal_dismissal_patterns(text: str) -> int:
    """Count low signal dismissal pattern matches in text."""
    return sum(len(pattern.findall(text)) for pattern in LOW_SIGNAL_DISMISSAL_PATTERNS)


def count_high_confidence_patterns(text: str) -> int:
    """Count high-confidence monetary context patterns that strongly suggest a primary award amount."""
    return len(HIGH_CONFIDENCE_PATTERNS.findall(text))


def count_amount_adjacent_keywords(text: str) -> int:
    """Count amount-adjacent keywords that suggest the amount is the main award."""
    return len(AMOUNT_ADJACENT_KEYWORDS.findall(text))


def compute_confidence_boost_score(text: str) -> float:
    """
    Compute a confidence boost score based on high-confidence patterns.
    This helps elevate amounts that are clearly the primary award/judgment.
    """
    score = 0.0

    # High-confidence patterns get a significant boost
    high_conf_count = count_high_confidence_patterns(text)
    score += high_conf_count * 5.0  # 5x boost for each high-confidence pattern

    # Amount-adjacent keywords get a moderate boost
    adjacent_count = count_amount_adjacent_keywords(text)
    score += adjacent_count * 2.0  # 2x boost for each adjacent keyword

    # Extra boost for multiple indicators
    if high_conf_count > 0 and adjacent_count > 0:
        score += 3.0  # Additional boost when both types are present

    return score


def extract_calculated_amounts(
    text: str, min_amount: float = 10000
) -> List[Dict[str, Any]]:
    """
    QUICK WIN: Extract calculated amounts from common legal document patterns.
    Targets specific calculation scenarios found in the gold standard.
    """
    calculated_amounts = []

    # 1. Class member calculations: "588,887 class members eligible for $175"
    for match in CLASS_MEMBER_CALCULATION.finditer(text):
        try:
            members = int(match.group(1).replace(",", ""))
            per_member = float(match.group(2).replace(",", ""))
            total = members * per_member

            if total >= min_amount:
                calculated_amounts.append(
                    {
                        "value": total,
                        "raw_text": f"{members:,} class members × ${per_member:,.2f}",
                        "context": text[
                            max(0, match.start() - 100) : match.end() + 100
                        ],
                        "calculation_type": "class_member_multiplication",
                        "start": match.start(),
                        "end": match.end(),
                        "calculation_boost": 300.0,  # Highest boost for class member calculations
                    }
                )
        except (ValueError, AttributeError):
            continue

    # 2. One-third calculations: "fees of $2,500,000 represent one-third"
    for match in ONE_THIRD_CALCULATION.finditer(text):
        try:
            amount_str = match.group(1).replace(",", "")
            amount = float(amount_str)

            # Apply multipliers
            if "million" in match.group(0).lower():
                amount *= 1_000_000
            elif "thousand" in match.group(0).lower():
                amount *= 1_000

            total = amount * 3  # If this is 1/3, total is 3x

            if total >= min_amount:
                calculated_amounts.append(
                    {
                        "value": total,
                        "raw_text": f"${amount:,.0f} × 3 (one-third calculation)",
                        "context": text[
                            max(0, match.start() - 100) : match.end() + 100
                        ],
                        "calculation_type": "one_third_multiplication",
                        "start": match.start(),
                        "end": match.end(),
                        "calculation_boost": 200.0,  # Major boost for calculations
                    }
                )
        except (ValueError, AttributeError):
            continue

    # 3. Multi-fund additions: "Fund A $23,750,000... Fund B $2,000,000"
    for match in FUND_ADDITION_PATTERN.finditer(text):
        try:
            amount1_str = match.group(1).replace(",", "")
            amount2_str = match.group(2).replace(",", "")
            amount1 = float(amount1_str)
            amount2 = float(amount2_str)

            # Apply multipliers
            match_text = match.group(0).lower()
            if "million" in match_text:
                amount1 *= 1_000_000
                amount2 *= 1_000_000
            elif "thousand" in match_text:
                amount1 *= 1_000
                amount2 *= 1_000

            total = amount1 + amount2

            if total >= min_amount:
                calculated_amounts.append(
                    {
                        "value": total,
                        "raw_text": f"${amount1:,.0f} + ${amount2:,.0f}",
                        "context": text[
                            max(0, match.start() - 100) : match.end() + 100
                        ],
                        "calculation_type": "multi_fund_addition",
                        "start": match.start(),
                        "end": match.end(),
                        "calculation_boost": 200.0,  # Major boost for calculations
                    }
                )
        except (ValueError, AttributeError):
            continue

    return calculated_amounts


def extract_smart_sum_amounts(
    text: str, min_amount: float = 10000
) -> List[Dict[str, Any]]:
    """
    Smart sum detection for complex multi-component calculations.
    Handles cases like: $14,532,418.31 + $731,986.71 = $15,264,405.02
    And: $5.73 million + $3 million + $500,000 + $2.2 million = $8.46 million
    """
    calculated_amounts = []

    # Find all monetary amounts in the text with their positions
    money_pattern = re.compile(
        r"\\?\$?([\d,]+(?:\.\d+)?)\s*(?:million|thousand|billion)?", re.IGNORECASE
    )
    amounts_found = []

    for match in money_pattern.finditer(text):
        amount_str = match.group(1).replace(",", "")
        try:
            amount = float(amount_str)

            # Apply multipliers based on context
            full_match = match.group(0).lower()
            if "billion" in full_match:
                amount *= 1_000_000_000
            elif "million" in full_match:
                amount *= 1_000_000
            elif "thousand" in full_match:
                amount *= 1_000

            amounts_found.append(
                {
                    "value": amount,
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(0),
                }
            )
        except ValueError:
            continue

    # Look for patterns that suggest these amounts should be summed
    if len(amounts_found) >= 2:
        # Check for additive language within reasonable distance
        additive_patterns = [
            r"\band\b",
            r"\bplus\b",
            r"\balong\s+with\b",
            r"\bin\s+addition\s+to\b",
            r"\btotal\b",
            r"\binclude[ds]?\b",
            r"\bcombined\b",
            r"\btogether\b",
        ]

        # For each potential combination of 2-4 amounts, check if they're connected by additive language
        for i in range(len(amounts_found)):
            for j in range(
                i + 1, min(i + 4, len(amounts_found))
            ):  # Check up to 4 amounts
                start_pos = amounts_found[i]["start"]
                end_pos = amounts_found[j]["end"]
                context = text[start_pos:end_pos]

                # Check if there's additive language between amounts
                has_additive_language = any(
                    re.search(pattern, context, re.IGNORECASE)
                    for pattern in additive_patterns
                )

                if has_additive_language:
                    # Calculate sum
                    sum_amounts = amounts_found[i : j + 1]

                    # Additional quality checks - only create sums for meaningful amounts
                    amounts_in_range = [
                        amt for amt in sum_amounts if amt["value"] >= 10000
                    ]  # Filter very small amounts
                    if len(amounts_in_range) < 2:
                        continue

                    total_value = sum(amt["value"] for amt in sum_amounts)

                    if total_value >= min_amount:
                        components_text = " + ".join(
                            f"${amt['value']:,.0f}" for amt in sum_amounts
                        )

                        calculated_amounts.append(
                            {
                                "value": total_value,
                                "raw_text": f"Smart sum: {components_text}",
                                "context": text[
                                    max(0, start_pos - 100) : end_pos + 100
                                ],
                                "calculation_type": "smart_sum_detection",
                                "start": start_pos,
                                "end": end_pos,
                                "calculation_boost": 180.0,
                            }
                        )
                        break  # Only create one sum per starting amount

    return calculated_amounts


def extract_attorney_fees_expenses(
    text: str, min_amount: float = 10000
) -> List[Dict[str, Any]]:
    """
    Extract attorney fees + expenses calculations.
    Example: "$14,532,418.31 in attorneys' fees and $731,986.71 in expenses"
    """
    calculated_amounts = []

    for match in ATTORNEY_FEES_EXPENSES.finditer(text):
        try:
            fees_str = match.group(1).replace(",", "")
            expenses_str = match.group(2).replace(",", "")
            fees = float(fees_str)
            expenses = float(expenses_str)

            # Apply multipliers
            match_text = match.group(0).lower()
            if "million" in match_text:
                fees *= 1_000_000
                expenses *= 1_000_000
            elif "thousand" in match_text:
                fees *= 1_000
                expenses *= 1_000

            total = fees + expenses

            if total >= min_amount:
                calculated_amounts.append(
                    {
                        "value": total,
                        "raw_text": f"${fees:,.0f} (fees) + ${expenses:,.0f} (expenses)",
                        "context": text[
                            max(0, match.start() - 100) : match.end() + 100
                        ],
                        "calculation_type": "attorney_fees_expenses",
                        "start": match.start(),
                        "end": match.end(),
                        "calculation_boost": 50.0,
                    }
                )
        except (ValueError, AttributeError):
            continue

    return calculated_amounts


def extract_multi_component_settlements(
    text: str, min_amount: float = 10000
) -> List[Dict[str, Any]]:
    """
    Extract multi-component settlement calculations (3+ amounts).
    Example: "$5.73 million... $3 million... $500,000... $2.2 million"
    """
    calculated_amounts = []

    for match in MULTI_COMPONENT_SETTLEMENT.finditer(text):
        try:
            amount1_str = match.group(1).replace(",", "")
            amount2_str = match.group(2).replace(",", "")
            amount3_str = match.group(3).replace(",", "")
            amount1 = float(amount1_str)
            amount2 = float(amount2_str)
            amount3 = float(amount3_str)

            # Apply multipliers
            match_text = match.group(0).lower()
            if "million" in match_text:
                amount1 *= 1_000_000
                amount2 *= 1_000_000
                amount3 *= 1_000_000
            elif "thousand" in match_text:
                amount1 *= 1_000
                amount2 *= 1_000
                amount3 *= 1_000

            total = amount1 + amount2 + amount3

            if total >= min_amount:
                calculated_amounts.append(
                    {
                        "value": total,
                        "raw_text": f"${amount1:,.0f} + ${amount2:,.0f} + ${amount3:,.0f}",
                        "context": text[
                            max(0, match.start() - 100) : match.end() + 100
                        ],
                        "calculation_type": "multi_component_settlement",
                        "start": match.start(),
                        "end": match.end(),
                        "calculation_boost": 50.0,
                    }
                )
        except (ValueError, AttributeError):
            continue

    return calculated_amounts


def extract_settlement_benefit_totals(
    text: str, min_amount: float = 10000
) -> List[Dict[str, Any]]:
    """
    Extract settlement benefit total calculations.
    Example: "total benefit to the class for fee purposes is thus $42.5 million"
    """
    calculated_amounts = []

    for match in SETTLEMENT_BENEFIT_TOTAL.finditer(text):
        try:
            amount_str = match.group(1).replace(",", "")
            amount = float(amount_str)

            # Apply multipliers
            match_text = match.group(0).lower()
            if "million" in match_text:
                amount *= 1_000_000
            elif "thousand" in match_text:
                amount *= 1_000

            if amount >= min_amount:
                calculated_amounts.append(
                    {
                        "value": amount,
                        "raw_text": f"Total benefit: ${amount:,.0f}",
                        "context": text[
                            max(0, match.start() - 100) : match.end() + 100
                        ],
                        "calculation_type": "settlement_benefit_total",
                        "start": match.start(),
                        "end": match.end(),
                        "calculation_boost": 150.0,  # High boost for benefit totals
                    }
                )
        except (ValueError, AttributeError):
            continue

    return calculated_amounts


def extract_damage_component_totals(
    text: str, min_amount: float = 10000
) -> List[Dict[str, Any]]:
    """
    QUICK WIN: Extract total damage amounts by finding and summing damage components.
    """
    damage_totals = []

    # Find all damage components in the text
    components = []
    for match in DAMAGE_COMPONENTS.finditer(text):
        try:
            amount_str = match.group(1).replace(",", "")
            amount = float(amount_str)

            # Apply multipliers
            if "million" in match.group(0).lower():
                amount *= 1_000_000
            elif "thousand" in match.group(0).lower():
                amount *= 1_000

            components.append(
                {
                    "amount": amount,
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
        except (ValueError, AttributeError):
            continue

    # If we have multiple damage components, calculate total
    if len(components) >= 2:
        total = sum(comp["amount"] for comp in components)

        if total >= min_amount:
            component_texts = [f"${comp['amount']:,.0f}" for comp in components]
            damage_totals.append(
                {
                    "value": total,
                    "raw_text": " + ".join(component_texts),
                    "context": text[
                        max(0, components[0]["start"] - 100) : components[-1]["end"]
                        + 100
                    ],
                    "calculation_type": "damage_component_sum",
                    "start": components[0]["start"],
                    "end": components[-1]["end"],
                    "calculation_boost": 50.0,  # Major boost for calculations
                }
            )

    return damage_totals


def extract_dependency_features(text: str, nlp: Optional[Any] = None) -> int:
    """
    Extract dependency parsing features for money-headed noun phrases.
    Uses spaCy's dependency parser to find monetary contexts.
    """
    if not nlp or not spacy_available:
        return 0

    try:
        doc = nlp(text)
        money_features = 0

        for token in doc:
            # Look for monetary entities and their dependencies
            if token.ent_type_ == "MONEY":
                money_features += 1
                # Check for syntactic dependencies that indicate legal monetary contexts
                for child in token.children:
                    if child.dep_ in ["nmod", "compound", "amod"] and any(
                        pattern in child.text.lower()
                        for pattern in [
                            "settle",
                            "award",
                            "damage",
                            "judgment",
                            "penalty",
                            "fine",
                        ]
                    ):
                        money_features += 1

            # Look for monetary terms with specific syntactic patterns
            if token.pos_ == "NUM" and token.head.text.lower() in [
                "million",
                "billion",
                "thousand",
            ]:
                # Check if this is in a monetary context
                if any(
                    pattern in token.sent.text.lower()
                    for pattern in [
                        "settlement",
                        "award",
                        "damage",
                        "judgment",
                        "penalty",
                    ]
                ):
                    money_features += 1

        return money_features
    except Exception:
        return 0


def extract_enhanced_fractions(text: str, weights=None) -> List[Dict[str, Any]]:
    """
    Extract enhanced fraction patterns including mixed numbers.
    Returns list of fraction matches with their computed values.
    """
    # 🚀 CRITICAL DROPOUT FIX - Skip expensive regex if disabled
    if weights and weights.fraction_extraction_weight == 0.0:
        return []

    fractions = []

    # Find fraction patterns
    for match in FRACTION_PATTERNS.finditer(text):
        fraction_text = match.group()
        try:
            # Parse mixed numbers and fractions
            value = parse_mixed_number_or_fraction(fraction_text)
            if value > 0:
                fractions.append(
                    {
                        "text": fraction_text,
                        "value": value,
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
        except Exception:
            continue

    return fractions


def extract_percentages_with_totals(text: str) -> List[Dict[str, Any]]:
    """
    Extract percentage patterns and compute implied totals when possible.
    """
    percentages = []

    for match in PERCENTAGE_PATTERNS.finditer(text):
        percentage_text = match.group()
        try:
            # Extract percentage value and base amount
            percent_value, base_amount = parse_percentage_with_base(percentage_text)
            if percent_value > 0 and base_amount > 0:
                implied_total = base_amount / (percent_value / 100.0)
                percentages.append(
                    {
                        "text": percentage_text,
                        "percentage": percent_value,
                        "base_amount": base_amount,
                        "implied_total": implied_total,
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
        except Exception:
            continue

    return percentages


def count_document_structure_features(text: str) -> int:
    """Count document structure features like tables, headers, sections."""
    structure_count = 0

    # Count table-like structures
    structure_count += len(TABLE_PATTERNS.findall(text))

    # Count header patterns
    structure_count += len(HEADER_PATTERNS.findall(text))

    # Count section boundaries
    structure_count += len(SECTION_BOUNDARIES.findall(text))

    return structure_count


def count_numeric_gazetteer_matches(text: str) -> int:
    """Count matches from the comprehensive numeric gazetteer.

    🚀 PERFORMANCE OPTIMIZED: Uses pre-compiled regex patterns for speed.
    """
    count = 0
    text_lower = text.lower()

    # Use pre-compiled patterns for much faster matching
    for word, compiled_pattern in _NUMERIC_GAZETTEER_COMPILED_PATTERNS.items():
        matches = compiled_pattern.findall(text_lower)
        count += len(matches)

    return count


def extract_sentence_boundary_context(
    text: str, match_start: int, match_end: int
) -> str:
    """
    Extract context that respects sentence boundaries.
    Returns full sentences containing the match rather than character-based windows.
    """
    try:
        # Simple sentence boundary detection
        sentences = re.split(r"[.!?]+\s+", text)
        match_text = text[match_start:match_end]

        # Find sentences containing the match
        context_sentences = []
        for sentence in sentences:
            if match_text in sentence:
                context_sentences.append(sentence.strip())

        # Also include adjacent sentences for broader context
        for i, sentence in enumerate(sentences):
            if match_text in sentence:
                # Add previous sentence
                if i > 0:
                    context_sentences.insert(-1, sentences[i - 1].strip())
                # Add next sentence
                if i < len(sentences) - 1:
                    context_sentences.append(sentences[i + 1].strip())
                break

        return " ".join(context_sentences)
    except Exception:
        # Fallback to character-based context
        return text[max(0, match_start - 200) : match_end + 200]


def extract_paragraph_boundary_context(
    text: str, match_start: int, match_end: int
) -> str:
    """
    Extract context that respects paragraph boundaries.
    Returns full paragraphs containing the match.
    """
    try:
        # Simple paragraph boundary detection
        paragraphs = re.split(r"\n\s*\n", text)
        match_text = text[match_start:match_end]

        # Find paragraph containing the match
        for paragraph in paragraphs:
            if match_text in paragraph:
                return paragraph.strip()

        # Fallback to sentence boundary context
        return extract_sentence_boundary_context(text, match_start, match_end)
    except Exception:
        # Fallback to character-based context
        return text[max(0, match_start - 300) : match_end + 300]


def parse_mixed_number_or_fraction(fraction_text: str) -> float:
    """
    Parse mixed numbers and fractions to float values.
    Examples: "one and a half million" -> 1.5
              "2 and 3/4 billion" -> 2.75
    """
    text_lower = fraction_text.lower()

    # Handle mixed numbers with spelled-out parts
    if "and" in text_lower:
        parts = text_lower.split("and")
        if len(parts) == 2:
            whole_part = parse_spelled_number(parts[0].strip())
            fraction_part = parse_spelled_fraction(parts[1].strip())
            return whole_part + fraction_part

    # Handle simple fractions
    if "/" in fraction_text:
        try:
            fraction = Fraction(fraction_text.strip())
            return float(fraction)
        except:
            pass

    # Handle spelled-out fractions
    return parse_spelled_fraction(text_lower)


def parse_spelled_number(text: str) -> float:
    """Parse spelled-out numbers to float values."""
    text_lower = text.lower().strip()

    if text_lower in NUMERIC_GAZETTEER:
        return float(NUMERIC_GAZETTEER[text_lower])

    # Handle compound numbers like "twenty-five"
    if "-" in text_lower:
        parts = text_lower.split("-")
        if len(parts) == 2 and all(part in NUMERIC_GAZETTEER for part in parts):
            return float(NUMERIC_GAZETTEER[parts[0]] + NUMERIC_GAZETTEER[parts[1]])

    return 0.0


def parse_spelled_fraction(text: str) -> float:
    """Parse spelled-out fractions to float values."""
    text_lower = text.lower().strip()

    # Handle common fraction phrases
    if "half" in text_lower:
        return 0.5
    elif "quarter" in text_lower:
        return 0.25
    elif "third" in text_lower:
        return 0.333
    elif "three quarters" in text_lower or "three-quarters" in text_lower:
        return 0.75

    # Handle numeric fractions with spelled multipliers
    match = re.search(r"(\d+)\s*/\s*(\d+)", text_lower)
    if match:
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        if denominator != 0:
            return numerator / denominator

    return 0.0


def parse_percentage_with_base(percentage_text: str) -> tuple[float, float]:
    """
    Parse percentage text to extract percentage value and base amount.
    Returns (percentage_value, base_amount).
    """
    # Extract percentage value
    percent_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:%|percent|per\s+cent)", percentage_text, re.IGNORECASE
    )
    if not percent_match:
        return 0.0, 0.0

    percent_value = float(percent_match.group(1))

    # Extract base amount
    amount_match = re.search(
        r"\$?([\d,]+(?:\.\d+)?)\s*(?:million|billion|thousand)?",
        percentage_text,
        re.IGNORECASE,
    )
    if amount_match:
        amount_str = amount_match.group(1).replace(",", "")
        base_amount = float(amount_str)

        # Apply multipliers
        if "million" in percentage_text.lower():
            base_amount *= 1000000
        elif "billion" in percentage_text.lower():
            base_amount *= 1000000000
        elif "thousand" in percentage_text.lower():
            base_amount *= 1000

        return percent_value, base_amount

    return percent_value, 0.0


def get_dismissal_score(text: str, document_type_weight: float = 1.0) -> float:
    """
    Calculate a weighted dismissal score based on multiple factors.

    Args:
        text: Document text to analyze
        document_type_weight: Weight multiplier for document type matches

    Returns:
        float: Weighted dismissal score (higher = more likely dismissed)
    """
    score = 0.0

    # Base pattern count
    base_patterns = count_dismissal_patterns(text)
    score += base_patterns * 1.0

    # Contextual pattern count (weighted higher)
    contextual_patterns = count_contextual_dismissal_patterns(text)
    score += contextual_patterns * 2.0

    # Document type weighting
    doc_type_matches = 0
    for pattern in DISMISSAL_DOCUMENT_TYPES:
        if pattern.search(text):
            doc_type_matches += 1

    score += doc_type_matches * document_type_weight

    return score


def is_case_dismissed(
    case_root: Path,
    dismissal_ratio_threshold: float = 0.3,
    use_weighted_scoring: bool = True,
    document_type_weight: float = 2.0,
) -> bool:
    """
    Determine if a case is dismissed by examining stage1 files for dismissal language.
    Enhanced with contextual filtering and weighted scoring.

    Args:
        case_root: Path to the case directory
        dismissal_ratio_threshold: Minimum ratio of documents with dismissal language (default: 0.05)
        use_weighted_scoring: Whether to use weighted scoring instead of simple pattern counting
        document_type_weight: Weight for document type matches when using weighted scoring

    Returns:
        bool: True if case appears to be dismissed, False otherwise
    """
    dismissal_documents = 0
    total_documents = 0
    total_dismissal_score = 0.0

    # Scan all stage1 files in the case
    for path in case_root.rglob("*_stage1.jsonl"):
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                try:
                    data = fast_json_loads(line)
                    text = data.get("text", "")
                    if not text:
                        continue

                    total_documents += 1

                    if use_weighted_scoring:
                        # Use weighted scoring
                        doc_score = get_dismissal_score(text, document_type_weight)
                        total_dismissal_score += doc_score
                        if doc_score > 0:
                            dismissal_documents += 1
                    else:
                        # Use simple pattern counting
                        dismissal_matches = count_dismissal_patterns(text)
                        if dismissal_matches > 0:
                            dismissal_documents += 1

                except json.JSONDecodeError:
                    continue

    # Calculate ratio and compare to threshold
    if total_documents == 0:
        return False

    if use_weighted_scoring:
        # Use average dismissal score instead of simple ratio
        avg_dismissal_score = total_dismissal_score / total_documents
        return avg_dismissal_score >= dismissal_ratio_threshold
    else:
        # Use traditional ratio-based approach
        dismissal_ratio = dismissal_documents / total_documents
        return dismissal_ratio >= dismissal_ratio_threshold


def is_case_definitively_dismissed(
    case_root: Path,
    strict_dismissal_threshold: float = 0.8,
    document_type_weight: float = 3.0,
) -> bool:
    """
    More nuanced dismissal detection - only flags cases that are CLEARLY dismissed.
    Uses higher thresholds and stronger patterns to avoid false positives.
    """
    if not case_root.exists():
        return False

    # Stronger dismissal patterns that indicate definitive dismissal
    DEFINITIVE_DISMISSAL_PATTERNS = [
        r"\b(?:case|action|matter|lawsuit|complaint)\s+(?:is\s+)?(?:hereby\s+)?dismissed\s+(?:with\s+prejudice|in\s+its\s+entirety)",
        r"\b(?:motion|petition)\s+(?:to\s+)?dismiss\s+(?:is\s+)?(?:hereby\s+)?granted",
        r"\bjudgment\s+(?:is\s+)?(?:hereby\s+)?entered\s+(?:in\s+favor\s+of\s+)?defendants?(?:\s+and\s+against\s+plaintiffs?)?",
        r"\b(?:final\s+)?judgment\s+(?:of\s+)?dismissal",
        r"\bcase\s+(?:is\s+)?(?:hereby\s+)?closed",
        r"\b(?:all\s+)?claims\s+(?:are\s+)?(?:hereby\s+)?dismissed",
        r"\bdefendants?\s+motion\s+for\s+summary\s+judgment\s+(?:is\s+)?granted",
        r"\bplaintiffs?\s+motion\s+for\s+class\s+certification\s+(?:is\s+)?denied.*with\s+prejudice",
    ]

    definitive_pattern = re.compile(
        "|".join(DEFINITIVE_DISMISSAL_PATTERNS), re.IGNORECASE
    )

    total_documents = 0
    definitive_dismissal_documents = 0
    weighted_dismissal_score = 0.0
    total_weighted_documents = 0.0

    for jsonl_path in case_root.rglob("*_stage1.jsonl"):
        total_documents += 1

        # Document type weighting
        doc_weight = 1.0
        filename = jsonl_path.name.lower()
        if any(
            keyword in filename
            for keyword in ["order", "judgment", "ruling", "decision"]
        ):
            doc_weight = document_type_weight

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = fast_json_loads(line)
                    text = data.get("text", "")

                    if definitive_pattern.search(text):
                        definitive_dismissal_documents += 1
                        weighted_dismissal_score += doc_weight
                        break  # Only count once per document

                    total_weighted_documents += doc_weight
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

    if total_documents == 0:
        return False

    # Calculate weighted ratio
    weighted_ratio = (
        weighted_dismissal_score / total_weighted_documents
        if total_weighted_documents > 0
        else 0
    )

    # More strict threshold - only flag cases with strong dismissal evidence
    return weighted_ratio >= strict_dismissal_threshold


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
                    data = fast_json_loads(line)
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
    case_root: Path, patent_ratio_threshold: float = 50.0
) -> bool:
    """
    Determine if a case contains many patent references above threshold ratio.

    Args:
        case_root: Path to the case directory
        patent_ratio_threshold: Minimum ratio of patent occurrences per document to flag (default: 50.0)

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
                    data = fast_json_loads(line)
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
    patent_ratio_threshold: float = 50.0,
    dismissal_ratio_threshold: float = 0.05,
    bankruptcy_ratio_threshold: float = 0.5,
    use_weighted_dismissal_scoring: bool = True,
    dismissal_document_type_weight: float = 2.0,
    fast_mode: bool = False,
) -> dict[str, bool]:
    """
    Get all flags for a case (dismissal, fee-shifting, large patent amounts, bankruptcy court).
    Enhanced with configurable dismissal detection.

    Args:
        case_root: Path to the case directory
        fee_shifting_ratio_threshold: Minimum ratio of fee-shifting occurrences per document to flag (default: 1.0)
        patent_ratio_threshold: Minimum ratio of patent occurrences per document to flag (default: 50.0)
        dismissal_ratio_threshold: Minimum ratio of documents with dismissal language (default: 0.05)
        bankruptcy_ratio_threshold: Minimum ratio of bankruptcy court documents to flag (default: 0.5)
        use_weighted_dismissal_scoring: Whether to use weighted scoring for dismissal detection (default: True)
        dismissal_document_type_weight: Weight for document type matches in dismissal scoring (default: 2.0)

    Returns:
        dict[str, bool]: Dictionary with flag information
    """
    # 🚀 FAST MODE - Skip expensive flag calculations during optimization
    if fast_mode:
        # Return minimal flags for optimization speed
        flags = {
            "is_dismissed": False,  # Skip expensive dismissal detection
            "has_fee_shifting": False,  # Skip fee shifting analysis
            "has_large_patent_amounts": False,  # Skip patent detection
            "is_bankruptcy_court": False,  # Skip court type analysis
        }
    else:
        # Full flag calculation for production use
        flags = {
            "is_dismissed": is_case_dismissed(
                case_root,
                dismissal_ratio_threshold,
                use_weighted_dismissal_scoring,
                dismissal_document_type_weight,
            ),
            "has_fee_shifting": has_fee_shifting(
                case_root, fee_shifting_ratio_threshold
            ),
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
            # Settlement Fund patterns
            {
                "label": "SETTLEMENT_CONTEXT",
                "pattern": [{"LOWER": "settlement"}, {"LOWER": "fund"}],
            },
            {
                "label": "SETTLEMENT_CONTEXT",
                "pattern": [{"LOWER": "released"}, {"LOWER": "claims"}],
            },
            {
                "label": "SETTLEMENT_CONTEXT",
                "pattern": [{"LOWER": "dismissal"}, {"LOWER": "actions"}],
            },
            {
                "label": "SETTLEMENT_CONTEXT",
                "pattern": [{"LOWER": "preliminary"}, {"LOWER": "approval"}],
            },
            {
                "label": "SETTLEMENT_CONTEXT",
                "pattern": [{"LOWER": "escrow"}, {"LOWER": "account"}],
            },
            # Settlement Fund of $X patterns
            {
                "label": "SETTLEMENT_AMOUNT",
                "pattern": [
                    {"LOWER": "settlement"},
                    {"LOWER": "fund"},
                    {"LOWER": "of"},
                    {"IS_CURRENCY": True},
                    {"LIKE_NUM": True},
                ],
            },
            {
                "label": "SETTLEMENT_AMOUNT",
                "pattern": [
                    {"LOWER": "settlement"},
                    {"LOWER": "fund"},
                    {"LOWER": "of"},
                    {"IS_CURRENCY": True},
                    {"LIKE_NUM": True},
                    {"TEXT": ","},
                    {"LIKE_NUM": True},
                ],
            },
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

            # Handle SETTLEMENT_AMOUNT entities (new)
            elif ent.label_ == "SETTLEMENT_AMOUNT":
                if not fast_mode:
                    print(f"🏛️ spaCy SETTLEMENT_AMOUNT entity: '{ent.text}'")

                # Extract the monetary value from the settlement amount
                # Pattern: "Settlement Fund of $9,750,000"
                monetary_match = re.search(r"\$([0-9,]+(?:\.[0-9]{2})?)", ent.text)
                if monetary_match:
                    try:
                        val = float(monetary_match.group(1).replace(",", ""))
                        if val >= min_amount:
                            start, end = ent.start_char, ent.end_char
                            ctx = text[
                                max(0, start - context_chars) : end + context_chars
                            ].replace("\n", " ")
                            candidates.append(
                                {
                                    "amount": monetary_match.group(
                                        0
                                    ),  # Just the $ amount
                                    "value": val,
                                    "context": ctx,
                                    "type": "spacy_settlement_amount",
                                }
                            )
                    except ValueError:
                        continue

            # Handle SETTLEMENT_CONTEXT entities (new) - these provide voting context
            elif ent.label_ == "SETTLEMENT_CONTEXT":
                if not fast_mode:
                    print(f"📋 spaCy SETTLEMENT_CONTEXT entity: '{ent.text}'")
                # These entities don't contain amounts themselves but indicate settlement context
                # They will be picked up by the proximity pattern voting system
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
    3. FINANCIAL_TERMS - comprehensive financial terminology
    4. SETTLEMENT_TERMS - settlement-specific vocabulary
    5. LEGAL_PROCEEDINGS - legal proceedings terminology
    6. MONETARY_PHRASES - enhanced monetary phrase patterns
    7. DOCUMENT_STRUCTURE - tables, headers, sections
    8. NUMERIC_GAZETTEER - spelled-out numbers and fractions

    Returns:
        int: Total number of pattern matches found
    """
    feature_count = 0.0

    # 🚨 ULTRA-FAST MODE - Bypass ALL expensive operations if most weights are 0
    expensive_weights = [
        weights.numeric_gazetteer_weight,
        weights.dependency_parsing_weight,
        weights.fraction_extraction_weight,
        weights.document_structure_weight,
        weights.table_detection_weight,
        weights.sentence_boundary_weight,
        weights.paragraph_boundary_weight,
    ]

    if all(w == 0.0 for w in expensive_weights):
        # 🚀 EMERGENCY SPEED MODE: Only count basic patterns
        feature_count = 0

        # Count basic monetary amounts (fastest possible)
        basic_amount_matches = len(re.findall(r"\$[\d,]+", context))
        if basic_amount_matches > 0:
            feature_count += (
                basic_amount_matches * 10.0
            )  # High weight for actual amounts

        # Quick settlement terms
        if "settlement" in context.lower():
            feature_count += 5.0
        if "damages" in context.lower():
            feature_count += 3.0

        return int(feature_count)

    # 🚀 CRITICAL FIX - Original features with DROPOUT protection
    if weights.proximity_pattern_weight > 0.0:
        proximity_matches = PROXIMITY_PATTERN.findall(context)
        feature_count += len(proximity_matches) * weights.proximity_pattern_weight

    if weights.judgment_verbs_weight > 0.0:
        judgment_matches = JUDGMENT_VERBS.findall(context)
        feature_count += len(judgment_matches) * weights.judgment_verbs_weight

    # 🚀 ULTRA-FAST DROPOUT OPTIMIZATION - Skip expensive regex when weight = 0
    # New enhanced features with high/low signal separation
    if weights.financial_terms_weight > 0.0:
        financial_count = count_financial_terms(context)
        feature_count += financial_count * weights.financial_terms_weight

    # High/Low signal financial terms
    if weights.high_signal_financial_weight > 0.0:
        high_signal_financial_count = count_high_signal_financial_terms(context)
        feature_count += (
            high_signal_financial_count * weights.high_signal_financial_weight
        )

    if weights.low_signal_financial_weight > 0.0:
        low_signal_financial_count = count_low_signal_financial_terms(context)
        feature_count += (
            low_signal_financial_count * weights.low_signal_financial_weight
        )

    if weights.settlement_terms_weight > 0.0:
        settlement_count = count_settlement_terms(context)
        feature_count += settlement_count * weights.settlement_terms_weight

    # High/Low signal settlement terms
    if weights.high_signal_settlement_weight > 0.0:
        high_signal_settlement_count = count_high_signal_settlement_terms(context)
        feature_count += (
            high_signal_settlement_count * weights.high_signal_settlement_weight
        )

    if weights.low_signal_settlement_weight > 0.0:
        low_signal_settlement_count = count_low_signal_settlement_terms(context)
        feature_count += (
            low_signal_settlement_count * weights.low_signal_settlement_weight
        )

    if weights.legal_proceedings_weight > 0.0:
        legal_proceedings_count = count_legal_proceedings(context)
        feature_count += legal_proceedings_count * weights.legal_proceedings_weight

    # High/Low signal legal proceedings (skip expensive regex if weight = 0)
    if weights.high_signal_financial_weight > 0.0:
        high_signal_legal_count = count_high_signal_legal_proceedings(context)
        feature_count += high_signal_legal_count * weights.high_signal_financial_weight

    if weights.low_signal_financial_weight > 0.0:
        low_signal_legal_count = count_low_signal_legal_proceedings(context)
        feature_count += low_signal_legal_count * weights.low_signal_financial_weight

    if weights.monetary_phrases_weight > 0.0:
        monetary_phrases_count = count_monetary_phrases(context)
        feature_count += monetary_phrases_count * weights.monetary_phrases_weight

    # High/Low signal monetary phrases (skip expensive regex if weight = 0)
    if weights.high_signal_settlement_weight > 0.0:
        high_signal_monetary_count = count_high_signal_monetary_phrases(context)
        feature_count += (
            high_signal_monetary_count * weights.high_signal_settlement_weight
        )

    if weights.low_signal_settlement_weight > 0.0:
        low_signal_monetary_count = count_low_signal_monetary_phrases(context)
        feature_count += (
            low_signal_monetary_count * weights.low_signal_settlement_weight
        )

    if weights.document_structure_weight > 0.0:
        document_structure_count = count_document_structure_features(context)
        feature_count += document_structure_count * weights.document_structure_weight

    if weights.numeric_gazetteer_weight > 0.0:
        numeric_gazetteer_count = count_numeric_gazetteer_matches(context)
        feature_count += numeric_gazetteer_count * weights.numeric_gazetteer_weight

    # 🚀 CRITICAL FIX - Add confidence boost features with DROPOUT protection
    if weights.high_confidence_patterns_weight > 0.0:
        high_confidence_count = count_high_confidence_patterns(context)
        feature_count += high_confidence_count * weights.high_confidence_patterns_weight

    if weights.amount_adjacent_keywords_weight > 0.0:
        adjacent_keywords_count = count_amount_adjacent_keywords(context)
        feature_count += (
            adjacent_keywords_count * weights.amount_adjacent_keywords_weight
        )

    # Add overall confidence boost score
    confidence_boost = compute_confidence_boost_score(context)
    feature_count += confidence_boost * weights.confidence_boost_weight

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
    nlp: Optional[Any] = None,
    full_text: str = "",
    match_start: int = 0,
    match_end: int = 0,
) -> int:
    """
    Enhanced feature voting that includes all new features: chronological position,
    title proximity, dependency parsing, enhanced fractions, percentages, and
    boundary-aware context extraction.

    Args:
        context: Text context around the amount
        file_path: Path to the current file for position calculation
        case_position_threshold: Threshold for case position voting
        docket_position_threshold: Threshold for docket position voting
        weights: VotingWeights with all feature weights
        header_chars: Number of characters to consider as header for document titles
        nlp: spaCy NLP model for dependency parsing
        full_text: Full document text for boundary-aware context extraction
        match_start: Start position of the match in full_text
        match_end: End position of the match in full_text

    Returns:
        int: Total number of feature votes including all enhanced features
    """
    # Get standard enhanced feature votes
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

    # Add dependency parsing features if available
    if nlp and spacy_available:
        dependency_votes = (
            extract_dependency_features(context, nlp)
            * weights.dependency_parsing_weight
        )
        votes += dependency_votes

    # Add enhanced fraction extraction features - with DROPOUT support
    if full_text and match_start >= 0 and match_end > match_start:
        fractions = extract_enhanced_fractions(context, weights)
        fraction_votes = len(fractions) * weights.fraction_extraction_weight
        votes += fraction_votes

        # Add percentage extraction with implied totals
        percentages = extract_percentages_with_totals(context)
        percentage_votes = len(percentages) * weights.percentage_extraction_weight
        votes += percentage_votes

        # Add implied totals votes
        implied_total_votes = (
            sum(1 for p in percentages if p.get("implied_total", 0) > 0)
            * weights.implied_totals_weight
        )
        votes += implied_total_votes

        # 🚀 CRITICAL FIX - Add boundary-aware context features with DROPOUT protection
        if weights.sentence_boundary_weight > 0.0:
            sentence_context = extract_sentence_boundary_context(
                full_text, match_start, match_end
            )
        else:
            sentence_context = context  # Skip expensive sentence boundary extraction
        if len(sentence_context) > len(
            context
        ):  # If sentence boundary provides more context
            sentence_votes = (
                (len(sentence_context.split()) - len(context.split()))
                * 0.1
                * weights.sentence_boundary_weight
            )
            votes += sentence_votes

        # 🚀 CRITICAL FIX - Paragraph boundary context with DROPOUT protection
        if weights.paragraph_boundary_weight > 0.0:
            paragraph_context = extract_paragraph_boundary_context(
                full_text, match_start, match_end
            )
        else:
            paragraph_context = sentence_context  # Skip expensive paragraph extraction
        if len(paragraph_context) > len(
            sentence_context
        ):  # If paragraph boundary provides even more context
            paragraph_votes = (
                (len(paragraph_context.split()) - len(sentence_context.split()))
                * 0.05
                * weights.paragraph_boundary_weight
            )
            votes += paragraph_votes

    return int(votes)


def passes_enhanced_feature_filter_with_titles(
    context: str,
    file_path: str,
    min_features: int,
    case_position_threshold: float = 0.5,
    docket_position_threshold: float = 0.5,
    weights: VotingWeights = DEFAULT_VOTING_WEIGHTS,
    header_chars: int = 2000,
    nlp: Optional[Any] = None,
    full_text: str = "",
    match_start: int = 0,
    match_end: int = 0,
) -> bool:
    """
    Enhanced feature filter that includes all new features: chronological position,
    title proximity, dependency parsing, enhanced fractions, percentages, and
    boundary-aware context extraction.

    Args:
        context: Text context around the amount
        file_path: Path to the current file for position calculation
        min_features: Minimum number of features required
        case_position_threshold: Threshold for case position voting
        docket_position_threshold: Threshold for docket position voting
        weights: VotingWeights with all feature weights
        header_chars: Number of characters to consider as header for document titles
        nlp: spaCy NLP model for dependency parsing
        full_text: Full document text for boundary-aware context extraction
        match_start: Start position of the match in full_text
        match_end: End position of the match in full_text

    Returns:
        bool: True if context has at least min_features with all enhancements
    """
    votes = compute_enhanced_feature_votes_with_titles(
        context,
        file_path,
        case_position_threshold,
        docket_position_threshold,
        weights,
        header_chars,
        nlp,
        full_text,
        match_start,
        match_end,
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


def compute_eligible_count(text: str, per_person_amount: Optional[float] = 175) -> int:
    """
    From `text`, extract all numbers that look like money (e.g. "103,055,225" or "$175"),
    pick the largest as the total fund and the next as the per-person cost (unless you
    override per_person_amount), then return floor(total / per_person_amount).
    """
    # 1. Find all things that look like numbers with optional commas/decimals
    matches = re.findall(r"\$?([\d,]+(?:\.\d+)?)", text)
    # 2. Normalize to floats
    nums = [float(m.replace(",", "")) for m in matches]
    if not nums:
        raise ValueError("No numeric values found in text")

    # 3. Determine total and per-person
    total = max(nums)
    if per_person_amount is None:
        # if per_person_amount not given, take second‐largest in text
        nums_sorted = sorted(nums, reverse=True)
        if len(nums_sorted) < 2:
            raise ValueError("Not enough numbers to infer per-person amount")
        per = nums_sorted[1]
    else:
        per = per_person_amount

    # 4. Compute floor division
    return int(total // per)


def extract_complex_amount_candidates(
    text: str, min_amount: float = 100
) -> List[Dict[str, Any]]:
    """
    Extract complex amount patterns that might be missed by standard extraction.

    Patterns:
    1. "X class members who are eligible for Y amount"
    2. "total of X million"
    3. "up to X dollars"
    4. "settlement fund of X"
    5. "awarded X in damages"
    6. "comprised of: (1) X million (2) Y million (3) Z million"
    """
    candidates = []

    # Pattern 1: "X class members who are eligible for Y amount"
    pattern1 = re.compile(
        r"(\d{1,3}(?:,\d{3})*)\s+class\s+members?\s+(?:who\s+are\s+)?eligible\s+for\s+(?:a\s+)?(?:single\s+)?\$?(\d{1,3}(?:,\d{3})*)\s+(?:TSB\s+)?(?:repair\s+)?subsidy",
        re.IGNORECASE,
    )

    for match in pattern1.finditer(text):
        class_count = float(match.group(1).replace(",", ""))
        per_person = float(match.group(2).replace(",", ""))
        total_fund = class_count * per_person

        if total_fund >= min_amount:
            candidates.append(
                {
                    "value": total_fund,
                    "amount": match.group(0),
                    "context": text[max(0, match.start() - 100) : match.end() + 100],
                    "confidence": 0.9,
                    "type": "complex_class_members",
                }
            )

    # Pattern 2: "total of X million/billion"
    pattern2 = re.compile(
        r"total\s+(?:value\s+)?(?:of\s+)?(?:the\s+)?(?:proposed\s+)?(?:settlement\s+)?(?:is\s+)?\$?(\d{1,3}(?:,\d{3})*)\s*(million|billion)",
        re.IGNORECASE,
    )

    for match in pattern2.finditer(text):
        amount = float(match.group(1).replace(",", ""))
        multiplier = 1_000_000 if match.group(2).lower() == "million" else 1_000_000_000
        total = amount * multiplier

        if total >= min_amount:
            candidates.append(
                {
                    "value": total,
                    "amount": match.group(0),
                    "context": text[max(0, match.start() - 100) : match.end() + 100],
                    "confidence": 0.8,
                    "type": "complex_total_value",
                }
            )

    # Pattern 3: "up to X dollars/million/billion"
    pattern3 = re.compile(
        r"up\s+to\s+\$?(\d{1,3}(?:,\d{3})*)\s*(dollars?|million|billion)", re.IGNORECASE
    )

    for match in pattern3.finditer(text):
        amount = float(match.group(1).replace(",", ""))
        unit = match.group(2).lower()
        total = amount  # Default to amount
        if unit == "dollars" or unit == "dollar":
            total = amount
        elif unit == "million":
            total = amount * 1_000_000
        elif unit == "billion":
            total = amount * 1_000_000_000

        if total >= min_amount:
            candidates.append(
                {
                    "value": total,
                    "amount": match.group(0),
                    "context": text[max(0, match.start() - 100) : match.end() + 100],
                    "confidence": 0.7,
                    "type": "complex_up_to",
                }
            )

    # Pattern 4: "settlement fund of X"
    pattern4 = re.compile(
        r"settlement\s+fund\s+(?:of\s+)?(?:is\s+)?\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(million|billion)?",
        re.IGNORECASE,
    )

    for match in pattern4.finditer(text):
        amount = float(match.group(1).replace(",", ""))
        multiplier = 1
        if match.group(2):
            multiplier = (
                1_000_000 if match.group(2).lower() == "million" else 1_000_000_000
            )
        total = amount * multiplier

        if total >= min_amount:
            candidates.append(
                {
                    "value": total,
                    "amount": match.group(0),
                    "context": text[max(0, match.start() - 100) : match.end() + 100],
                    "confidence": 0.85,
                    "type": "complex_settlement_fund",
                }
            )

    # Pattern 5: "awarded X in damages/compensation"
    pattern5 = re.compile(
        r"awarded\s+\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(million|billion)?\s+in\s+(damages|compensation|fees)",
        re.IGNORECASE,
    )

    for match in pattern5.finditer(text):
        amount = float(match.group(1).replace(",", ""))
        multiplier = 1
        if match.group(2):
            multiplier = (
                1_000_000 if match.group(2).lower() == "million" else 1_000_000_000
            )
        total = amount * multiplier

        if total >= min_amount:
            candidates.append(
                {
                    "value": total,
                    "amount": match.group(0),
                    "context": text[max(0, match.start() - 100) : match.end() + 100],
                    "confidence": 0.9,
                    "type": "complex_awarded",
                }
            )

    # Pattern 6: "comprised of: (1) X million (2) Y million (3) Z million"
    pattern6 = re.compile(
        r"comprised\s+of:\s*(?:\(\d+\)\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(million|billion)?\s*[^)]*)+",
        re.IGNORECASE,
    )

    for match in pattern6.finditer(text):
        # Extract all amounts from the comprised list
        amounts = re.findall(
            r"\(\d+\)\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(million|billion)?",
            match.group(0),
        )
        total = 0
        for amount_str, unit in amounts:
            amount = float(amount_str.replace(",", ""))
            multiplier = 1
            if unit:
                multiplier = 1_000_000 if unit.lower() == "million" else 1_000_000_000
            total += amount * multiplier

        if total >= min_amount:
            candidates.append(
                {
                    "value": total,
                    "amount": match.group(0),
                    "context": text[max(0, match.start() - 100) : match.end() + 100],
                    "confidence": 0.8,
                    "type": "complex_comprised",
                }
            )

    # Pattern 7: "X class members who are eligible for Y amount per repair"
    pattern7 = re.compile(
        r"(\d{1,3}(?:,\d{3})*)\s+class\s+members?\s+(?:who\s+are\s+)?eligible\s+for\s+(?:a\s+)?(?:single\s+)?\$?(\d{1,3}(?:,\d{3})*)\s+(?:TSB\s+)?(?:repair\s+)?subsidy",
        re.IGNORECASE,
    )

    for match in pattern7.finditer(text):
        class_count = float(match.group(1).replace(",", ""))
        per_person = float(match.group(2).replace(",", ""))
        total_fund = class_count * per_person

        if total_fund >= min_amount:
            candidates.append(
                {
                    "value": total_fund,
                    "amount": match.group(0),
                    "context": text[max(0, match.start() - 100) : match.end() + 100],
                    "confidence": 0.95,
                    "type": "complex_class_members_calc",
                }
            )

    # Pattern 8: "X class members × Y amount = Z total"
    pattern8 = re.compile(
        r"(\d{1,3}(?:,\d{3})*)\s+class\s+members?\s*[×x]\s*\$?(\d{1,3}(?:,\d{3})*)\s*[=]\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)",
        re.IGNORECASE,
    )

    for match in pattern8.finditer(text):
        total_fund = float(match.group(3).replace(",", ""))

        if total_fund >= min_amount:
            candidates.append(
                {
                    "value": total_fund,
                    "amount": match.group(0),
                    "context": text[max(0, match.start() - 100) : match.end() + 100],
                    "confidence": 0.95,
                    "type": "complex_class_members_equation",
                }
            )

    # Pattern 9: "total of X million/billion in settlement value"
    pattern9 = re.compile(
        r"total\s+(?:value\s+)?(?:of\s+)?(?:the\s+)?(?:proposed\s+)?(?:settlement\s+)?(?:is\s+)?(?:estimated\s+at\s+)?\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(million|billion)?\s+(?:in\s+)?(?:settlement\s+)?(?:value|benefits?)",
        re.IGNORECASE,
    )

    for match in pattern9.finditer(text):
        amount = float(match.group(1).replace(",", ""))
        multiplier = 1
        if match.group(2):
            multiplier = (
                1_000_000 if match.group(2).lower() == "million" else 1_000_000_000
            )
        total = amount * multiplier

        if total >= min_amount:
            candidates.append(
                {
                    "value": total,
                    "amount": match.group(0),
                    "context": text[max(0, match.start() - 100) : match.end() + 100],
                    "confidence": 0.9,
                    "type": "complex_settlement_value",
                }
            )

    return candidates


def extract_candidate_from_fraction(text: str) -> int:
    """
    From an unstructured `text` like:
      "requested fees of $2,500,000 will represent approximately one-third of the common fund"
    this returns the implied total fund size (fees ÷ fraction), e.g. 7500000.
    """
    # 1. Find the fee amount - look for patterns like "fees of $X" or "attorney fees of $X"
    fee_patterns = [
        r"fees?\s+of\s+\$([\d,]+(?:\.\d+)?)",
        r"attorney\s+fees?\s+of\s+\$([\d,]+(?:\.\d+)?)",
        r"counsel\s+fees?\s+of\s+\$([\d,]+(?:\.\d+)?)",
        r"requested\s+fees?\s+of\s+\$([\d,]+(?:\.\d+)?)",
        r"award\s+of\s+\$([\d,]+(?:\.\d+)?)",
    ]

    fee = None
    for pattern in fee_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            fee = float(match.group(1).replace(",", ""))
            break

    # Fallback: if no specific fee pattern found, take the first money value
    if fee is None:
        money_match = re.search(r"\$([\d,]+(?:\.\d+)?)", text)
        if not money_match:
            raise ValueError("No monetary value found")
        fee = float(money_match.group(1).replace(",", ""))

    # 2. Look for a spelled-out fraction
    #    support hyphens or spaces, common fractions up to thirds
    frac_map = {
        "one[ -]?half": Fraction(1, 2),
        "one[ -]?third": Fraction(1, 3),
        "two[ -]?thirds": Fraction(2, 3),
        "one[ -]?quarter": Fraction(1, 4),
        "three[ -]?quarters": Fraction(3, 4),
    }
    frac = None
    for pat, f in frac_map.items():
        if re.search(pat, text, re.IGNORECASE):
            frac = f
            break
    if frac is None:
        raise ValueError("No supported fraction phrase found")

    # 3. Compute implied total: fee ÷ fraction
    total = fee / float(frac)
    return int(round(total))


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
                    text = fast_json_loads(line).get("text", "")

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

                    # Process fraction-based implied amounts (new)
                    try:
                        implied_total = extract_candidate_from_fraction(text)
                        if implied_total >= min_amount:
                            # Create context around the fraction pattern
                            fraction_patterns = [
                                "one-third",
                                "one-half",
                                "one-quarter",
                                "two-thirds",
                                "three-quarters",
                            ]
                            for pattern in fraction_patterns:
                                if pattern in text.lower():
                                    # Find the pattern in the text
                                    pattern_match = re.search(
                                        pattern, text, re.IGNORECASE
                                    )
                                    if pattern_match:
                                        start, end = pattern_match.span()
                                        ctx = text[
                                            max(0, start - context_chars) : end
                                            + context_chars
                                        ].replace("\n", " ")

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
                                                    "amount": f"${implied_total:,}",
                                                    "context": ctx,
                                                    "simhash": sim,
                                                    "type": "fraction_implied",
                                                    "case_id": case_id,
                                                }
                                                out_f.write(
                                                    json.dumps(candidate_data) + "\n"
                                                )
                                                if csv_output:
                                                    candidates_for_csv.append(
                                                        candidate_data
                                                    )
                                                break  # Only process the first fraction pattern found
                    except (ValueError, Exception):
                        # Skip if fraction extraction fails
                        pass

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
