"""Enhanced interpretable feature engineering for legal risk classification.

This module provides sophisticated yet fully interpretable features including
risk lexicons, sequence modeling without embeddings, and linguistic analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple, Optional
import re
from collections import defaultdict
import numpy as np
from numpy.typing import NDArray


# Risk lexicons for legal domain
RISK_LEXICONS = {
    "deception": {
        "misleading",
        "false",
        "deceptive",
        "misrepresent",
        "fabricate",
        "distort",
        "manipulate",
        "deceive",
        "fraud",
        "dishonest",
        "untrue",
        "inaccurate",
        "misstate",
        "conceal",
        "hide",
    },
    "guarantee": {
        "guarantee",
        "assure",
        "promise",
        "commit",
        "ensure",
        "warrant",
        "pledge",
        "vow",
        "swear",
        "certify",
        "will",
        "definitely",
        "certainly",
        "absolutely",
    },
    "pricing_claims": {
        "free",
        "no fee",
        "no cost",
        "zero",
        "complimentary",
        "discount",
        "save",
        "bargain",
        "deal",
        "offer",
        "lowest",
        "cheapest",
        "best price",
        "reduced",
    },
    "reliability": {
        "secure",
        "safe",
        "protected",
        "reliable",
        "trusted",
        "proven",
        "tested",
        "verified",
        "certified",
        "guaranteed",
        "risk-free",
        "foolproof",
        "fail-safe",
    },
    "scienter": {
        "knew",
        "aware",
        "conscious",
        "deliberate",
        "intentional",
        "willful",
        "reckless",
        "negligent",
        "should have known",
        "bad faith",
        "malicious",
        "purposeful",
    },
    "superlatives": {
        "best",
        "always",
        "never",
        "all",
        "every",
        "most",
        "least",
        "only",
        "unique",
        "exclusive",
        "unmatched",
        "unparalleled",
        "incomparable",
    },
    "hedges": {
        "may",
        "might",
        "could",
        "possibly",
        "perhaps",
        "potential",
        "probable",
        "likely",
        "appear",
        "seem",
        "suggest",
        "indicate",
        "approximately",
        "about",
    },
    "disclaimers": {
        "not intended",
        "subject to",
        "except",
        "excluding",
        "limitations apply",
        "terms and conditions",
        "disclaimer",
        "not liable",
        "no warranty",
        "as is",
        "void where",
    },
}

# Discourse markers for sequence modeling
DISCOURSE_MARKERS = {
    "causal": {"because", "therefore", "thus", "hence", "consequently", "so"},
    "contrast": {"however", "but", "although", "despite", "nevertheless", "yet"},
    "temporal": {"then", "next", "after", "before", "subsequently", "meanwhile"},
    "conditional": {"if", "unless", "provided", "assuming", "given that"},
    "additive": {"moreover", "furthermore", "additionally", "also", "besides"},
}

# Negation patterns
NEGATION_PATTERNS = re.compile(
    r"\b(not|no|never|none|nothing|nowhere|neither|nobody|cannot|can\'t|won\'t|"
    r"wouldn\'t|shouldn\'t|couldn\'t|didn\'t|doesn\'t|don\'t|isn\'t|aren\'t|"
    r"wasn\'t|weren\'t|hardly|scarcely|barely)\b",
    re.IGNORECASE,
)


class InterpretableFeatureExtractor:
    """Extract interpretable features for legal text classification."""

    def __init__(
        self,
        include_lexicons: bool = True,
        include_sequence: bool = True,
        include_linguistic: bool = True,
        include_structural: bool = True,
        lexicon_weights: Optional[Dict[str, float]] = None,
    ):
        self.include_lexicons = include_lexicons
        self.include_sequence = include_sequence
        self.include_linguistic = include_linguistic
        self.include_structural = include_structural
        self.lexicon_weights = lexicon_weights or {}

    def extract_features(
        self, text: str, context: Optional[str] = None
    ) -> Dict[str, float]:
        """Extract all interpretable features from text."""
        features = {}

        # Combine text and context if provided
        full_text = text
        if context:
            full_text = f"{text} [CONTEXT] {context}"

        # Tokenize
        tokens = self._tokenize(full_text)
        text_tokens = self._tokenize(text)

        if self.include_lexicons:
            features.update(self._extract_lexicon_features(tokens, text_tokens))

        if self.include_sequence:
            features.update(self._extract_sequence_features(tokens))

        if self.include_linguistic:
            features.update(self._extract_linguistic_features(text, tokens))

        if self.include_structural:
            features.update(self._extract_structural_features(text))

        return features

    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        return re.findall(r"\b\w+\b", text.lower())

    def _extract_lexicon_features(
        self, tokens: List[str], quote_tokens: List[str]
    ) -> Dict[str, float]:
        """Extract risk lexicon features."""
        features = {}
        token_set = set(tokens)
        quote_token_set = set(quote_tokens)

        for lexicon_name, lexicon_terms in RISK_LEXICONS.items():
            # Raw counts
            count = sum(1 for token in tokens if token in lexicon_terms)
            quote_count = sum(1 for token in quote_tokens if token in lexicon_terms)

            # Normalized by length
            norm_count = count / max(len(tokens), 1)
            quote_norm_count = quote_count / max(len(quote_tokens), 1)

            # Binary presence
            presence = int(bool(token_set & lexicon_terms))

            # Apply optional weighting
            weight = self.lexicon_weights.get(lexicon_name, 1.0)

            features[f"lex_{lexicon_name}_count"] = count * weight
            features[f"lex_{lexicon_name}_norm"] = norm_count * weight
            features[f"lex_{lexicon_name}_present"] = presence * weight
            features[f"lex_{lexicon_name}_quote_count"] = quote_count * weight
            features[f"lex_{lexicon_name}_quote_norm"] = quote_norm_count * weight

        return features

    def _extract_sequence_features(self, tokens: List[str]) -> Dict[str, float]:
        """Extract sequence features without embeddings."""
        features = {}

        if len(tokens) < 2:
            return features

        # Transition probabilities between risk categories
        transitions = defaultdict(int)
        for i in range(len(tokens) - 1):
            curr_cats = self._get_token_categories(tokens[i])
            next_cats = self._get_token_categories(tokens[i + 1])

            for c1 in curr_cats:
                for c2 in next_cats:
                    transitions[f"{c1}_to_{c2}"] += 1

        # Normalize and add as features
        total_transitions = sum(transitions.values()) or 1
        for trans, count in transitions.items():
            features[f"seq_trans_{trans}"] = count / total_transitions

        # Positional encoding of risk terms
        risk_positions: List[float] = []
        for i, token in enumerate(tokens):
            if any(token in RISK_LEXICONS[lex] for lex in RISK_LEXICONS):
                risk_positions.append(i / len(tokens))

        if risk_positions:
            features["seq_risk_mean_pos"] = np.mean(risk_positions)
            features["seq_risk_std_pos"] = np.std(risk_positions)
            features["seq_risk_first_pos"] = risk_positions[0]
            features["seq_risk_last_pos"] = risk_positions[-1]

        # Discourse marker patterns
        for marker_type, markers in DISCOURSE_MARKERS.items():
            marker_count = sum(1 for token in tokens if token in markers)
            features[f"seq_discourse_{marker_type}"] = marker_count

        return features

    def _get_token_categories(self, token: str) -> Set[str]:
        """Get risk categories for a token."""
        categories = set()
        for lex_name, lex_terms in RISK_LEXICONS.items():
            if token in lex_terms:
                categories.add(lex_name)
        if not categories:
            categories.add("neutral")
        return categories

    def _extract_linguistic_features(
        self, text: str, tokens: List[str]
    ) -> Dict[str, float]:
        """Extract linguistic features."""
        features = {}

        # Negation features
        negations = NEGATION_PATTERNS.findall(text)
        features["ling_negation_count"] = len(negations)
        features["ling_negation_norm"] = len(negations) / max(len(tokens), 1)

        # Modal verbs (certainty levels)
        high_certainty_modals = {"will", "must", "shall", "cannot"}
        low_certainty_modals = {"may", "might", "could", "would"}

        features["ling_high_certainty"] = sum(
            1 for t in tokens if t in high_certainty_modals
        )
        features["ling_low_certainty"] = sum(
            1 for t in tokens if t in low_certainty_modals
        )

        # Numbers and financial indicators
        numbers = re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b", text)
        features["ling_number_count"] = len(numbers)

        # Money amounts
        money_pattern = re.compile(
            r"\$[\d,]+(?:\.\d{2})?|\b\d+\s*(?:dollars|cents|USD)\b", re.I
        )
        money_amounts = money_pattern.findall(text)
        features["ling_money_count"] = len(money_amounts)

        if money_amounts:
            # Extract numeric values for binning
            amounts: List[float] = []
            for amt in money_amounts:
                try:
                    value = float(re.sub(r"[^\d.]", "", amt))
                    amounts.append(value)
                except (ValueError, TypeError):
                    pass
            if amounts:
                features["ling_money_max"] = float(np.log1p(max(amounts)))
                features["ling_money_mean"] = float(np.log1p(np.mean(amounts)))

        # Time references
        time_pattern = re.compile(
            r"\b(?:year|month|week|day|hour|minute|immediately|soon|"
            r"deadline|expire|within|before|after)\b",
            re.I,
        )
        features["ling_time_ref_count"] = len(time_pattern.findall(text))

        return features

    def _extract_structural_features(self, text: str) -> Dict[str, float]:
        """Extract structural text features."""
        features = {}

        # Basic counts
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        features["struct_sentence_count"] = len(sentences)
        features["struct_word_count"] = len(text.split())
        features["struct_char_count"] = len(text)

        # Punctuation density
        punct_count = len(re.findall(r"[^\w\s]", text))
        features["struct_punct_density"] = punct_count / max(len(text), 1)

        # Capitalization ratio
        upper_count = sum(1 for c in text if c.isupper())
        features["struct_caps_ratio"] = upper_count / max(len(text), 1)

        # Question marks and exclamations
        features["struct_questions"] = text.count("?")
        features["struct_exclamations"] = text.count("!")

        # Parenthetical statements (often disclaimers)
        features["struct_parentheses"] = len(re.findall(r"\([^)]+\)", text))

        # Average sentence length
        if sentences:
            sentence_lengths = [len(s.split()) for s in sentences]
            features["struct_avg_sent_len"] = np.mean(sentence_lengths)
            features["struct_std_sent_len"] = np.std(sentence_lengths)

        return features


def create_feature_matrix(
    records: List[Dict[str, Any]],
    extractor: InterpretableFeatureExtractor,
    text_key: str = "text",
    context_key: str = "context",
) -> Tuple[NDArray[np.float64], List[str]]:
    """Create feature matrix from records."""
    all_features = []

    for record in records:
        text = record.get(text_key, "")
        context = record.get(context_key, "")
        features = extractor.extract_features(text, context)
        all_features.append(features)

    # Get all feature names
    feature_names = sorted(set(key for feat_dict in all_features for key in feat_dict))

    # Create matrix
    matrix = np.zeros((len(records), len(feature_names)))
    for i, features in enumerate(all_features):
        for j, name in enumerate(feature_names):
            matrix[i, j] = features.get(name, 0.0)

    return matrix, feature_names
