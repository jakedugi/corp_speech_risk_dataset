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
    "compliance": {
        "compliant",
        "adhere",
        "conform",
        "regulate",
        "standard",
        "ethical",
        "transparent",
        "disclose",
        "report",
        "audit",
        "verify",
        "certify",
        "accordance",
        "pursuant",
        "documented",
        "recorded",
        "filed",
        "submitted",
    },
    "positive_qualifiers": {
        "based on",
        "in accordance with",
        "subject to approval",
        "as permitted",
        "consistent with",
        "aligned with",
        "under review",
        "preliminary",
        "according to",
        "as reported by",
        "data from",
        "study shows",
        "report finds",
    },
    "scope_limiters": {
        "only",
        "limited",
        "except",
        "excluding",
        "subject to",
        "conditional upon",
        "provided that",
        "other than",
        "save for",
        "solely",
        "exclusively",
        "restricted",
    },
    "ambiguity": {
        "reasonable",
        "substantially",
        "material",
        "approximately",
        "generally",
        "often",
        "typically",
        "vague",
        "uncertain",
        "ambiguous",
        "unclear",
        "indefinite",
        "unspecified",
        "subject to interpretation",
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
    "obligation": {
        "must",
        "shall",
        "required",
        "obligated",
        "compelled",
        "mandatory",
        "compulsory",
        "binding",
        "duty",
        "obligation",
    },
    "permission": {
        "may",
        "can",
        "permitted",
        "allowed",
        "optional",
        "discretionary",
        "eligible",
        "authorized",
        "entitled",
        "voluntary",
    },
    "clarity": {
        "clear",
        "precise",
        "specific",
        "defined",
        "explicit",
        "exact",
        "detailed",
        "unambiguous",
        "definitive",
        "concrete",
    },
    "policy_procedure": {
        "policy",
        "procedure",
        "guideline",
        "protocol",
        "standard",
        "framework",
        "methodology",
        "process",
        "workflow",
        "governance",
        "audit",
        "compliance",
        "oversight",
        "monitoring",
        "review",
    },
    "accountability": {
        "accountable",
        "responsible",
        "liable",
        "answerable",
        "culpable",
        "ownership",
        "accountability",
        "responsibility",
        "liability",
        "fault",
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

# Additional patterns for low-risk signal detection
SAFE_HARBOR_PHRASES = [
    "forward-looking statements",
    "forward looking statements",
    "cautionary statements",
    "private securities litigation reform act",
    "pslra",
    "safe harbor",
    "risk factors",
    "undue reliance",
    "actual results may differ",
    "no obligation to update",
]

NON_ADVICE_PHRASES = [
    "not investment advice",
    "no investment advice",
    "do not rely",
    "informational purposes only",
    "educational purposes only",
    "does not constitute an offer",
    "no solicitation",
]

EVIDENTIAL_PHRASES = [
    "according to",
    "as reported by",
    "as per",
    "data from",
    "study shows",
    "report finds",
    "survey indicates",
    "third-party",
    "independent review",
]

# Compiled patterns for efficiency
SEC_REF_PATTERN = re.compile(
    r"\b(10-k|10-q|8-k|20-f|6-k|s-1|s-3|form\s+[a-z0-9-]+|sec)\b", re.I
)
REGULATORY_REF_PATTERN = re.compile(
    r"\b(rule|section|§|cfr|u\.s\.c\.|iso|pci|sox|fcpa|gdpr|reg\.|regulation|circular|directive|guidance|sec|finra|pcaob|fasb|gaap|ifrs)\b",
    re.I,
)
URL_PATTERN = re.compile(r"https?://|www\.")
TENSE_PATTERNS = {
    "past": re.compile(
        r"\b(was|were|had|did|went|said|made|took|got|saw|came|knew|thought|found|gave|told|became|showed|left|felt|put)\b",
        re.I,
    ),
    "future": re.compile(r"\b(will|shall|going to|would)\b", re.I),
    "conditional": re.compile(
        r"\b(if|unless|provided that|in case|assuming|whether)\b", re.I
    ),
}

# New pattern sets for features
FACTIVITY_VERBS = {
    "show",
    "demonstrate",
    "indicate",
    "find",
    "observe",
    "estimate",
    "calculate",
    "measured",
    "audited",
    "verified",
    "confirmed",
    "documented",
    "recorded",
    "established",
    "determined",
    "concluded",
    "reported",
    "disclosed",
    "revealed",
}

TIME_ANCHOR_PATTERN = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{4}|q[1-4]|fy\d{2,4}|quarter|fiscal|year)\b",
    re.I,
)

ENUMERATION_PATTERN = re.compile(r"\b(\d+\.|[-*•])\s", re.MULTILINE)

THIRD_PERSON_PRONOUNS = {
    "he",
    "she",
    "they",
    "him",
    "her",
    "them",
    "his",
    "hers",
    "their",
    "theirs",
    "company",
    "firm",
    "organization",
    "corporation",
    "entity",
}
FIRST_PERSON_PRONOUNS = {
    "i",
    "we",
    "us",
    "our",
    "ours",
    "my",
    "mine",
    "myself",
    "ourselves",
}

ACTIVE_VOICE_PATTERN = re.compile(r"\b(am|is|are|was|were)\s+\w+ing\b", re.I)
ATTRIBUTION_PATTERN = re.compile(
    r"\b(said|stated|reported|according to|as reported by|noted|mentioned|indicated|disclosed|announced|confirmed)\b",
    re.I,
)

# New comprehensive patterns for 20 features
DATE_MARKER_PATTERN = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december|q[1-4]|fy\d{2,4}|fiscal\s+year|\d{4}|quarter)\b",
    re.I,
)

POLICY_REFERENCE_PATTERN = re.compile(
    r"\b(policy|procedure|terms|conditions|guidelines|code\s+of\s+conduct|privacy\s+policy|terms\s+of\s+use|tou|terms\s+of\s+service)\b",
    re.I,
)

RANGE_EXPRESSION_PATTERN = re.compile(
    r"\b(between\s+\w+\s+and\s+\w+|from\s+\w+\s+to\s+\w+|at\s+least|no\s+more\s+than|up\s+to|in\s+the\s+range|ranging\s+from)\b",
    re.I,
)

INTENSIFIER_TERMS = {
    "very",
    "really",
    "extremely",
    "highly",
    "completely",
    "totally",
    "absolutely",
    "entirely",
    "quite",
    "rather",
    "significantly",
    "substantially",
}

VAGUE_QUANTIFIER_TERMS = {
    "some",
    "many",
    "several",
    "few",
    "numerous",
    "various",
    "multiple",
    "certain",
    "certain",
    "considerable",
}

NEGATED_CLAIM_PATTERN = re.compile(
    r"\b(no\s+guarantee|cannot\s+guarantee|not\s+responsible|cannot\s+ensure|no\s+warranty|not\s+liable|do\s+not\s+guarantee)\b",
    re.I,
)

WE_FUTURE_COMMIT_PATTERN = re.compile(
    r"\b(we\s+will|we\s+shall|we\s+plan\s+to|we\s+intend\s+to|our\s+plan|our\s+intention)\b",
    re.I,
)

EXTERNAL_AUTHORITY_TERMS = {
    "sec",
    "ftc",
    "fda",
    "doj",
    "court",
    "judge",
    "consent order",
    "settlement",
    "regulator",
    "regulatory",
    "commission",
    "agency",
    "enforcement",
}

DEFINITION_MARKER_PATTERN = re.compile(
    r"\b(means|includes|shall\s+mean|defined\s+as|refers\s+to|definition|hereby\s+defined)\b",
    re.I,
)

SENTENCE_INITIAL_HEDGE_PATTERN = re.compile(
    r"^(may|might|could|subject\s+to|assuming|provided\s+that|if)", re.I
)

# Social media patterns
EMOJI_PATTERN = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]",
    re.UNICODE,
)
HASHTAG_PATTERN = re.compile(r"#\w+")
MENTION_PATTERN = re.compile(r"@\w+")

# Sets for proximity analysis
FUTURE_MODALS = {"will", "shall", "going", "gonna"}
NEG_WORDS = {
    "not",
    "no",
    "never",
    "none",
    "nothing",
    "nowhere",
    "neither",
    "nobody",
    "cannot",
    "can't",
    "won't",
    "wouldn't",
    "shouldn't",
    "couldn't",
    "didn't",
    "doesn't",
    "don't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "hardly",
    "scarcely",
    "barely",
}

# Additional patterns for new features
APPROXIMATION_WORDS = {
    "about",
    "around",
    "roughly",
    "circa",
    "approximately",
    "near",
    "close",
    "estimate",
    "estimated",
    "approx",
    "~",
    "≈",
}

CONTRAST_DISCOURSE_MARKERS = {
    "however",
    "but",
    "although",
    "despite",
    "nevertheless",
    "yet",
    "whereas",
    "while",
    "though",
    "on the other hand",
    "in contrast",
    "conversely",
}

# Compiled patterns for new features
NUMERIC_PATTERN = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
PERCENTAGE_PATTERN = re.compile(r"\b\d[\d,]*(?:\.\d+)?%\b")
MONEY_PATTERN_EXTENDED = re.compile(
    r"\$[\d,]+(?:\.\d+)?|\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:dollars|cents|USD|EUR|GBP)\b",
    re.I,
)
RANGE_PATTERN = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s*(?:–|-|to)\s*\d[\d,]*(?:\.\d+)?\b|from\s+\d[\d,]*(?:\.\d+)?\s+to\s+\d[\d,]*(?:\.\d+)?\b",
    re.I,
)
EXPLICIT_DATE_PATTERN = re.compile(
    r"\b(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december)\b|\b20\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    re.I,
)
VAGUE_TIME_PATTERN = re.compile(
    r"\b(?:soon|later|eventually|sometime|future|past|recently|lately|earlier)\b", re.I
)
ATTRIBUTION_VERBS = {
    "said",
    "stated",
    "noted",
    "announced",
    "according",
    "reported",
    "indicated",
    "disclosed",
    "revealed",
    "mentioned",
    "confirmed",
    "declared",
    "asserted",
}

# Additional patterns and lexicons for new features
ORG_HINTS = {
    "inc",
    "corp",
    "llc",
    "co",
    "company",
    "board",
    "corporation",
    "firm",
    "organization",
    "entity",
    "sec",
    "commission",
    "agency",
    "bureau",
}

DISCLAIMER_PHRASES = {
    "subject to",
    "limitations",
    "except",
    "only",
    "as is",
    "no warranty",
    "terms and conditions",
    "disclaimer",
    "not liable",
    "void where",
    "limitations apply",
    "excluding",
    "save for",
}

# Remediation terms for risk mitigation context
REMEDIATION_TERMS = {
    "remediate",
    "mitigate",
    "mitigating",
    "mitigation",
    "corrective",
    "amended",
    "updated",
    "restated",
    "rectify",
    "addressed",
    "enhanced",
    "trained",
    "audited",
    "disclosed",
    "corrected",
    "fixed",
    "resolved",
    "improved",
    "strengthened",
    "reinforced",
}

# Deontic modality patterns
DEONTIC_OBLIGATION = {
    "must",
    "shall",
    "required",
    "obligated",
    "mandatory",
    "compulsory",
}
DEONTIC_PERMISSION = {"may", "can", "permitted", "allowed", "optional", "discretionary"}

# Claim and qualifier markers for argument structure
CLAIM_MARKERS = {"claim", "assert", "state", "declare", "maintain", "contend"}
QUALIFIER_MARKERS = {"because", "since", "given", "provided", "if", "when", "where"}

# Additional patterns for the 7 new features
CONDITIONAL_MARKERS = {
    "if",
    "unless",
    "provided that",
    "assuming",
    "whether",
    "subject to",
}
SCOPE_LIMITERS_EXTENDED = {
    "only",
    "subject to",
    "except",
    "provided that",
    "limited to",
    "solely",
    "exclusively",
    "conditional upon",
    "save for",
    "other than",
    "excluding",
}


def _count_phrases(text_lower: str, phrases: List[str]) -> int:
    """Count occurrences of phrases in text using word boundaries."""
    count = 0
    for phrase in phrases:
        phrase_escaped = re.escape(phrase)
        pattern = rf"(?<!\w){phrase_escaped}(?!\w)"
        count += len(re.findall(pattern, text_lower))
    return count


def _window_proximity_count(
    tokens: List[str], set_a: set[str], set_b: set[str], radius: int = 3
) -> int:
    """Count how many times tokens from set_a appear within radius of tokens from set_b."""
    if not tokens or not set_a or not set_b:
        return 0

    # Find indices of tokens in each set
    idx_a = [i for i, t in enumerate(tokens) if t in set_a]
    idx_b = [i for i, t in enumerate(tokens) if t in set_b]

    if not idx_a or not idx_b:
        return 0

    count = 0
    j = 0
    for i in idx_a:
        # Advance j while B[j] < i - radius
        while j < len(idx_b) and idx_b[j] < i - radius:
            j += 1
        k = j
        while k < len(idx_b) and idx_b[k] <= i + radius:
            count += 1
            k += 1

    return count


def _count_syllables(word: str) -> int:
    """Approximate syllable count using vowel groups."""
    word = word.lower()
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e"):
        count -= 1
    return max(count, 1)


def _detect_person_name(tokens: List[str], i: int) -> bool:
    """Detect if token position i starts a person name (Proper Noun + Proper Noun pattern)."""
    if i + 1 >= len(tokens):
        return False
    return (
        tokens[i].istitle()
        and len(tokens[i]) > 1
        and tokens[i + 1].istitle()
        and len(tokens[i + 1]) > 1
        and tokens[i] not in ORG_HINTS
        and tokens[i + 1] not in ORG_HINTS
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
        features: Dict[str, float] = {}

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

        # Add low-risk signal features (NEW)
        low_risk_features = self._extract_low_risk_signals(
            text, tokens, text_tokens, features
        )
        features.update(low_risk_features)

        # Add derived features (ratios and interactions)
        if features:  # Only extract derived features if we have base features
            derived_features = self._extract_derived_features(features)
            features.update(derived_features)

        # Add the 20 new features
        new_features = self._extract_new_20_features(
            text, tokens, text_tokens, features
        )
        features.update(new_features)

        # Add the additional 19 new features
        additional_features = self._extract_additional_19_features(
            text, tokens, text_tokens, features
        )
        features.update(additional_features)

        # Add the final 7 new features
        final_features = self._extract_final_7_features(
            text, tokens, text_tokens, features
        )
        features.update(final_features)

        # Add the 24 additional new features
        additional_24_features = self._extract_24_additional_features(
            text, tokens, text_tokens, features
        )
        features.update(additional_24_features)

        # Add the 20 additional new features
        additional_20_features = self._extract_20_additional_features(
            text, tokens, text_tokens, features
        )
        features.update(additional_20_features)

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
        total_transitions = sum(count for count in transitions.values()) or 1
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

        # NEW SEQUENCE FEATURES

        # 9. Conditional Transition Frequency - transitions from conditional to claims
        conditional_to_claim_count = transitions.get("conditional_to_guarantee", 0)
        conditional_to_claim_count += transitions.get("conditional_to_superlatives", 0)
        features["seq_trans_conditional_to_claim"] = conditional_to_claim_count / max(
            total_transitions, 1
        )

        # 4. Compliance Transition Share - compliance with safe categories
        compliance_transitions = 0
        safe_cats = {"neutral", "hedges", "scope_limiters", "disclaimers"}
        for trans, count in transitions.items():
            if "compliance" in trans and any(cat in trans for cat in safe_cats):
                compliance_transitions += count
        features["seq_compliance_transition_share"] = compliance_transitions / max(
            total_transitions, 1
        )

        # NEW SEQUENCE FEATURES

        # 6. Permission to Obligation Transition - sequence from permissive to obligatory
        permission_to_obligation_count = transitions.get("permission_to_obligation", 0)
        features["seq_trans_permission_to_obligation"] = (
            permission_to_obligation_count / max(total_transitions, 1)
        )

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

        # Grammar-based features (NEW)
        # Tense patterns for temporal classification
        past_matches = TENSE_PATTERNS["past"].findall(text)
        future_matches = TENSE_PATTERNS["future"].findall(text)
        conditional_matches = TENSE_PATTERNS["conditional"].findall(text)

        features["ling_past_tense_count"] = len(past_matches)
        features["ling_future_tense_count"] = len(future_matches)
        features["ling_conditional_count"] = len(conditional_matches)

        # Tense ratio (higher = more factual/conditional, less speculative)
        features["ling_tense_ratio"] = (
            len(past_matches) + len(conditional_matches) + 0.0001
        ) / (len(future_matches) + 0.0001)

        # Passive voice detection (avoids responsibility)
        passive_pattern = re.compile(
            r"\b(was|were|been|being|is|are|am)\s+\w+ed\b", re.I
        )
        passive_matches = passive_pattern.findall(text)
        features["ling_passive_voice_count"] = len(passive_matches)
        features["ling_passive_voice_norm"] = len(passive_matches) / max(len(tokens), 1)

        # Unitized number rate (NEW) - facticity indicator for low-risk
        unitized_pattern = re.compile(
            r"(?:[$€£¥]\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:,\d{3})*(?:\.\d+)?%|\d+(?:,\d{3})*(?:\.\d+)?\s*(?:bps?|basis\s*points?|million|billion|m|bn|usd|eur|gbp))",
            re.I,
        )
        unitized_numbers = unitized_pattern.findall(text)
        features["ling_unitized_number_count"] = len(unitized_numbers)
        features["ling_unitized_number_norm"] = len(unitized_numbers) / max(
            len(tokens), 1
        )

        # Fact-sentence share (NEW) - sentences with numbers AND past/reported verbs
        reporting_verbs = {
            "said",
            "stated",
            "reported",
            "according",
            "showed",
            "indicated",
            "found",
            "demonstrated",
        }
        past_verbs = {"was", "were", "had", "did", "made", "took", "became", "got"}

        # Split into sentences for analysis
        text_sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        fact_sentences = 0
        if text_sentences:
            for sentence in text_sentences:
                sentence_lower = sentence.lower()
                has_number = bool(re.search(r"\d+", sentence))
                has_reporting = any(verb in sentence_lower for verb in reporting_verbs)
                has_past = any(verb in sentence_lower for verb in past_verbs)

                if has_number and (has_reporting or has_past):
                    fact_sentences += 1

        features["ling_fact_sentence_count"] = fact_sentences
        features["ling_fact_sentence_share"] = (
            fact_sentences / max(len(text_sentences), 1) if text_sentences else 0.0
        )

        # Deontic modality balance (NEW) - obligations vs permissions for risk assessment
        oblig_count = sum(1 for token in tokens if token in DEONTIC_OBLIGATION)
        perm_count = sum(1 for token in tokens if token in DEONTIC_PERMISSION)
        total_deontic = oblig_count + perm_count

        features["ling_deontic_oblig_count"] = oblig_count
        features["ling_deontic_perm_count"] = perm_count
        features["ling_deontic_balance_norm"] = (perm_count + 0.0001) / (
            total_deontic + 0.0001
        )  # Higher = more permissive (low-risk)

        # NEW FEATURES: 20 Enhanced Interpretable Features

        # 3. Temporal Reference Share - sentence-level temporal density
        time_ref_matches = TIME_ANCHOR_PATTERN.findall(text)
        features["ling_temporal_ref_share"] = (
            len(time_ref_matches) / max(len(text_sentences), 1)
            if text_sentences
            else 0.0
        )

        # 4. Fact-Oriented Sentence Ratio - evidential qualifiers per sentence
        evidential_sentences = 0
        if text_sentences:
            for sentence in text_sentences:
                sentence_lower = sentence.lower()
                if any(phrase in sentence_lower for phrase in EVIDENTIAL_PHRASES):
                    evidential_sentences += 1
        features["ling_fact_sentence_ratio"] = evidential_sentences / max(
            len(text_sentences), 1
        )

        # 6. Factivity Verbs Norm - evidential/factivity verb rate
        factivity_count = sum(1 for token in tokens if token in FACTIVITY_VERBS)
        features["ling_factivity_verbs_norm"] = factivity_count / max(len(tokens), 1)

        # 7. Time Anchor Norm - specific time anchors rate
        features["ling_time_anchor_norm"] = len(time_ref_matches) / max(len(tokens), 1)

        # 8. Third Person Ratio - third person vs first person mentions
        third_person_count = sum(
            1 for token in tokens if token in THIRD_PERSON_PRONOUNS
        )
        first_person_count = sum(
            1 for token in tokens if token in FIRST_PERSON_PRONOUNS
        )
        features["ling_third_person_ratio"] = (third_person_count + 0.0001) / (
            first_person_count + 0.0001
        )

        # 8. Passive vs Active Voice Balance - ratio of passive to total voice
        active_matches = ACTIVE_VOICE_PATTERN.findall(text)
        active_count = len(active_matches)
        total_voice = len(passive_matches) + active_count
        features["ling_passive_active_balance"] = len(passive_matches) / max(
            total_voice, 1
        )

        # 9. Negation Near Risk Rate - negations within ±3 tokens of any risk terms
        all_risk_terms: Set[str] = set()
        for lexicon_terms in RISK_LEXICONS.values():
            all_risk_terms.update(lexicon_terms)
        negation_near_risk = _window_proximity_count(
            tokens, NEG_WORDS, all_risk_terms, radius=3
        )
        risk_term_count = sum(1 for token in tokens if token in all_risk_terms)
        features["ling_negation_near_risk_rate"] = negation_near_risk / max(
            risk_term_count, 1
        )

        # NEW 20 ENHANCED FEATURES

        # 1. Date Marker Norm - explicit dates/periods normalized by tokens
        date_matches = DATE_MARKER_PATTERN.findall(text)
        features["ling_date_marker_norm"] = len(date_matches) / max(len(tokens), 1)

        # 4. Intensifier Norm - adverbial intensifiers density
        intensifier_count = sum(1 for token in tokens if token in INTENSIFIER_TERMS)
        features["ling_intensifier_norm"] = intensifier_count / max(len(tokens), 1)

        # 5. Vague Quantifier Norm - vague quantifiers density
        vague_quantifier_count = sum(
            1 for token in tokens if token in VAGUE_QUANTIFIER_TERMS
        )
        features["ling_vague_quantifier_norm"] = vague_quantifier_count / max(
            len(tokens), 1
        )

        # 6. Negated Claim Bigram Norm - exact bigrams like "no guarantee"
        negated_claim_matches = NEGATED_CLAIM_PATTERN.findall(text)
        features["ling_negated_claim_bigram_norm"] = len(negated_claim_matches) / max(
            len(tokens), 1
        )

        # 7. We Future Commit Norm - first-person future commitments
        we_future_matches = WE_FUTURE_COMMIT_PATTERN.findall(text)
        features["ling_we_future_commit_norm"] = len(we_future_matches) / max(
            len(tokens), 1
        )

        # 8. External Authority Ref Norm - regulator/court mentions
        authority_count = sum(
            1 for token in tokens if token in EXTERNAL_AUTHORITY_TERMS
        )
        features["ling_external_authority_ref_norm"] = authority_count / max(
            len(tokens), 1
        )

        # 9. Definition Marker Norm - legalese definers
        definition_matches = DEFINITION_MARKER_PATTERN.findall(text)
        features["ling_definition_marker_norm"] = len(definition_matches) / max(
            len(tokens), 1
        )

        # 10. Sentence Initial Hedge Share - sentences starting with hedges
        hedge_initial_sentences = 0
        if text_sentences:
            for sentence in text_sentences:
                if SENTENCE_INITIAL_HEDGE_PATTERN.search(sentence.strip()):
                    hedge_initial_sentences += 1
        features["ling_sentence_initial_hedge_share"] = hedge_initial_sentences / max(
            len(text_sentences), 1
        )

        # 8. Fact vs Opinion Ratio - evidential vs superlatives/hedges
        hedge_count = sum(
            1 for token in tokens if token in RISK_LEXICONS.get("hedges", set())
        )
        superlative_count = sum(
            1 for token in tokens if token in RISK_LEXICONS.get("superlatives", set())
        )
        evidential_count = sum(
            1 for phrase in EVIDENTIAL_PHRASES if phrase in text.lower()
        )
        fact_count = evidential_count
        opinion_count = hedge_count + superlative_count
        features["ling_fact_opinion_ratio"] = (fact_count + 0.0001) / (
            opinion_count + fact_count + 0.0001
        )

        # 7. Clarity Lexicon Density - normalized clarity terms
        clarity_count = sum(
            1 for token in tokens if token in RISK_LEXICONS.get("clarity", set())
        )
        features["ling_clarity_density"] = clarity_count / max(len(tokens), 1)

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

        # Social media-specific structural features (NEW)
        features["struct_emoji_count"] = len(EMOJI_PATTERN.findall(text))
        features["struct_hashtag_count"] = len(HASHTAG_PATTERN.findall(text))
        features["struct_mention_count"] = len(MENTION_PATTERN.findall(text))
        features["struct_link_presence"] = int(bool(URL_PATTERN.search(text)))

        # Disclaimer hashtags (compliance signals)
        hashtags = HASHTAG_PATTERN.findall(text.lower())
        disclaimer_tags = [
            tag
            for tag in hashtags
            if any(disc in tag for disc in ["ad", "sponsor", "disclaim", "terms"])
        ]
        features["struct_disclaimer_hashtag"] = int(bool(disclaimer_tags))

        # Readability approximation (Flesch-Kincaid inspired)
        if sentences and len(text.split()) > 0:
            words = text.split()
            syllables = sum(_count_syllables(w) for w in words)
            features["struct_flesch_score"] = (
                206.835
                - 1.015 * (len(words) / len(sentences))
                - 84.6 * (syllables / len(words))
            )
        else:
            features["struct_flesch_score"] = 0.0

        # Disclaimer cluster density (NEW) - clustered disclaimers indicate low-risk
        tokens = text.lower().split()  # Simple tokenization for position tracking
        disclaimer_terms = set()
        for phrase in SAFE_HARBOR_PHRASES + NON_ADVICE_PHRASES:
            disclaimer_terms.update(phrase.split())

        disclaimer_positions = [
            i
            for i, token in enumerate(tokens)
            if any(d in token for d in disclaimer_terms)
        ]

        if len(disclaimer_positions) > 1:
            # Low std = clustered (good for low-risk)
            cluster_density = np.std(disclaimer_positions) / max(len(tokens), 1)
            features["lr_disclaimer_cluster"] = max(
                0.0, 1.0 - cluster_density
            )  # High = clustered
        elif len(disclaimer_positions) == 1:
            features["lr_disclaimer_cluster"] = (
                0.5  # Single disclaimer = medium clustering
            )
        else:
            features["lr_disclaimer_cluster"] = 0.0  # No disclaimers = no clustering

        # NEW STRUCTURAL FEATURES

        # 5. Enumeration Density - lists/bullets normalized by length
        enum_matches = ENUMERATION_PATTERN.findall(text)
        features["struct_enumeration_density"] = len(enum_matches) / max(len(text), 1)

        # 10. Entity Attribution Density - sentences with attribution patterns
        attribution_sentences = 0
        if sentences:
            for sentence in sentences:
                if ATTRIBUTION_PATTERN.search(sentence) or re.search(
                    r'"[^"]+"', sentence
                ):
                    attribution_sentences += 1
        features["struct_entity_attribution_density"] = attribution_sentences / max(
            len(sentences), 1
        )

        # 5. Regulatory Reference Norm - broader regulatory references
        reg_ref_matches = REGULATORY_REF_PATTERN.findall(text)
        features["struct_regulatory_reference_norm"] = len(reg_ref_matches) / max(
            len(text.split()), 1
        )

        # NEW STRUCTURAL FEATURES

        # 2. Policy Reference Norm - policy/procedure terms density
        policy_ref_matches = POLICY_REFERENCE_PATTERN.findall(text)
        features["struct_policy_reference_norm"] = len(policy_ref_matches) / max(
            len(text.split()), 1
        )

        # 3. Range Expression Norm - bounded ranges density
        range_matches = RANGE_EXPRESSION_PATTERN.findall(text)
        features["struct_range_expression_norm"] = len(range_matches) / max(
            len(text.split()), 1
        )

        # 5. Enumeration in Disclaimers - enumerations within disclaimer sentences
        disclaimer_terms = {
            "disclaimer",
            "not intended",
            "subject to",
            "except",
            "excluding",
            "limitations apply",
            "terms and conditions",
            "not liable",
            "no warranty",
        }
        disclaimer_sentences = [
            s for s in sentences if any(disc in s.lower() for disc in disclaimer_terms)
        ]
        enum_in_disclaimers = sum(
            len(ENUMERATION_PATTERN.findall(s)) for s in disclaimer_sentences
        )
        features["struct_enum_in_disclaimers"] = enum_in_disclaimers / max(
            len(sentences), 1
        )

        return features

    def _extract_low_risk_signals(
        self,
        text: str,
        tokens: List[str],
        quote_tokens: List[str],
        base_features: Dict[str, float],
    ) -> Dict[str, float]:
        """Extract signals that typically characterize low-risk (Class 0) quotes."""
        features = {}
        n_tokens = max(len(tokens), 1)
        text_lower = text.lower()

        # 1. Safe harbor / cautionary boilerplate
        safe_harbor_count = _count_phrases(text_lower, SAFE_HARBOR_PHRASES)
        features["lr_safe_harbor_count"] = safe_harbor_count
        features["lr_safe_harbor_norm"] = safe_harbor_count / n_tokens
        features["lr_safe_harbor_present"] = int(safe_harbor_count > 0)

        # 2. Non-advice disclaimers
        non_advice_count = _count_phrases(text_lower, NON_ADVICE_PHRASES)
        features["lr_non_advice_count"] = non_advice_count
        features["lr_non_advice_norm"] = non_advice_count / n_tokens
        features["lr_non_advice_present"] = int(non_advice_count > 0)

        # 3. Evidential attribution (phrases + URLs + SEC refs)
        evidential_count = _count_phrases(text_lower, EVIDENTIAL_PHRASES)
        url_count = len(URL_PATTERN.findall(text))
        sec_count = len(SEC_REF_PATTERN.findall(text))
        total_evidential = evidential_count + url_count + sec_count

        features["lr_evidential_count"] = evidential_count
        features["lr_url_count"] = url_count
        features["lr_sec_ref_count"] = sec_count
        features["lr_evidential_norm"] = total_evidential / n_tokens
        features["lr_evidential_present"] = int(total_evidential > 0)

        # 4. Negated guarantees (negation within ±3 tokens of guarantee terms)
        guarantee_terms = RISK_LEXICONS.get("guarantee", set())
        negated_guarantee_count = _window_proximity_count(
            tokens, NEG_WORDS, guarantee_terms, radius=3
        )
        features["lr_negated_guarantee_count"] = negated_guarantee_count
        features["lr_negated_guarantee_norm"] = negated_guarantee_count / n_tokens

        # 5. Hedge near guarantee (cautious commitments)
        hedge_terms = RISK_LEXICONS.get("hedges", set())
        hedge_near_guarantee = _window_proximity_count(
            tokens, hedge_terms, guarantee_terms, radius=3
        )
        features["lr_hedge_near_guarantee_count"] = hedge_near_guarantee
        features["lr_hedge_near_guarantee_norm"] = hedge_near_guarantee / n_tokens

        # 6. Conditional sentence share
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if sentences:
            conditional_sentences = sum(
                1
                for s in sentences
                if re.search(r"\b(if|unless|provided|assuming|subject to)\b", s, re.I)
            )
            features["lr_conditional_sentence_share"] = conditional_sentences / len(
                sentences
            )
        else:
            features["lr_conditional_sentence_share"] = 0.0

        # 7. Disclaimer end-position (last 20% of tokens)
        if n_tokens > 5:
            tail_start = int(0.8 * n_tokens)
            tail_tokens = tokens[tail_start:]
            tail_text = " ".join(tail_tokens)
            tail_disclaimers = _count_phrases(
                tail_text, SAFE_HARBOR_PHRASES + NON_ADVICE_PHRASES
            )
            features["lr_disclaimer_endpos_present"] = int(tail_disclaimers > 0)
        else:
            features["lr_disclaimer_endpos_present"] = 0

        # 8. Quote vs overall hedge contrast (speaker hedges more than context)
        hedge_norm_overall = base_features.get("lex_hedges_norm", 0.0)
        hedge_norm_quote = base_features.get("lex_hedges_quote_norm", 0.0)
        features["lr_quote_minus_overall_hedge"] = hedge_norm_quote - hedge_norm_overall

        # 9. Scope limiter density
        scope_limiters = RISK_LEXICONS.get("scope_limiters", set())
        scope_count = sum(1 for token in tokens if token in scope_limiters)
        features["lr_scope_limiter_count"] = scope_count
        features["lr_scope_limiter_norm"] = scope_count / n_tokens

        # 10. Future with caution proximity (softened forecasts)
        future_with_caution = _window_proximity_count(
            tokens, FUTURE_MODALS, hedge_terms.union(scope_limiters), radius=3
        )
        features["lr_future_with_caution_count"] = future_with_caution
        features["lr_future_with_caution_norm"] = future_with_caution / n_tokens

        # 11. Entity and quotation patterns (provenance signals)
        entity_pattern = re.compile(
            r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b"
        )  # Proper nouns
        quote_pattern = re.compile(r'"[^"]+"|\'[^\']+\'')

        entity_count = len(entity_pattern.findall(text))
        quote_count = len(quote_pattern.findall(text))

        features["lr_entity_count"] = entity_count
        features["lr_quote_presence"] = int(quote_count > 0)
        features["lr_sourced_ratio"] = (quote_count + 0.0001) / (entity_count + 0.0001)

        # 12. Compliance vs risk balance indicators
        compliance_norm = base_features.get("lex_compliance_norm", 0.0)
        deception_norm = base_features.get("lex_deception_norm", 0.0)
        features["lr_compliance_vs_deception"] = (compliance_norm + 0.0001) / (
            deception_norm + 0.0001
        )

        # 13. Scoped-claim proximity (NEW) - scope limiters near strong claims
        guarantee_terms = RISK_LEXICONS.get("guarantee", set())
        superlative_terms = RISK_LEXICONS.get("superlatives", set())
        scope_limiters = RISK_LEXICONS.get("scope_limiters", set())

        # Scope limiters within ±5 tokens of guarantees
        scope_near_guarantee = _window_proximity_count(
            tokens, scope_limiters, guarantee_terms, radius=5
        )
        features["lr_scope_near_guarantee_count"] = scope_near_guarantee
        features["lr_scope_near_guarantee_norm"] = scope_near_guarantee / n_tokens

        # Scope limiters within ±5 tokens of superlatives
        scope_near_superlative = _window_proximity_count(
            tokens, scope_limiters, superlative_terms, radius=5
        )
        features["lr_scope_near_superlative_count"] = scope_near_superlative
        features["lr_scope_near_superlative_norm"] = scope_near_superlative / n_tokens

        # 14. Passive voice near claims (NEW) - avoids responsibility
        passive_terms = {"was", "were", "been", "being", "is", "are", "am"}
        claim_terms = guarantee_terms.union(superlative_terms)
        passive_near_claims = _window_proximity_count(
            tokens, passive_terms, claim_terms, radius=5
        )
        features["lr_passive_near_claims_count"] = passive_near_claims
        features["lr_passive_near_claims_norm"] = passive_near_claims / n_tokens

        # 15. Evidential strength score (NEW) - weighted evidential attribution
        evidential_weights = {
            "independent": 2.0,
            "third-party": 1.5,
            "study": 1.2,
            "report": 1.0,
            "survey": 1.1,
            "according": 1.0,
            "data": 1.0,
        }
        evidential_strength = 0.0
        for phrase in EVIDENTIAL_PHRASES:
            if phrase in text_lower:
                first_word = phrase.split()[0] if phrase.split() else ""
                weight = evidential_weights.get(first_word, 1.0)
                evidential_strength += weight

        # Weight external references higher
        evidential_strength += 1.5 * url_count + 2.0 * sec_count

        features["lr_evidential_strength"] = evidential_strength
        features["lr_evidential_strength_norm"] = evidential_strength / n_tokens

        # 16. Uncautioned-Claim Proportion (UCP) (NEW) - separate High (2) from Medium/Low
        # Among all strong claims, what proportion lack hedge/conditional/scope within ±5 tokens and are not negated
        strong_claim_terms = RISK_LEXICONS.get("guarantee", set()).union(
            RISK_LEXICONS.get("superlatives", set()),
            RISK_LEXICONS.get("pricing_claims", set()),
        )
        caution_terms = RISK_LEXICONS.get("hedges", set()).union(
            RISK_LEXICONS.get("scope_limiters", set())
        )

        claim_indices = [i for i, t in enumerate(tokens) if t in strong_claim_terms]
        caution_indices = [i for i, t in enumerate(tokens) if t in caution_terms]
        neg_indices = [i for i, t in enumerate(tokens) if t in NEG_WORDS]

        def is_near(pos: int, indices: List[int], radius: int = 5) -> bool:
            return any(abs(idx - pos) <= radius for idx in indices)

        uncautioned_claims = sum(
            1
            for i in claim_indices
            if not is_near(i, caution_indices, 5) and not is_near(i, neg_indices, 5)
        )

        ucp_rate = uncautioned_claims / max(len(claim_indices), 1)
        features["lr_uncautioned_claim_proportion"] = ucp_rate
        features["lr_uncautioned_claim_count"] = uncautioned_claims

        # 17. Documented-Fact Sentence Share (DFSS) (NEW) - bolster Low (0) vs {1,2}
        # Fraction of sentences with (number+unit) AND (past-tense or evidential cue)
        unit_num_pattern = re.compile(
            r"(\$[\d,]+(?:\.\d+)?|\d+(?:,\d{3})*(?:\.\d+)?%|\d+(?:,\d{3})*(?:\.\d+)?\s*(?:bps?|bp|m|bn|million|billion))",
            re.I,
        )
        reporting_pattern = re.compile(
            r"\b(reported|stated|said|according to|as reported by|showed|found|indicated)\b",
            re.I,
        )

        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        documented_fact_sentences = 0

        if sentences:
            for sentence in sentences:
                has_unit_number = bool(unit_num_pattern.search(sentence))
                has_past_tense = bool(TENSE_PATTERNS["past"].search(sentence))
                has_reporting = bool(reporting_pattern.search(sentence))
                has_sec_url = bool(
                    SEC_REF_PATTERN.search(sentence) or URL_PATTERN.search(sentence)
                )

                if has_unit_number and (has_past_tense or has_reporting or has_sec_url):
                    documented_fact_sentences += 1

        dfss_rate = documented_fact_sentences / max(len(sentences), 1)
        features["lr_documented_fact_sentence_share"] = dfss_rate
        features["lr_documented_fact_sentence_count"] = documented_fact_sentences

        # 18. Remediation-Action Proximity (RAP) (NEW) - clarify Medium (1) vs High (2)
        # Count remediation verbs within ±7 tokens of risk cues
        risk_near_terms = RISK_LEXICONS.get("deception", set()).union(
            RISK_LEXICONS.get("scienter", set()),
            RISK_LEXICONS.get("guarantee", set()),
            RISK_LEXICONS.get("superlatives", set()),
        )

        remediation_indices = [
            i for i, t in enumerate(tokens) if t in REMEDIATION_TERMS
        ]
        risk_indices = [i for i, t in enumerate(tokens) if t in risk_near_terms]

        remediation_proximity_count = sum(
            1 for i in risk_indices if any(abs(j - i) <= 7 for j in remediation_indices)
        )

        features["lr_remediation_action_proximity_count"] = remediation_proximity_count
        features["lr_remediation_action_proximity_norm"] = (
            remediation_proximity_count / n_tokens
        )

        # 19. Qualified Argument Count (NEW) - claims with backing/qualifiers
        # Count claims that have qualifiers/causal reasoning within ±5 tokens
        all_claim_terms = strong_claim_terms.union(CLAIM_MARKERS)
        qualifier_indices = [i for i, t in enumerate(tokens) if t in QUALIFIER_MARKERS]
        claim_indices_all = [i for i, t in enumerate(tokens) if t in all_claim_terms]

        qualified_arguments = sum(
            1
            for i in claim_indices_all
            if any(abs(j - i) <= 5 for j in qualifier_indices)
        )

        features["lr_qualified_argument_count"] = qualified_arguments
        features["lr_qualified_argument_norm"] = qualified_arguments / n_tokens

        # NEW LOW-RISK FEATURES

        # 3. Claims Scoped Proportion - strong claims with scope limiters nearby
        strong_claim_terms = RISK_LEXICONS.get("guarantee", set()).union(
            RISK_LEXICONS.get("superlatives", set())
        )
        scope_limiters = RISK_LEXICONS.get("scope_limiters", set())

        claim_indices = [i for i, t in enumerate(tokens) if t in strong_claim_terms]
        scope_indices = [i for i, t in enumerate(tokens) if t in scope_limiters]

        scoped_claims = sum(
            1 for i in claim_indices if any(abs(j - i) <= 3 for j in scope_indices)
        )
        features["lr_claims_scoped_proportion"] = scoped_claims / max(
            len(claim_indices), 1
        )

        # 10. Evidential Link Density - evidential phrases near URLs
        link_terms = {"http", "www", "url", "link", "website"}
        link_indices = [
            i
            for i, t in enumerate(tokens)
            if any(link in t.lower() for link in link_terms)
        ]
        evidential_indices = [
            i
            for i, t in enumerate(tokens)
            if any(phrase.split()[0] == t for phrase in EVIDENTIAL_PHRASES)
        ]

        evidential_near_links = sum(
            1 for i in evidential_indices if any(abs(j - i) <= 5 for j in link_indices)
        )
        features["lr_evidential_link_density"] = evidential_near_links / n_tokens

        return features

    def _extract_derived_features(
        self, base_features: Dict[str, float]
    ) -> Dict[str, float]:
        """Extract derived features (ratios and interactions) from base features."""
        derived = {}
        EPS = 1e-4  # Small epsilon to avoid division by zero

        # Extract normalized lexicon values
        guarantee_norm = base_features.get("lex_guarantee_norm", 0.0)
        hedges_norm = base_features.get("lex_hedges_norm", 0.0)
        deception_norm = base_features.get("lex_deception_norm", 0.0)
        superlatives_present = base_features.get("lex_superlatives_present", 0.0)
        high_certainty = base_features.get("ling_high_certainty", 0.0)

        # Ratios (directional, unit-free)
        derived["ratio_guarantee_vs_hedge"] = (guarantee_norm + EPS) / (
            hedges_norm + EPS
        )

        derived["ratio_deception_vs_hedge"] = (deception_norm + EPS) / (
            hedges_norm + EPS
        )

        derived["ratio_guarantee_vs_superlative"] = (guarantee_norm + EPS) / (
            superlatives_present + EPS
        )

        # Interactions (hypothesized nonlinearity compressible into LR)
        derived["interact_guarantee_x_cert"] = guarantee_norm * high_certainty
        derived["interact_superlative_x_cert"] = superlatives_present * high_certainty
        derived["interact_hedge_x_guarantee"] = hedges_norm * guarantee_norm

        # NEW: Low-risk composite ratios for class 0 discrimination
        # Disclaimer vs guarantee ratio (higher = safer)
        disclaimer_total = (
            base_features.get("lr_safe_harbor_norm", 0.0)
            + base_features.get("lr_non_advice_norm", 0.0)
            + base_features.get("lr_evidential_norm", 0.0)
        )
        derived["ratio_disclaimer_vs_guarantee"] = (disclaimer_total + EPS) / (
            guarantee_norm + EPS
        )

        # Cautioned future vs uncautioned future
        future_with_caution = base_features.get("lr_future_with_caution_norm", 0.0)
        future_certainty = base_features.get("ling_future_tense_count", 0.0) / max(
            len(base_features), 1
        )
        derived["ratio_future_cautioned"] = (future_with_caution + EPS) / (
            future_certainty + EPS
        )

        # Negated guarantees share of total guarantees
        negated_guarantee = base_features.get("lr_negated_guarantee_norm", 0.0)
        derived["ratio_negated_guarantee"] = (negated_guarantee + EPS) / (
            guarantee_norm + EPS
        )

        # Compliance vs deception balance (low-risk indicator)
        compliance_norm = base_features.get("lex_compliance_norm", 0.0)
        derived["ratio_compliance_vs_deception"] = (compliance_norm + EPS) / (
            deception_norm + EPS
        )

        # Conditional language saturation
        conditional_share = base_features.get("lr_conditional_sentence_share", 0.0)
        derived["ratio_conditional_saturation"] = conditional_share

        # Scope limitation vs strong claims
        scope_norm = base_features.get("lr_scope_limiter_norm", 0.0)
        strong_claims = guarantee_norm + superlatives_present
        derived["ratio_scope_vs_claims"] = (scope_norm + EPS) / (strong_claims + EPS)

        # NEW DERIVED FEATURES

        # 2. Modal Hedge Ratio - low certainty vs high certainty modals
        low_certainty = base_features.get("ling_low_certainty", 0.0)
        high_certainty = base_features.get("ling_high_certainty", 0.0)
        derived["ratio_modal_hedge_ratio"] = (low_certainty + EPS) / (
            high_certainty + EPS
        )

        # 7. Ambiguity vs Compliance Ratio
        ambiguity_norm = base_features.get("lex_ambiguity_norm", 0.0)
        derived["ratio_ambiguity_vs_compliance"] = (ambiguity_norm + EPS) / (
            compliance_norm + EPS
        )

        # NEW: Policy vs Obligation Balance (policy should reduce obligation risk)
        policy_norm = base_features.get("lex_policy_procedure_norm", 0.0)
        obligation_norm = base_features.get("lex_obligation_norm", 0.0)
        derived["ratio_policy_vs_obligation"] = (policy_norm + EPS) / (
            obligation_norm + EPS
        )

        # NEW: Permission vs Obligation Balance (permission should balance obligation)
        permission_norm = base_features.get("lex_permission_norm", 0.0)
        derived["ratio_permission_vs_obligation"] = (permission_norm + EPS) / (
            obligation_norm + EPS
        )

        # NEW: Clarity vs Ambiguity Ratio (clarity should reduce ambiguity risk)
        clarity_norm = base_features.get("lex_clarity_norm", 0.0)
        derived["ratio_clarity_vs_ambiguity"] = (clarity_norm + EPS) / (
            ambiguity_norm + EPS
        )

        # NEW 20 ENHANCED DERIVED FEATURES

        # 10. Policy vs Ambiguity Ratio
        derived["ratio_policy_vs_ambiguity"] = (policy_norm + EPS) / (
            ambiguity_norm + EPS
        )

        # Bonus ratios for better discrimination
        # Policy vs Claims Ratio
        strong_claims = guarantee_norm + superlatives_present
        derived["ratio_policy_vs_claims"] = (policy_norm + EPS) / (strong_claims + EPS)

        # Dates vs Commit Ratio
        date_norm = base_features.get("ling_date_marker_norm", 0.0)
        we_commit_norm = base_features.get("ling_we_future_commit_norm", 0.0)
        derived["ratio_dates_vs_commit"] = (date_norm + EPS) / (we_commit_norm + EPS)

        # Additional coverage-focused ratios
        # Range vs Intensifier (cautious vs aggressive language)
        range_norm = base_features.get("struct_range_expression_norm", 0.0)
        intensifier_norm = base_features.get("ling_intensifier_norm", 0.0)
        derived["ratio_range_vs_intensifier"] = (range_norm + EPS) / (
            intensifier_norm + EPS
        )

        # Vague vs Definitive (cautious vs specific)
        vague_norm = base_features.get("ling_vague_quantifier_norm", 0.0)
        definition_norm = base_features.get("ling_definition_marker_norm", 0.0)
        derived["ratio_vague_vs_definitive"] = (vague_norm + EPS) / (
            definition_norm + EPS
        )

        return derived

    def _extract_new_20_features(
        self,
        text: str,
        tokens: List[str],
        text_tokens: List[str],
        base_features: Dict[str, float],
    ) -> Dict[str, float]:
        """Extract the 20 new interpretable features."""
        features = {}
        n_tokens = max(len(tokens), 1)
        text_lower = text.lower()

        # Helper function for proximity analysis
        def _proximity_count(
            tokens: List[str], set_a: set, set_b: set, radius: int = 3
        ) -> int:
            return _window_proximity_count(tokens, set_a, set_b, radius)

        # --- FIRST SET OF 10 FEATURES ---

        # 1. Deception Near First Person Proximity
        deception_terms = RISK_LEXICONS.get("deception", set())
        deception_near_firstperson = _proximity_count(
            tokens, deception_terms, FIRST_PERSON_PRONOUNS, radius=3
        )
        features["new_deception_near_first_person_count"] = deception_near_firstperson
        features["new_deception_near_first_person_norm"] = (
            deception_near_firstperson / n_tokens
        )

        # 2. Number Near Guarantee Proximity
        guarantee_terms = RISK_LEXICONS.get("guarantee", set())
        numeric_tokens = [i for i, t in enumerate(tokens) if re.search(r"\d", t)]
        guarantee_indices = [i for i, t in enumerate(tokens) if t in guarantee_terms]

        number_near_guarantee = 0
        for num_idx in numeric_tokens:
            if any(abs(num_idx - g_idx) <= 5 for g_idx in guarantee_indices):
                number_near_guarantee += 1

        features["new_number_near_guarantee_count"] = number_near_guarantee
        features["new_number_near_guarantee_norm"] = number_near_guarantee / max(
            len(guarantee_indices), 1
        )

        # 3. Caps Near Superlatives Density
        superlative_terms = RISK_LEXICONS.get("superlatives", set())
        superlative_indices = [
            i for i, t in enumerate(tokens) if t in superlative_terms
        ]

        caps_near_superlatives = 0
        for sup_idx in superlative_indices:
            window_start = max(0, sup_idx - 3)
            window_end = min(len(tokens), sup_idx + 4)
            for i in range(window_start, window_end):
                if i < len(tokens) and tokens[i].isupper():
                    caps_near_superlatives += 1
                    break  # Count once per superlative

        features["new_caps_near_superlatives_count"] = caps_near_superlatives
        features["new_caps_near_superlatives_norm"] = caps_near_superlatives / max(
            len(superlative_indices), 1
        )

        # 4. Contrast Discourse Near Deception
        contrast_near_deception = _proximity_count(
            tokens, CONTRAST_DISCOURSE_MARKERS, deception_terms, radius=4
        )
        features["new_contrast_discourse_near_deception_count"] = (
            contrast_near_deception
        )
        features["new_contrast_discourse_near_deception_norm"] = (
            contrast_near_deception
            / max(len([t for t in tokens if t in deception_terms]), 1)
        )

        # 5. Low Certainty Modal Clustering
        low_certainty_modals = RISK_LEXICONS.get("hedges", set()).intersection(
            {"may", "might", "could", "possibly"}
        )
        modal_positions = [i for i, t in enumerate(tokens) if t in low_certainty_modals]

        if len(modal_positions) > 1:
            # Calculate clustering as inverse of standard deviation of positions
            clustering_score = 1.0 - min(1.0, np.std(modal_positions) / n_tokens)
        elif len(modal_positions) == 1:
            clustering_score = 0.5  # Medium clustering for single modal
        else:
            clustering_score = 0.0

        features["new_low_certainty_modal_clustering"] = clustering_score

        # 6. Hedges Near Future Tense Proximity
        hedge_terms = RISK_LEXICONS.get("hedges", set())
        future_indices = [i for i, t in enumerate(tokens) if t in FUTURE_MODALS]
        hedge_indices = [i for i, t in enumerate(tokens) if t in hedge_terms]

        hedges_near_future = sum(
            1
            for h_idx in hedge_indices
            if any(abs(h_idx - f_idx) <= 3 for f_idx in future_indices)
        )
        features["new_hedges_near_future_tense_count"] = hedges_near_future
        features["new_hedges_near_future_tense_norm"] = hedges_near_future / max(
            len(future_indices), 1
        )

        # 7. Sentence Length Variation Near Risks
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        # Get sentences containing risk terms
        all_risk_terms = set()
        for lexicon_terms in RISK_LEXICONS.values():
            all_risk_terms.update(lexicon_terms)

        risk_sentence_lengths = []
        all_sentence_lengths = []
        for sentence in sentences:
            sent_tokens = self._tokenize(sentence)
            all_sentence_lengths.append(len(sent_tokens))
            if any(token in all_risk_terms for token in sent_tokens):
                risk_sentence_lengths.append(len(sent_tokens))

        if risk_sentence_lengths and all_sentence_lengths:
            avg_risk_length = np.mean(risk_sentence_lengths)
            avg_all_length = np.mean(all_sentence_lengths)
            features["new_sentence_length_variation_near_risks"] = (
                avg_risk_length / max(avg_all_length, 1)
            )
        else:
            features["new_sentence_length_variation_near_risks"] = 1.0

        # 8. Neutral to Compliance Transition Rate
        compliance_terms = RISK_LEXICONS.get("compliance", set())
        neutral_to_compliance_transitions = 0
        total_transitions = 0

        for i in range(len(tokens) - 1):
            curr_cats = self._get_token_categories(tokens[i])
            next_cats = self._get_token_categories(tokens[i + 1])

            if "neutral" in curr_cats and "compliance" in next_cats:
                neutral_to_compliance_transitions += 1
            total_transitions += 1

        features["new_neutral_to_compliance_transition_rate"] = (
            neutral_to_compliance_transitions / max(total_transitions, 1)
        )

        # 9. Word Count in Conditional Sentences
        conditional_word_count = 0
        total_word_count = len(tokens)

        conditional_starts = ["if", "unless", "provided", "assuming", "whether"]
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if any(sentence_lower.startswith(cond) for cond in conditional_starts):
                conditional_word_count += len(sentence.split())

        features["new_word_count_in_conditional_sentences"] = conditional_word_count
        features["new_conditional_word_share"] = conditional_word_count / max(
            total_word_count, 1
        )

        # 10. Deception to Contrast Transition Frequency
        deception_to_contrast_transitions = 0
        deception_count = len([t for t in tokens if t in deception_terms])

        for i in range(len(tokens) - 1):
            curr_cats = self._get_token_categories(tokens[i])
            next_cats = self._get_token_categories(tokens[i + 1])

            if "deception" in curr_cats and any(
                tokens[i + 1] in CONTRAST_DISCOURSE_MARKERS
                for marker in CONTRAST_DISCOURSE_MARKERS
            ):
                deception_to_contrast_transitions += 1

        features["new_deception_to_contrast_transition_count"] = (
            deception_to_contrast_transitions
        )
        features["new_deception_to_contrast_transition_norm"] = (
            deception_to_contrast_transitions / max(deception_count, 1)
        )

        # --- SECOND SET OF 10 FEATURES ---

        # 1. num_approx_modifier_proximity
        numeric_matches = NUMERIC_PATTERN.findall(text)
        approx_near_numbers = 0

        for match in numeric_matches:
            # Find position of this number in text
            match_pos = text.find(match)
            if match_pos != -1:
                # Check surrounding context (±20 characters)
                start = max(0, match_pos - 20)
                end = min(len(text), match_pos + len(match) + 20)
                context = text[start:end].lower()

                if any(approx_word in context for approx_word in APPROXIMATION_WORDS):
                    approx_near_numbers += 1

        features["new_num_approx_modifier_proximity"] = approx_near_numbers
        features["new_num_approx_modifier_proximity_norm"] = approx_near_numbers / max(
            len(numeric_matches) * 100, 1
        )

        # 2. num_range_expression_count
        range_matches = RANGE_PATTERN.findall(text)
        features["new_num_range_expression_count"] = len(range_matches)
        features["new_num_range_expression_norm"] = len(range_matches) / max(
            n_tokens / 100, 1
        )

        # 3. percent_share_of_numbers
        percent_matches = PERCENTAGE_PATTERN.findall(text)
        total_numeric = len(numeric_matches) + len(percent_matches)
        features["new_percent_share_of_numbers"] = len(percent_matches) / max(
            total_numeric, 1
        )

        # 4. hedged_number_ratio
        hedge_near_number = 0
        for match in numeric_matches:
            match_pos = text.find(match)
            if match_pos != -1:
                start = max(0, match_pos - 15)
                end = min(len(text), match_pos + len(match) + 15)
                context = text[start:end].lower()

                if any(hedge in context for hedge in hedge_terms):
                    hedge_near_number += 1

        hedged_total = approx_near_numbers + len(range_matches) + hedge_near_number
        features["new_hedged_number_ratio"] = hedged_total / max(total_numeric, 1)

        # 5. date_specificity_index
        explicit_dates = EXPLICIT_DATE_PATTERN.findall(text)
        vague_time_refs = VAGUE_TIME_PATTERN.findall(text)
        total_time_refs = len(explicit_dates) + len(vague_time_refs)

        features["new_date_specificity_index"] = len(explicit_dates) / max(
            total_time_refs, 1
        )

        # 6. parenthetical_token_share
        parenthetical_content = re.findall(r"\(([^)]+)\)", text)
        parenthetical_tokens = sum(
            len(content.split()) for content in parenthetical_content
        )
        features["new_parenthetical_token_share"] = parenthetical_tokens / max(
            n_tokens / 100, 1
        )

        # 7. clause_initial_conditional_share
        conditional_initial_sentences = 0
        conditional_starters = re.compile(
            r"^(if|unless|provided that|subject to|assuming)\b", re.I
        )

        for sentence in sentences:
            if conditional_starters.search(sentence.strip()):
                conditional_initial_sentences += 1

        features["new_clause_initial_conditional_share"] = (
            conditional_initial_sentences / max(len(sentences), 1)
        )

        # 8. contrast_then_hedge_count
        contrast_then_hedge = 0

        for sentence in sentences:
            sentence_tokens = self._tokenize(sentence)
            contrast_positions = [
                i
                for i, t in enumerate(sentence_tokens)
                if t in CONTRAST_DISCOURSE_MARKERS
            ]
            hedge_positions = [
                i for i, t in enumerate(sentence_tokens) if t in hedge_terms
            ]

            # Check if any contrast marker comes before any hedge in the same sentence
            for c_pos in contrast_positions:
                if any(h_pos > c_pos for h_pos in hedge_positions):
                    contrast_then_hedge += 1
                    break  # Count once per sentence

        features["new_contrast_then_hedge_count"] = contrast_then_hedge
        features["new_contrast_then_hedge_norm"] = contrast_then_hedge / max(
            len(sentences), 1
        )

        # 9. negated_superlative_proximity
        superlative_terms = RISK_LEXICONS.get("superlatives", set())
        negated_superlatives = _proximity_count(
            tokens, NEG_WORDS, superlative_terms, radius=3
        )
        features["new_negated_superlative_proximity"] = negated_superlatives
        features["new_negated_superlative_proximity_norm"] = negated_superlatives / max(
            n_tokens / 100, 1
        )

        # 10. attribution_verb_density
        attribution_count = sum(1 for token in tokens if token in ATTRIBUTION_VERBS)
        features["new_attribution_verb_density"] = attribution_count / max(
            n_tokens / 100, 1
        )

        return features

    def _extract_additional_19_features(
        self,
        text: str,
        tokens: List[str],
        text_tokens: List[str],
        base_features: Dict[str, float],
    ) -> Dict[str, float]:
        """Extract the additional 19 new interpretable features."""
        features = {}
        n_tokens = max(len(tokens), 1)
        text_lower = text.lower()
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        # Get key term indices
        deception_terms = RISK_LEXICONS.get("deception", set())
        attribution_indices = [
            i for i, t in enumerate(tokens) if t in ATTRIBUTION_VERBS
        ]
        deception_indices = [i for i, t in enumerate(tokens) if t in deception_terms]
        hedge_terms = RISK_LEXICONS.get("hedges", set())

        # --- FIRST SET: 10 FEATURES ---

        # 1. attr_org_vs_person_ratio
        org_near_attr = 0
        person_near_attr = 0

        for attr_idx in attribution_indices:
            window_start = max(0, attr_idx - 3)
            window_end = min(len(tokens), attr_idx + 4)

            # Count org hints in window
            for i in range(window_start, window_end):
                if tokens[i].lower() in ORG_HINTS:
                    org_near_attr += 1
                    break  # Count once per attribution

            # Count person names in window
            for i in range(window_start, window_end):
                if _detect_person_name(tokens, i):
                    person_near_attr += 1
                    break  # Count once per attribution

        features["new2_attr_org_vs_person_ratio"] = (org_near_attr + 0.0001) / (
            person_near_attr + 0.0001
        )

        # 2. attr_sentence_hedge_rate
        attr_sentence_hedge_rates = []
        for sentence in sentences:
            sent_tokens = self._tokenize(sentence)
            has_attribution = any(t in ATTRIBUTION_VERBS for t in sent_tokens)
            if has_attribution:
                hedge_count = sum(1 for t in sent_tokens if t in hedge_terms)
                attr_sentence_hedge_rates.append(hedge_count / max(len(sent_tokens), 1))

        features["new2_attr_sentence_hedge_rate"] = (
            np.mean(attr_sentence_hedge_rates) if attr_sentence_hedge_rates else 0.0
        )

        # 3. negated_deception_norm
        negated_deception = _window_proximity_count(
            tokens, NEG_WORDS, deception_terms, radius=3
        )
        features["new2_negated_deception_norm"] = negated_deception / n_tokens

        # 4. deception_near_disclaimer_norm
        disclaimer_tokens = set()
        for phrase in DISCLAIMER_PHRASES:
            disclaimer_tokens.update(phrase.split())

        deception_near_disclaimer = _window_proximity_count(
            tokens, deception_terms, disclaimer_tokens, radius=5
        )
        features["new2_deception_near_disclaimer_norm"] = (
            deception_near_disclaimer / n_tokens
        )

        # 5. neutral_run_max_ratio
        # Get all risk terms
        all_risk_terms = set()
        for lexicon_terms in RISK_LEXICONS.values():
            all_risk_terms.update(lexicon_terms)

        # Find runs of neutral tokens
        neutral_runs = []
        current_run = 0
        for token in tokens:
            if token not in all_risk_terms:
                current_run += 1
            else:
                if current_run > 0:
                    neutral_runs.append(current_run)
                current_run = 0
        if current_run > 0:
            neutral_runs.append(current_run)

        max_neutral_run = max(neutral_runs) if neutral_runs else 0
        features["new2_neutral_run_max_ratio"] = max_neutral_run / n_tokens

        # 6. neutral_run_mean
        features["new2_neutral_run_mean"] = (
            np.mean(neutral_runs) if neutral_runs else 0.0
        )

        # 7. risk_category_entropy
        risk_token_categories = []
        for token in tokens:
            for lex_name, lex_terms in RISK_LEXICONS.items():
                if token in lex_terms:
                    risk_token_categories.append(lex_name)
                    break

        if risk_token_categories:
            from collections import Counter

            category_counts = Counter(risk_token_categories)
            total_risk_tokens = len(risk_token_categories)
            entropy = 0.0
            for count in category_counts.values():
                p = count / total_risk_tokens
                if p > 0:
                    entropy -= p * np.log(p)
            features["new2_risk_category_entropy"] = entropy
        else:
            features["new2_risk_category_entropy"] = 0.0

        # 8. deception_attr_min_dist
        if deception_indices and attribution_indices:
            min_dist = float("inf")
            for d_idx in deception_indices:
                for a_idx in attribution_indices:
                    dist = abs(d_idx - a_idx)
                    min_dist = min(min_dist, dist)
            features["new2_deception_attr_min_dist"] = (
                min(min_dist, 30) / 30.0
            )  # Normalized by cap
        else:
            features["new2_deception_attr_min_dist"] = (
                1.0  # Max distance when no co-occurrence
            )

        # 9. modal_balance_index
        low_certainty = base_features.get("ling_low_certainty", 0.0)
        high_certainty = base_features.get("ling_high_certainty", 0.0)
        modal_sum = low_certainty + high_certainty
        features["new2_modal_balance_index"] = (low_certainty - high_certainty) / (
            1 + modal_sum
        )

        # 10. hedged_deception_share
        deception_in_hedged_sentences = 0
        total_deception_terms = len(deception_indices)

        if total_deception_terms > 0:
            for sentence in sentences:
                sent_tokens = self._tokenize(sentence)
                has_hedge = any(t in hedge_terms for t in sent_tokens)
                has_deception = any(t in deception_terms for t in sent_tokens)
                if has_hedge and has_deception:
                    # Count deception terms in this hedged sentence
                    deception_in_hedged_sentences += sum(
                        1 for t in sent_tokens if t in deception_terms
                    )

            features["new2_hedged_deception_share"] = (
                deception_in_hedged_sentences / total_deception_terms
            )
        else:
            features["new2_hedged_deception_share"] = 0.0

        # --- SECOND SET: 9 FEATURES (excluding redundant #6) ---

        # 1. Deception Near Attribution Proximity
        deception_near_attribution = _window_proximity_count(
            tokens, deception_terms, ATTRIBUTION_VERBS, radius=4
        )
        features["new2_deception_near_attribution_prox"] = (
            deception_near_attribution / max(len(attribution_indices), 1)
        )

        # 2. Attribution Verb Near Neutral Transition
        neutral_transitions_near_attr = 0
        total_neutral_transitions = 0

        for i in range(len(tokens) - 1):
            curr_cats = self._get_token_categories(tokens[i])
            next_cats = self._get_token_categories(tokens[i + 1])

            if "neutral" in curr_cats and "neutral" in next_cats:
                total_neutral_transitions += 1
                # Check if this transition is near attribution verb
                if any(abs(i - attr_idx) <= 3 for attr_idx in attribution_indices):
                    neutral_transitions_near_attr += 1

        features["new2_attr_verb_near_neutral_transition"] = (
            neutral_transitions_near_attr / max(total_neutral_transitions, 1)
        )

        # 3. Deception Cluster Density
        if len(deception_indices) > 1:
            deception_positions = np.array(deception_indices) / n_tokens
            clustering_score = 1.0 - min(1.0, np.std(deception_positions))
        elif len(deception_indices) == 1:
            clustering_score = 0.5
        else:
            clustering_score = 0.0
        features["new2_deception_cluster_density"] = clustering_score

        # 4. Attribution Verb in Deception Sentences
        total_attr_per_sentence = []
        deception_sentence_attr = []

        for sentence in sentences:
            sent_tokens = self._tokenize(sentence)
            attr_count = sum(1 for t in sent_tokens if t in ATTRIBUTION_VERBS)
            total_attr_per_sentence.append(attr_count)

            has_deception = any(t in deception_terms for t in sent_tokens)
            if has_deception:
                deception_sentence_attr.append(attr_count)

        avg_overall_attr = (
            np.mean(total_attr_per_sentence) if total_attr_per_sentence else 0.0
        )
        avg_deception_attr = (
            np.mean(deception_sentence_attr) if deception_sentence_attr else 0.0
        )

        features["new2_attr_verb_in_deception_sentences"] = avg_deception_attr / max(
            avg_overall_attr, 0.0001
        )

        # 5. Neutral to Deception Transition Rate
        neutral_to_deception_transitions = 0
        total_transitions = 0

        for i in range(len(tokens) - 1):
            curr_cats = self._get_token_categories(tokens[i])
            next_cats = self._get_token_categories(tokens[i + 1])
            total_transitions += 1

            if "neutral" in curr_cats and "deception" in next_cats:
                neutral_to_deception_transitions += 1

        features["new2_neutral_to_deception_transition_rate"] = (
            neutral_to_deception_transitions / max(total_transitions, 1)
        )

        # 7. Attribution Verb Clustering Score
        if len(attribution_indices) > 1:
            attr_positions = np.array(attribution_indices) / n_tokens
            attr_clustering_score = 1.0 - min(1.0, np.std(attr_positions))
        elif len(attribution_indices) == 1:
            attr_clustering_score = 0.5
        else:
            attr_clustering_score = 0.0
        features["new2_attribution_verb_clustering_score"] = attr_clustering_score

        # 8. Deception to Attribution Transition Frequency
        deception_to_attr_transitions = 0

        for i in range(len(tokens) - 1):
            curr_cats = self._get_token_categories(tokens[i])
            if "deception" in curr_cats and tokens[i + 1] in ATTRIBUTION_VERBS:
                deception_to_attr_transitions += 1

        features["new2_deception_to_attribution_transition"] = (
            deception_to_attr_transitions / max(len(deception_indices), 1)
        )

        # 9. Neutral Transition Near Deception
        neutral_transitions_near_deception = 0

        for i in range(len(tokens) - 1):
            curr_cats = self._get_token_categories(tokens[i])
            next_cats = self._get_token_categories(tokens[i + 1])

            if "neutral" in curr_cats and "neutral" in next_cats:
                # Check if this transition is near deception term
                if any(abs(i - d_idx) <= 5 for d_idx in deception_indices):
                    neutral_transitions_near_deception += 1

        features["new2_neutral_transition_near_deception"] = (
            neutral_transitions_near_deception / max(total_neutral_transitions, 1)
        )

        # 10. Attributed Deception Ratio
        attributed_deception = 0
        for d_idx in deception_indices:
            if any(abs(d_idx - a_idx) <= 4 for a_idx in attribution_indices):
                attributed_deception += 1

        features["new2_attributed_deception_ratio"] = attributed_deception / max(
            len(deception_indices), 1
        )

        return features

    def _extract_final_7_features(
        self,
        text: str,
        tokens: List[str],
        text_tokens: List[str],
        base_features: Dict[str, float],
    ) -> Dict[str, float]:
        """Extract the final 7 new interpretable features (non-redundant)."""
        features = {}
        n_tokens = max(len(tokens), 1)
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        # Get key term indices
        attribution_indices = [
            i for i, t in enumerate(tokens) if t in ATTRIBUTION_VERBS
        ]

        # Get all risk terms
        all_risk_terms = set()
        for lexicon_terms in RISK_LEXICONS.values():
            all_risk_terms.update(lexicon_terms)
        risk_indices = [i for i, t in enumerate(tokens) if t in all_risk_terms]

        # --- FEATURE 2: attributed_sentence_share ---
        attributed_sentence_tokens = 0
        for sentence in sentences:
            sent_tokens = self._tokenize(sentence)
            has_attribution = any(t in ATTRIBUTION_VERBS for t in sent_tokens)
            if has_attribution:
                attributed_sentence_tokens += len(sent_tokens)

        features["new3_attributed_sentence_share"] = (
            attributed_sentence_tokens / n_tokens
        )

        # --- FEATURE 3: attribution_lead_vs_first_risk ---
        if attribution_indices and risk_indices:
            first_attr = min(attribution_indices)
            first_risk = min(risk_indices)
            lead_distance = (first_attr - first_risk) / n_tokens
            # Winsorize at ±0.1
            lead_distance = max(-0.1, min(0.1, lead_distance))
            features["new3_attribution_lead_vs_first_risk"] = lead_distance
        else:
            features["new3_attribution_lead_vs_first_risk"] = 0.0

        # --- FEATURE 4: attributed_risk_share ---
        risk_in_attributed_sentences = 0
        total_risk_tokens = len(risk_indices)

        if total_risk_tokens > 0:
            for sentence in sentences:
                sent_tokens = self._tokenize(sentence)
                has_attribution = any(t in ATTRIBUTION_VERBS for t in sent_tokens)
                if has_attribution:
                    risk_in_attributed_sentences += sum(
                        1 for t in sent_tokens if t in all_risk_terms
                    )

            features["new3_attributed_risk_share"] = (
                risk_in_attributed_sentences / total_risk_tokens
            )
        else:
            features["new3_attributed_risk_share"] = 0.0

        # --- FEATURE 5: neutral_edge_coverage ---
        # First 20% and last 20% of tokens
        head_end = max(1, int(0.2 * n_tokens))
        tail_start = max(head_end, int(0.8 * n_tokens))

        head_tokens = tokens[:head_end]
        tail_tokens = tokens[tail_start:]

        head_neutral = sum(1 for t in head_tokens if t not in all_risk_terms)
        tail_neutral = sum(1 for t in tail_tokens if t not in all_risk_terms)

        head_coverage = head_neutral / max(len(head_tokens), 1)
        tail_coverage = tail_neutral / max(len(tail_tokens), 1)

        features["new3_neutral_edge_coverage"] = (head_coverage + tail_coverage) / 2.0

        # --- FEATURE 6: neutral_sandwich_flag ---
        # Check for neutral start (≥5 tokens) + risk middle + neutral end (≥5 tokens)
        neutral_sandwich = 0
        if n_tokens >= 15:  # Need at least 15 tokens for pattern
            # Check first 5+ tokens are neutral
            head_neutral_run = 0
            for t in tokens[:10]:  # Check up to first 10
                if t not in all_risk_terms:
                    head_neutral_run += 1
                else:
                    break

            # Check last 5+ tokens are neutral
            tail_neutral_run = 0
            for t in reversed(tokens[-10:]):  # Check last 10
                if t not in all_risk_terms:
                    tail_neutral_run += 1
                else:
                    break

            # Check if there's risk in the middle
            middle_start = max(5, head_neutral_run)
            middle_end = min(n_tokens - 5, n_tokens - tail_neutral_run)

            if (
                head_neutral_run >= 5
                and tail_neutral_run >= 5
                and middle_start < middle_end
                and any(
                    tokens[i] in all_risk_terms for i in range(middle_start, middle_end)
                )
            ):
                neutral_sandwich = 1

        features["new3_neutral_sandwich_flag"] = neutral_sandwich

        # --- FEATURE 9: conditionalized_risk_share ---
        conditional_risk_tokens = 0

        if total_risk_tokens > 0:
            for sentence in sentences:
                sent_tokens = self._tokenize(sentence)
                sentence_lower = sentence.lower()

                # Check if sentence contains conditional markers
                is_conditional = any(
                    marker in sentence_lower for marker in CONDITIONAL_MARKERS
                )

                if is_conditional:
                    conditional_risk_tokens += sum(
                        1 for t in sent_tokens if t in all_risk_terms
                    )

            features["new3_conditionalized_risk_share"] = (
                conditional_risk_tokens / total_risk_tokens
            )
        else:
            features["new3_conditionalized_risk_share"] = 0.0

        # --- FEATURE 10: scope_gated_risk_share ---
        scope_gated_risk = 0

        if total_risk_tokens > 0:
            # For each risk token, check if there's a scope limiter within ±6 tokens
            scope_indices = [
                i for i, t in enumerate(tokens) if t in SCOPE_LIMITERS_EXTENDED
            ]

            for risk_idx in risk_indices:
                # Check if any scope limiter is within ±6 tokens
                if any(abs(risk_idx - scope_idx) <= 6 for scope_idx in scope_indices):
                    scope_gated_risk += 1

            features["new3_scope_gated_risk_share"] = (
                scope_gated_risk / total_risk_tokens
            )
        else:
            features["new3_scope_gated_risk_share"] = 0.0

        return features

    def _extract_24_additional_features(
        self,
        text: str,
        tokens: List[str],
        text_tokens: List[str],
        base_features: Dict[str, float],
    ) -> Dict[str, float]:
        """Extract 24 additional unique interpretable features (non-redundant)."""
        features = {}
        n_tokens = max(len(tokens), 1)
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        # Get key term sets
        hedge_terms = RISK_LEXICONS.get("hedges", set())
        compliance_terms = RISK_LEXICONS.get("compliance", set())
        disclaimer_terms = RISK_LEXICONS.get("disclaimers", set())
        scope_limiter_terms = RISK_LEXICONS.get("scope_limiters", set())
        deception_terms = RISK_LEXICONS.get("deception", set())
        attribution_indices = [
            i for i, t in enumerate(tokens) if t in ATTRIBUTION_VERBS
        ]

        # Get all risk terms
        all_risk_terms = set()
        for lexicon_terms in RISK_LEXICONS.values():
            all_risk_terms.update(lexicon_terms)

        # Helper function for clustering
        def _compute_clustering_score(indices: List[int]) -> float:
            if len(indices) > 1:
                positions = np.array(indices) / n_tokens
                clustering_score = 1.0 - min(1.0, np.std(positions))
                return clustering_score
            elif len(indices) == 1:
                return 0.5
            else:
                return 0.0

        # Helper function for consecutive runs
        def _compute_run_mean(term_set: set) -> float:
            runs = []
            current_run = 0
            for token in tokens:
                if token in term_set:
                    current_run += 1
                else:
                    if current_run > 0:
                        runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                runs.append(current_run)
            return np.mean(runs) if runs else 0.0

        # --- CLUSTERING FEATURES (4) ---

        # 1. hedge_cluster_density
        hedge_indices = [i for i, t in enumerate(tokens) if t in hedge_terms]
        features["new4_hedge_cluster_density"] = _compute_clustering_score(
            hedge_indices
        )

        # 2. compliance_cluster_density
        compliance_indices = [i for i, t in enumerate(tokens) if t in compliance_terms]
        features["new4_compliance_cluster_density"] = _compute_clustering_score(
            compliance_indices
        )

        # 3. disclaimer_cluster_density
        disclaimer_indices = [i for i, t in enumerate(tokens) if t in disclaimer_terms]
        features["new4_disclaimer_cluster_density"] = _compute_clustering_score(
            disclaimer_indices
        )

        # 4. scope_limiter_cluster_density
        scope_limiter_indices = [
            i for i, t in enumerate(tokens) if t in scope_limiter_terms
        ]
        features["new4_scope_limiter_cluster_density"] = _compute_clustering_score(
            scope_limiter_indices
        )

        # --- TRANSITION RATE FEATURES (6) ---

        total_transitions = max(len(tokens) - 1, 1)

        # 5. neutral_to_hedge_transition_rate
        neutral_to_hedge_transitions = 0
        for i in range(len(tokens) - 1):
            curr_cats = self._get_token_categories(tokens[i])
            next_cats = self._get_token_categories(tokens[i + 1])
            if "neutral" in curr_cats and any(
                tokens[i + 1] in hedge_terms for _ in next_cats
            ):
                neutral_to_hedge_transitions += 1
        features["new4_neutral_to_hedge_transition_rate"] = (
            neutral_to_hedge_transitions / total_transitions
        )

        # 6. neutral_to_scope_limiter_transition_rate
        neutral_to_scope_transitions = 0
        for i in range(len(tokens) - 1):
            curr_cats = self._get_token_categories(tokens[i])
            if "neutral" in curr_cats and tokens[i + 1] in scope_limiter_terms:
                neutral_to_scope_transitions += 1
        features["new4_neutral_to_scope_limiter_transition_rate"] = (
            neutral_to_scope_transitions / total_transitions
        )

        # 7. hedge_to_neutral_transition_rate
        hedge_to_neutral_transitions = 0
        for i in range(len(tokens) - 1):
            next_cats = self._get_token_categories(tokens[i + 1])
            if tokens[i] in hedge_terms and "neutral" in next_cats:
                hedge_to_neutral_transitions += 1
        features["new4_hedge_to_neutral_transition_rate"] = (
            hedge_to_neutral_transitions / total_transitions
        )

        # 8. neutral_to_disclaimer_transition_rate
        neutral_to_disclaimer_transitions = 0
        for i in range(len(tokens) - 1):
            curr_cats = self._get_token_categories(tokens[i])
            if "neutral" in curr_cats and tokens[i + 1] in disclaimer_terms:
                neutral_to_disclaimer_transitions += 1
        features["new4_neutral_to_disclaimer_transition_rate"] = (
            neutral_to_disclaimer_transitions / total_transitions
        )

        # 9. disclaimer_to_neutral_transition_rate
        disclaimer_to_neutral_transitions = 0
        for i in range(len(tokens) - 1):
            next_cats = self._get_token_categories(tokens[i + 1])
            if tokens[i] in disclaimer_terms and "neutral" in next_cats:
                disclaimer_to_neutral_transitions += 1
        features["new4_disclaimer_to_neutral_transition_rate"] = (
            disclaimer_to_neutral_transitions / total_transitions
        )

        # 10. neutral_transition_symmetry
        neutral_to_risk = 0
        risk_to_neutral = 0
        for i in range(len(tokens) - 1):
            curr_cats = self._get_token_categories(tokens[i])
            next_cats = self._get_token_categories(tokens[i + 1])

            if "neutral" in curr_cats and any(cat != "neutral" for cat in next_cats):
                neutral_to_risk += 1
            elif any(cat != "neutral" for cat in curr_cats) and "neutral" in next_cats:
                risk_to_neutral += 1

        symmetry_denom = neutral_to_risk + risk_to_neutral + 1e-6
        features["new4_neutral_transition_symmetry"] = (
            risk_to_neutral - neutral_to_risk
        ) / symmetry_denom

        # --- ATTRIBUTION PROXIMITY FEATURES (4) ---

        # 11. attribution_verb_near_compliance
        attr_near_compliance = 0
        for attr_idx in attribution_indices:
            if any(abs(attr_idx - comp_idx) <= 4 for comp_idx in compliance_indices):
                attr_near_compliance += 1
        features["new4_attribution_verb_near_compliance"] = attr_near_compliance / max(
            len(attribution_indices), 1
        )

        # 12. attribution_verb_near_scope_limiter
        attr_near_scope = 0
        for attr_idx in attribution_indices:
            if any(
                abs(attr_idx - scope_idx) <= 3 for scope_idx in scope_limiter_indices
            ):
                attr_near_scope += 1
        features["new4_attribution_verb_near_scope_limiter"] = attr_near_scope / max(
            len(scope_limiter_indices), 1
        )

        # 13. attribution_verb_near_disclaimer
        attr_near_disclaimer = 0
        for attr_idx in attribution_indices:
            if any(abs(attr_idx - disc_idx) <= 4 for disc_idx in disclaimer_indices):
                attr_near_disclaimer += 1
        features["new4_attribution_verb_near_disclaimer"] = attr_near_disclaimer / max(
            len(attribution_indices), 1
        )

        # 14. attribution_verb_near_hedge_transition
        neutral_transitions_near_hedge = 0
        total_neutral_transitions = 0
        for i in range(len(tokens) - 1):
            curr_cats = self._get_token_categories(tokens[i])
            next_cats = self._get_token_categories(tokens[i + 1])

            if "neutral" in curr_cats and "neutral" in next_cats:
                total_neutral_transitions += 1
                # Check if this transition is near any hedge term
                if any(abs(i - hedge_idx) <= 3 for hedge_idx in hedge_indices):
                    neutral_transitions_near_hedge += 1
        features["new4_attribution_verb_near_hedge_transition"] = (
            neutral_transitions_near_hedge / max(total_neutral_transitions, 1)
        )

        # --- RUN MEAN FEATURES (2) ---

        # 15. scope_limiter_run_mean
        features["new4_scope_limiter_run_mean"] = _compute_run_mean(scope_limiter_terms)

        # 16. disclaimer_run_mean
        features["new4_disclaimer_run_mean"] = _compute_run_mean(disclaimer_terms)

        # --- EDGE COVERAGE FEATURES (2) ---

        # 17. compliance_edge_coverage
        head_end = max(1, int(0.2 * n_tokens))
        tail_start = max(head_end, int(0.8 * n_tokens))
        head_tokens = tokens[:head_end]
        tail_tokens = tokens[tail_start:]

        head_compliance = sum(1 for t in head_tokens if t in compliance_terms)
        tail_compliance = sum(1 for t in tail_tokens if t in compliance_terms)
        head_coverage = head_compliance / max(len(head_tokens), 1)
        tail_coverage = tail_compliance / max(len(tail_tokens), 1)
        features["new4_compliance_edge_coverage"] = (
            head_coverage + tail_coverage
        ) / 2.0

        # 18. hedge_edge_coverage
        head_hedge = sum(1 for t in head_tokens if t in hedge_terms)
        tail_hedge = sum(1 for t in tail_tokens if t in hedge_terms)
        head_hedge_coverage = head_hedge / max(len(head_tokens), 1)
        tail_hedge_coverage = tail_hedge / max(len(tail_tokens), 1)
        features["new4_hedge_edge_coverage"] = (
            head_hedge_coverage + tail_hedge_coverage
        ) / 2.0

        # --- ADVANCED RISK ANALYSIS FEATURES (6) ---

        # 19. attribution_lead_lag_signed (different from new3_attribution_lead_vs_first_risk)
        risk_indices = [i for i, t in enumerate(tokens) if t in all_risk_terms]
        if attribution_indices and risk_indices:
            first_attr = min(attribution_indices)
            first_risk = min(risk_indices)
            lead_lag = (first_attr - first_risk) / n_tokens
            # Winsorize at ±0.2 (different from the ±0.1 in new3)
            features["new4_attribution_lead_lag_signed"] = max(-0.2, min(0.2, lead_lag))
        else:
            features["new4_attribution_lead_lag_signed"] = 0.0

        # 20. risk_core_span_ratio
        non_neutral_indices = [i for i, t in enumerate(tokens) if t in all_risk_terms]
        if non_neutral_indices:
            span_start = min(non_neutral_indices)
            span_end = max(non_neutral_indices)
            span_length = span_end - span_start + 1
            features["new4_risk_core_span_ratio"] = span_length / n_tokens
        else:
            features["new4_risk_core_span_ratio"] = 0.0

        # 21. deception_isolation_score
        deception_indices = [i for i, t in enumerate(tokens) if t in deception_terms]
        isolated_deception = 0
        for d_idx in deception_indices:
            # Check if deception token is isolated (surrounded by neutral)
            left_neutral = d_idx == 0 or tokens[d_idx - 1] not in all_risk_terms
            right_neutral = (
                d_idx == len(tokens) - 1 or tokens[d_idx + 1] not in all_risk_terms
            )
            if left_neutral and right_neutral:
                isolated_deception += 1
        features["new4_deception_isolation_score"] = isolated_deception / max(
            len(deception_indices), 1
        )

        # 22. neutral_recovery_after_deception
        deception_cluster_ends = []
        in_deception_cluster = False
        for i, token in enumerate(tokens):
            if token in deception_terms:
                in_deception_cluster = True
            elif in_deception_cluster and token not in deception_terms:
                deception_cluster_ends.append(i)
                in_deception_cluster = False

        recovery_scores = []
        for cluster_end in deception_cluster_ends:
            window_end = min(len(tokens), cluster_end + 8)
            window_tokens = tokens[cluster_end:window_end]
            if window_tokens:
                neutral_share = sum(
                    1 for t in window_tokens if t not in all_risk_terms
                ) / len(window_tokens)
                recovery_scores.append(neutral_share)
        features["new4_neutral_recovery_after_deception"] = (
            np.mean(recovery_scores) if recovery_scores else 0.0
        )

        # 23. conditionalized_attr_share
        attributed_sentences = 0
        conditional_attributed_sentences = 0
        for sentence in sentences:
            sent_tokens = self._tokenize(sentence)
            has_attribution = any(t in ATTRIBUTION_VERBS for t in sent_tokens)
            if has_attribution:
                attributed_sentences += 1
                # Check for conditional markers
                sentence_lower = sentence.lower()
                if any(marker in sentence_lower for marker in CONDITIONAL_MARKERS):
                    conditional_attributed_sentences += 1
        features["new4_conditionalized_attr_share"] = (
            conditional_attributed_sentences / max(attributed_sentences, 1)
        )

        # 24. tail_neutral_gain
        head_neutral = sum(1 for t in head_tokens if t not in all_risk_terms)
        tail_neutral = sum(1 for t in tail_tokens if t not in all_risk_terms)
        head_neutral_pct = head_neutral / max(len(head_tokens), 1)
        tail_neutral_pct = tail_neutral / max(len(tail_tokens), 1)
        features["new4_tail_neutral_gain"] = tail_neutral_pct - head_neutral_pct

        return features

    def _extract_20_additional_features(
        self,
        text: str,
        tokens: List[str],
        text_tokens: List[str],
        base_features: Dict[str, float],
    ) -> Dict[str, float]:
        """Extract 20 additional unique interpretable features."""
        features = {}
        n_tokens = max(len(tokens), 1)
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        # Get key term sets
        positive_qualifiers = RISK_LEXICONS.get("positive_qualifiers", set())
        clarity_terms = RISK_LEXICONS.get("clarity", set())
        policy_procedure_terms = RISK_LEXICONS.get("policy_procedure", set())
        ambiguity_terms = RISK_LEXICONS.get("ambiguity", set())
        hedge_terms = RISK_LEXICONS.get("hedges", set())
        deception_terms = RISK_LEXICONS.get("deception", set())
        scope_limiter_terms = RISK_LEXICONS.get("scope_limiters", set())
        attribution_indices = [
            i for i, t in enumerate(tokens) if t in ATTRIBUTION_VERBS
        ]

        # Get all risk terms
        all_risk_terms = set()
        for lexicon_terms in RISK_LEXICONS.values():
            all_risk_terms.update(lexicon_terms)

        # Helper function for clustering (reused)
        def _compute_clustering_score(indices: List[int]) -> float:
            if len(indices) > 1:
                positions = np.array(indices) / n_tokens
                clustering_score = 1.0 - min(1.0, np.std(positions))
                return clustering_score
            elif len(indices) == 1:
                return 0.5
            else:
                return 0.0

        # Helper function for edge coverage (reused)
        def _compute_edge_coverage(term_set: set) -> float:
            head_end = max(1, int(0.2 * n_tokens))
            tail_start = max(head_end, int(0.8 * n_tokens))
            head_tokens = tokens[:head_end]
            tail_tokens = tokens[tail_start:]

            head_coverage = sum(1 for t in head_tokens if t in term_set) / max(
                len(head_tokens), 1
            )
            tail_coverage = sum(1 for t in tail_tokens if t in term_set) / max(
                len(tail_tokens), 1
            )
            return (head_coverage + tail_coverage) / 2.0

        # Helper function for run mean (reused)
        def _compute_run_mean(term_set: set) -> float:
            runs = []
            current_run = 0
            for token in tokens:
                if token in term_set:
                    current_run += 1
                else:
                    if current_run > 0:
                        runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                runs.append(current_run)
            return np.mean(runs) if runs else 0.0

        # --- FIRST SET: 10 FEATURES ---

        # 1. positive_qualifiers_cluster_density
        pq_indices = [i for i, t in enumerate(tokens) if t in positive_qualifiers]
        features["new5_positive_qualifiers_cluster_density"] = (
            _compute_clustering_score(pq_indices)
        )

        # 2. clarity_cluster_density
        clarity_indices = [i for i, t in enumerate(tokens) if t in clarity_terms]
        features["new5_clarity_cluster_density"] = _compute_clustering_score(
            clarity_indices
        )

        # 3. neutral_to_positive_qualifiers_transition_rate
        neutral_to_pq_transitions = 0
        total_transitions = max(len(tokens) - 1, 1)
        for i in range(len(tokens) - 1):
            curr_cats = self._get_token_categories(tokens[i])
            if "neutral" in curr_cats and tokens[i + 1] in positive_qualifiers:
                neutral_to_pq_transitions += 1
        features["new5_neutral_to_positive_qualifiers_transition_rate"] = (
            neutral_to_pq_transitions / total_transitions
        )

        # 4. positive_qualifiers_to_neutral_transition_rate
        pq_to_neutral_transitions = 0
        for i in range(len(tokens) - 1):
            next_cats = self._get_token_categories(tokens[i + 1])
            if tokens[i] in positive_qualifiers and "neutral" in next_cats:
                pq_to_neutral_transitions += 1
        features["new5_positive_qualifiers_to_neutral_transition_rate"] = (
            pq_to_neutral_transitions / total_transitions
        )

        # 5. attribution_verb_near_positive_qualifiers
        attr_near_pq = 0
        for attr_idx in attribution_indices:
            if any(abs(attr_idx - pq_idx) <= 4 for pq_idx in pq_indices):
                attr_near_pq += 1
        features["new5_attribution_verb_near_positive_qualifiers"] = attr_near_pq / max(
            len(attribution_indices), 1
        )

        # 6. policy_procedure_edge_coverage
        features["new5_policy_procedure_edge_coverage"] = _compute_edge_coverage(
            policy_procedure_terms
        )

        # 7. remediation_edge_coverage
        features["new5_remediation_edge_coverage"] = _compute_edge_coverage(
            REMEDIATION_TERMS
        )

        # 8. policy_procedure_run_mean
        features["new5_policy_procedure_run_mean"] = _compute_run_mean(
            policy_procedure_terms
        )

        # 9. neutral_recovery_after_ambiguity
        ambiguity_indices = [i for i, t in enumerate(tokens) if t in ambiguity_terms]
        ambiguity_cluster_ends = []
        in_ambiguity_cluster = False
        for i, token in enumerate(tokens):
            if token in ambiguity_terms:
                in_ambiguity_cluster = True
            elif in_ambiguity_cluster and token not in ambiguity_terms:
                ambiguity_cluster_ends.append(i)
                in_ambiguity_cluster = False

        recovery_scores = []
        for cluster_end in ambiguity_cluster_ends:
            window_end = min(len(tokens), cluster_end + 8)
            window_tokens = tokens[cluster_end:window_end]
            if window_tokens:
                neutral_share = sum(
                    1 for t in window_tokens if t not in all_risk_terms
                ) / len(window_tokens)
                recovery_scores.append(neutral_share)
        features["new5_neutral_recovery_after_ambiguity"] = (
            np.mean(recovery_scores) if recovery_scores else 0.0
        )

        # 10. conditionalized_positive_share
        total_pq_terms = len(pq_indices)
        pq_in_conditional_sentences = 0
        if total_pq_terms > 0:
            for sentence in sentences:
                sent_tokens = self._tokenize(sentence)
                sentence_lower = sentence.lower()
                is_conditional = any(
                    marker in sentence_lower for marker in CONDITIONAL_MARKERS
                )
                if is_conditional:
                    pq_in_conditional_sentences += sum(
                        1 for t in sent_tokens if t in positive_qualifiers
                    )
        features["new5_conditionalized_positive_share"] = (
            pq_in_conditional_sentences / max(total_pq_terms, 1)
        )

        # --- SECOND SET: 10 FEATURES ---

        # 1. attr_edge_span_coverage
        edge_15_end = max(1, int(0.15 * n_tokens))
        edge_85_start = max(edge_15_end, int(0.85 * n_tokens))
        edge_tokens = list(range(edge_15_end)) + list(range(edge_85_start, n_tokens))

        attributed_spans = set()
        for sentence in sentences:
            sent_tokens = self._tokenize(sentence)
            has_attribution = any(t in ATTRIBUTION_VERBS for t in sent_tokens)
            if has_attribution:
                # Mark this sentence's token positions as attributed
                start_pos = text.lower().find(sentence.lower())
                if start_pos != -1:
                    sent_word_count = len(sent_tokens)
                    # Approximate token positions (simple heuristic)
                    for i in range(sent_word_count):
                        attributed_spans.add(start_pos // 5 + i)  # Rough approximation

        edge_attributed = sum(1 for i in edge_tokens if i in attributed_spans)
        features["new5_attr_edge_span_coverage"] = edge_attributed / max(
            len(edge_tokens), 1
        )

        # 2. attr_hedge_cohesion
        hedges_near_attr = 0
        hedges_elsewhere = 0
        window_tokens_near = 0
        window_tokens_elsewhere = 0

        for i, token in enumerate(tokens):
            near_attr = any(abs(i - attr_idx) <= 3 for attr_idx in attribution_indices)
            if near_attr:
                window_tokens_near += 1
                if token in hedge_terms:
                    hedges_near_attr += 1
            else:
                window_tokens_elsewhere += 1
                if token in hedge_terms:
                    hedges_elsewhere += 1

        hedge_density_near = hedges_near_attr / max(window_tokens_near, 1)
        hedge_density_elsewhere = hedges_elsewhere / max(window_tokens_elsewhere, 1)
        features["new5_attr_hedge_cohesion"] = hedge_density_near / max(
            hedge_density_elsewhere, 0.0001
        )

        # 3. attr_contrast_after_rate
        contrast_markers = CONTRAST_DISCOURSE_MARKERS
        contrast_after_attr = 0
        for attr_idx in attribution_indices:
            window_start = attr_idx + 1
            window_end = min(len(tokens), attr_idx + 7)
            for i in range(window_start, window_end):
                if tokens[i] in contrast_markers:
                    contrast_after_attr += 1
                    break  # Count once per attribution
        features["new5_attr_contrast_after_rate"] = contrast_after_attr / max(
            len(attribution_indices), 1
        )

        # 4. neutral_slope_5bins
        bin_size = max(1, n_tokens // 5)
        bin_neutral_rates = []
        for bin_idx in range(5):
            start_idx = bin_idx * bin_size
            end_idx = min(n_tokens, (bin_idx + 1) * bin_size)
            if start_idx < end_idx:
                bin_tokens = tokens[start_idx:end_idx]
                neutral_rate = sum(
                    1 for t in bin_tokens if t not in all_risk_terms
                ) / len(bin_tokens)
                bin_neutral_rates.append(neutral_rate)

        if len(bin_neutral_rates) >= 2:
            # Simple slope calculation
            x_vals = list(range(len(bin_neutral_rates)))
            slope = np.polyfit(x_vals, bin_neutral_rates, 1)[0]
            # Winsorize slope
            slope = max(-0.5, min(0.5, slope))
        else:
            slope = 0.0
        features["new5_neutral_slope_5bins"] = slope

        # 5. deesc_half_life
        deception_indices = [i for i, t in enumerate(tokens) if t in deception_terms]
        if deception_indices:
            last_deception = max(deception_indices)
            half_life_distance = 1.0  # Default to max if never stabilizes

            window_size = 8
            for start_pos in range(last_deception + 1, len(tokens) - window_size + 1):
                window_tokens = tokens[start_pos : start_pos + window_size]
                neutral_share = sum(
                    1 for t in window_tokens if t not in all_risk_terms
                ) / len(window_tokens)
                if neutral_share >= 0.75:
                    half_life_distance = (start_pos - last_deception) / n_tokens
                    break
            features["new5_deesc_half_life"] = half_life_distance
        else:
            features["new5_deesc_half_life"] = 0.0

        # 6. risk_gate_lead_distance
        gate_distances = []
        for sentence in sentences:
            sent_tokens = self._tokenize(sentence)
            has_risk = any(t in all_risk_terms for t in sent_tokens)
            if has_risk:
                # Find first scope limiter in sentence
                first_gate_pos = None
                for i, token in enumerate(sent_tokens):
                    if token in scope_limiter_terms:
                        first_gate_pos = i
                        break
                if first_gate_pos is not None:
                    gate_distances.append(first_gate_pos / max(len(sent_tokens), 1))
        features["new5_risk_gate_lead_distance"] = (
            np.mean(gate_distances) if gate_distances else 1.0
        )

        # 7. risk_clause_hedge_ratio
        risk_clauses = 0
        hedged_risk_clauses = 0
        for sentence in sentences:
            # Split by commas and semicolons
            clauses = [c.strip() for c in re.split(r"[,;]", sentence) if c.strip()]
            for clause in clauses:
                clause_tokens = self._tokenize(clause)
                has_risk = any(t in all_risk_terms for t in clause_tokens)
                if has_risk:
                    risk_clauses += 1
                    has_hedge = any(t in hedge_terms for t in clause_tokens)
                    if has_hedge:
                        hedged_risk_clauses += 1
        features["new5_risk_clause_hedge_ratio"] = hedged_risk_clauses / max(
            risk_clauses, 1
        )

        # 8. epistemic_over_deontic_in_attr
        epistemic_modals = {"may", "might", "could"}
        deontic_modals = {"must", "shall", "cannot"}

        epistemic_in_attr = 0
        deontic_in_attr = 0
        for sentence in sentences:
            sent_tokens = self._tokenize(sentence)
            has_attribution = any(t in ATTRIBUTION_VERBS for t in sent_tokens)
            if has_attribution:
                epistemic_in_attr += sum(
                    1 for t in sent_tokens if t in epistemic_modals
                )
                deontic_in_attr += sum(1 for t in sent_tokens if t in deontic_modals)
        features["new5_epistemic_over_deontic_in_attr"] = (
            epistemic_in_attr + 0.0001
        ) / (deontic_in_attr + 0.0001)

        # 9. negated_risk_in_attr_rate
        risk_in_attr_sentences = 0
        negated_risk_in_attr = 0
        for sentence in sentences:
            sent_tokens = self._tokenize(sentence)
            has_attribution = any(t in ATTRIBUTION_VERBS for t in sent_tokens)
            if has_attribution:
                for i, token in enumerate(sent_tokens):
                    if token in all_risk_terms:
                        risk_in_attr_sentences += 1
                        # Check for negation in prior 3 tokens
                        start_check = max(0, i - 3)
                        prior_tokens = sent_tokens[start_check:i]
                        if any(t in NEG_WORDS for t in prior_tokens):
                            negated_risk_in_attr += 1
        features["new5_negated_risk_in_attr_rate"] = negated_risk_in_attr / max(
            risk_in_attr_sentences, 1
        )

        # 10. neutral_bridge_rate
        risk_risk_pairs = 0
        bridged_pairs = 0
        window_size = 3

        for i in range(len(tokens) - window_size + 1):
            window = tokens[i : i + window_size]
            if (
                window[0] in all_risk_terms
                and window[2] in all_risk_terms
                and window[1] not in all_risk_terms
            ):  # risk-neutral-risk pattern
                bridged_pairs += 1
                risk_risk_pairs += 1
            elif window[0] in all_risk_terms and window[2] in all_risk_terms:
                risk_risk_pairs += 1

        features["new5_neutral_bridge_rate"] = bridged_pairs / max(risk_risk_pairs, 1)

        return features


def create_feature_matrix(
    records: List[Dict[str, Any]],
    extractor: InterpretableFeatureExtractor,
    text_key: str = "text",
    context_key: str = "context",
) -> Tuple[NDArray[np.float64], List[str]]:
    """Create feature matrix from records."""
    all_features: List[Dict[str, float]] = []

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

    return matrix, list(feature_names)
