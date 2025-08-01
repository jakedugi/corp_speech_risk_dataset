#!/usr/bin/env python3
"""
fast_text_processing.py

Ultra-fast text processing optimizations using vectorized operations,
FlashText for faster pattern matching, and optimized regex patterns.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
import multiprocessing as mp

# Try to import FlashText for faster keyword extraction
try:
    from flashtext import KeywordProcessor

    FLASHTEXT_AVAILABLE = True
except ImportError:
    FLASHTEXT_AVAILABLE = False

# Compiled regex patterns for maximum speed
COMPILED_PATTERNS = {
    "amount": re.compile(
        r"\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion))?", re.IGNORECASE
    ),
    "proximity": re.compile(
        r"\b(?:settlement|judgment|damages|award|penalty|fine|amount|paid|cost|price|fee)\b",
        re.IGNORECASE,
    ),
    "judgment_verbs": re.compile(
        r"\b(?:award(?:ed)?|order(?:ed)?|grant(?:ed)?|enter(?:ed)?|assess(?:ed)?)\b",
        re.IGNORECASE,
    ),
    "financial_terms": re.compile(
        r"\b(?:dollar|USD|settlement|damages|compensation|restitution)\b", re.IGNORECASE
    ),
    "dismissal": re.compile(
        r"\b(?:dismiss(?:al|ed)?|denied|rejected|motion\s+to\s+dismiss)\b",
        re.IGNORECASE,
    ),
}


class FastTextProcessor:
    """Ultra-fast text processing using optimized patterns and vectorized operations."""

    def __init__(self, fast_mode: bool = True):
        """Initialize fast text processor."""
        self.fast_mode = fast_mode
        self.keyword_processor = None

        if FLASHTEXT_AVAILABLE and fast_mode:
            self._setup_flashtext()

    def _setup_flashtext(self):
        """Setup FlashText keyword processor for ultra-fast pattern matching."""
        self.keyword_processor = KeywordProcessor(case_sensitive=False)

        # Add financial keywords
        financial_keywords = [
            "settlement",
            "judgment",
            "damages",
            "award",
            "penalty",
            "fine",
            "amount",
            "paid",
            "cost",
            "price",
            "fee",
            "compensation",
            "restitution",
            "dollar",
            "USD",
            "million",
            "billion",
        ]

        for keyword in financial_keywords:
            self.keyword_processor.add_keyword(keyword, f"FINANCIAL_{keyword.upper()}")

        # Add legal keywords
        legal_keywords = [
            "awarded",
            "ordered",
            "granted",
            "entered",
            "assessed",
            "dismissed",
            "denied",
            "rejected",
            "motion to dismiss",
        ]

        for keyword in legal_keywords:
            self.keyword_processor.add_keyword(keyword, f"LEGAL_{keyword.upper()}")

    def extract_amounts_vectorized(
        self, texts: List[str]
    ) -> List[List[Dict[str, Any]]]:
        """Extract amounts from multiple texts using vectorized operations."""
        if not texts:
            return []

        results = []

        # Process texts in batches for better memory efficiency
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_results = []

            for text in batch:
                text_amounts = []

                # Fast regex extraction
                for match in COMPILED_PATTERNS["amount"].finditer(text):
                    amount_str = match.group(0)
                    start, end = match.span()

                    # Quick amount parsing
                    value = self._parse_amount_fast(amount_str)
                    if value > 0:
                        text_amounts.append(
                            {
                                "value": value,
                                "raw_text": amount_str,
                                "start": start,
                                "end": end,
                                "context": text[max(0, start - 100) : end + 100],
                            }
                        )

                batch_results.append(text_amounts)

            results.extend(batch_results)

        return results

    def _parse_amount_fast(self, amount_str: str) -> float:
        """Fast amount parsing using optimized string operations."""
        try:
            # Remove common prefixes/suffixes
            clean_str = amount_str.replace("$", "").replace(",", "").strip().lower()

            # Handle millions/billions
            multiplier = 1
            if "million" in clean_str:
                multiplier = 1_000_000
                clean_str = clean_str.replace("million", "").strip()
            elif "billion" in clean_str:
                multiplier = 1_000_000_000
                clean_str = clean_str.replace("billion", "").strip()

            # Parse number
            return float(clean_str) * multiplier

        except (ValueError, AttributeError):
            return 0.0

    def compute_feature_scores_vectorized(
        self, contexts: List[str], weights: Dict[str, float]
    ) -> np.ndarray:
        """Compute feature scores for multiple contexts using vectorized operations."""
        if not contexts:
            return np.array([])

        # Initialize score matrix
        scores = np.zeros(len(contexts))

        if self.keyword_processor and FLASHTEXT_AVAILABLE:
            # Use FlashText for ultra-fast keyword matching
            for i, context in enumerate(contexts):
                keywords_found = self.keyword_processor.extract_keywords(context)

                # Calculate score based on found keywords
                financial_count = sum(
                    1 for kw in keywords_found if kw.startswith("FINANCIAL_")
                )
                legal_count = sum(1 for kw in keywords_found if kw.startswith("LEGAL_"))

                scores[i] = financial_count * weights.get(
                    "financial_terms_weight", 1.0
                ) + legal_count * weights.get("legal_proceedings_weight", 1.0)
        else:
            # Fallback to regex patterns (still optimized)
            for pattern_name, pattern in COMPILED_PATTERNS.items():
                if pattern_name in ["proximity", "judgment_verbs", "financial_terms"]:
                    weight = weights.get(f"{pattern_name}_weight", 1.0)

                    # Vectorized pattern matching
                    for i, context in enumerate(contexts):
                        matches = len(pattern.findall(context))
                        scores[i] += matches * weight

        return scores

    def parallel_text_processing(
        self, texts: List[str], process_func, n_jobs: int = None
    ) -> List[Any]:
        """Process texts in parallel for maximum throughput."""
        if n_jobs is None:
            n_jobs = min(mp.cpu_count(), 4)  # Don't overwhelm the system

        if n_jobs <= 1 or len(texts) < 10:
            # Not worth parallelizing for small datasets
            return [process_func(text) for text in texts]

        # Split texts into chunks
        chunk_size = max(1, len(texts) // n_jobs)
        chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]

        # Process chunks in parallel
        with mp.Pool(n_jobs) as pool:
            chunk_results = pool.map(
                lambda chunk: [process_func(text) for text in chunk], chunks
            )

        # Flatten results
        results = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)

        return results

    def optimize_regex_patterns(self) -> Dict[str, re.Pattern]:
        """Return optimized regex patterns for better performance."""
        # Pre-compiled patterns with optimizations
        optimized_patterns = {
            # More specific amount pattern to reduce false positives
            "amount_precise": re.compile(
                r"\$(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{2})?(?:\s*(?:million|billion|M|B))?",
                re.IGNORECASE,
            ),
            # Optimized proximity pattern with word boundaries
            "proximity_optimized": re.compile(
                r"\b(?:settlement|judgment|judgement|damages?|awards?|penalt(?:y|ies)|fines?|amounts?|paid|costs?|prices?|fees?|compensation|restitution)\b",
                re.IGNORECASE,
            ),
            # More precise legal action verbs
            "legal_actions": re.compile(
                r"\b(?:award(?:ed|ing)?|order(?:ed|ing)?|grant(?:ed|ing)?|enter(?:ed|ing)?|assess(?:ed|ing)?|recover(?:ed|y)?|dismiss(?:ed|al)?)\b",
                re.IGNORECASE,
            ),
            # Dismissal patterns with context
            "dismissal_context": re.compile(
                r"\b(?:motion\s+(?:to\s+)?dismiss|dismiss(?:al|ed)|denied|rejected|summary\s+judgment)\b",
                re.IGNORECASE,
            ),
        }

        return optimized_patterns

    def extract_numbers_batch(self, texts: List[str]) -> List[List[float]]:
        """Extract all numbers from texts in batch for faster processing."""
        if not texts:
            return []

        # Vectorized number extraction
        number_pattern = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b")

        results = []
        for text in texts:
            numbers = []
            for match in number_pattern.finditer(text):
                try:
                    num_str = match.group(0).replace(",", "")
                    numbers.append(float(num_str))
                except ValueError:
                    continue
            results.append(numbers)

        return results


def get_fast_processor(fast_mode: bool = True) -> FastTextProcessor:
    """Get a fast text processor instance."""
    return FastTextProcessor(fast_mode=fast_mode)


# Vectorized utility functions
def compute_position_scores_vectorized(
    file_paths: List[str], case_threshold: float = 0.5, docket_threshold: float = 0.5
) -> np.ndarray:
    """Compute position scores for multiple files using vectorized operations."""
    scores = np.zeros(len(file_paths))

    for i, file_path in enumerate(file_paths):
        # Extract case and docket position quickly
        path_parts = file_path.split("/")

        # Simple heuristic: later parts of path = later in case
        if len(path_parts) > 2:
            case_pos = len(path_parts) / 10.0  # Normalize
            docket_pos = hash(file_path) % 100 / 100.0  # Pseudo-random position

            case_score = 1.0 if case_pos > case_threshold else 0.0
            docket_score = 1.0 if docket_pos > docket_threshold else 0.0

            scores[i] = case_score + docket_score

    return scores


def batch_context_extraction(
    texts: List[str], positions: List[Tuple[int, int]], context_chars: int = 500
) -> List[str]:
    """Extract context around positions from multiple texts efficiently."""
    contexts = []

    for text, (start, end) in zip(texts, positions):
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)
        context = text[context_start:context_end].replace("\n", " ")
        contexts.append(context)

    return contexts
