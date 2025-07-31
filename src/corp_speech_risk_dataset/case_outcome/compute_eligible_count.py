#!/usr/bin/env python3
"""
compute_eligible_count.py

Extract candidates from sentences with complex amount patterns.
Can compute eligible counts and extract amounts that might be missed by standard patterns.
"""

import re
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AmountCandidate:
    """Represents a potential amount found in text."""

    value: float
    raw_text: str
    context: str
    confidence: float = 1.0


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
) -> List[AmountCandidate]:
    """
    Extract complex amount patterns that might be missed by standard extraction.

    Patterns:
    1. "X class members who are eligible for Y amount"
    2. "total of X million"
    3. "up to X dollars"
    4. "settlement fund of X"
    5. "awarded X in damages"
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
                AmountCandidate(
                    value=total_fund,
                    raw_text=match.group(0),
                    context=text[max(0, match.start() - 100) : match.end() + 100],
                    confidence=0.9,
                )
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
                AmountCandidate(
                    value=total,
                    raw_text=match.group(0),
                    context=text[max(0, match.start() - 100) : match.end() + 100],
                    confidence=0.8,
                )
            )

    # Pattern 3: "up to X dollars/million/billion"
    pattern3 = re.compile(
        r"up\s+to\s+\$?(\d{1,3}(?:,\d{3})*)\s*(dollars?|million|billion)", re.IGNORECASE
    )

    for match in pattern3.finditer(text):
        amount = float(match.group(1).replace(",", ""))
        unit = match.group(2).lower()
        if unit == "dollars" or unit == "dollar":
            total = amount
        elif unit == "million":
            total = amount * 1_000_000
        elif unit == "billion":
            total = amount * 1_000_000_000

        if total >= min_amount:
            candidates.append(
                AmountCandidate(
                    value=total,
                    raw_text=match.group(0),
                    context=text[max(0, match.start() - 100) : match.end() + 100],
                    confidence=0.7,
                )
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
                AmountCandidate(
                    value=total,
                    raw_text=match.group(0),
                    context=text[max(0, match.start() - 100) : match.end() + 100],
                    confidence=0.85,
                )
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
                AmountCandidate(
                    value=total,
                    raw_text=match.group(0),
                    context=text[max(0, match.start() - 100) : match.end() + 100],
                    confidence=0.9,
                )
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
                AmountCandidate(
                    value=total,
                    raw_text=match.group(0),
                    context=text[max(0, match.start() - 100) : match.end() + 100],
                    confidence=0.8,
                )
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
                AmountCandidate(
                    value=total_fund,
                    raw_text=match.group(0),
                    context=text[max(0, match.start() - 100) : match.end() + 100],
                    confidence=0.95,
                )
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
                AmountCandidate(
                    value=total_fund,
                    raw_text=match.group(0),
                    context=text[max(0, match.start() - 100) : match.end() + 100],
                    confidence=0.95,
                )
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
                AmountCandidate(
                    value=total,
                    raw_text=match.group(0),
                    context=text[max(0, match.start() - 100) : match.end() + 100],
                    confidence=0.9,
                )
            )

    return candidates


def test_complex_extraction():
    """Test the complex extraction patterns."""
    test_texts = [
        "That leaves at least 588,887 class members who are eligible for a single $175 TSB repair subsidy should they demonstrate an exhaust smell.",
        "The total value of the proposed settlement is $5.73 million, comprised of: (1) $3 million made available to pay Class member claims; (2) $500,000 made available to pay identity theft losses.",
        "Plaintiffs were awarded $89,600,000 in damages for the breach of contract.",
        "The settlement fund of $108,055,225 will be distributed among class members.",
        "Defendant agreed to pay up to $7,600,000 in settlement costs.",
    ]

    print("🧪 Testing Complex Amount Extraction")
    print("=" * 60)

    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}:")
        print(f"Text: {text}")

        candidates = extract_complex_amount_candidates(text)

        if candidates:
            print("✅ Found candidates:")
            for j, candidate in enumerate(candidates, 1):
                print(
                    f"   {j}. ${candidate.value:,.0f} (confidence: {candidate.confidence:.2f})"
                )
                print(f"      Raw: '{candidate.raw_text}'")
                print(f"      Context: ...{candidate.context[:80]}...")
        else:
            print("❌ No candidates found")

        # Also test the eligible count function
        try:
            eligible_count = compute_eligible_count(text)
            print(f"   📊 Eligible count: {eligible_count:,}")
        except ValueError as e:
            print(f"   📊 Eligible count: {e}")


if __name__ == "__main__":
    test_complex_extraction()
