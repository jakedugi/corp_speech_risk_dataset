#!/usr/bin/env python3
"""
case_outcome_imputer.py

Adds `final_judgement_real` to every `*_stage4.jsonl` record, producing
`*_stage5.jsonl` files.  Can run in automatic (largest amount) or manual
(prompt‐driven) modes, and supports overriding context size and minimum
amount thresholds.
"""

from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, NamedTuple
from fractions import Fraction
import re

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


from src.corp_speech_risk_dataset.case_outcome.extract_cash_amounts_stage1 import (
    AMOUNT_REGEX,
    PROXIMITY_PATTERN,
    JUDGMENT_VERBS,
    SPELLED_OUT_AMOUNTS,
    USD_AMOUNTS,
    extract_spelled_out_amount,
    extract_usd_amount,
    extract_calculated_amounts,
    extract_damage_component_totals,
    extract_attorney_fees_expenses,
    extract_multi_component_settlements,
    extract_settlement_benefit_totals,
    extract_smart_sum_amounts,
    is_case_definitively_dismissed,
    get_spacy_nlp,
    extract_spacy_amounts,
    passes_feature_filter,
    passes_enhanced_feature_filter,
    passes_enhanced_feature_filter_with_titles,
    compute_feature_votes,
    compute_enhanced_feature_votes,
    compute_enhanced_feature_votes_with_titles,
    CONTEXT_CHARS as DEFAULT_CONTEXT,
    DEFAULT_MIN_AMOUNT as DEFAULT_MIN,
    get_case_court_type,
    is_case_dismissed,
    get_case_flags,
    VotingWeights,
    DEFAULT_VOTING_WEIGHTS,
)

# 🚀 OPTIMIZATION: Import fast text processing for vectorized operations
try:
    from src.corp_speech_risk_dataset.case_outcome.fast_text_processing import (
        FastTextProcessor,
        get_fast_processor,
        compute_position_scores_vectorized,
    )

    FAST_PROCESSING_AVAILABLE = True
except ImportError:
    FAST_PROCESSING_AVAILABLE = False

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


# ------------------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------------------


class Candidate(NamedTuple):
    value: float
    raw_text: str
    context: str
    feature_votes: int


class AmountSelector:
    def choose(self, candidates: List[Candidate]) -> float | None:
        if not candidates:
            return None
        # Sort by feature votes (descending), then by value (descending)
        sorted_candidates = sorted(
            candidates, key=lambda c: (c.feature_votes, c.value), reverse=True
        )
        return sorted_candidates[0].value


class ManualAmountSelector(AmountSelector):
    def choose(self, candidates: List[Candidate]) -> float | None:
        if not candidates:
            print("⚠ No candidate amounts found.")
            return None

        # Sort by feature votes (descending), then by value (descending)
        sorted_candidates = sorted(
            candidates, key=lambda c: (c.feature_votes, c.value), reverse=True
        )

        print("\n── Candidates (ranked by feature votes) ───────────────────────")
        for i, c in enumerate(sorted_candidates, 1):
            print(f"[{i}] {c.value:,.0f} (votes: {c.feature_votes})\t…{c.context}…")
        while True:
            choice = input("\nPick #, 's' to skip, or custom » ").strip()
            if choice.lower() == "s":
                return None
            if choice.isdigit() and 1 <= int(choice) <= len(sorted_candidates):
                return sorted_candidates[int(choice) - 1].value
            try:
                return float(choice.replace(",", ""))
            except ValueError:
                print("Invalid input—try again.")


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def scan_stage1(
    case_root: Path,
    min_amount: float,
    context_chars: int,
    min_features: int = 2,
    case_position_threshold: float = 0.5,
    docket_position_threshold: float = 0.5,
    voting_weights: VotingWeights = DEFAULT_VOTING_WEIGHTS,
    header_chars: int = 2000,
    fast_mode: bool = False,  # New parameter for optimization speed
) -> List[Candidate]:
    """
    Scan stage1 JSONL files for cash amounts. Each line in stage1 files is a JSON object
    with a 'text' field containing the actual document text.
    Enhanced with spaCy EntityRuler, spelled-out amounts, USD prefixes, judgment-verb filtering,
    and chronological position-based voting.

    Args:
        fast_mode: If True, disables print statements and reduces I/O for faster optimization
    """
    seen = set()
    out = []

    # 🚀 OPTIMIZATION: Completely disable spaCy in fast mode for maximum speed
    nlp = None if fast_mode else get_spacy_nlp()

    # 🚀 ULTRA-FAST MODE: Use vectorized processing for maximum speed
    if fast_mode and FAST_PROCESSING_AVAILABLE:
        fast_processor = get_fast_processor(fast_mode=True)

        # Collect all texts for batch processing
        all_texts = []
        text_metadata = []

        for path in case_root.rglob("*_stage1.jsonl"):
            with open(path, "r", encoding="utf8") as f:
                for line_num, line in enumerate(f):
                    try:
                        data = fast_json_loads(line)
                        text = data.get("text", "")
                        if text:
                            all_texts.append(text)
                            text_metadata.append(
                                {"path": str(path), "line_num": line_num, "data": data}
                            )
                    except json.JSONDecodeError:
                        continue

        if all_texts:
            # Batch extract amounts using vectorized operations
            batch_amounts = fast_processor.extract_amounts_vectorized(all_texts)

            # Process results
            for text_amounts, metadata in zip(batch_amounts, text_metadata):
                for amount_info in text_amounts:
                    if amount_info["value"] >= min_amount:
                        # Quick feature scoring (simplified for speed)
                        feature_votes = 2  # Base score for fast mode

                        sig = f"{amount_info['raw_text']}:{amount_info['context'][:60]}"
                        if sig not in seen:
                            seen.add(sig)
                            out.append(
                                Candidate(
                                    amount_info["value"],
                                    amount_info["raw_text"],
                                    amount_info["context"],
                                    feature_votes,
                                )
                            )

        return out

    # Standard processing mode
    for path in case_root.rglob("*_stage1.jsonl"):
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                try:
                    data = fast_json_loads(line)
                    text = data.get("text", "")
                    if not text:
                        continue

                    # Process spaCy EntityRuler amounts (skip in fast mode)
                    if not fast_mode:
                        spacy_candidates = extract_spacy_amounts(
                            text, nlp, min_amount, context_chars, fast_mode
                        )
                        for candidate in spacy_candidates:
                            ctx = candidate["context"]
                            # Enhanced filtering: require minimum feature votes including all new features
                            feature_votes = compute_enhanced_feature_votes_with_titles(
                                ctx,
                                str(path),
                                case_position_threshold,
                                docket_position_threshold,
                                voting_weights,
                                header_chars,
                                nlp,
                                text,
                                candidate.get("start", 0),
                                candidate.get("end", 0),
                            )
                            if not fast_mode:
                                print(
                                    f"🔍 spaCy candidate '{candidate['amount']}' -> {feature_votes} votes (need {min_features})"
                                )

                            if passes_enhanced_feature_filter_with_titles(
                                ctx,
                                str(path),
                                min_features,
                                case_position_threshold,
                                docket_position_threshold,
                                voting_weights,
                                header_chars,
                                nlp,
                                text,
                                candidate.get("start", 0),
                                candidate.get("end", 0),
                            ):
                                sig = f"{candidate['amount']}:{ctx[:60]}"
                                if sig not in seen:
                                    seen.add(sig)
                                    out.append(
                                        Candidate(
                                            candidate["value"],
                                            candidate["amount"],
                                            ctx,
                                            feature_votes,
                                        )
                                    )
                            elif not fast_mode:
                                print(
                                    f"❌ Filtered out: '{candidate['amount']}' (votes: {feature_votes}, need: {min_features})"
                                )

                    # Process enhanced spelled-out amounts (new)
                    for m in SPELLED_OUT_AMOUNTS.finditer(text):
                        val = extract_spelled_out_amount(text, m)
                        if val >= min_amount:
                            start, end = m.span()
                            ctx = text[
                                max(0, start - context_chars) : end + context_chars
                            ].replace("\n", " ")
                            # Enhanced filtering: require minimum feature votes including all new features
                            feature_votes = compute_enhanced_feature_votes_with_titles(
                                ctx,
                                str(path),
                                case_position_threshold,
                                docket_position_threshold,
                                voting_weights,
                                header_chars,
                                nlp,
                                text,
                                start,
                                end,
                            )
                            if feature_votes >= min_features:
                                sig = f"{m.group(0)}:{ctx[:60]}"
                                if sig not in seen:
                                    seen.add(sig)
                                    out.append(
                                        Candidate(val, m.group(0), ctx, feature_votes)
                                    )

                    # Process enhanced USD amounts (new)
                    for m in USD_AMOUNTS.finditer(text):
                        val = extract_usd_amount(text, m)
                        if val >= min_amount:
                            start, end = m.span()
                            ctx = text[
                                max(0, start - context_chars) : end + context_chars
                            ].replace("\n", " ")
                            # Enhanced filtering: require minimum feature votes including all new features
                            feature_votes = compute_enhanced_feature_votes_with_titles(
                                ctx,
                                str(path),
                                case_position_threshold,
                                docket_position_threshold,
                                voting_weights,
                                header_chars,
                                nlp,
                                text,
                                start,
                                end,
                            )
                            if feature_votes >= min_features:
                                sig = f"{m.group(0)}:{ctx[:60]}"
                                if sig not in seen:
                                    seen.add(sig)
                                    out.append(
                                        Candidate(val, m.group(0), ctx, feature_votes)
                                    )

                    # QUICK WIN: Extract calculated amounts (class member multiplication, one-third, multi-fund)
                    calculated_candidates = extract_calculated_amounts(text, min_amount)
                    for calc in calculated_candidates:
                        start, end = calc["start"], calc["end"]
                        ctx = text[
                            max(0, start - context_chars) : end + context_chars
                        ].replace("\n", " ")

                        feature_votes = compute_enhanced_feature_votes_with_titles(
                            ctx,
                            str(path),
                            case_position_threshold,
                            docket_position_threshold,
                            voting_weights,
                            header_chars,
                            nlp,
                            text,
                            start,
                            end,
                        )

                        # Apply calculation boost to improve ranking
                        if "calculation_boost" in calc:
                            feature_votes += calc["calculation_boost"]

                        if feature_votes >= min_features:
                            sig = f"{calc['raw_text']}:{ctx[:60]}"
                            if sig not in seen:
                                seen.add(sig)
                                out.append(
                                    Candidate(
                                        calc["value"],
                                        calc["raw_text"],
                                        ctx,
                                        feature_votes,
                                    )
                                )

                    # QUICK WIN: Extract damage component totals
                    damage_candidates = extract_damage_component_totals(
                        text, min_amount
                    )
                    for damage in damage_candidates:
                        start, end = damage["start"], damage["end"]
                        ctx = text[
                            max(0, start - context_chars) : end + context_chars
                        ].replace("\n", " ")

                        feature_votes = compute_enhanced_feature_votes_with_titles(
                            ctx,
                            str(path),
                            case_position_threshold,
                            docket_position_threshold,
                            voting_weights,
                            header_chars,
                            nlp,
                            text,
                            start,
                            end,
                        )

                        # Apply calculation boost to improve ranking
                        if "calculation_boost" in damage:
                            feature_votes += damage["calculation_boost"]

                        if feature_votes >= min_features:
                            sig = f"{damage['raw_text']}:{ctx[:60]}"
                            if sig not in seen:
                                seen.add(sig)
                                out.append(
                                    Candidate(
                                        damage["value"],
                                        damage["raw_text"],
                                        ctx,
                                        feature_votes,
                                    )
                                )

                    # NEW: Extract attorney fees + expenses calculations
                    attorney_candidates = extract_attorney_fees_expenses(
                        text, min_amount
                    )
                    for attorney in attorney_candidates:
                        start, end = attorney["start"], attorney["end"]
                        ctx = text[
                            max(0, start - context_chars) : end + context_chars
                        ].replace("\n", " ")

                        feature_votes = compute_enhanced_feature_votes_with_titles(
                            ctx,
                            str(path),
                            case_position_threshold,
                            docket_position_threshold,
                            voting_weights,
                            header_chars,
                            nlp,
                            text,
                            start,
                            end,
                        )

                        if "calculation_boost" in attorney:
                            feature_votes += attorney["calculation_boost"]

                        if feature_votes >= min_features:
                            sig = f"{attorney['raw_text']}:{ctx[:60]}"
                            if sig not in seen:
                                seen.add(sig)
                                out.append(
                                    Candidate(
                                        attorney["value"],
                                        attorney["raw_text"],
                                        ctx,
                                        feature_votes,
                                    )
                                )

                    # NEW: Extract multi-component settlements
                    multi_candidates = extract_multi_component_settlements(
                        text, min_amount
                    )
                    for multi in multi_candidates:
                        start, end = multi["start"], multi["end"]
                        ctx = text[
                            max(0, start - context_chars) : end + context_chars
                        ].replace("\n", " ")

                        feature_votes = compute_enhanced_feature_votes_with_titles(
                            ctx,
                            str(path),
                            case_position_threshold,
                            docket_position_threshold,
                            voting_weights,
                            header_chars,
                            nlp,
                            text,
                            start,
                            end,
                        )

                        if "calculation_boost" in multi:
                            feature_votes += multi["calculation_boost"]

                        if feature_votes >= min_features:
                            sig = f"{multi['raw_text']}:{ctx[:60]}"
                            if sig not in seen:
                                seen.add(sig)
                                out.append(
                                    Candidate(
                                        multi["value"],
                                        multi["raw_text"],
                                        ctx,
                                        feature_votes,
                                    )
                                )

                    # NEW: Extract settlement benefit totals
                    benefit_candidates = extract_settlement_benefit_totals(
                        text, min_amount
                    )
                    for benefit in benefit_candidates:
                        start, end = benefit["start"], benefit["end"]
                        ctx = text[
                            max(0, start - context_chars) : end + context_chars
                        ].replace("\n", " ")

                        feature_votes = compute_enhanced_feature_votes_with_titles(
                            ctx,
                            str(path),
                            case_position_threshold,
                            docket_position_threshold,
                            voting_weights,
                            header_chars,
                            nlp,
                            text,
                            start,
                            end,
                        )

                        if "calculation_boost" in benefit:
                            feature_votes += benefit["calculation_boost"]

                        if feature_votes >= min_features:
                            sig = f"{benefit['raw_text']}:{ctx[:60]}"
                            if sig not in seen:
                                seen.add(sig)
                                out.append(
                                    Candidate(
                                        benefit["value"],
                                        benefit["raw_text"],
                                        ctx,
                                        feature_votes,
                                    )
                                )

                    # NEW: Extract smart sum amounts (multiple amounts with additive language)
                    smart_sum_candidates = extract_smart_sum_amounts(text, min_amount)
                    for smart_sum in smart_sum_candidates:
                        start, end = smart_sum["start"], smart_sum["end"]
                        ctx = text[
                            max(0, start - context_chars) : end + context_chars
                        ].replace("\n", " ")

                        feature_votes = compute_enhanced_feature_votes_with_titles(
                            ctx,
                            str(path),
                            case_position_threshold,
                            docket_position_threshold,
                            voting_weights,
                            header_chars,
                            nlp,
                            text,
                            start,
                            end,
                        )

                        if "calculation_boost" in smart_sum:
                            feature_votes += smart_sum["calculation_boost"]

                        if feature_votes >= min_features:
                            sig = f"{smart_sum['raw_text']}:{ctx[:60]}"
                            if sig not in seen:
                                seen.add(sig)
                                out.append(
                                    Candidate(
                                        smart_sum["value"],
                                        smart_sum["raw_text"],
                                        ctx,
                                        feature_votes,
                                    )
                                )

                    # Continue with existing regex extraction (enhanced with judgment-verb filtering)
                    for m in AMOUNT_REGEX.finditer(text):
                        amt = m.group(0)
                        # strip punctuation but leave "million"/"billion" suffix attached
                        norm = (
                            amt.lower()
                            .replace(",", "")
                            .replace("$", "")
                            .replace("usd", "")
                            .strip()
                        )

                        # Calculate actual value
                        if "million" in norm:
                            multiplier = 1_000_000
                            num_str = norm.replace("million", "").strip()
                        elif "billion" in norm:
                            multiplier = 1_000_000_000
                            num_str = norm.replace("billion", "").strip()
                        else:
                            multiplier = 1
                            num_str = norm

                        try:
                            val = float(num_str) * multiplier
                        except ValueError:
                            continue

                        # Apply minimum threshold
                        if val < min_amount:
                            continue

                        start, end = m.span()
                        ctx = text[
                            max(0, start - context_chars) : end + context_chars
                        ].replace("\n", " ")

                        # Enhanced filtering: require minimum feature votes including all new features
                        feature_votes = compute_enhanced_feature_votes_with_titles(
                            ctx,
                            str(path),
                            case_position_threshold,
                            docket_position_threshold,
                            voting_weights,
                            header_chars,
                            nlp,
                            text,
                            start,
                            end,
                        )
                        if not fast_mode:
                            print(
                                f"🔍 Regex candidate '{amt}' -> {feature_votes} votes (need {min_features})"
                            )

                        if feature_votes < min_features:
                            if not fast_mode:
                                print(
                                    f"❌ Filtered out: '{amt}' (votes: {feature_votes}, need: {min_features})"
                                )
                            continue

                        sig = f"{amt}:{ctx[:60]}"
                        if sig in seen:
                            continue
                        seen.add(sig)

                        out.append(Candidate(val, amt, ctx, feature_votes))

                except json.JSONDecodeError:
                    continue
    return out


def rewrite_stage_file(
    input_file: Path,
    amount: float | None,
    outdir: Path | None,
    tokenized_root: Path,
    input_stage: int,
    output_stage: int,
):
    rel = input_file.relative_to(tokenized_root)
    rel_tok = input_file.relative_to(tokenized_root)
    target = (outdir or tokenized_root) / rel_tok.parent
    target.mkdir(parents=True, exist_ok=True)
    outname = input_file.name.replace(
        f"_stage{input_stage}.jsonl", f"_stage{output_stage}.jsonl"
    )
    tmp = target / (outname + ".tmp")

    with input_file.open(encoding="utf8") as fin, tmp.open(
        "w", encoding="utf8"
    ) as fout:
        for line in fin:
            rec = fast_json_loads(line)
            rec["final_judgement_real"] = amount
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(target / outname)


def impute_for_case(
    case_root: Path,
    selector: AmountSelector,
    min_amount: float,
    context_chars: int,
    min_features: int,
    tokenized_root: Path,
    extracted_root: Path,
    outdir: Path | None,
    input_stage: int,
    output_stage: int,
    case_position_threshold: float = 0.5,
    docket_position_threshold: float = 0.5,
    fee_shifting_ratio_threshold: float = 1.0,
    patent_ratio_threshold: float = 20.0,
    dismissal_ratio_threshold: float = 0.05,
    dismissal_use_weighted_scoring: bool = True,
    dismissal_document_type_weight: float = 2.0,
    bankruptcy_ratio_threshold: float = 0.5,
    voting_weights: VotingWeights = DEFAULT_VOTING_WEIGHTS,
    header_chars: int = 2000,
):
    # map this tokenized-case back to its extracted-location
    # map this case folder directly into the extracted tree by name
    extracted_case_root = extracted_root / case_root.name

    # Get case flags with configurable thresholds
    flags = get_case_flags(
        extracted_case_root,
        fee_shifting_ratio_threshold,
        patent_ratio_threshold,
        dismissal_ratio_threshold,
        bankruptcy_ratio_threshold,
        dismissal_use_weighted_scoring,
        dismissal_document_type_weight,
    )

    # Check if this is a bankruptcy court case
    court_type = get_case_court_type(extracted_case_root, bankruptcy_ratio_threshold)
    if court_type == "BANKRUPTCY":
        amount = None
        print(
            f"▶ {case_root.relative_to(tokenized_root)} → BANKRUPTCY COURT (auto-null)"
        )
    # Check if this is a high patent ratio case (hard filter)
    elif flags["has_large_patent_amounts"]:
        amount = None
        print(
            f"▶ {case_root.relative_to(tokenized_root)} → HIGH PATENT RATIO (auto-null)"
        )
    else:
        # Check if this is a DEFINITIVELY dismissed case (using more nuanced logic)
        is_definitively_dismissed = is_case_definitively_dismissed(
            extracted_case_root,
            strict_dismissal_threshold=0.9,  # Very high threshold
            document_type_weight=3.0,
        )

        if is_definitively_dismissed:
            amount = 0.0
            print(
                f"▶ {case_root.relative_to(tokenized_root)} → DEFINITIVELY DISMISSED (auto-zero)"
            )
        # For regular dismissal flags, use reduced thresholds but still proceed with extraction
        elif flags["is_dismissed"]:
            # Proceed with extraction but use more permissive parameters
            print(
                f"▶ {case_root.relative_to(tokenized_root)} → Possible dismissal (proceeding with relaxed parameters)"
            )
            candidates = scan_stage1(
                extracted_case_root,
                min_amount * 0.5,  # Lower minimum amount
                context_chars,
                min_features - 1,  # Lower feature threshold
                case_position_threshold,
                docket_position_threshold,
                voting_weights,
                header_chars,
            )
            amount = selector.choose(candidates)
        else:
            candidates = scan_stage1(
                extracted_case_root,
                min_amount,
                context_chars,
                min_features,
                case_position_threshold,
                docket_position_threshold,
                voting_weights,
                header_chars,
            )
            amount = selector.choose(candidates)
            print(f"▶ {case_root.relative_to(tokenized_root)} → {amount!r}")

    # Print flags if any are raised (excluding the ones that trigger hard filters)
    flag_messages = []
    if flags["has_fee_shifting"]:
        flag_messages.append("🚩 FEE-SHIFTING")

    if flag_messages:
        print(f"   {' | '.join(flag_messages)}")

    for input_file in case_root.rglob(f"*_stage{input_stage}.jsonl"):
        rewrite_stage_file(
            input_file, amount, outdir, tokenized_root, input_stage, output_stage
        )


def parse_args():
    p = argparse.ArgumentParser(description="Add final_judgement_real to stage-5 JSONL")
    p.add_argument(
        "--root", type=Path, required=True, help="Tokenized stage4 tree root"
    )
    p.add_argument(
        "--stage1-root",
        type=Path,
        default=None,
        help="Alternative tree to scan stage1 JSONL",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Mirror stage5 output under this directory",
    )
    p.add_argument(
        "--mode",
        choices=["auto", "manual"],
        default="auto",
        help="Automatic vs manual selection",
    )
    p.add_argument(
        "--context-chars",
        type=int,
        default=DEFAULT_CONTEXT,
        help="Chars of context around each amount",
    )
    p.add_argument(
        "--min-amount",
        type=float,
        default=DEFAULT_MIN,
        help="Skip values below this threshold",
    )
    p.add_argument(
        "--min-features",
        type=int,
        default=2,
        help="Minimum number of heuristic feature votes needed to pass filter (each pattern match = +1)",
    )
    p.add_argument(
        "--input-stage",
        type=int,
        default=4,
        help="Input stage number (default: 4)",
    )
    p.add_argument(
        "--output-stage",
        type=int,
        default=5,
        help="Output stage number (default: 5)",
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
    p.add_argument(
        "--fee-shifting-ratio-threshold",
        type=float,
        default=1.0,
        help="Minimum ratio of fee-shifting occurrences per document to flag (default: 1.0)",
    )
    p.add_argument(
        "--patent-ratio-threshold",
        type=float,
        default=50.0,  # Increased from 20.0 to 50.0 for hard filter
        help="Minimum ratio of patent occurrences per document to trigger hard filter (default: 50.0)",
    )
    p.add_argument(
        "--dismissal-ratio-threshold",
        type=float,
        default=0.05,
        help="Minimum ratio of documents with dismissal language to flag (default: 0.05)",
    )
    p.add_argument(
        "--use-weighted-dismissal-scoring",
        action="store_true",
        default=True,
        help="Use weighted scoring for dismissal detection (default: True)",
    )
    p.add_argument(
        "--dismissal-document-type-weight",
        type=float,
        default=2.0,
        help="Weight for document type matches in dismissal scoring (default: 2.0)",
    )
    p.add_argument(
        "--bankruptcy-ratio-threshold",
        type=float,
        default=0.5,
        help="Minimum ratio of bankruptcy court documents to flag (default: 0.5)",
    )
    # Voting weights arguments
    p.add_argument(
        "--proximity-pattern-weight",
        type=float,
        default=1.0,
        help="Weight for monetary context words (default: 1.0)",
    )
    p.add_argument(
        "--judgment-verbs-weight",
        type=float,
        default=1.0,
        help="Weight for legal action verbs (default: 1.0)",
    )
    p.add_argument(
        "--case-position-weight",
        type=float,
        default=1.0,
        help="Weight for chronological position within case (default: 1.0)",
    )
    p.add_argument(
        "--docket-position-weight",
        type=float,
        default=1.0,
        help="Weight for chronological position within docket (default: 1.0)",
    )
    p.add_argument(
        "--all-caps-titles-weight",
        type=float,
        default=1.0,
        help="Weight for ALL CAPS section titles (default: 1.0)",
    )
    p.add_argument(
        "--document-titles-weight",
        type=float,
        default=1.0,
        help="Weight for document titles (default: 1.0)",
    )
    p.add_argument(
        "--header-chars",
        type=int,
        default=2000,
        help="Number of characters to consider as header for document titles (default: 2000)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    selector = ManualAmountSelector() if args.mode == "manual" else AmountSelector()
    tokenized_root = args.root
    extracted_root = args.stage1_root or args.root

    # Create voting weights from command line arguments
    voting_weights = VotingWeights(
        proximity_pattern_weight=args.proximity_pattern_weight,
        judgment_verbs_weight=args.judgment_verbs_weight,
        case_position_weight=args.case_position_weight,
        docket_position_weight=args.docket_position_weight,
        all_caps_titles_weight=args.all_caps_titles_weight,
        document_titles_weight=args.document_titles_weight,
    )

    # discover each CASE folder by taking the first path segment under --root
    all_input_files = list(tokenized_root.rglob(f"*_stage{args.input_stage}.jsonl"))
    roots: set[Path] = set()
    for input_file in all_input_files:
        rel = input_file.relative_to(
            tokenized_root
        )  # e.g. "0:14-cv-61344_flsd/entries/…/doc.jsonl"
        case_name = rel.parts[0]  # "0:14-cv-61344_flsd"
        roots.add(
            tokenized_root / case_name
        )  # → data/tokenized/courtlistener/0:14-cv-61344_flsd

    for case_root in sorted(roots):
        impute_for_case(
            case_root,
            selector,
            args.min_amount,
            args.context_chars,
            args.min_features,
            tokenized_root,
            extracted_root,
            args.outdir,
            args.input_stage,
            args.output_stage,
            args.case_position_threshold,
            args.docket_position_threshold,
            args.fee_shifting_ratio_threshold,
            args.patent_ratio_threshold,
            args.dismissal_ratio_threshold,
            args.use_weighted_dismissal_scoring,
            args.dismissal_document_type_weight,
            args.bankruptcy_ratio_threshold,
            voting_weights,
            args.header_chars,
        )


if __name__ == "__main__":
    sys.exit(main())
