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

from src.corp_speech_risk_dataset.case_outcome.extract_cash_amounts_stage1 import (
    AMOUNT_REGEX,
    PROXIMITY_PATTERN,
    JUDGMENT_VERBS,
    SPELLED_OUT_AMOUNTS,
    USD_AMOUNTS,
    extract_spelled_out_amount,
    extract_usd_amount,
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

    # Initialize spaCy pipeline once for reuse (skip in fast mode for speed)
    nlp = None if fast_mode else get_spacy_nlp()

    for path in case_root.rglob("*_stage1.jsonl"):
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                try:
                    data = json.loads(line)
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
                            # Enhanced filtering: require minimum feature votes including position and titles
                            feature_votes = compute_enhanced_feature_votes_with_titles(
                                ctx,
                                str(path),
                                case_position_threshold,
                                docket_position_threshold,
                                voting_weights,
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
                            # Enhanced filtering: require minimum feature votes including position and titles
                            if passes_enhanced_feature_filter_with_titles(
                                ctx,
                                str(path),
                                min_features,
                                case_position_threshold,
                                docket_position_threshold,
                                voting_weights,
                                header_chars,
                            ):
                                sig = f"{m.group(0)}:{ctx[:60]}"
                                if sig not in seen:
                                    seen.add(sig)
                                    feature_votes = (
                                        compute_enhanced_feature_votes_with_titles(
                                            ctx,
                                            str(path),
                                            case_position_threshold,
                                            docket_position_threshold,
                                            voting_weights,
                                            header_chars,
                                        )
                                    )
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
                            # Enhanced filtering: require minimum feature votes including position and titles
                            if passes_enhanced_feature_filter_with_titles(
                                ctx,
                                str(path),
                                min_features,
                                case_position_threshold,
                                docket_position_threshold,
                                voting_weights,
                                header_chars,
                            ):
                                sig = f"{m.group(0)}:{ctx[:60]}"
                                if sig not in seen:
                                    seen.add(sig)
                                    feature_votes = (
                                        compute_enhanced_feature_votes_with_titles(
                                            ctx,
                                            str(path),
                                            case_position_threshold,
                                            docket_position_threshold,
                                            voting_weights,
                                            header_chars,
                                        )
                                    )
                                    out.append(
                                        Candidate(val, m.group(0), ctx, feature_votes)
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

                        # Enhanced filtering: require minimum feature votes including position and titles
                        feature_votes = compute_enhanced_feature_votes_with_titles(
                            ctx,
                            str(path),
                            case_position_threshold,
                            docket_position_threshold,
                            voting_weights,
                            header_chars,
                        )
                        if not fast_mode:
                            print(
                                f"🔍 Regex candidate '{amt}' -> {feature_votes} votes (need {min_features})"
                            )

                        if not passes_enhanced_feature_filter_with_titles(
                            ctx,
                            str(path),
                            min_features,
                            case_position_threshold,
                            docket_position_threshold,
                            voting_weights,
                            header_chars,
                        ):
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
            rec = json.loads(line)
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
    dismissal_ratio_threshold: float = 0.5,
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
    )

    # Check if this is a bankruptcy court case
    court_type = get_case_court_type(extracted_case_root, bankruptcy_ratio_threshold)
    if court_type == "BANKRUPTCY":
        amount = None
        print(
            f"▶ {case_root.relative_to(tokenized_root)} → BANKRUPTCY COURT (auto-null)"
        )
    else:
        # Check if this is a dismissed case
        if flags["is_dismissed"]:
            amount = 0.0
            print(
                f"▶ {case_root.relative_to(tokenized_root)} → DISMISSED CASE (auto-zero)"
            )
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

    # Print flags if any are raised
    flag_messages = []
    if flags["has_fee_shifting"]:
        flag_messages.append("🚩 FEE-SHIFTING")
    if flags["has_large_patent_amounts"]:
        flag_messages.append("🚩 LARGE PATENT AMOUNTS")

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
        default=20.0,
        help="Minimum ratio of patent occurrences per document to flag (default: 20.0)",
    )
    p.add_argument(
        "--dismissal-ratio-threshold",
        type=float,
        default=0.5,
        help="Minimum ratio of documents with dismissal language to flag (default: 0.5)",
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
            args.bankruptcy_ratio_threshold,
            voting_weights,
            args.header_chars,
        )


if __name__ == "__main__":
    sys.exit(main())
