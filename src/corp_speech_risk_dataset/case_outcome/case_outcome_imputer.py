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
import orjson as json
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, NamedTuple

from corp_speech_risk_dataset.case_outcome.extract_cash_amounts_stage1 import (
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
    disable_spacy: bool = False,
    disable_spelled: bool = False,
    disable_usd: bool = False,
    disable_calcs: bool = False,
    disable_regex: bool = False,
) -> List[Candidate]:
    """
    Scan stage1 JSONL files for cash amounts. Each line in stage1 files is a JSON object
    with a 'text' field containing the actual document text.
    Enhanced with spaCy EntityRuler, spelled-out amounts, USD prefixes, judgment-verb filtering,
    and chronological position-based voting.
    """
    seen = set()
    out = []
    all_raw = []  # Track all candidates before filtering

    # Initialize spaCy pipeline once for reuse
    nlp = get_spacy_nlp()

    # Debug: Check if stage1 files exist
    stage1_files = list(case_root.rglob("*_stage1.jsonl"))

    for path in case_root.rglob("*_stage1.jsonl"):
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if not text:
                        continue

                    # Process spaCy EntityRuler amounts (new - highest priority)
                    if not disable_spacy:
                        spacy_candidates = extract_spacy_amounts(
                            text, nlp, min_amount, context_chars
                        )
                        all_raw.extend(spacy_candidates)  # Track raw candidates
                        for candidate in spacy_candidates:
                            ctx = candidate["context"]
                            # Enhanced filtering: require minimum feature votes including position and titles
                            if passes_enhanced_feature_filter_with_titles(
                                ctx,
                                str(path),
                                min_features,
                                case_position_threshold,
                                docket_position_threshold,
                                voting_weights,
                            ):
                                sig = f"{candidate['amount']}:{ctx[:60]}"
                                if sig not in seen:
                                    seen.add(sig)
                                    feature_votes = (
                                        compute_enhanced_feature_votes_with_titles(
                                            ctx,
                                            str(path),
                                            case_position_threshold,
                                            docket_position_threshold,
                                            voting_weights,
                                        )
                                    )
                                    out.append(
                                        Candidate(
                                            candidate["value"],
                                            candidate["amount"],
                                            ctx,
                                            feature_votes,
                                        )
                                    )

                    # Process enhanced spelled-out amounts (new)
                    if not disable_spelled:
                        spelled_matches = list(SPELLED_OUT_AMOUNTS.finditer(text))
                        for m in spelled_matches:
                            val = extract_spelled_out_amount(text, m)
                            all_raw.append(
                                {"amount": m.group(0), "value": val}
                            )  # Track raw
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
                                            )
                                        )
                                        out.append(
                                            Candidate(
                                                val, m.group(0), ctx, feature_votes
                                            )
                                        )

                    # Process enhanced USD amounts (new)
                    if not disable_usd:
                        usd_matches = list(USD_AMOUNTS.finditer(text))
                        for m in usd_matches:
                            val = extract_usd_amount(text, m)
                            all_raw.append(
                                {"amount": m.group(0), "value": val}
                            )  # Track raw
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
                                            )
                                        )
                                        out.append(
                                            Candidate(
                                                val, m.group(0), ctx, feature_votes
                                            )
                                        )

                    # Continue with existing regex extraction (enhanced with judgment-verb filtering)
                    if not disable_regex:
                        regex_matches = list(AMOUNT_REGEX.finditer(text))
                        for m in regex_matches:
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
                                all_raw.append(
                                    {"amount": amt, "value": val}
                                )  # Track raw
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
                            if not passes_enhanced_feature_filter_with_titles(
                                ctx,
                                str(path),
                                min_features,
                                case_position_threshold,
                                docket_position_threshold,
                                voting_weights,
                            ):
                                continue

                            sig = f"{amt}:{ctx[:60]}"
                            if sig in seen:
                                continue
                            seen.add(sig)

                            feature_votes = compute_enhanced_feature_votes_with_titles(
                                ctx,
                                str(path),
                                case_position_threshold,
                                docket_position_threshold,
                                voting_weights,
                            )
                            out.append(Candidate(val, amt, ctx, feature_votes))

                except json.JSONDecodeError:
                    continue

    # Debug summary
    if len(stage1_files) == 0 or len(out) == 0:
        print(
            f"[DEBUG] {case_root.name}: {len(stage1_files)} stage1 files, {len(out)} final candidates"
        )

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
            fout.write(json.dumps(rec).decode() + "\n")
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
    disable_spacy: bool = False,
    disable_spelled: bool = False,
    disable_usd: bool = False,
    disable_calcs: bool = False,
    disable_regex: bool = False,
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
                disable_spacy,
                disable_spelled,
                disable_usd,
                disable_calcs,
                disable_regex,
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
        choices=["auto", "manual", "coverage"],
        default="auto",
        help="Automatic vs manual selection",
    )
    p.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="(coverage mode) CSV of hand-annotated amounts with columns ['case_id','final_amount']",
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
    # Disable flags for pipeline stages
    p.add_argument(
        "--disable-spacy",
        action="store_true",
        help="Turn off all spaCy-based extraction",
    )
    p.add_argument(
        "--disable-spelled",
        action="store_true",
        help="Turn off spelled-out-number extraction",
    )
    p.add_argument(
        "--disable-usd", action="store_true", help="Turn off USD-prefix extraction"
    )
    p.add_argument(
        "--disable-calcs",
        action="store_true",
        help="Turn off any calculation-based extraction (fractions, sums, etc.)",
    )
    p.add_argument(
        "--disable-regex",
        action="store_true",
        help="Turn off the standard AMOUNT_REGEX pass",
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

    def coverage_analysis():
        import numpy as _np

        # load annotations if provided
        ann_map = {}
        if args.annotations:
            import pandas as _pd

            df_ann = _pd.read_csv(args.annotations, dtype={"case_id": str})
            df_ann["final_amount"] = (
                df_ann["final_amount"].astype(str).str.replace(",", "").astype(float)
            )
            # Extract just the case name from paths like "data/extracted/courtlistener/0:14-cv-61344_flsd/"
            df_ann["case_name"] = df_ann["case_id"].apply(
                lambda x: (
                    x.rstrip("/").split("/")[-1] if _pd.notna(x) and x.strip() else None
                )
            )
            # Filter out null/empty case names and create mapping
            valid_rows = df_ann.dropna(subset=["case_name", "final_amount"])
            ann_map = dict(zip(valid_rows["case_name"], valid_rows["final_amount"]))

        print("\nCANDIDATE COVERAGE PER CASE\n" + "=" * 60)
        coverages = []
        top5s = []
        hits = []

        # Only process cases that are in the annotation file
        if len(ann_map) > 0:
            print(
                f"[DEBUG] Annotation mapping contains {len(ann_map)} cases: {list(ann_map.keys())[:5]}..."
            )
            annotated_case_names = list(ann_map.keys())
        else:
            print("[DEBUG] No annotations loaded - processing all cases")
            # Fallback: discover all cases if no annotations
            all_input_files = list(
                tokenized_root.rglob(f"*_stage{args.input_stage}.jsonl")
            )
            annotated_case_names = []
            for input_file in all_input_files:
                rel = input_file.relative_to(tokenized_root)
                case_name = rel.parts[0]
                annotated_case_names.append(case_name)

        print(f"[DEBUG] Processing {len(annotated_case_names)} annotated cases")

        for case_name in sorted(annotated_case_names):
            # run your scan
            candidates = scan_stage1(
                extracted_root / case_name,
                args.min_amount,
                args.context_chars,
                args.min_features,
                args.case_position_threshold,
                args.docket_position_threshold,
                voting_weights,
                disable_spacy=args.disable_spacy,
                disable_spelled=args.disable_spelled,
                disable_usd=args.disable_usd,
                disable_calcs=args.disable_calcs,
                disable_regex=args.disable_regex,
            )
            cnt = len(candidates)
            coverages.append(cnt)
            # avg top‐5 vote‐scores
            votes = sorted([c.feature_votes for c in candidates], reverse=True)[:5]
            avg5 = sum(votes) / len(votes) if votes else 0.0
            top5s.append(avg5)

            # check true‐amount membership
            true_amt = ann_map.get(case_name)
            found = False
            if true_amt is not None:
                import math

                for c in candidates:
                    if math.isclose(c.value, true_amt, rel_tol=1e-4, abs_tol=0.01):
                        found = True
                        break
                hits.append(found)
                flag = "✓" if found else "✗"
                true_str = f"{true_amt:,.0f}"
            else:
                flag = "–"
                true_str = "n/a"

            print(
                f"{case_name:<30}  #cand={cnt:>4}  avg5={avg5:>5.2f}  true={true_str:>10}  hit={flag}"
            )

        total = len(coverages)
        hit_count = sum(hits)
        print("\nSUMMARY\n" + "-" * 60)

        # Handle empty data gracefully
        if coverages:
            print(f"Avg #candidates/case: {_np.mean(coverages):.2f}")
        else:
            print("Avg #candidates/case: 0.00 (no cases processed)")

        if top5s:
            print(f"Avg top-5 votes/case: {_np.mean(top5s):.2f}")
        else:
            print("Avg top-5 votes/case: 0.00 (no cases processed)")

        if hits:
            print(
                f"True‐amt coverage: {hit_count}/{len(hits)} cases ({100*hit_count/len(hits):.1f}%)"
            )
        else:
            print("True‐amt coverage: No annotated cases found")

    if args.mode == "coverage":
        coverage_analysis()
        return

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
            args.disable_spacy,
            args.disable_spelled,
            args.disable_usd,
            args.disable_calcs,
            args.disable_regex,
        )


if __name__ == "__main__":
    sys.exit(main())
