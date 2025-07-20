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
    CONTEXT_CHARS as DEFAULT_CONTEXT,
    DEFAULT_MIN_AMOUNT as DEFAULT_MIN,
)

# ------------------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------------------


class Candidate(NamedTuple):
    value: float
    raw_text: str
    context: str


class AmountSelector:
    def choose(self, candidates: List[Candidate]) -> float | None:
        if not candidates:
            return None
        return max(c.value for c in candidates)


class ManualAmountSelector(AmountSelector):
    def choose(self, candidates: List[Candidate]) -> float | None:
        if not candidates:
            print("⚠ No candidate amounts found.")
            return None
        print("\n── Candidates ───────────────────────────────────")
        for i, c in enumerate(candidates, 1):
            print(f"[{i}] {c.value:,.0f}\t…{c.context}…")
        while True:
            choice = input("\nPick #, 's' to skip, or custom » ").strip()
            if choice.lower() == "s":
                return None
            if choice.isdigit() and 1 <= int(choice) <= len(candidates):
                return candidates[int(choice) - 1].value
            try:
                return float(choice.replace(",", ""))
            except ValueError:
                print("Invalid input—try again.")


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def scan_stage1(
    case_root: Path, min_amount: float, context_chars: int
) -> List[Candidate]:
    seen = set()
    out = []
    for path in case_root.rglob("*_stage1.jsonl"):
        text = path.read_text(encoding="utf8")
        for m in AMOUNT_REGEX.finditer(text):
            amt = m.group(0)
            norm = amt.lower().replace(",", "").replace("$", "")
            if "million" not in norm and "billion" not in norm:
                try:
                    val = float(norm)
                    if val < min_amount:
                        continue
                except ValueError:
                    continue
            start, end = m.span()
            ctx = text[max(0, start - context_chars) : end + context_chars].replace(
                "\n", " "
            )
            if PROXIMITY_PATTERN.search(ctx) is None:
                continue
            sig = f"{amt}:{ctx[:60]}"
            if sig in seen:
                continue
            seen.add(sig)
            val = float(norm.split()[0]) * (
                1_000_000
                if "million" in norm
                else 1_000_000_000 if "billion" in norm else 1
            )
            out.append(Candidate(val, amt, ctx))
    return out


def rewrite_stage4(
    stage4: Path, amount: float | None, outdir: Path | None, tokenized_root: Path
):
    rel = stage4.relative_to(tokenized_root)
    rel_tok = stage4.relative_to(tokenized_root)
    target = (outdir or tokenized_root) / rel_tok.parent
    target.mkdir(parents=True, exist_ok=True)
    outname = stage4.name.replace("_stage4.jsonl", "_stage5.jsonl")
    tmp = target / (outname + ".tmp")

    with stage4.open(encoding="utf8") as fin, tmp.open("w", encoding="utf8") as fout:
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
    tokenized_root: Path,
    extracted_root: Path,
    outdir: Path | None,
):
    # map this tokenized-case back to its extracted-location
    rel = case_root.relative_to(tokenized_root)
    extracted_case_root = extracted_root / rel
    candidates = scan_stage1(extracted_case_root, min_amount, context_chars)
    amount = selector.choose(candidates)
    print(f"▶ {case_root.relative_to(tokenized_root)} → {amount!r}")
    for stage4 in case_root.rglob("*_stage4.jsonl"):
        rewrite_stage4(stage4, amount, outdir, tokenized_root)


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
    return p.parse_args()


def main():
    args = parse_args()
    selector = ManualAmountSelector() if args.mode == "manual" else AmountSelector()
    tokenized_root = args.root
    extracted_root = args.stage1_root or args.root
    roots = {p.parent for p in tokenized_root.rglob("*_stage4.jsonl")}

    for case_root in roots:
        impute_for_case(
            case_root,
            selector,
            args.min_amount,
            args.context_chars,
            tokenized_root,
            extracted_root,
            args.outdir,
        )


if __name__ == "__main__":
    sys.exit(main())
