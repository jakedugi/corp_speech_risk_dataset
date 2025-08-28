#!/usr/bin/env python3
"""
Quick Legal-BERT Embedding Data Diagnostic

Examines data quality issues found in the leakage validation.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Fast JSON loading
try:
    import orjson as _json

    def _loads_bytes(data: bytes):
        return _json.loads(data)

except ImportError:
    import json as _json

    def _loads_bytes(data: bytes):
        return _json.loads(data.decode("utf-8"))


def diagnose_legal_bert_data():
    """Diagnose Legal-BERT embedding data quality."""
    print("🔍 Legal-BERT Embedding Data Diagnostic")
    print("=" * 50)

    # Load data
    data_dir = Path(
        "data/final_stratified_kfold_splits_binary_quote_balanced_with_graphsage"
    )
    fold_path = data_dir / "fold_0" / "train.jsonl"

    print(f"📁 Loading from: {fold_path}")

    data_rows = []
    sample_size = 1000  # Quick sample

    with open(fold_path, "rb") as f:
        for i, line_bytes in enumerate(f):
            if i >= sample_size:
                break
            line_bytes = line_bytes.strip()
            if line_bytes:
                try:
                    data_rows.append(_loads_bytes(line_bytes))
                except Exception as e:
                    print(f"⚠️ Failed to parse line {i}: {e}")
                    continue

    df = pd.DataFrame(data_rows)
    print(f"✅ Loaded {len(df)} samples")

    # Check legal_bert_emb column
    print("\n📊 Legal-BERT Embedding Analysis:")

    if "legal_bert_emb" not in df.columns:
        print("❌ legal_bert_emb column not found!")
        print(f"Available columns: {list(df.columns)}")
        return

    # Basic stats
    emb_col = df["legal_bert_emb"]
    print(f"- Total samples: {len(emb_col)}")
    print(f"- Non-null samples: {emb_col.notna().sum()}")
    print(f"- Null samples: {emb_col.isna().sum()}")

    # Check embedding structure
    valid_embeddings = []
    issues = []

    for i, emb in enumerate(emb_col):
        if emb is None:
            issues.append(f"Row {i}: null embedding")
            continue

        if not isinstance(emb, list):
            issues.append(f"Row {i}: embedding is {type(emb)}, not list")
            continue

        if len(emb) != 768:
            issues.append(f"Row {i}: embedding has {len(emb)} dims, not 768")
            continue

        # Convert to numpy and check for issues
        try:
            emb_array = np.array(emb, dtype=float)
            if np.any(np.isnan(emb_array)):
                issues.append(f"Row {i}: contains NaN values")
                continue
            if np.any(np.isinf(emb_array)):
                issues.append(f"Row {i}: contains infinity values")
                continue
            if np.all(emb_array == 0):
                issues.append(f"Row {i}: all zeros")
                continue

            valid_embeddings.append(emb_array)

        except Exception as e:
            issues.append(f"Row {i}: conversion error - {e}")

    print(f"- Valid embeddings: {len(valid_embeddings)}")
    print(f"- Issues found: {len(issues)}")

    if issues:
        print("\n⚠️ Top 10 Issues:")
        for issue in issues[:10]:
            print(f"  - {issue}")

    # Analyze valid embeddings
    if valid_embeddings:
        print(f"\n📈 Valid Embedding Statistics:")
        emb_matrix = np.array(valid_embeddings)
        print(f"- Shape: {emb_matrix.shape}")
        print(f"- Mean magnitude: {np.mean(np.linalg.norm(emb_matrix, axis=1)):.3f}")
        print(f"- Std magnitude: {np.std(np.linalg.norm(emb_matrix, axis=1)):.3f}")
        print(f"- Mean value: {np.mean(emb_matrix):.6f}")
        print(f"- Std value: {np.std(emb_matrix):.6f}")

        # Check for duplicate embeddings
        unique_embeddings = len(np.unique(emb_matrix.round(6), axis=0))
        print(f"- Unique embeddings: {unique_embeddings}/{len(valid_embeddings)}")

        if unique_embeddings < len(valid_embeddings):
            duplicates = len(valid_embeddings) - unique_embeddings
            print(f"  ⚠️ {duplicates} duplicate embeddings detected!")

    # Check outcome distribution
    print(f"\n🎯 Outcome Analysis:")
    if "outcome_bin" in df.columns:
        outcome_counts = df["outcome_bin"].value_counts()
        print(f"- Outcome distribution: {dict(outcome_counts)}")

        # Check case distribution
        if "case_id" in df.columns:
            case_counts = df["case_id"].value_counts()
            print(f"- Unique cases: {len(case_counts)}")
            print(
                f"- Samples per case - Mean: {case_counts.mean():.1f}, Std: {case_counts.std():.1f}"
            )
            print(f"- Max samples per case: {case_counts.max()}")

            # Cases with single sample (problematic for GroupKFold)
            single_sample_cases = (case_counts == 1).sum()
            print(
                f"- Single-sample cases: {single_sample_cases} ({single_sample_cases/len(case_counts):.1%})"
            )

    else:
        print("❌ outcome_bin column not found!")

    # Summary recommendation
    print(f"\n💡 Diagnostic Summary:")
    if len(issues) > len(valid_embeddings) * 0.1:
        print("❌ CRITICAL: >10% of embeddings have quality issues")
        print("   Recommendation: Investigate embedding generation pipeline")
    elif len(valid_embeddings) < 100:
        print("⚠️ WARNING: Very few valid embeddings for analysis")
        print("   Recommendation: Increase sample size or fix data issues")
    else:
        print("✅ Data quality appears acceptable for analysis")

    if unique_embeddings < len(valid_embeddings) * 0.8:
        print("⚠️ WARNING: High number of duplicate embeddings")
        print("   Recommendation: Check for repeated text or processing issues")


if __name__ == "__main__":
    diagnose_legal_bert_data()
