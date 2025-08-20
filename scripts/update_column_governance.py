#!/usr/bin/env python3
"""Update column governance to block dropped features from analysis."""

import sys
from pathlib import Path
import pandas as pd


def update_column_governance():
    """Add dropped features to column governance blocklist."""
    print("🔄 Updating column governance with dropped features...")

    # Read dropped features
    dropped_df = pd.read_csv(
        "docs/feature_analysis/final_feature_set/dropped_features.csv"
    )
    dropped_features = dropped_df["feature"].tolist()

    print(f"📋 Found {len(dropped_features)} features to block")

    # Read current column governance
    governance_file = Path(
        "src/corp_speech_risk_dataset/fully_interpretable/column_governance.py"
    )
    with open(governance_file, "r") as f:
        content = f.read()

    # Find the BLOCKLIST_PATTERNS section
    patterns_start = content.find("BLOCKLIST_PATTERNS = [")
    patterns_end = content.find("]", patterns_start) + 1

    if patterns_start == -1:
        print("❌ Could not find BLOCKLIST_PATTERNS in column governance file")
        return

    # Extract current patterns
    patterns_section = content[patterns_start:patterns_end]

    # Create regex patterns for dropped features
    new_patterns = []
    for feature in dropped_features:
        # Escape the feature name for regex
        escaped_feature = feature.replace("_", r"\_")
        pattern = f'    r"^{escaped_feature}$",'
        new_patterns.append(pattern)

    # Insert new patterns before the closing bracket
    # Find the last pattern before closing bracket
    patterns_lines = patterns_section.split("\n")
    closing_bracket_line = patterns_lines[-1]  # The ']' line
    other_lines = patterns_lines[:-1]

    # Add comment and new patterns
    new_content_lines = (
        other_lines
        + [
            "    # DROPPED FEATURES FROM ANALYSIS (automatically generated)",
        ]
        + new_patterns
        + [closing_bracket_line]
    )

    new_patterns_section = "\n".join(new_content_lines)

    # Replace the patterns section in the full content
    new_content = (
        content[:patterns_start] + new_patterns_section + content[patterns_end:]
    )

    # Write back to file
    with open(governance_file, "w") as f:
        f.write(new_content)

    print(f"✅ Updated column governance with {len(dropped_features)} blocked features")
    print(f"📁 File updated: {governance_file}")

    # Create summary of what was blocked
    summary_file = Path(
        "docs/feature_analysis/final_feature_set/governance_update_summary.txt"
    )
    with open(summary_file, "w") as f:
        f.write("# Column Governance Update Summary\n\n")
        f.write(f"Added {len(dropped_features)} features to BLOCKLIST_PATTERNS:\n\n")
        for feature in sorted(dropped_features):
            reason = dropped_df[dropped_df["feature"] == feature]["drop_reason"].iloc[0]
            f.write(f"- {feature} ({reason})\n")
        f.write(f"\nThese features are now permanently blocked from training.\n")
        f.write(
            f"They failed quality thresholds for missing%, sparsity, drift, or temporal stability.\n"
        )

    print(f"📄 Summary saved to: {summary_file}")


if __name__ == "__main__":
    update_column_governance()
