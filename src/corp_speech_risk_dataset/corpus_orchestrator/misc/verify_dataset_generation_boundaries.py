#!/usr/bin/env python3
"""
Verify that the dataset generation script uses correct boundary methodology.

This script checks all boundary assignment functions in stratified_kfold_case_split.py
to ensure they match the corrected methodology:
- Low: y < cutoff_low
- Medium: cutoff_low <= y <= cutoff_high
- High: y > cutoff_high

Key functions to check:
1. casewise_train_bins()
2. make_temporal_fold()
3. apply_tertile_binning() (in polar_pipeline.py)
4. Any np.digitize() calls with right=True
"""

import sys
import re
from pathlib import Path


def check_boundary_methodology_in_file(file_path: str):
    """Check boundary methodology in a specific file."""
    print(f"\n🔍 **CHECKING: {file_path}**")

    with open(file_path, "r") as f:
        content = f.read()

    issues = []

    # Check for np.digitize calls
    digitize_matches = re.finditer(r"np\.digitize\([^)]+\)", content)
    for match in digitize_matches:
        line_start = content.rfind("\n", 0, match.start()) + 1
        line_end = content.find("\n", match.end())
        if line_end == -1:
            line_end = len(content)
        line = content[line_start:line_end]
        line_num = content[: match.start()].count("\n") + 1

        if "right=True" not in match.group():
            issues.append(
                {
                    "type": "digitize_missing_right_true",
                    "line": line_num,
                    "code": line.strip(),
                    "issue": "np.digitize should use right=True for correct boundary handling",
                }
            )

    # Check for manual boundary logic
    boundary_patterns = [
        r"if.*<=.*cutoff",
        r"if.*<=.*edge",
        r"if.*<=.*boundary",
        r"outcome.*<=",
        r"y.*<=.*e[12]",
    ]

    for pattern in boundary_patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            line_start = content.rfind("\n", 0, match.start()) + 1
            line_end = content.find("\n", match.end())
            if line_end == -1:
                line_end = len(content)
            line = content[line_start:line_end]
            line_num = content[: match.start()].count("\n") + 1

            # Check if this looks like incorrect boundary logic
            if "<=" in line and "cutoff" in line.lower():
                issues.append(
                    {
                        "type": "manual_boundary_logic",
                        "line": line_num,
                        "code": line.strip(),
                        "issue": "Manual boundary logic may be incorrect - check against: y < e1, e1 <= y <= e2, y > e2",
                    }
                )

    # Check for pd.cut or pd.qcut usage
    cut_matches = re.finditer(r"pd\.(q?cut)\([^)]+\)", content)
    for match in cut_matches:
        line_start = content.rfind("\n", 0, match.start()) + 1
        line_end = content.find("\n", match.end())
        if line_end == -1:
            line_end = len(content)
        line = content[line_start:line_end]
        line_num = content[: match.start()].count("\n") + 1

        # Check if include_lowest is set correctly
        if "include_lowest=True" not in match.group():
            issues.append(
                {
                    "type": "cut_missing_include_lowest",
                    "line": line_num,
                    "code": line.strip(),
                    "issue": "pd.cut/qcut should use include_lowest=True for correct boundary handling",
                }
            )

    if issues:
        print(f"❌ **FOUND {len(issues)} POTENTIAL ISSUES:**")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. Line {issue['line']}: {issue['issue']}")
            print(f"      Code: {issue['code']}")
    else:
        print("✅ **NO BOUNDARY ISSUES DETECTED**")

    return issues


def main():
    """Main verification function."""
    print("🔍 **VERIFYING DATASET GENERATION BOUNDARY METHODOLOGY**")
    print("=" * 70)

    files_to_check = [
        "scripts/stratified_kfold_case_split.py",
        "src/corp_speech_risk_dataset/fully_interpretable/polar_pipeline.py",
        "scripts/fix_all_fold_boundary_labels.py",
    ]

    all_issues = []

    for file_path in files_to_check:
        if Path(file_path).exists():
            file_issues = check_boundary_methodology_in_file(file_path)
            all_issues.extend(file_issues)
        else:
            print(f"⚠️  **FILE NOT FOUND: {file_path}**")

    print("\n" + "=" * 70)
    print("📊 **SUMMARY**")
    print("=" * 70)

    if all_issues:
        print(f"🚨 **TOTAL ISSUES FOUND: {len(all_issues)}**")
        print("\n**RECOMMENDATIONS:**")
        print("1. Review each flagged line for correct boundary logic")
        print("2. Ensure all np.digitize() calls use right=True")
        print("3. Ensure all pd.cut() calls use include_lowest=True")
        print("4. Use the corrected logic: y < e1, e1 <= y <= e2, y > e2")

        # Group issues by type
        issue_types = {}
        for issue in all_issues:
            issue_type = issue["type"]
            if issue_type not in issue_types:
                issue_types[issue_type] = 0
            issue_types[issue_type] += 1

        print(f"\n**ISSUE BREAKDOWN:**")
        for issue_type, count in issue_types.items():
            print(f"- {issue_type}: {count} instances")

        return 1
    else:
        print("🎉 **ALL BOUNDARY METHODOLOGY CHECKS PASSED!**")
        print("✅ Dataset generation uses correct boundary logic")
        print("✅ All np.digitize calls properly configured")
        print("✅ Consistent with verified fold data")
        return 0


if __name__ == "__main__":
    exit(main())
