import sys
from pathlib import Path

sys.path.insert(0, "src")
from corp_speech_risk_dataset.case_outcome.extract_cash_amounts_stage1 import (
    get_case_flags,
    get_case_court_type,
)

# Test case flags for the specific case
case_path = Path("data/extracted/courtlistener/0:14-cv-61344_flsd")

# Check flags with optimization parameters
flags = get_case_flags(
    case_path,
    fee_shifting_ratio_threshold=1.0,
    patent_ratio_threshold=50.0,
    dismissal_ratio_threshold=0.5667,  # From optimization
    bankruptcy_ratio_threshold=0.5,
    use_weighted_dismissal_scoring=True,
    dismissal_document_type_weight=2.0,
)

print("Case flags:")
for key, value in flags.items():
    print(f"  {key}: {value}")

# Check court type
court_type = get_case_court_type(case_path, bankruptcy_ratio_threshold=0.5)
print(f"\nCourt type: {court_type}")

# Check dismissal specifically
print(f"\nDismissal check:")
print(f"  is_dismissed: {flags['is_dismissed']}")
print(f"  has_large_patent_amounts: {flags['has_large_patent_amounts']}")

# The prediction logic:
if court_type == "BANKRUPTCY":
    print("❌ FILTERED OUT: Bankruptcy court")
elif flags["has_large_patent_amounts"]:
    print("❌ FILTERED OUT: Large patent amounts")
elif flags["is_dismissed"]:
    print("❌ FILTERED OUT: Case dismissed (would return 0.0)")
else:
    print("✅ PASSES: Case should proceed to extraction")
