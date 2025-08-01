import sys
from pathlib import Path

sys.path.insert(0, "src")
from corp_speech_risk_dataset.case_outcome.case_outcome_imputer import (
    scan_stage1,
    VotingWeights,
    DEFAULT_VOTING_WEIGHTS,
)

# Test one specific case
case_path = Path("data/extracted/courtlistener/0:14-cv-61344_flsd")
print(f"Testing case: {case_path}")
print(f"Case exists: {case_path.exists()}")

# Count stage1 files
stage1_files = list(case_path.rglob("*_stage1.jsonl"))
print(f"Found {len(stage1_files)} stage1 files")

# Test with simple parameters
candidates = scan_stage1(
    case_path,
    min_amount=1000,  # Low threshold
    context_chars=200,
    min_features=1,  # Very low threshold
    case_position_threshold=0.5,
    docket_position_threshold=0.5,
    voting_weights=DEFAULT_VOTING_WEIGHTS,
    header_chars=2000,
    fast_mode=True,
)

print(f"Found {len(candidates)} candidates")
for i, candidate in enumerate(candidates[:3]):
    print(
        f"  {i+1}. Value: ${candidate.value:,.0f}, Context: {candidate.context[:100]}..."
    )
