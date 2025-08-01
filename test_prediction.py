import sys
from pathlib import Path

sys.path.insert(0, "src")
from corp_speech_risk_dataset.case_outcome.case_outcome_imputer import (
    scan_stage1,
    VotingWeights,
    DEFAULT_VOTING_WEIGHTS,
)

# Test with optimization parameters that are failing
case_path = Path("data/extracted/courtlistener/0:14-cv-61344_flsd")

# Use parameters from the failing optimization
candidates = scan_stage1(
    case_path,
    min_amount=1119,  # From optimization
    context_chars=437,  # From optimization
    min_features=8,  # From optimization - THIS IS THE PROBLEM!
    case_position_threshold=0.67,
    docket_position_threshold=0.82,
    voting_weights=DEFAULT_VOTING_WEIGHTS,
    header_chars=1705,
    fast_mode=True,
)

print(f"With optimization params (min_features=8): {len(candidates)} candidates")

# Now test with lower min_features
candidates2 = scan_stage1(
    case_path,
    min_amount=1119,
    context_chars=437,
    min_features=2,  # Lower threshold
    case_position_threshold=0.67,
    docket_position_threshold=0.82,
    voting_weights=DEFAULT_VOTING_WEIGHTS,
    header_chars=1705,
    fast_mode=True,
)

print(f"With min_features=2: {len(candidates2)} candidates")

if candidates2:
    # Sort by feature votes (descending), then by value (descending)
    sorted_candidates = sorted(
        candidates2, key=lambda c: (c.feature_votes, c.value), reverse=True
    )
    top_candidate = sorted_candidates[0]
    print(
        f"Top candidate: ${top_candidate.value:,.0f} (votes: {top_candidate.feature_votes})"
    )
