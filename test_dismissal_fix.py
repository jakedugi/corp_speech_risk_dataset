import sys
from pathlib import Path

sys.path.insert(0, "src")
from corp_speech_risk_dataset.case_outcome.extract_cash_amounts_stage1 import (
    get_case_flags,
)

case_path = Path("data/extracted/courtlistener/0:14-cv-61344_flsd")

# Test different dismissal thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for threshold in thresholds:
    flags = get_case_flags(
        case_path,
        dismissal_ratio_threshold=threshold,
        use_weighted_dismissal_scoring=True,
        dismissal_document_type_weight=2.0,
    )
    status = "DISMISSED" if flags["is_dismissed"] else "NOT DISMISSED"
    print(f"Threshold {threshold}: {status}")
