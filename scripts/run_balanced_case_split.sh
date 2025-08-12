#!/bin/bash
"""
Run the balanced case split script with the specified parameters.

This script creates balanced train/val/test splits by case ID while:
- Separating outlier cases (>5 billion)
- Filtering excluded speakers
- Maintaining balanced distribution across splits
- Dumping comprehensive statistics and metadata
"""

uv run python scripts/balanced_case_split.py \
  --input "/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/final_destination/courtlistener_v6_fused_raw_coral_pred/doc_*_text_stage15.jsonl" \
  --output-dir /Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/balanced_case_splits \
  --outlier-threshold 5000000000 \
  --exclude-speakers "Unknown,Court,FTC,Fed,Plaintiff,State,Commission,Congress,Circuit,FDA" \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --random-seed 42
