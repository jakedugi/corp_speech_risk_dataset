# ----------------------------------
# scripts/example_coral_run.sh
# ----------------------------------
#!/usr/bin/env bash
set -e
python -m coral_ordinal.cli \
  --data data/train.jsonl \
  --feature-key fused_emb \
  --label-key bucket \
  --buckets Low Medium High \
  --epochs 10 --plot-cm
