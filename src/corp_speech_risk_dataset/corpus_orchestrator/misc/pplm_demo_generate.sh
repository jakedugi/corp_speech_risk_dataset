# =============================
# scripts/demo_generate.sh
# =============================
#!/usr/bin/env bash
set -e
python -m corp_speech_risk_dataset.pplm_ordinal.cli \
  --prompt "The court finds that" \
  --class-id 2 \
  --classifier-path runs/classifier/best.pt \
  --num-classes 3 \
  --length 50 --num-steps 3 --step-size 0.05 --gm-scale 0.95 --kl-scale 0.01
