# =============================
# README_PPLM.md
# =============================
"""
# PPLM Ordinal Steering (No CORAL in this module)

## TL;DR
- Steer GPT-2 with PPLM using an ordinal *multi-class* classifier. Buckets are categorical & ordered, but optimization is through standard CE loss (PPLM-compatible).
- Bring your own classifier (BYOC). If you trained a CORAL model upstream, wrap it; otherwise use the provided SoftmaxOrdinalClassifier.
- Includes go/no-go gate: ≥40% exact, ≥80% off-by-one, Spearman ρ>0.5.

## Install
```bash
pip install torch transformers numpy scipy scikit-learn matplotlib
```

## Train / Load Classifier
Train your ordinal classifier elsewhere (e.g., coral_ordinal). Save as:
```python
torch.save({"state_dict": model.state_dict(), "meta": {"in_dim": H, "num_classes": K}}, "runs/classifier/best.pt")
```

## Generate
```bash
python -m pplm_ordinal.cli --prompt "The settlement amount" --class-id 0 \
  --classifier-path runs/classifier/best.pt --num-classes 3 --length 60
```

## Why This Design?
- **Only PPLM hooks**: We perturb `past_key_values` per token, just like Dathathri et al., avoiding any new plumbing.
- **Ordinal but CE-friendly**: We keep ordinal info in label indices; continuity can later be enforced by your loss choice (e.g., CORAL) without changing PPLM code.
- **Swap-in classifiers**: Abstract base lets you drop in CORAL-head, logistic, SVM-on-hidden, etc.
- **Metrics-first gate**: Utilities to compute exact/off-by-one/ρ keep you honest before burning GPU hours.
- **M1-safe**: `device.py` auto-picks mps; AMP disabled by default.
- **Small & modular**: One file per concern; unit tests ensure nothing silently breaks.

## Extending
- Multi-attribute steering: run perturb_past with summed losses from multiple classifiers.
- Different reps: switch `input_rep` (CLS token for decoder-only LMs: use last hidden; mean pooling works fine too).
- More stable optimization: add second-order approx or line search inside `perturb_past`.


"""
