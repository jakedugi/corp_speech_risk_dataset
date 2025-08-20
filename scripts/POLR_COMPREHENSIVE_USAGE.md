# 🚀 POLR Comprehensive Pipeline Usage Guide

This document provides complete usage instructions for the comprehensive POLR pipeline runner system. **All scripts ensure proper weight inheritance from the authoritative dataset and never recompute weights, tertiles, or labels.**

## 📋 Quick Start

### 1. Validate Your Data (Always Start Here!)
```bash
# Quick validation
python scripts/quick_validate.py

# Detailed validation
python scripts/quick_validate.py --detailed

# Validate custom data directory
python scripts/quick_validate.py --data-dir path/to/your/data
```

### 2. Run a Quick Test
```bash
# Fast test run for development
python scripts/polr_commands.py run quick_test
```

### 3. Production Run
```bash
# Full production pipeline with paper assets
python scripts/polr_commands.py run paper_ready
```

## 📁 Script Overview

### Core Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_polr_comprehensive.py` | Main pipeline runner with multiple modes | Direct execution with full control |
| `polr_commands.py` | Helper with pre-configured commands | Easy access to common workflows |
| `quick_validate.py` | Standalone data validation | Fast integrity checks |

### Supporting Scripts (Existing)
- `make_paper_tables.py` - Generate LaTeX tables
- `make_paper_figures.py` - Generate publication figures
- `final_polish_assets.py` - Additional publication assets

## 🎯 Pipeline Modes

### 1. **validate** - Data Integrity Check
```bash
# Basic validation
python scripts/run_polr_comprehensive.py --mode validate

# Skip confirmations for automation
python scripts/run_polr_comprehensive.py --mode validate --no-confirmation
```
**Purpose**: Verify data integrity and weight inheritance before training.

### 2. **cv-only** - Cross-Validation Only
```bash
# Standard CV for hyperparameter search
python scripts/run_polr_comprehensive.py --mode cv-only --output-dir runs/my_cv_run

# With thorough hyperparameter search
python scripts/run_polr_comprehensive.py --mode cv-only --hyperparameters thorough
```
**Purpose**: Run hyperparameter search on folds 0, 1, 2 only.

### 3. **final-only** - Final Model Training
```bash
# Train final model (requires existing CV results)
python scripts/run_polr_comprehensive.py --mode final-only --output-dir runs/existing_cv_run
```
**Purpose**: Train final model on fold 3 and evaluate on OOF test set.

### 4. **full** - Complete Pipeline
```bash
# Complete pipeline with default settings
python scripts/run_polr_comprehensive.py --mode full

# Production run with thorough hyperparameters
python scripts/run_polr_comprehensive.py --mode full --hyperparameters thorough --output-dir runs/production
```
**Purpose**: Complete workflow (CV + final training + evaluation + assets).

### 5. **assets** - Generate Paper Assets Only
```bash
# Generate assets from existing results
python scripts/run_polr_comprehensive.py --mode assets --output-dir runs/existing_results
```
**Purpose**: Generate LaTeX tables and PDF figures from existing model outputs.

## ⚙️ Hyperparameter Presets

### **fast** - Minimal Search (2-3 minutes)
```json
{"C": [1.0], "solver": ["lbfgs"], "max_iter": [200], "tol": [1e-4]}
```

### **default** - Balanced Search (10-15 minutes)
```json
{"C": [0.01, 1, 100], "solver": ["lbfgs"], "max_iter": [200], "tol": [1e-4]}
```

### **thorough** - Extensive Search (30-60 minutes)
```json
{
  "C": [0.001, 0.01, 0.1, 1.0, 10, 100],
  "solver": ["lbfgs", "newton-cg"],
  "max_iter": [200, 500, 1000],
  "tol": [1e-4, 1e-5]
}
```

### **custom** - User-Defined
```bash
python scripts/run_polr_comprehensive.py \
  --mode full \
  --hyperparameters custom \
  --custom-hyperparameters '{"C": [0.1, 1.0], "solver": ["lbfgs"], "max_iter": [500]}'
```

## 🛠️ Common Workflows

### Development Workflow
```bash
# 1. Validate data
python scripts/quick_validate.py

# 2. Quick test
python scripts/polr_commands.py run quick_test

# 3. Check results
ls runs/polr_quick_test_*/
```

### Research Workflow
```bash
# 1. Validate data
python scripts/polr_commands.py run validate

# 2. Cross-validation experiment
python scripts/run_polr_comprehensive.py \
  --mode cv-only \
  --hyperparameters thorough \
  --output-dir runs/cv_experiment_$(date +%Y%m%d)

# 3. Train final model
python scripts/run_polr_comprehensive.py \
  --mode final-only \
  --output-dir runs/cv_experiment_$(date +%Y%m%d)
```

### Publication Workflow
```bash
# Single command for complete paper-ready results
python scripts/polr_commands.py run paper_ready

# Or step-by-step with custom settings
python scripts/run_polr_comprehensive.py \
  --mode full \
  --hyperparameters default \
  --generate-assets \
  --output-dir runs/paper_final_$(date +%Y%m%d_%H%M)
```

### Automation Workflow
```bash
# Fully automated run (no prompts)
python scripts/run_polr_comprehensive.py \
  --mode full \
  --hyperparameters fast \
  --no-confirmation \
  --skip-validation \
  --output-dir runs/automated_$(date +%Y%m%d_%H%M)
```

## 📊 Output Structure

After running any training mode, your output directory will contain:

```
runs/your_output_dir/
├── polr_comprehensive_*.log              # Detailed logs
├── cv_results.json                       # Cross-validation results
├── final_polar_model.joblib              # Trained POLR model
├── final_preprocessor.joblib             # Feature preprocessing pipeline
├── final_calibrators.joblib              # Isotonic calibration models
├── final_oof_predictions.jsonl           # OOF predictions (polr_ prefix)
├── final_oof_metrics.json                # OOF evaluation metrics
├── final_model_metadata.json             # Complete model metadata
└── paper_assets/                         # Publication materials
    ├── latex/                           # LaTeX tables (T1-T10)
    ├── figures/oof/                     # PDF figures (F1-F18)
    ├── tables/                          # Additional tables
    └── macros/                          # LaTeX macros
```

## 🔍 Validation and Safety Features

### Data Integrity Checks
- ✅ Verifies authoritative data directory structure
- ✅ Validates `per_fold_metadata.json` format and completeness
- ✅ Checks for precomputed `outcome_bin` labels in all folds
- ✅ Verifies precomputed `sample_weight` values are present
- ✅ Validates fold-specific tertile boundaries and class weights
- ✅ Checks class distribution balance across folds

### Weight Inheritance Verification
```bash
# The pipeline NEVER recomputes:
# ❌ outcome_bin labels (uses precomputed values)
# ❌ sample_weight values (uses precomputed values)
# ❌ tertile boundaries (uses per_fold_metadata.json)
# ❌ class weights (uses per_fold_metadata.json)
```

### Safety Prompts
The pipeline will ask for confirmation before:
- Running cross-validation
- Training final model
- Generating assets
- Any operation that could overwrite existing results

## 🚨 Troubleshooting

### Common Issues

#### 1. **Validation Fails**
```bash
# Check specific issues
python scripts/quick_validate.py --detailed

# Common fixes:
# - Ensure data directory exists: data/final_stratified_kfold_splits_authoritative/
# - Check per_fold_metadata.json format
# - Verify all fold directories (fold_0, fold_1, fold_2, fold_3, oof_test)
```

#### 2. **Missing Precomputed Fields**
```bash
# Error: "No precomputed outcome_bin field found"
# Solution: Regenerate authoritative data with proper labels
python scripts/stratified_kfold_case_split.py --regenerate-labels
```

#### 3. **OOF Test Set Issues**
```bash
# Error: "OOF test file not found"
# Check: data/final_stratified_kfold_splits_authoritative/oof_test/test.jsonl
```

#### 4. **Hyperparameter Search Fails**
```bash
# Try simpler hyperparameters
python scripts/run_polr_comprehensive.py --mode cv-only --hyperparameters fast
```

#### 5. **Asset Generation Fails**
```bash
# Generate assets separately
python scripts/run_polr_comprehensive.py --mode assets --output-dir runs/your_results
```

### Debug Mode
```bash
# Run with extensive logging
python scripts/run_polr_comprehensive.py \
  --mode full \
  --hyperparameters fast \
  --output-dir runs/debug_$(date +%Y%m%d_%H%M)

# Check logs
tail -f runs/debug_*/polr_comprehensive_*.log
```

## 🎯 Command Examples

### Quick Commands via Helper
```bash
# List all available commands
python scripts/polr_commands.py list

# Get detailed pipeline info
python scripts/polr_commands.py info

# Interactive command builder
python scripts/polr_commands.py custom

# Run specific command
python scripts/polr_commands.py run production
```

### Direct Pipeline Control
```bash
# Minimal fast run
python scripts/run_polr_comprehensive.py \
  --mode full \
  --hyperparameters fast \
  --no-confirmation

# Thorough research run
python scripts/run_polr_comprehensive.py \
  --mode full \
  --hyperparameters thorough \
  --output-dir runs/research_$(date +%Y%m%d)

# CV only with custom hyperparameters
python scripts/run_polr_comprehensive.py \
  --mode cv-only \
  --hyperparameters custom \
  --custom-hyperparameters '{"C": [0.01, 0.1, 1, 10], "max_iter": [500]}'
```

### Batch Processing
```bash
# Run multiple experiments
for preset in fast default thorough; do
  python scripts/run_polr_comprehensive.py \
    --mode full \
    --hyperparameters $preset \
    --no-confirmation \
    --output-dir runs/experiment_${preset}_$(date +%Y%m%d)
done
```

## 📈 Expected Runtime

| Mode | Hyperparameters | Estimated Time |
|------|----------------|----------------|
| validate | N/A | 30 seconds |
| cv-only | fast | 3-5 minutes |
| cv-only | default | 10-15 minutes |
| cv-only | thorough | 30-60 minutes |
| final-only | N/A | 5-10 minutes |
| full | fast | 5-8 minutes |
| full | default | 20-35 minutes |
| full | thorough | 45-90 minutes |
| assets | N/A | 2-5 minutes |

## 🔬 Advanced Usage

### Custom Data Directory
```bash
python scripts/run_polr_comprehensive.py \
  --data-dir path/to/custom/authoritative/data \
  --mode full
```

### Disable Specific Assets
```bash
python scripts/run_polr_comprehensive.py \
  --mode full \
  --no-figures \
  --no-tables
```

### Multiple Output Formats
```bash
# Save validation report
python scripts/quick_validate.py --output validation_report.json

# Custom output directory with timestamp
python scripts/run_polr_comprehensive.py \
  --mode full \
  --output-dir "runs/custom_$(hostname)_$(date +%Y%m%d_%H%M%S)"
```

## 🎉 Success Indicators

### Validation Success
```
✅ All validation checks PASSED!
✅ Data is ready for POLR pipeline execution
```

### Training Success
```
✅ POLAR TRAINING COMPLETE
📁 Results saved to: runs/your_output_dir/
🎯 Publication package now complete!
```

### Expected Files Generated
- ✅ `final_oof_predictions.jsonl` with `polr_` prefixed predictions
- ✅ LaTeX tables T1-T10 in `paper_assets/latex/`
- ✅ PDF figures F1-F18 in `paper_assets/figures/oof/`
- ✅ Model artifacts (`.joblib` files)
- ✅ Comprehensive logs and metadata

---

## 🛡️ Critical Safety Reminders

1. **NEVER modify** the authoritative data directory during training
2. **ALWAYS validate** data integrity before training (`python scripts/quick_validate.py`)
3. **VERIFY** that `outcome_bin` and `sample_weight` fields exist in your data
4. **CHECK** logs for any warnings about weight inheritance issues
5. **BACKUP** your results before running new experiments

---

*This pipeline is designed to be production-ready for academic publication while maintaining complete reproducibility and data integrity.*
