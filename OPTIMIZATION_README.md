# Unified Optimization System

## Overview

This unified optimization system brings together all optimization approaches for the Corporate Speech Risk Dataset case outcome imputer. It preserves all existing functionality while providing a clean, unified interface.

## Features

### 🎯 **Optimization Types**
- **Bayesian Optimization**: Intelligent hyperparameter search using Gaussian processes
- **Grid Search**: Exhaustive search over hyperparameter combinations
- **Fast Optimization**: Quick Bayesian optimization with reduced evaluations

### 🔧 **High/Low Signal Pattern Support**
- Granular control over pattern matching weights
- Separate tuning for high-confidence vs low-confidence patterns
- Enhanced financial, settlement, legal, and monetary phrase patterns
- Dismissal pattern classification (definitive vs procedural)

### 📊 **Advanced Features**
- Real-time progress monitoring with ETA estimation
- Comprehensive logging system
- Parallel processing support (grid search)
- Fast mode for reduced I/O overhead
- Automatic result saving with timestamps
- Cross-validation using Leave-One-Out (LOOCV)

## Quick Start

### Simple Commands

```bash
# Full Bayesian optimization (recommended)
./run_optimization.sh bayesian 100

# Fast optimization for testing
./run_optimization.sh fast 30

# Grid search optimization
./run_optimization.sh grid 50
```

### Advanced Usage

```bash
# Custom data paths and settings
python3 src/corp_speech_risk_dataset/case_outcome/unified_optimizer.py \
  --type bayesian \
  --max-evaluations 200 \
  --gold-standard /path/to/gold_standard.csv \
  --extracted-data /path/to/extracted_data \
  --fast-mode \
  --output-dir custom_results
```

## Hyperparameters Being Optimized

**🎯 Total: 40 Hyperparameters (100% VotingWeights Coverage)**

### Core Extraction Parameters (4)
- `min_amount`: Minimum monetary amount threshold (800 - 1200)
- `context_chars`: Characters of context around amounts (400 - 600)
- `min_features`: Minimum feature votes needed (5 - 9)
- `header_chars`: Header character limit for title detection (1500 - 1800)

### Position-Based Voting (2)
- `case_position_threshold`: Chronological position within case (0.55 - 0.75)
- `docket_position_threshold`: Position within docket sequence (0.75 - 0.90)

### Case Flag Thresholds (4)
- `fee_shifting_ratio_threshold`: Fee-shifting detection threshold (0.9 - 1.2)
- `patent_ratio_threshold`: Patent case detection threshold (0.9 - 1.3)
- `dismissal_ratio_threshold`: Dismissal detection threshold (0.5 - 0.7)
- `bankruptcy_ratio_threshold`: Bankruptcy court threshold (0.45 - 0.65)

### Original Voting Weights (6)
- `proximity_pattern_weight`: Monetary context words (0.8 - 1.2)
- `judgment_verbs_weight`: Legal action verbs (0.5 - 0.9)
- `case_position_weight`: Chronological case position (1.5 - 2.0)
- `docket_position_weight`: Chronological docket position (2.2 - 3.0)
- `all_caps_titles_weight`: ALL CAPS section titles (1.5 - 2.0)
- `document_titles_weight`: Document title matching (0.8 - 1.3)

### Enhanced Feature Weights (12)
- `financial_terms_weight`: Financial terminology (0.5 - 2.0)
- `settlement_terms_weight`: Settlement-specific terms (0.5 - 2.0)
- `legal_proceedings_weight`: Legal proceedings vocabulary (0.5 - 2.0)
- `monetary_phrases_weight`: Enhanced monetary phrases (0.5 - 2.0)
- `dependency_parsing_weight`: Dependency parsing features (0.3 - 1.5)
- `fraction_extraction_weight`: Fraction extraction (0.3 - 1.5)
- `percentage_extraction_weight`: Percentage extraction (0.3 - 1.5)
- `implied_totals_weight`: Implied total calculations (0.3 - 1.5)
- `document_structure_weight`: Document structure features (0.5 - 1.5)
- `table_detection_weight`: Table detection (0.3 - 1.5)
- `header_detection_weight`: Header detection (0.3 - 1.5)
- `section_boundaries_weight`: Section boundary detection (0.3 - 1.5)
- `numeric_gazetteer_weight`: Numeric gazetteer matches (0.5 - 1.5)
- `mixed_numbers_weight`: Mixed number patterns (0.3 - 1.5)
- `sentence_boundary_weight`: Sentence boundary context (0.3 - 1.5)
- `paragraph_boundary_weight`: Paragraph boundary context (0.3 - 1.5)

### Confidence Boosting Features (3)
- `high_confidence_patterns_weight`: High-confidence patterns (0.8 - 2.5)
- `amount_adjacent_keywords_weight`: Amount-adjacent keywords (0.5 - 2.0)
- `confidence_boost_weight`: Overall confidence boost (0.5 - 2.0)

### High/Low Signal Pattern Weights (5) - NEW!
- `high_signal_financial_weight`: High-confidence financial terms (0.5 - 2.5)
- `low_signal_financial_weight`: General financial vocabulary (0.2 - 1.0)
- `high_signal_settlement_weight`: Direct settlement terms (0.5 - 2.5)
- `low_signal_settlement_weight`: Procedural settlement terms (0.2 - 1.0)
- `calculation_boost_multiplier`: Calculation pattern boost (1.0 - 3.0)

## Pattern Classification Examples

### High Signal Financial Terms
- `settlement_fund`, `common_fund`, `escrow_account`
- `purchase_price`, `transaction_value`, `merger_consideration`
- `liquidation_value`, `breakup_fee`, `termination_fee`

### Low Signal Financial Terms
- `gross_revenue`, `net_income`, `operating_expense`
- `accounts_receivable`, `debt_service`, `interest_expense`
- `working_capital`, `retained_earnings`, `book_value`

### High Signal Settlement Terms
- `settlement_agreement`, `final_approval`, `arbitration_award`
- `class_action_settlement`, `attorney_fees_award`
- `qualified_settlement_fund`, `incentive_award`

### Low Signal Settlement Terms
- `claims_administrator`, `opt-out_period`, `fairness_hearing`
- `settlement_website`, `claim_form`, `notice_to_class_members`
- `distribution_plan`, `representative_enhancement`

## Monitoring

### Real-time Log Monitoring
```bash
# In a separate terminal
python3 src/corp_speech_risk_dataset/case_outcome/monitor_optimization.py
```

### Log Files Location
- `logs/unified_optimization_YYYYMMDD_HHMMSS.log`
- Real-time progress updates with ETA
- Best parameter discoveries highlighted

## Results

### Output Files
- `optimization_results/bayesian_results_YYYYMMDD_HHMMSS.json`
- `optimization_results/best_parameters_TYPE_YYYYMMDD_HHMMSS.json`
- Comprehensive evaluation metrics and hyperparameter combinations

### Metrics Tracked
- **MSE Loss**: Mean squared error (primary optimization target)
- **Precision**: Exact match precision
- **Recall**: Coverage of actual cases
- **F1 Score**: Harmonic mean of precision and recall
- **Exact Matches**: Cases with perfect predictions

## Dependencies

### Required
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `polars` - Fast data processing (case outcome imputer)

### Optional
- `scikit-optimize` - Bayesian optimization (install: `pip install scikit-optimize`)
- `spacy` - Enhanced NLP features (install: `pip install spacy`)

## System Architecture

```
unified_optimizer.py          # Main orchestration system
├── bayesian_optimizer.py     # Bayesian optimization with scikit-optimize
├── grid_search_optimizer.py  # Grid search with parallel processing
├── monitor_optimization.py   # Real-time log monitoring
├── case_outcome_imputer.py   # Core extraction logic
└── extract_cash_amounts_stage1.py  # Pattern matching and voting
```

## Best Practices

### For Production Optimization
```bash
# Use Bayesian optimization with high evaluation count
./run_optimization.sh bayesian 200
```

### For Development/Testing
```bash
# Use fast mode for quick iterations
./run_optimization.sh fast 20
```

### For Comprehensive Search
```bash
# Use grid search for exhaustive coverage
./run_optimization.sh grid 100
```

## Troubleshooting

### Missing Dependencies
```bash
pip install pandas numpy scikit-optimize
```

### Path Issues
Ensure you're running from the project root directory containing:
- `src/corp_speech_risk_dataset/`
- `data/gold_standard/`
- `data/extracted/`

### Memory Issues
- Use `--fast-mode` flag
- Reduce `--max-evaluations`
- Reduce `--max-workers` for grid search

### Performance Issues
- Enable fast mode: `--fast-mode`
- Use fewer workers: `--max-workers 2`
- Monitor system resources during optimization

---

## Command Reference

### Full Bayesian Optimization (Recommended)
```bash
./run_optimization.sh bayesian 100
```

This command runs a comprehensive Bayesian optimization with:
- 100 evaluations
- All high/low signal patterns enabled
- Intelligent hyperparameter space exploration
- Comprehensive logging and monitoring
- Automatic result saving

**Estimated Runtime**: 2-4 hours depending on dataset size
**Expected Results**: MSE loss improvements of 10-30% over default parameters
