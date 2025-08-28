# Minimal Hyperparameter Grid Extension

To add configurable hyperparameter grids to `scripts/train_validated_features_comprehensive.py`, add this code:

## Add after line 56 (imports section):
```python
from sklearn.model_selection import GridSearchCV
```

## Replace the `define_models()` method (lines 157-191) with:
```python
def define_models(self, use_grid_search=True) -> Dict[str, Any]:
    """Define all models to evaluate with optional hyperparameter search."""

    # Define hyperparameter grids
    hyperparameter_grids = {
        'logistic_l1': {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['liblinear'],
            'max_iter': [1000, 2000]
        },
        'logistic_l2': {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['lbfgs', 'newton-cg'],
            'max_iter': [1000, 2000]
        },
        'logistic_elasticnet': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'solver': ['saga'],
            'max_iter': [1000, 2000]
        },
        'svm_linear': {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'max_iter': [2000, 5000]
        },
        'polr_champion': {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    }

    models = {}

    if use_grid_search:
        # Create GridSearchCV models
        models['logistic_l1'] = GridSearchCV(
            LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
            hyperparameter_grids['logistic_l1'],
            cv=3, scoring='roc_auc', n_jobs=-1
        )

        models['logistic_l2'] = GridSearchCV(
            LogisticRegression(penalty='l2', random_state=42),
            hyperparameter_grids['logistic_l2'],
            cv=3, scoring='roc_auc', n_jobs=-1
        )

        models['logistic_elasticnet'] = GridSearchCV(
            LogisticRegression(penalty='elasticnet', random_state=42),
            hyperparameter_grids['logistic_elasticnet'],
            cv=3, scoring='roc_auc', n_jobs=-1
        )

        models['svm_linear'] = GridSearchCV(
            LinearSVC(random_state=42, dual=False),
            hyperparameter_grids['svm_linear'],
            cv=3, scoring='roc_auc', n_jobs=-1
        )

        # MLR variants (no grid search, use balanced)
        models['mlr_enhanced'] = OneVsRestClassifier(
            LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        )
        models['mlr_balanced'] = OneVsRestClassifier(
            LogisticRegression(C=1.0, class_weight='balanced', random_state=42, max_iter=1000)
        )

        # POLR with grid search
        if MORD_AVAILABLE:
            models['polr_champion'] = GridSearchCV(
                mord.LogisticAT(),
                hyperparameter_grids['polr_champion'],
                cv=3, scoring='roc_auc', n_jobs=-1
            )

    else:
        # Original fixed models (fallback)
        # ... (keep original code)

    logger.info(f"Defined {len(models)} models: {list(models.keys())}")
    return models
```

## Add command line argument in main() function:
```python
parser.add_argument("--hyperparameter-search", action="store_true",
                   help="Enable hyperparameter grid search")

# Then pass it to evaluator:
evaluator = ComprehensiveModelEvaluator(
    data_dir=args.data_dir,
    output_dir=args.output_dir,
    fold=args.fold,
    use_grid_search=args.hyperparameter_search
)
```

## Usage with hyperparameter search:
```bash
uv run python scripts/train_validated_features_comprehensive.py \
    --data-dir data/final_stratified_kfold_splits_binary_quote_balanced_with_graphsage \
    --output-dir results/optimized_models \
    --fold 4 \
    --hyperparameter-search
```

This adds configurable hyperparameter grids while keeping the existing functionality intact.
