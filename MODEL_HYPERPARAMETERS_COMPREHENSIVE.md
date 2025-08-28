# Complete Hyperparameter Guide for 6 Models

## 1. **Logistic Regression L1** (`logistic_l1`)
**Base:** `LogisticRegression(penalty='l1', solver='liblinear')`

### Key Hyperparameters:
```python
{
    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],              # Regularization strength (inverse)
    'solver': ['liblinear', 'saga'],                                  # Optimization algorithm
    'max_iter': [100, 200, 500, 1000, 2000, 5000],                  # Maximum iterations
    'tol': [1e-4, 1e-3, 1e-2],                                      # Tolerance for optimization
    'fit_intercept': [True, False],                                  # Whether to fit intercept
    'class_weight': [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}], # Class weights
    'random_state': [42],                                            # Fixed for reproducibility
    'warm_start': [True, False],                                     # Reuse previous solution
    'intercept_scaling': [1.0, 0.1, 10.0]                          # Intercept scaling
}
```

## 2. **Logistic Regression L2** (`logistic_l2`)
**Base:** `LogisticRegression(penalty='l2')`

### Key Hyperparameters:
```python
{
    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],              # Regularization strength
    'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],                 # Optimization algorithm
    'max_iter': [100, 200, 500, 1000, 2000, 5000],                  # Maximum iterations
    'tol': [1e-4, 1e-3, 1e-2],                                      # Tolerance
    'fit_intercept': [True, False],                                  # Fit intercept
    'class_weight': [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}], # Class weights
    'random_state': [42],                                            # Fixed
    'warm_start': [True, False],                                     # Warm start
    'intercept_scaling': [1.0, 0.1, 10.0],                          # Intercept scaling
    'multi_class': ['auto', 'ovr', 'multinomial']                   # Multi-class strategy
}
```

## 3. **Logistic Regression ElasticNet** (`logistic_elasticnet`)
**Base:** `LogisticRegression(penalty='elasticnet', solver='saga')`

### Key Hyperparameters:
```python
{
    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],                      # Regularization strength
    'l1_ratio': [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0],    # ElasticNet mixing (0=L2, 1=L1)
    'solver': ['saga'],                                               # Only saga supports elasticnet
    'max_iter': [100, 200, 500, 1000, 2000, 5000],                  # Maximum iterations
    'tol': [1e-4, 1e-3, 1e-2],                                      # Tolerance
    'fit_intercept': [True, False],                                  # Fit intercept
    'class_weight': [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}], # Class weights
    'random_state': [42],                                            # Fixed
    'warm_start': [True, False],                                     # Warm start
    'intercept_scaling': [1.0, 0.1, 10.0]                          # Intercept scaling
}
```

## 4. **Linear SVM** (`svm_linear`)
**Base:** `LinearSVC()`

### Key Hyperparameters:
```python
{
    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],              # Regularization parameter
    'loss': ['squared_hinge', 'hinge'],                              # Loss function
    'penalty': ['l1', 'l2'],                                         # Regularization type
    'dual': [True, False],                                           # Dual/primal formulation
    'tol': [1e-4, 1e-3, 1e-2],                                      # Tolerance
    'fit_intercept': [True, False],                                  # Fit intercept
    'intercept_scaling': [1.0, 0.1, 10.0],                          # Intercept scaling
    'class_weight': [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}], # Class weights
    'random_state': [42],                                            # Fixed
    'max_iter': [1000, 2000, 5000, 10000]                           # Maximum iterations
}
```

## 5. **Multinomial Logistic Regression Enhanced** (`mlr_enhanced`)
**Base:** `OneVsRestClassifier(LogisticRegression())`

### Key Hyperparameters:
```python
{
    # Inner LogisticRegression parameters (prefixed with 'estimator__')
    'estimator__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],          # Regularization
    'estimator__penalty': ['l1', 'l2', 'elasticnet', 'none'],       # Penalty type
    'estimator__solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'], # Solver
    'estimator__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],             # ElasticNet ratio (if penalty='elasticnet')
    'estimator__max_iter': [100, 200, 500, 1000, 2000],             # Max iterations
    'estimator__tol': [1e-4, 1e-3, 1e-2],                          # Tolerance
    'estimator__fit_intercept': [True, False],                      # Fit intercept
    'estimator__class_weight': [None, 'balanced'],                  # Class weights
    'estimator__random_state': [42],                                # Fixed

    # OneVsRestClassifier parameters
    'n_jobs': [1, -1]                                               # Parallel jobs
}
```

## 6. **Multinomial Logistic Regression Balanced** (`mlr_balanced`)
**Base:** `OneVsRestClassifier(LogisticRegression(class_weight='balanced'))`

### Key Hyperparameters:
```python
{
    # Inner LogisticRegression parameters (prefixed with 'estimator__')
    'estimator__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],          # Regularization
    'estimator__penalty': ['l1', 'l2', 'elasticnet', 'none'],       # Penalty type
    'estimator__solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'], # Solver
    'estimator__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],             # ElasticNet ratio
    'estimator__max_iter': [100, 200, 500, 1000, 2000],             # Max iterations
    'estimator__tol': [1e-4, 1e-3, 1e-2],                          # Tolerance
    'estimator__fit_intercept': [True, False],                      # Fit intercept
    'estimator__class_weight': ['balanced'],                        # Fixed to balanced
    'estimator__random_state': [42],                                # Fixed

    # OneVsRestClassifier parameters
    'n_jobs': [1, -1]                                               # Parallel jobs
}
```

## 7. **Proportional Odds Logistic Regression** (`polr_champion`)
**Base:** `mord.LogisticAT()` (if mord available)

### Key Hyperparameters:
```python
{
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],         # Regularization parameter
    'fit_intercept': [True, False],                                  # Fit intercept
    'normalize': [True, False],                                      # Normalize features
    'copy_X': [True, False],                                         # Copy input data
    'max_iter': [100, 200, 500, 1000, 2000],                       # Maximum iterations
    'verbose': [0, 1],                                               # Verbosity level
    'tol': [1e-4, 1e-3, 1e-2]                                      # Tolerance
}
```

---

## 🎯 **Recommended Grid Search Configurations**

### **Fast Grid (for testing):**
```python
FAST_GRIDS = {
    'logistic_l1': {'C': [0.1, 1.0, 10.0], 'max_iter': [1000]},
    'logistic_l2': {'C': [0.1, 1.0, 10.0], 'solver': ['lbfgs'], 'max_iter': [1000]},
    'logistic_elasticnet': {'C': [0.1, 1.0, 10.0], 'l1_ratio': [0.3, 0.5, 0.7], 'max_iter': [1000]},
    'svm_linear': {'C': [0.1, 1.0, 10.0], 'max_iter': [2000]},
    'mlr_enhanced': {'estimator__C': [0.1, 1.0, 10.0], 'estimator__max_iter': [1000]},
    'mlr_balanced': {'estimator__C': [0.1, 1.0, 10.0], 'estimator__max_iter': [1000]},
    'polr_champion': {'alpha': [0.1, 1.0, 10.0]}
}
```

### **Comprehensive Grid (for production):**
```python
COMPREHENSIVE_GRIDS = {
    'logistic_l1': {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 2000],
        'class_weight': [None, 'balanced']
    },
    'logistic_l2': {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'solver': ['lbfgs', 'newton-cg', 'saga'],
        'max_iter': [1000, 2000],
        'class_weight': [None, 'balanced']
    },
    'logistic_elasticnet': {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'max_iter': [1000, 2000],
        'class_weight': [None, 'balanced']
    },
    'svm_linear': {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'loss': ['squared_hinge', 'hinge'],
        'max_iter': [2000, 5000],
        'class_weight': [None, 'balanced']
    },
    'mlr_enhanced': {
        'estimator__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'estimator__penalty': ['l1', 'l2', 'elasticnet'],
        'estimator__solver': ['lbfgs', 'liblinear', 'saga'],
        'estimator__max_iter': [1000, 2000]
    },
    'mlr_balanced': {
        'estimator__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'estimator__penalty': ['l1', 'l2', 'elasticnet'],
        'estimator__solver': ['lbfgs', 'liblinear', 'saga'],
        'estimator__max_iter': [1000, 2000]
    },
    'polr_champion': {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        'max_iter': [500, 1000, 2000]
    }
}
```

---

## ⚠️ **Important Notes:**

1. **Solver Constraints:**
   - L1 penalty: Use `liblinear` or `saga`
   - L2 penalty: Any solver (`lbfgs` recommended)
   - ElasticNet: Only `saga` solver

2. **Class Weight Options:**
   - `None`: No weighting
   - `'balanced'`: Automatic inverse frequency weighting
   - `{0: w0, 1: w1}`: Custom weights

3. **Performance Considerations:**
   - Start with fast grids for initial testing
   - Use comprehensive grids for final optimization
   - `n_jobs=-1` for parallel processing

4. **POLR Availability:**
   - Requires `mord` package installation
   - Falls back gracefully if not available
