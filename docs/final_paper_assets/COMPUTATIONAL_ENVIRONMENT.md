# Computational Environment

## Software Versions
- **Python**: 3.11.13
- **NumPy**: 1.26.4
- **Pandas**: 2.3.1
- **Scikit-learn**: 1.7.0

## Hardware
- **Platform**: macOS-15.5-arm64-arm-64bit
- **Architecture**: 64bit

## Reproducibility
- **Random Seed**: 42 (fixed across all analyses)
- **Cross-validation**: Temporal splits with fixed case assignments
- **Feature Selection**: Deterministic rules applied to training data only

## Key Methodological Choices
- **Tertile Boundaries**: Computed on train data only, applied to dev/test
- **Feature Preprocessing**: StandardScaler fit on train, applied to dev/test
- **Weight Computation**: √N case discount + tempered class reweighting (train only)
- **Model Selection**: Hyperparameters selected via 3-fold CV on folds 0,1,2
- **Final Evaluation**: Independent OOF test set, never used for training or selection
