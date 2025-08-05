# CORAL Ordinal Regression for Corporate Speech Risk

This document explains how to train and use the CORAL (Consistent Rank Logits) ordinal regression model for predicting corporate speech risk levels from fused embeddings.

## Overview

The CORAL ordinal regression system:
- **Learns** ordinal relationships between Low/Medium/High risk categories
- **Processes** fused embeddings (combining sentence-transformer + graph embeddings)
- **Maintains** full traceability back to original text and embeddings
- **Optimized** for efficiency on low-RAM M1 Macs
- **Prevents** overfitting with proper train/validation/test splits

## Quick Start

### 1. Prepare Training Data

First, convert your fused embeddings and outcomes into CORAL-compatible format:

```bash
uv run python scripts/prepare_coral_data.py \
    --input "data/outcomes/courtlistener_v1/*/doc_*_text_stage9.jsonl" \
    --output data/coral_training_data.jsonl \
    --max-threshold 15500000000 \
    --exclude-speakers "Unknown,Court,FTC,Fed,Plaintiff,State,Commission,Congress,Circuit,FDA"
```

This will:
- Load JSONL files with `fused_emb` and `final_judgement_real` fields
- Filter out excluded speakers and outcomes above threshold
- Create ordinal buckets (Low: 0-33%, Medium: 33-67%, High: 67-100%)
- Save prepared data for training

### 2. Train CORAL Model

Train the ordinal regression model with comprehensive tracking:

```bash
uv run python scripts/train_coral_model.py \
    --data data/coral_training_data.jsonl \
    --output runs/coral_experiment \
    --epochs 50 \
    --batch-size 64 \
    --lr 3e-4 \
    --val-split 0.2 \
    --test-split 0.1 \
    --seed 42
```

This will:
- Split data into train/validation/test sets
- Train CORAL MLP with progress tracking
- Save best model based on validation accuracy
- Generate training curves and confusion matrix
- Evaluate final performance on test set

### 3. Run Inference

Use trained model to predict on new data:

```bash
uv run python scripts/coral_inference.py \
    --model runs/coral_experiment/best_model.pt \
    --input "data/new_data/*/doc_*_text_stage9.jsonl" \
    --output predictions/coral_predictions.jsonl \
    --batch-size 64
```

This will:
- Load trained model and configuration
- Process new data with fused embeddings
- Generate ordinal predictions with confidence scores
- Maintain full traceability to original embeddings
- Save detailed results with metadata

### 4. Complete Pipeline

Run the entire pipeline end-to-end:

```bash
uv run python scripts/run_coral_pipeline.py \
    --input "data/outcomes/courtlistener_v1/*/doc_*_text_stage9.jsonl" \
    --output-dir runs/coral_complete_pipeline \
    --max-threshold 15500000000 \
    --exclude-speakers "Unknown,Court,FTC,Fed,Plaintiff,State,Commission,Congress,Circuit,FDA" \
    --epochs 50 \
    --batch-size 64 \
    --seed 42
```

## Data Format

### Input Data (JSONL)
Each line should contain:
```json
{
  "doc_id": "doc_125129554_text",
  "text": "the product was not well advertised about its true nature.",
  "speaker": "Owoc",
  "fused_emb": [1.23, -0.45, ...],  // 384-dim fused embedding
  "final_judgement_real": 5000000.0,
  "context": "...",
  "_src": "path/to/source/file.txt"
}
```

### Output Predictions (JSONL)
Each prediction includes:
```json
{
  "predicted_bucket": "medium",
  "predicted_class": 1,
  "confidence": 0.78,
  "class_probabilities": {
    "low": 0.15,
    "medium": 0.78,
    "high": 0.07
  },
  "prediction_uncertainty": 0.45,
  "ordinal_scores": [0.85, 0.22],

  // Original data preserved for traceability
  "doc_id": "...",
  "text": "...",
  "st_emb": [...],
  "gph_emb": [...],
  // ... all original fields

  "inference_timestamp": 1703123456.789
}
```

## Model Architecture

### CORAL MLP
- **Input**: 384-dimensional fused embeddings
- **Hidden layers**: [512, 128] with ReLU activation
- **Dropout**: 0.1 for regularization
- **Output**: K-1 ordinal thresholds (for K classes)
- **Loss**: CORAL ordinal regression loss

### Embedding Fusion
The model uses pre-computed fused embeddings that combine:
- **Sentence-Transformer**: 384-dim semantic embeddings
- **Graph embeddings**: 256-dim structural embeddings (GraphSAGE)
- **Fusion method**: Learned linear combination

## Training Configuration

### Recommended Settings
- **Epochs**: 50-100 (with early stopping)
- **Batch size**: 64 (memory efficient for M1 Macs)
- **Learning rate**: 3e-4 with AdamW optimizer
- **Weight decay**: 1e-4 for regularization
- **Validation split**: 20% of training data
- **Test split**: 10% held out for final evaluation

### Memory Optimization
- Uses mixed precision when available
- Efficient data loading with PyTorch DataLoader
- Gradient accumulation for larger effective batch sizes
- Model checkpointing to save memory

## Evaluation Metrics

### Primary Metrics
- **Exact Accuracy**: Percentage of exactly correct predictions
- **Off-by-One Accuracy**: Allowing one ordinal level error
- **Spearman Correlation**: Rank correlation between predictions and true labels

### Secondary Metrics
- **Confusion Matrix**: Visual breakdown of prediction errors
- **Class-wise Precision/Recall**: Performance per risk level
- **Prediction Confidence**: Model uncertainty estimation

## File Structure

After running the complete pipeline:

```
runs/coral_complete_pipeline/
├── coral_training_data.jsonl          # Prepared training data
├── model/                             # Trained model artifacts
│   ├── best_model.pt                  # Best model checkpoint
│   ├── final_model.pt                 # Final model checkpoint
│   ├── config.json                    # Model configuration
│   ├── training_history.json          # Training metrics
│   ├── training_curves.png            # Training visualizations
│   ├── confusion_matrix.png           # Confusion matrix plot
│   ├── test_results.json              # Final test metrics
│   ├── classification_report.txt      # Detailed classification report
│   └── training.log                   # Training logs
├── predictions/                       # Inference results
│   ├── coral_predictions.jsonl        # Detailed predictions
│   └── coral_predictions_analysis.json # Prediction statistics
├── pipeline_config.json               # Pipeline configuration
├── pipeline_report.json               # Complete pipeline report
└── pipeline.log                       # Pipeline execution log
```

## Advanced Usage

### Custom Buckets
```bash
--buckets very_low low medium high very_high
```

### Model Architecture
```bash
--hidden-dims 1024 512 256  # Deeper network
--dropout 0.2               # More regularization
```

### Training Control
```bash
--val-split 0.15     # Smaller validation set
--test-split 0.05    # Smaller test set
--threshold 0.6      # Higher decision threshold
```

### Device Selection
```bash
--device mps         # Apple Silicon GPU
--device cuda        # NVIDIA GPU
--device cpu         # CPU only
```

## Traceability Features

### Original Embedding Access
All predictions maintain links to:
- Original sentence-transformer embeddings (`st_emb`)
- Original graph embeddings (`gph_emb`)
- Dependency parse information (`deps`)
- Source document paths (`_src`)

### Reversibility
The system preserves:
- Token-level information (`sp_ids`)
- Original text and context
- Speaker attribution
- Processing stage metadata

## Performance Tips

### For M1 Macs
- Use `--device mps` for GPU acceleration
- Keep batch size ≤ 64 to avoid memory issues
- Use efficient data loading with multiple workers

### For Large Datasets
- Increase batch size on high-memory systems
- Use gradient accumulation for memory-constrained training
- Consider data sharding for very large datasets

### Model Optimization
- Start with smaller networks and scale up
- Use early stopping to prevent overfitting
- Monitor validation metrics closely

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce batch size or use CPU
2. **Poor convergence**: Adjust learning rate or add regularization
3. **Imbalanced data**: Use class weights or data augmentation
4. **Missing embeddings**: Check data preparation step

### Performance Debugging
- Check training curves for overfitting
- Examine confusion matrix for systematic errors
- Review prediction confidence distributions
- Validate data quality and preprocessing

## Integration with Existing Pipeline

The CORAL system integrates seamlessly with the existing corporate speech risk pipeline:

1. **Data Flow**: Uses output from `fused_embeddings` stage
2. **Speaker Filtering**: Respects existing speaker exclusion lists
3. **Outcome Processing**: Works with existing judgment amount extraction
4. **Visualization**: Compatible with existing analysis tools

This allows you to enhance the existing similarity-based analysis with learned ordinal risk prediction while maintaining full compatibility with your current workflow.
