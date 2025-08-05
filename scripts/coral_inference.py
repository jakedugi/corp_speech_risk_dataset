#!/usr/bin/env python3
"""
Run CORAL model inference on new data with traceability.

This script:
1. Loads trained CORAL model and configuration
2. Processes new JSONL data with fused embeddings
3. Generates ordinal predictions with confidence scores
4. Maintains traceability back to original embeddings and sentences
5. Saves results with detailed metadata

Usage:
    python scripts/coral_inference.py \
        --model runs/coral_experiment/best_model.pt \
        --input "data/new_data/*/doc_*_text_stage9.jsonl" \
        --output predictions/coral_predictions.jsonl \
        --batch-size 64
"""

import argparse
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import time

# Import CORAL components
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))
from corp_speech_risk_dataset.coral_ordinal.config import Config, load_config
from corp_speech_risk_dataset.coral_ordinal.model import CORALMLP
from corp_speech_risk_dataset.coral_ordinal.utils import choose_device


class InferenceDataset(Dataset):
    """Dataset for CORAL inference with full metadata preservation."""

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        self.features = []
        self.metadata = []

        for record in data:
            if "fused_emb" not in record:
                logger.warning(
                    f"Skipping record without fused_emb: {record.get('doc_id', 'unknown')}"
                )
                continue

            # Extract features
            features = torch.tensor(record["fused_emb"], dtype=torch.float32)
            self.features.append(features)

            # Preserve all metadata for traceability
            metadata = {
                "doc_id": record.get("doc_id"),
                "text": record.get("text"),
                "speaker": record.get("speaker"),
                "score": record.get("score"),
                "context": record.get("context"),
                "st_emb": record.get(
                    "st_emb"
                ),  # Original sentence transformer embedding
                "gph_emb": record.get("gph_emb"),  # Original graph embedding
                "gph_method": record.get("gph_method"),
                "deps": record.get("deps"),  # Dependency parse info
                "sp_ids": record.get("sp_ids"),  # SentencePiece token IDs
                "final_judgement_real": record.get("final_judgement_real"),
                "_src": record.get("_src"),
                "urls": record.get("urls", []),
                "stage": record.get("stage"),
            }
            self.metadata.append(metadata)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], idx  # Return index for metadata lookup

    def get_metadata(self, idx):
        """Get metadata for traceability."""
        return self.metadata[idx]


def load_data(pattern: str) -> List[Dict[str, Any]]:
    """Load JSONL data files matching the pattern."""
    files = glob.glob(pattern) if "*" in pattern else [pattern]

    all_data = []
    for file_path in files:
        logger.info(f"Loading {file_path}...")
        try:
            with open(file_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        all_data.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"JSON decode error in {file_path}:{line_num}: {e}"
                        )
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue

    logger.info(f"Loaded {len(all_data)} records from {len(files)} files")
    return all_data


def load_model(model_path: str, device: torch.device) -> Tuple[CORALMLP, Config]:
    """Load trained CORAL model and configuration."""
    model_path = Path(model_path)
    config_path = model_path.parent / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load configuration
    config = load_config(config_path)

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Create model
    input_dim = checkpoint.get("input_dim")
    if input_dim is None:
        raise ValueError("Model checkpoint missing input_dim information")

    model = CORALMLP(
        in_dim=input_dim,
        num_classes=len(config.buckets),
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, config


def run_inference(
    model: CORALMLP,
    dataloader: DataLoader,
    dataset: InferenceDataset,
    config: Config,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Run inference and generate predictions with metadata."""
    model.eval()

    results = []
    total_batches = len(dataloader)

    logger.info(f"Running inference on {len(dataset)} samples...")

    with torch.no_grad():
        for batch_idx, (features, indices) in enumerate(dataloader):
            features = features.to(device)

            # Forward pass
            logits = model(features)
            probabilities = torch.sigmoid(logits)

            # Generate predictions
            predictions = (probabilities > config.prob_threshold).sum(1)

            # Convert to numpy
            probs_np = probabilities.cpu().numpy()
            preds_np = predictions.cpu().numpy()
            indices_np = indices.numpy()

            # Process each sample in batch
            for i, idx in enumerate(indices_np):
                metadata = dataset.get_metadata(idx)

                # Calculate confidence scores
                class_probs = []
                for class_idx in range(len(config.buckets)):
                    if class_idx == 0:
                        # P(class 0) = 1 - P(first threshold)
                        prob = 1.0 - probs_np[i, 0]
                    elif class_idx == len(config.buckets) - 1:
                        # P(last class) = P(last threshold)
                        prob = probs_np[i, class_idx - 1]
                    else:
                        # P(middle class) = P(threshold) - P(previous threshold)
                        prob = probs_np[i, class_idx - 1] - probs_np[i, class_idx]

                    class_probs.append(float(prob))

                # Get predicted class and confidence
                predicted_class = int(preds_np[i])
                predicted_bucket = config.buckets[predicted_class]
                confidence = class_probs[predicted_class]

                # Calculate prediction entropy (uncertainty measure)
                epsilon = 1e-10  # Small constant to avoid log(0)
                class_probs_safe = np.array(class_probs) + epsilon
                class_probs_safe = (
                    class_probs_safe / class_probs_safe.sum()
                )  # Renormalize
                entropy = -np.sum(class_probs_safe * np.log(class_probs_safe))

                # Create result record
                result = {
                    # Prediction results
                    "predicted_bucket": predicted_bucket,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "class_probabilities": {
                        config.buckets[j]: class_probs[j]
                        for j in range(len(config.buckets))
                    },
                    "prediction_uncertainty": float(entropy),
                    "ordinal_scores": probs_np[i].tolist(),
                    # Model metadata
                    "model_threshold": config.prob_threshold,
                    "model_buckets": config.buckets,
                    # Original data with full traceability
                    **metadata,
                    # Inference metadata
                    "inference_timestamp": time.time(),
                    "batch_index": batch_idx,
                    "sample_index": idx,
                }

                results.append(result)

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed batch {batch_idx + 1}/{total_batches}")

    logger.info(f"Inference complete. Generated {len(results)} predictions.")
    return results


def analyze_predictions(
    results: List[Dict[str, Any]], config: Config
) -> Dict[str, Any]:
    """Analyze prediction distribution and confidence statistics."""
    # Count predictions by bucket
    bucket_counts = {bucket: 0 for bucket in config.buckets}
    confidences = []
    uncertainties = []

    for result in results:
        bucket_counts[result["predicted_bucket"]] += 1
        confidences.append(result["confidence"])
        uncertainties.append(result["prediction_uncertainty"])

    # Calculate statistics
    total_predictions = len(results)
    bucket_percentages = {
        bucket: 100 * count / total_predictions
        for bucket, count in bucket_counts.items()
    }

    analysis = {
        "total_predictions": total_predictions,
        "bucket_distribution": {
            "counts": bucket_counts,
            "percentages": bucket_percentages,
        },
        "confidence_stats": {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "median": float(np.median(confidences)),
        },
        "uncertainty_stats": {
            "mean": float(np.mean(uncertainties)),
            "std": float(np.std(uncertainties)),
            "min": float(np.min(uncertainties)),
            "max": float(np.max(uncertainties)),
            "median": float(np.median(uncertainties)),
        },
    }

    return analysis


def save_results(
    results: List[Dict[str, Any]], analysis: Dict[str, Any], output_path: str
) -> None:
    """Save inference results and analysis."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    logger.info(f"Saved {len(results)} predictions to {output_path}")

    # Save analysis summary
    analysis_path = output_path.parent / f"{output_path.stem}_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"Saved analysis to {analysis_path}")

    # Print summary
    logger.info("Prediction Summary:")
    for bucket, count in analysis["bucket_distribution"]["counts"].items():
        pct = analysis["bucket_distribution"]["percentages"][bucket]
        logger.info(f"  {bucket}: {count} ({pct:.1f}%)")

    conf_stats = analysis["confidence_stats"]
    logger.info(
        f"Confidence: mean={conf_stats['mean']:.3f}, std={conf_stats['std']:.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run CORAL model inference")
    parser.add_argument(
        "--model", required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument("--input", required=True, help="Input JSONL pattern or file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for inference"
    )
    parser.add_argument("--device", default=None, help="Device (cpu/cuda/mps/auto)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override model threshold (default: use model's threshold)",
    )

    args = parser.parse_args()

    logger.info("Starting CORAL inference")

    # Setup device
    device = choose_device(args.device)
    logger.info(f"Using device: {device}")

    # Load model and config
    model, config = load_model(args.model, device)

    # Override threshold if specified
    if args.threshold is not None:
        config.prob_threshold = args.threshold
        logger.info(f"Using custom threshold: {args.threshold}")

    # Load data
    data = load_data(args.input)
    if not data:
        logger.error("No data loaded!")
        return

    # Create dataset and dataloader
    dataset = InferenceDataset(data)
    if len(dataset) == 0:
        logger.error("No valid samples found!")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"Processing {len(dataset)} samples with batch size {args.batch_size}")

    # Run inference
    results = run_inference(model, dataloader, dataset, config, device)

    # Analyze results
    analysis = analyze_predictions(results, config)

    # Save results
    save_results(results, analysis, args.output)

    logger.success("Inference completed successfully!")


if __name__ == "__main__":
    main()
