"""Example usage of the enhanced fully interpretable module.

This script demonstrates how to use the advanced features for publication-ready
legal risk classification.
"""

from pathlib import Path
import json
from loguru import logger

from corp_speech_risk_dataset.fully_interpretable import (
    InterpretableConfig,
    train_and_eval,
    save_model,
    load_model,
    predict_directory,
)


def main():
    """Run a complete example workflow."""

    # 1. Configure the model with all enhancements
    config = InterpretableConfig(
        data_path="data/interpretable_training_data.jsonl",
        label_key="bucket",
        buckets=("Low", "Medium", "High"),
        # Text features
        text_keys=("text", "context"),  # Use both quote and context
        include_text=True,
        # Enhanced interpretable features
        include_lexicons=True,  # Risk lexicons
        include_sequence=True,  # Sequence modeling
        include_linguistic=True,  # Linguistic analysis
        include_structural=True,  # Structural features
        # Original features
        include_keywords=True,
        include_speaker=True,
        include_numeric=True,
        include_scalars=True,
        # Model selection - POLR for ordinal
        model="polr",
        model_params={
            "C": 1.0,
            "penalty": "l2",
            "max_iter": 1000,
        },
        # Optimization
        calibrate=True,
        calibration_method="isotonic",
        feature_selection=True,
        n_features=5000,
        use_sparse=True,
        n_jobs=-1,
        # Output options
        generate_report=True,
        output_dir="runs/polr_full_example",
        # Training
        val_split=0.2,
        seed=42,
    )

    logger.info("Starting enhanced interpretable training...")

    # 2. Train and evaluate with full validation
    results = train_and_eval(
        config,
        run_validation=True,
        case_outcomes=None,  # Would load from file if available
    )

    # 3. Display key metrics
    print("\n=== Training Results ===")
    print(f"Accuracy: {results['metrics']['accuracy']:.3f}")
    print(f"QWK: {results['metrics']['qwk']:.3f}")
    print(f"MAE: {results['metrics']['mae']:.3f}")
    print(f"ECE: {results['metrics'].get('ece', 'N/A')}")

    # 4. Save the final model
    from corp_speech_risk_dataset.fully_interpretable.pipeline import (
        build_dataset,
        make_model,
    )

    rows, y, labels, feature_extractor = build_dataset(config)

    model = make_model(
        config,
        enable_text_branch=True,
        enable_scalar_branch=True,
        enable_keywords=True,
        enable_speaker=True,
    )

    # Train on full dataset
    model.fit(rows, y)

    # Save with all components
    model_path = "runs/polr_full_example/model.joblib"
    save_model(model, labels, config, model_path, feature_extractor)

    print(f"\nModel saved to: {model_path}")

    # 5. Example: Load and use for prediction
    print("\n=== Example Prediction ===")

    # Load the model
    loaded_model, loaded_labels, loaded_config, loaded_extractor = load_model(
        model_path
    )

    # Example quote for prediction
    example_quote = {
        "text": "We guarantee the lowest prices in the market with absolutely no hidden fees.",
        "context": "The CEO stated during the earnings call that...",
        "speaker_raw": "CEO",
        "quote_top_keywords": ["guarantee", "lowest", "prices"],
        "context_top_keywords": ["earnings", "call"],
    }

    # Extract features
    from corp_speech_risk_dataset.fully_interpretable.pipeline import (
        _row_to_feature_dict,
    )

    feature_row = _row_to_feature_dict(example_quote, loaded_config, loaded_extractor)

    # Make prediction
    prediction = loaded_model.predict([feature_row])[0]
    proba = loaded_model.predict_proba([feature_row])[0]

    print(f"Quote: '{example_quote['text'][:50]}...'")
    print(f"Predicted risk: {loaded_labels[prediction]}")
    print("Probabilities:")
    for label, prob in zip(loaded_labels, proba):
        print(f"  {label}: {prob:.3f}")

    # 6. Show validation results if available
    if "validation" in results:
        print("\n=== Validation Experiments ===")
        val_results = results["validation"]

        if "negative_control" in val_results:
            nc = val_results["negative_control"]
            print(f"Negative Control - Accuracy drop: {nc['accuracy_drop']:.3f}")
            print(
                f"Negative Control - Significant: {nc['significance_test']['significant']}"
            )

        if "feature_ablation" in val_results:
            print("\nFeature Ablation - Top feature groups by importance:")
            for i, group in enumerate(
                val_results["feature_ablation"]["feature_importance_ranking"][:3]
            ):
                print(
                    f"  {i+1}. {group['feature_group']}: {group['relative_drop']:.1f}% drop"
                )

    print("\n=== Complete ===")
    print(f"Reports saved to: {config.output_dir}")
    print("Check the following directories for publication-ready outputs:")
    print("  - interpretability/: Feature importance plots, confusion matrices")
    print("  - validation/: Negative control and ablation experiments")


def example_batch_prediction():
    """Example of batch prediction with interpretability."""

    # Predict on a directory
    output_dir = predict_directory(
        model_path="runs/polr_full_example/model.joblib",
        input_root="data/test",
        output_root="data/test_predictions",
        batch_size=100,
    )

    print(f"Predictions saved to: {output_dir}")

    # Read and inspect a prediction
    prediction_file = list(output_dir.glob("*.jsonl"))[0]
    with open(prediction_file) as f:
        pred = json.loads(f.readline())

    print(f"\nExample prediction fields:")
    for key in pred:
        if key.startswith("fi_"):
            print(f"  {key}: {pred[key]}")


if __name__ == "__main__":
    main()
    # example_batch_prediction()  # Uncomment to test batch prediction
