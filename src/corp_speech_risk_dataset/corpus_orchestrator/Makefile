.PHONY: demo_e2e repro_paper smoke_test clean setup help

# Default target
help:
	@echo "Corpus Orchestrator - End-to-End Pipeline Runner"
	@echo ""
	@echo "Available targets:"
	@echo "  demo_e2e     - Run end-to-end demo on 10-doc fixture"
	@echo "  repro_paper  - Reproduce full paper pipeline"
	@echo "  smoke_test   - Run smoke test across all modules"
	@echo "  clean        - Clean temporary files"
	@echo "  setup        - Setup development environment"
	@echo ""

# End-to-end demo pipeline (10 docs → final case vectors & quick CV report)
demo_e2e:
	@echo "--- Running E2E Demo Pipeline ---"
	uv run python orchestrator.py demo --input demo/fixtures/input --output demo/results

# Reproduce full paper pipeline with parametrization
repro_paper:
	@echo "--- Reproducing Paper Results ---"
	uv run python orchestrator.py paper --config configs/pipeline/paper.yaml

# Smoke test across all modules
smoke_test:
	@echo "--- Running E2E Smoke Test ---"
	uv run python orchestrator.py smoke --input demo/fixtures

# Clean temporary files
clean:
	@echo "--- Cleaning temporary files ---"
	rm -rf demo/results/*
	rm -rf demo/temp/*
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Setup development environment
setup:
	@echo "--- Setting up development environment ---"
	uv sync --dev
	@echo "Environment setup complete. Run 'make demo_e2e' to test the pipeline."

# Alternative way to run demo
demo: demo_e2e

# Alternative way to run paper reproduction
paper: repro_paper

# Alternative way to run smoke test
smoke: smoke_test
