.PHONY: run setup_notebook visualize all optimize test_optimization

# Default command: runs the whole pipeline from data extraction to visualization
all: run setup_notebook visualize

# Step 1: Run the Python-based quote extraction pipeline
run:
	@echo "--- Running Quote Extraction Pipeline ---"
	uv run python3 run_extraction.py --visualize

# Step 2: (One-time setup) Create the visualization notebook from the script
setup_notebook:
	@echo "--- Setting up Visualization Notebook ---"
	uv run python3 scripts/create_notebook.py

# Step 3: Open the generated notebook
visualize DATA_DIR?=extracted_quotes
visualize:
	@echo "--- Launching notebook for $(DATA_DIR) ---"
	uv run python3 scripts/create_notebook.py --data-root $(DATA_DIR) \
	          --out notebooks/reports/pipeline_visualization.ipynb && \
	uv run python3 -m notebook notebooks/reports/pipeline_visualization.ipynb

# Step 4: Test the optimization setup
test_optimization:
	@echo "--- Testing Optimization Setup ---"
	cd src/corp_speech_risk_dataset/case_outcome && \
	uv run python3 test_optimization.py

# Step 5: Run hyperparameter optimization (reduced grid for testing)
optimize:
	@echo "--- Running Hyperparameter Optimization ---"
	cd src/corp_speech_risk_dataset/case_outcome && \
	uv run python3 run_optimization.py

# Step 6: Run full hyperparameter optimization (comprehensive grid)
optimize_full:
	@echo "--- Running Full Hyperparameter Optimization ---"
	cd src/corp_speech_risk_dataset/case_outcome && \
	uv run python3 run_optimization.py --full-grid --max-workers 4

# Step 7: Run Bayesian hyperparameter optimization (intelligent search)
optimize_bayesian:
	@echo "--- Running Bayesian Hyperparameter Optimization ---"
	cd src/corp_speech_risk_dataset/case_outcome && \
	uv run python3 run_optimization.py --bayesian --max-combinations 50

# Step 8: Run quick Bayesian optimization (limited evaluations)
optimize_bayesian_quick:
	@echo "--- Running Quick Bayesian Optimization (Fast Mode) ---"
	cd src/corp_speech_risk_dataset/case_outcome && \
	uv run python3 run_optimization.py --bayesian --max-combinations 150 --fast-mode

# Step 9: Monitor optimization progress
monitor_optimization:
	@echo "--- Monitoring Optimization Progress ---"
	cd src/corp_speech_risk_dataset/case_outcome && \
	uv run python3 monitor_optimization.py --interval 15
