.PHONY: run setup_notebook visualize all

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
