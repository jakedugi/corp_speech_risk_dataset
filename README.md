# Corporate Speech Risk Dataset

! Work in Progress: This repository is under active development. Some modules are still being tested and refactored. !

This repository contains code and data for analyzing corporate speech risk across multiple regulatory and legal sources, including FTC, SEC, and CourtListener data.

This project follows **Clean Architecture** with Domain-Driven Design patterns:

```
 Project Root
├── src/corp_speech_risk_dataset/     # Main source code (Clean Architecture)
│   ├── domain/                       # Business logic (innermost layer)
│   ├── application/                  # Use cases and services
│   ├── adapters/                     # Interface adapters
│   ├── infrastructure/               # External frameworks/tools
│   ├── shared/                       # Cross-cutting utilities
│   ├── api/                          # External API clients
│   ├── extractors/                   # Text processing pipeline
│   ├── orchestrators/                # Workflow coordination
│   └── workflows/                    # Business workflows
├── data/                             # Data organization
│   ├── raw/                          # Original source data
│   ├── processed/                    # Cleaned/transformed data
│   └── output/                       # Final results
├── tests/                            # Test suite
│   ├── unit/                         # Unit tests
│   └── integration/                  # Integration tests
├── scripts/                          # Utility scripts
├── docs/                             # Documentation and diagrams
├── logs/                             # Application logs
├── temp/                             # Temporary files
└── artifacts/                        # Historical data and builds
```

## Core Features

### Data Sources
- **CourtListener API**: Legal case data and court documents
- **FTC/SEC APIs**: Regulatory filings and enforcement actions (planned)
- **S&P 500 Scraper**: Company metadata and executive information

### Text Processing Pipeline
1. **Quote Extraction**: Multi-sieve attribution system using regex, NLP, and semantic analysis
2. **Speaker Attribution**: Identifies corporate speakers using company aliases and role keywords
3. **Risk Encoding**: Converts text to 2048-dimension Weisfeiler-Lehman feature vectors
4. **Semantic Reranking**: Filters quotes by similarity to seed examples

### Architecture Benefits
- **Clean separation of concerns** between business logic and infrastructure
- **Testable components** with dependency injection
- **Modular design** enabling easy extension and modification
- **Modern Python patterns** with type hints and async/await

## Quick Start

### Installation
```bash
# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env with your API tokens
```

### Basic Usage
```bash
# Run quote extraction pipeline
python -m corp_speech_risk_dataset.cli orchestrate \
  --statutes "FTC Section 5" \
  --pages 5 \
  --outdir data/raw/courtlistener

# Extract and process quotes
python scripts/run_extraction.py

# Run tests
pytest tests/ -v
```

## Testing

The project includes comprehensive test coverage with **46 out of 49 tests currently passing**:

```bash
# Run all tests
pytest tests/ -v --cov=src/corp_speech_risk_dataset

# Run specific test categories
pytest tests/unit/ -v                    # Unit tests
pytest tests/integration/ -v             # Integration tests
```

### Test Status
- **46 tests passing** - Core functionality is working
- **3 tests failing** - Minor configuration and test data issues
- **Import issues resolved** - Fixed missing infrastructure.nlp module references

## Data Flow

1. **Collection**: Legal cases from CourtListener API
2. **Processing**: Quote extraction and speaker attribution
3. **Encoding**: Text → vector representations for analysis
4. **Storage**: Organized in `data/` with versioning

## Configuration

- **API tokens**: Set in `.env` file
- **Pipeline config**: `src/corp_speech_risk_dataset/orchestrators/quote_extraction_config.py`
- **Layer boundaries**: Enforced via `.importlinter` configuration

## Development

### Code Quality
- **Type checking**: `mypy src/`
- **Linting**: `flake8 src/`
- **Import boundaries**: `import-linter` (when available)
- **Pre-commit hooks**: Automated formatting and checks

### Architecture Principles
- Domain layer has **zero external dependencies**
- Application layer orchestrates **pure business logic**
- Adapters translate between **external formats and domain objects**
- Infrastructure handles **framework-specific implementations**

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## COMMANDS

---

### Orchestrate CourtListener Pipeline (API)

```bash
python -m corp_speech_risk_dataset.cli_api orchestrate \
  --statutes "FTC Section 5" \
  --company-file data/sp500_official_names_cleaned.csv \
  --outdir data/testing_apis/courtlistener_refactor \
  --pages 1 \
  --page-size 1 \
  --async
```

# Example log output for orchestration
# (Shows progress and file locations)

```
2025-07-15 18:52:17.482 | INFO     | ... Starting ASYNC CourtListener orchestration (search + full hydrate)
2025-07-15 18:52:25.331 | INFO     | ... Saved search results to data/testing_apis/courtlistener_refactor/search/search_api_results.json
2025-07-15 18:52:25.836 | INFO     | ... Saved 1 dockets to data/testing_apis/courtlistener_refactor/15-16585_ca9/dockets
2025-07-15 18:52:25.837 | DEBUG    | ... Loaded JSON from data/testing_apis/courtlistener_refactor/15-16585_ca9/dockets/dockets_0.json
2025-07-15 18:52:25.838 | DEBUG    | ... Loaded JSON from data/testing_apis/courtlistener_refactor/15-16585_ca9/ia_dump.json
2025-07-15 18:52:25.877 | INFO     | ... Fetching docket entries with docket_id: 6155026, query: all
2025-07-15 18:52:27.853 | INFO     | ... Retrieved 31 docket entries
2025-07-15 18:52:27.873 | INFO     | ... Saved 31 docket entries to data/testing_apis/courtlistener_refactor/15-16585_ca9/entries
2025-07-15 18:52:27.876 | DEBUG    | ... Loaded JSON from data/testing_apis/courtlistener_refactor/15-16585_ca9/entries/entry_20951687_metadata.json
...
Aborted.
```

---

### Quote Extraction

```bash
python scripts/run_extraction.py
```

# Example log output for quote extraction

```
2025-07-15 19:10:14.692 | INFO     | __main__:main:20 - Starting quote extraction process...
2025-07-15 19:10:14.692 | INFO     | ... Initializing Quote Extraction Pipeline...
2025-07-15 19:10:15.205 | WARNING  | ... fastcoref not available, skipping: [E002] Can't find factory for 'fastcoref' ...
2025-07-15 19:10:20.947 | INFO     | ... Pipeline initialized.
2025-07-15 19:10:20.950 | INFO     | ... Saving results to extracted_quotes.jsonl...
2025-07-15 19:10:20.951 | DEBUG    | ... Starting pipeline run...
2025-07-15 19:10:20.951 | DEBUG    | ... Pipeline run finished.
2025-07-15 19:10:20.952 | INFO     | ... Saved 0 documents with quotes.
```

---

### Encode Quotes

```bash
python encode_quotes.py data/extracted/ --recursive
```

---

### Make Vectors for Clustering

```bash
python src/corp_speech_risk_dataset/clustering/make_vectors.py \
  --meta data/clustering/metadata.json \
  --out data/clustering/concat_vectors.npy
```

# Example log output for vector creation

```
→ 25% done at 340.6s elapsed
→ 50% done at 660.8s elapsed
→ 75% done at 1423.3s elapsed
Wrote (96732, 2816) → data/clustering/concat_vectors.npy
```

---

### Run Clustering

```bash
python -m corp_speech_risk_dataset.cli_cluster \
  --vec data/clustering/concat_vectors.npy \
  --meta data/clustering/metadata.json \
  --out data/clustering/clusters.html
```

---

### Dump Clusters (for downstream analysis)

```bash
uv run scripts/dump_clusters.py
```

# Example: dump cluster labels to JSON for analysis, plotting, or audit logs

```python
#!/usr/bin/env python3
'''
For use after the clustering pipeline has been run. Dumps idx→cluster mapping to JSON.
Automates downstream analysis, hyperparameter sweeps, and audit logs.
'''
from pathlib import Path
import argparse
from corp_speech_risk_dataset.clustering.pipeline import ClusterPipeline

def main():
    p = argparse.ArgumentParser(description="Dump idx→cluster mapping to JSON")
    # ...
```

---

### Prepare Metadata for Clustering

```bash
python -m corp_speech_risk_dataset.clustering.utils.prepare_metadata \
  --input-dir data/tokenized \
  --output-path data/clustering/metadata.json \
  --exclude-speakers Unknown Court FTC Fed Plaintiff State Commission Congress Circuit FDA
```

# Example log output

```
Dropped 24265 entries for excluded speakers
Wrote lossless metadata with 72457 entries to data/clustering/metadata.json
```

---

### Create Notebook from Extracted Data

```bash
python scripts/create_notebook.py \
  --data-root data/extracted/rss \
  --out notebooks/reports/rss_pipeline.ipynb
```

---

### Encode RSS Data

```bash
python -m corp_speech_risk_dataset.cli_encode \
  data/extracted/rss \
  -r \
  --extracted-root data/extracted/rss \
  --tokenized-root data/tokenized/rss
```

---

### Prepare Metadata with Heuristics

```bash
python -m corp_speech_risk_dataset.clustering.utils.prepare_metadata \
  --input-dir data/outcomes \
  --output-path data/clustering/metadata.json \
  --apply-heuristics
```

# Example log output

```
Dropped 50434 entries by heuristic filters
Wrote lossless metadata with 47081 entries to data/clustering/metadata.json
```

---

### Make Vectors (with environment variables for performance)

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
python -m corp_speech_risk_dataset.clustering.utils.make_vectors \
  --meta data/clustering/metadata.json \
  --out  data/clustering/concat_vectors.npy
```

# Example log output

```
→ 25% done at 4.2s elapsed
→ 50% done at 7.9s elapsed
→ 75% done at 12.5s elapsed
Wrote (976, 2816) → data/clustering/concat_vectors.npy
```

---

### Case Outcome Imputation (Final Step)

# Appends amounts and percentiles of the final judgement amount in favor of plaintiffs.
# Positive values = plaintiff, low/negative = defendant.

```bash
uv run python -m corp_speech_risk_dataset.case_outcome.case_outcome_imputer \
  --root        data/tokenized/courtlistener \
  --stage1-root data/extracted/courtlistener \
  --outdir      data/outcomes/courtlistener \
  --mode        manual \
  --context-chars 400 \
  --min-amount    0
```

---

### Supervised Clustering (Categorical and Continuous)

```bash
python -m src.corp_speech_risk_dataset.cli_cluster \
  --vec data/clustering/concat_vectors.npy \
  --meta data/clustering/metadata.json \
  --supervision categorical \
  --out data/clustering/supervised_cat.html

python -m src.corp_speech_risk_dataset.cli_cluster \
  --vec data/clustering/concat_vectors.npy \
  --meta data/clustering/metadata.json \
  --supervision continuous \
  --out data/clustering/supervised_cat.html
```


# PyG extensions must be installed manually (not via pyproject.toml or uv)

```bash
pip install pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)")+cpu.html
```

1. **Let UV handle only Python dependencies:**

```bash
rm uv.lock
uv lock
uv sync --extra cpu
```

2. **Manually install PyG extensions:**

```bash
pip install pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-$(python -c "import torch;print(torch.__version__)")+cpu.html
```

3. **On CUDA (e.g. Colab):**

Switch the URL suffix to +cu118 (or your CUDA version) and run:

```bash
pip install ... -f ...+cu118.html
```

FOR COLAB: # 1. Clone your repo
!git clone https://github.com/YOUR-USERNAME/corp_speech_risk_dataset.git
%cd corp_speech_risk_dataset

# 2. Install or upgrade uv
!pip install --upgrade uv

# 3. (Option A) Ensure gpu extra exists:
!uv add "torch>=2.7.1" --optional gpu
!uv add "torchvision>=0.14.1" --optional gpu

# 4. Sync all deps, including gpu
!uv sync --extra gpu

# 5. Manually install PyG extensions
!pip install pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-$(python -c "import torch;print(torch.__ve



python3 src/corp_speech_risk_dataset/clustering/utils/summary_y_stats.py data/tokenized/courtlistener_v5_final_with_outcome --field final_judgement_real --output-format text

uv run pip install spacy




 # CORAL Ordinal MLP (No PPLM)

## Quick Start
1. Put your dataset as JSONL, one object per line containing keys `fused_emb` (list[float]) and `bucket` (e.g. "Low"/"Medium"/"High").
2. Install deps: `pip install torch numpy scipy scikit-learn matplotlib`.
3. Train:
```bash
python -m coral_ordinal.cli --data path/to/data.jsonl --buckets Low Medium High --plot-cm
```
4. Metrics (Exact, Off-by-one, Spearman ρ) will print, confusion matrix saved to `runs/coral/cm.png`.
5. Use `--eval-only --model-path runs/coral/best.pt` to just evaluate.

### Why CORAL?
- Outputs K-1 cumulative logits → enforces order consistency.
- Simple BCE-with-logits loss, easy to backprop in downstream steering.
- Predict by counting thresholds surpassed → exact/off-by-one metrics trivial to compute.

### Design Choices
- **Shared-weight CORAL head** to guarantee monotonic logits (lightweight & provable).
- **Config dataclass** to keep hyperparams tidy & serializable.
- **Device auto-select** (MPS/cuda/cpu) works on
- **Unit tests** for buckets, loss, metrics, and forward pass ensure correctness & easy refactors.
- **Visualization** via confusion matrix for quick sanity-checks.

### Extending
- Swap MLP backbone for Transformer/Graph nets w/out touching CORAL head.
- Increase buckets (e.g. deciles) by just listing more labels.
- Add calibration or ordinal-specific metrics (MAE, Kendall τ) in `metrics.py`.

### Downstream Compatibility
- The `predict` method returns integer ranks [0..K-1]; plug directly into PPLM later as classifier logits/gradients.
- Check gate: script prints exact, off-by-one, ρ so you can enforce ≥40/80/0.5 thresholds.





(corp_speech_risk_dataset)  corp_speech_risk_dataset % uv run python src/corp_speech_risk_dataset/case_outcome/final_evaluate.py \
    --annotations data/gold_standard/case_outcome_amounts_hand_annotated.csv \
    --extracted-root data/extracted/courtlistener

uv run python src/corp_speech_risk_dataset/case_outcome/case_outcome_imputer.py \
  --root data/tokenized/courtlistener_v5_final \
  --stage1-root data/extracted/courtlistener \
  --outdir data/outcomes/courtlistener_v1 \
  --mode auto \
  --context-chars 561 \
  --min-amount 29309.97970771781 \
  --min-features 15 \
  --case-position-threshold 0.5423630428751168 \
  --docket-position-threshold 0.7947200838693315 \
  --dismissal-ratio-threshold 200.0 \
  --bankruptcy-ratio-threshold 6e22 \
  --patent-ratio-threshold 6e22 \


# Run complete analysis
uv run python scripts/outcome_summary_statistics.py \
    --outcomes-dir data/outcomes/courtlistener_v1 \
    --max-threshold 5000000000.00 \
    --export-csv data/outcomes/detailed_export.csv

# Generate visualizations
uv run python scripts/outcome_distribution_plots.py \
    --outcomes-dir data/outcomes/courtlistener_v1 \
    --output-dir plots/outcomes \
    --max-threshold 5000000000.00


uv run python scripts/visualize_similarity_by_outcomes.py --input "data/outcomes/courtlistener_v1/*/doc_*_text_stage9.jsonl" --output similarity_by_outcomes.html --max-threshold 15500000000 --exclude-speakers "Unknown,Court,FTC,Fed,Plaintiff,State,Commission,Congress,Circuit,FDA"


uv run python scripts/visualize_similarity_by_outcomes.py --input "data/outcomes/courtlistener_v1/*/doc_*_text_stage9.jsonl" --output similarity_by_outcomes_no_missing.html --max-threshold 15500000000 --exclude-speakers "Unknown,Court,FTC,Fed,Plaintiff,State,Commission,Congress,Circuit,FDA" --exclude-missing


!python -m src.corp_speech_risk_dataset.cli_encode \
  --out-root /content/corp_speech_risk_dataset/data/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/outcomes/courtlistener_v2_legal_bert \
  --recursive \
  --stage 9 \
  --text-embedder legal-bert \
  --embed-batch-size 32 \
  --use-amp \
  /content/corp_speech_risk_dataset/data/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/outcomes/courtlistener_v1 \
  embed



!python -m src.corp_speech_risk_dataset.cli_encode \
  --out-root /content/corp_speech_risk_dataset/data/outcomes/courtlistener_v3_legal_bert_graphsage \
  --recursive \
  --stage 10 \
  --hidden-dim 768 \
  --epochs 40 \
  /content/corp_speech_risk_dataset/data/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/outcomes/courtlistener_v2_legal_bert \
  graph \
    --graph-embed graphsage \
    --batch-size 512 \
    --loss-type hybrid \
    --eval-graph \
    --use-amp
"""
