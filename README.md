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

COMMANDS: corp_speech_risk_dataset % python -m corp_speech_risk_dataset.cli_api orchestrate \
  --statutes "FTC Section 5" \
  --company-file data/sp500_official_names_cleaned.csv \
  --outdir data/testing_apis/courtlistener_refactor \
  --pages 1 \
  --page-size 1 \
  --async \

2025-07-15 18:52:17.482 | INFO     | corp_speech_risk_dataset.orchestrators.courtlistener_orchestrator:run_async:72 - Starting ASYNC CourtListener orchestration (search + full hydrate)
2025-07-15 18:52:25.331 | INFO     | corp_speech_risk_dataset.orchestrators.courtlistener_orchestrator:_search_and_hydrate_async:105 - Saved search results to data/testing_apis/courtlistener_refactor/search/search_api_results.json
2025-07-15 18:52:25.836 | INFO     | corp_speech_risk_dataset.api.courtlistener.core:process_and_save:42 - Saved 1 dockets to data/testing_apis/courtlistener_refactor/15-16585_ca9/dockets
2025-07-15 18:52:25.837 | DEBUG    | corp_speech_risk_dataset.infrastructure.file_io:load_json:44 - Loaded JSON from data/testing_apis/courtlistener_refactor/15-16585_ca9/dockets/dockets_0.json
2025-07-15 18:52:25.838 | DEBUG    | corp_speech_risk_dataset.infrastructure.file_io:load_json:44 - Loaded JSON from data/testing_apis/courtlistener_refactor/15-16585_ca9/ia_dump.json
2025-07-15 18:52:25.877 | INFO     | corp_speech_risk_dataset.api.courtlistener.core:process_docket_entries:215 - Fetching docket entries with docket_id: 6155026, query: all
^C2025-07-15 18:52:27.853 | INFO     | corp_speech_risk_dataset.api.courtlistener.core:process_docket_entries:247 - Retrieved 31 docket entries
2025-07-15 18:52:27.873 | INFO     | corp_speech_risk_dataset.api.courtlistener.core:process_docket_entries:273 - Saved 31 docket entries to data/testing_apis/courtlistener_refactor/15-16585_ca9/entries
2025-07-15 18:52:27.876 | DEBUG    | corp_speech_risk_dataset.infrastructure.file_io:load_json:44 - Loaded JSON from data/testing_apis/courtlistener_refactor/15-16585_ca9/entries/entry_20951687_metadata.json
2025-07-15 18:52:27.876 | DEBUG    | corp_speech_risk_dataset.infrastructure.file_io:load_json:44 - Loaded JSON from data/testing_apis/courtlistener_refactor/15-16585_ca9/entries/entry_20951690_metadata.json
2025-07-15 18:52:27.876 | DEBUG    | corp_speech_risk_dataset.infrastructure.file_io:load_json:44 - Loaded JSON from data/testing_apis/courtlistener_refactor/15-16585_ca9/entries/entry_20951694_metadata.json
2025-07-15 18:52:27.876 | DEBUG    | corp_speech_risk_dataset.infrastructure.file_io:load_json:44 - Loaded JSON from data/testing_apis/courtlistener_refactor/15-16585_ca9/entries/entry_20951693_metadata.json
2025-07-15 18:52:27.877 | DEBUG    | corp_speech_risk_dataset.infrastructure.file_io:load_json:44 - Loaded JSON from data/testing_apis/courtlistener_refactor/15-16585_ca9/entries/entry_20951697_metadata.json
2025-07-15 18:52:27.881 | DEBUG    | corp_speech_risk_dataset.infrastructure.file_io:load_json:44 - Loaded JSON from data/testing_apis/courtlistener_refactor/15-16585_ca9/entries/entry_20951691_metadata.json
2025-07-15 18:52:27.881 | DEBUG    | corp_speech_risk_dataset.infrastructure.file_io:load_json:44 - Loaded JSON from data/testing_apis/courtlistener_refactor/15-16585_ca9/entries/entry_20951686_metadata.json

Aborted.
 corp_speech_risk_dataset % python run_quote_extractor.py

python: can't open file '': [Errno 2] No such file or directory
 corp_speech_risk_dataset % python scripts/run_extraction.py
2025-07-15 19:10:14.692 | INFO     | __main__:main:20 - Starting quote extraction process...
2025-07-15 19:10:14.692 | INFO     | corp_speech_risk_dataset.orchestrators.quote_extraction_pipeline:__init__:31 - Initializing Quote Extraction Pipeline...
2025-07-15 19:10:15.205 | WARNING  | corp_speech_risk_dataset.infrastructure.nlp:get_nlp:15 - fastcoref not available, skipping: [E002] Can't find factory for 'fastcoref' for language English (en). This usually happens when spaCy calls `nlp.create_pipe` with a custom component name that's not registered on the current language class. If you're using a custom component, make sure you've added the decorator `@Language.component` (for function components) or `@Language.factory` (for class components).

Available factories: attribute_ruler, tok2vec, merge_noun_chunks, merge_entities, merge_subtokens, token_splitter, doc_cleaner, parser, beam_parser, lemmatizer, trainable_lemmatizer, entity_linker, entity_ruler, tagger, morphologizer, ner, beam_ner, senter, sentencizer, spancat, spancat_singlelabel, span_finder, future_entity_ruler, span_ruler, textcat, textcat_multilabel, en.lemmatizer
2025-07-15 19:10:20.947 | INFO     | corp_speech_risk_dataset.orchestrators.quote_extraction_pipeline:__init__:55 - Pipeline initialized.
2025-07-15 19:10:20.950 | INFO     | corp_speech_risk_dataset.orchestrators.quote_extraction_pipeline:save_results:162 - Saving results to extracted_quotes.jsonl...
2025-07-15 19:10:20.951 | DEBUG    | corp_speech_risk_dataset.orchestrators.quote_extraction_pipeline:run:81 - Starting pipeline run...
2025-07-15 19:10:20.951 | DEBUG    | corp_speech_risk_dataset.orchestrators.quote_extraction_pipeline:run:153 - Pipeline run finished.
2025-07-15 19:10:20.952 | INFO     | corp_speech_risk_dataset.orchestrators.quote_extraction_pipeline:save_results:172 - Saved 0 documents with quotes.
 corp_speech_risk_dataset % python encode_quotes.py data/extracted/ --recursive

corp_speech_risk_dataset % python src/corp_speech_risk_dataset/clustering/make_vectors.py \
  --meta data/clustering/metadata.json \
  --out data/clustering/concat_vectors.npy
→ 25% done at 340.6s elapsed
→ 50% done at 660.8s elapsed
→ 75% done at 1423.3s elapsed
Wrote (96732, 2816) → data/clustering/concat_vectors.npy

corp_speech_risk_dataset % python -m corp_speech_risk_dataset.cli_cluster \
  --vec data/clustering/concat_vectors.npy \
  --meta data/clustering/metadata.json \
  --out data/clustering/clusters.html



Dump clusters: uv run scripts/dump_clusters.py
#!/usr/bin/env python3
''' For use after the clustering pipeline has been run. In order to get the cluster labels for each document, we need to dump the cluster labels to a JSON file.
it’s very handy for:
	•	Automating downstream analysis or plotting (e.g. grouping sentences by risk-level in Python/R).
	•	Hyperparameter sweeps: you can quickly diff two runs’ JSONs to see how cluster assignments shifted.
	•	Audit logs: regulators often want a raw data dump, not just visuals.
'''
from pathlib import Path
import argparse
from corp_speech_risk_dataset.clustering.pipeline import ClusterPipeline

def main():
    p = argparse.ArgumentParser(
        description="Dump idx→cluster mapping to JSON"
    )
     corp_speech_risk_dataset % python -m corp_speech_risk_dataset.clustering.utils.prepare_metadata \
  --input-dir data/tokenized \
  --output-path data/clustering/metadata.json \
  --exclude-speakers Unknown Court FTC Fed Plaintiff State Commission Congress Circuit FDA
2025-07-17 12:14:13.214 | INFO     | corp_speech_risk_dataset.encoding.tokenizer:<module>:82 - Loaded GPT-2 byte-level BPE tokenizer (50,257 tokens) once at startup
Dropped 24265 entries for excluded speakers
Wrote lossless metadata with 72457 entries to data/clustering/metadata.json




(corp_speech_risk_dataset) corp_speech_risk_dataset % python scripts/create_notebook.py \
  --data-root data/extracted/rss \
  --out notebooks/reports/rss_pipeline.ipynb


  corp_speech_risk_dataset % python -m corp_speech_risk_dataset.cli_encode \
    data/extracted/rss \
    -r \
    --extracted-root data/extracted/rss \
    --tokenized-root data/tokenized/rss


(corp_speech_risk_dataset)  corp_speech_risk_dataset % python -m corp_speech_risk_dataset.clustering.utils.prepare_metadata \
  --input-dir data/outcomes \
  --output-path data/clustering/metadata.json \
--apply-heuristics
2025-07-22 08:53:56.257 | INFO     | corp_speech_risk_dataset.encoding.tokenizer:<module>:82 - Loaded GPT-2 byte-level BPE tokenizer (50,257 tokens) once at startup
Dropped 50434 entries by heuristic filters
Wrote lossless metadata with 47081 entries to data/clustering/metadata.json

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
python -m corp_speech_risk_dataset.clustering.utils.make_vectors \
  --meta data/clustering/metadata.json \
  --out  data/clustering/concat_vectors.npy
2025-07-19 16:00:31.423 | INFO     | corp_speech_risk_dataset.encoding.tokenizer:<module>:82 - Loaded GPT-2 byte-level BPE tokenizer (50,257 tokens) once at startup
→ 25% done at 4.2s elapsed
→ 50% done at 7.9s elapsed
→ 75% done at 12.5s elapsed
Wrote (976, 2816) → data/clustering/concat_vectors.npy




THE OUTCOMES IS THE FINAL STEP TO APPEND THE AMOUNTS AND PERCENTILES OF THE FINAL JUDGEMENT AMOUNT IN FAVOR OF THE PLANTIFFS Positive values in favor of plantiff low or negative values in favor of defendant

(corp_speech_risk_dataset)  corp_speech_risk_dataset % uv run python -m corp_speech_risk_dataset.case_outcome.case_outcome_imputer \
  --root        data/tokenized/courtlistener \
  --stage1-root data/extracted/courtlistener \
  --outdir      data/outcomes/courtlistener \
  --mode        manual \
  --context-chars 400 \
  --min-amount    0



(corp_speech_risk_dataset) corp_speech_risk_dataset % python -m src.corp_speech_risk_dataset.cli_cluster \
  --vec data/clustering/concat_vectors.npy \
  --meta data/clustering/metadata.json \
  --supervision categorical \
  --out data/clustering/supervised_cat.html

  (corp_speech_risk_dataset) corp_speech_risk_dataset % python -m src.corp_speech_risk_dataset.cli_cluster \
  --vec data/clustering/concat_vectors.npy \
  --meta data/clustering/metadata.json \
  --supervision continuous \
  --out data/clustering/supervised_cat.html
