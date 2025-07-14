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

---

This reorganization improves maintainability, reduces confusion, and follows modern Python project patterns.
