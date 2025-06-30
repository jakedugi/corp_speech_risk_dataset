# Corporate Speech Risk Dataset

This repository contains code and data for analyzing corporate speech risk across multiple regulatory and legal sources, including FTC, SEC, and CourtListener data.

## Project Structure

```
corp_speech_risk_dataset/
├── data/               # Data storage
│   ├── raw/           # Raw data from sources
│   ├── processed/     # Processed and transformed data
│   └── metadata.jsonl # Dataset metadata
├── notebooks/         # Jupyter notebooks for exploration and analysis
├── src/              # Source code
│   ├── api/          # API clients for data sources
│   ├── extractors/   # Data extraction and processing
│   ├── orchestrators/# Pipeline orchestration
│   ├── utils/        # Utility functions
│   └── types/        # Type definitions
├── tests/            # Test suite
└── logs/             # Log files
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure API credentials in the appropriate client files
2. Run the data pipeline:
```bash
python src/main.py
```

## Development

- Follow PEP 8 style guide
- Write tests for new features
- Update documentation as needed

## License

[License details here]

# CourtListener API Client

A Python client for batch searching CourtListener's RECAP database for corporate speech cases.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/courtlistener-client.git
cd courtlistener-client
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure your API token:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your CourtListener API token:
     ```
     COURTLISTENER_API_TOKEN=your_token_here
     ```

## Usage

Run a search for specific statutes:
```bash
python -m src.cli search --statutes "FTC Section 5" "SEC Rule 10b-5" --pages 4 --page-size 50
```

Or use all default statutes:
```bash
python -m src.cli search
```

## Configuration

The client can be configured through:
1. Environment variables (prefixed with `COURTLISTENER_`)
2. `.env` file
3. Command-line arguments

See `.env.example` for all available options.

## Output

Results are saved to `data/raw/courtlistener/YYYY-MM-DD/<statute-slug>/`:
- `dockets.json`: Case metadata
- `opinions.json`: Full opinion texts (if `--opinions` flag used)

## Development

- Add new statutes by editing `STATUTE_QUERIES` in `src/courtlistener.py`
- Run tests: `pytest tests/`
- Format code: `black src/ tests/`
- Type check: `mypy src/ tests/`

## Company Name Chunking

When you use the `--company-file` option, the CLI will automatically split your company list into safe-size chunks (about 200 names per chunk) to avoid exceeding the CourtListener server's URL length limit. Each chunk is queried separately, and results are saved in subdirectories (e.g., `chunk_1`, `chunk_2`, ...). This ensures you only download the opinions you need, without hitting server errors or wasting disk space.