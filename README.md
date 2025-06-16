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