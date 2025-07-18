# src/corp_speech_risk_dataset/cli_rss.py

import typer
from pathlib import Path

from corp_speech_risk_dataset.rss_config import load_config
from corp_speech_risk_dataset.orchestrators.rss_orchestrator import RSSOrchestrator

app = typer.Typer()


@app.command()
def rss_orchestrate(
    tickers: list[str] = typer.Option(
        None,
        "--tickers",
        "-t",
        help="List of tickers to fetch (default: all in config)",
    ),
    outdir: Path = typer.Option(
        Path("data/raw/rss"), "--outdir", "-o", help="Base output directory"
    ),
    limit: int = typer.Option(
        None, "--limit", "-l", help="Max entries per ticker (client-side limit)"
    ),
    no_dedupe: bool = typer.Option(
        False, "--no-dedupe", help="Disable deduplication of entries"
    ),
):
    """
    Run full RSS fetch + save workflow for specified S&P 500 tickers.
    """
    config = load_config()
    orch = RSSOrchestrator(
        config=config,
        tickers=tickers,
        outdir=str(outdir),
        limit_per_ticker=limit,
        dedupe=not no_dedupe,
    )
    orch.run()


if __name__ == "__main__":
    import sys

    # optional: configure loguru to show INFO on stderr
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True)
    app()
