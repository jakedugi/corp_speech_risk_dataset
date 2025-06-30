import pytest
from pathlib import Path
from corp_speech_risk_dataset.api.courtlistener.queries import build_queries
from corp_speech_risk_dataset.orchestrators.courtlistener_orchestrator import CourtListenerOrchestrator

class DummyOrchestrator(CourtListenerOrchestrator):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("statutes", ["FTC Section 5"])
        super().__init__(*args, **kwargs)
        self.queries_processed = []

    def _process_query(self, statute, query):
        self.queries_processed.append((statute, query))

def test_orchestrator_builds_all_chunks(tmp_path):
    companies = tmp_path / "csv"
    companies.write_text("official_name\nA\nB\nC\nD\nE\n")
    orch = DummyOrchestrator(
        config=object(),
        company_file=companies,
        outdir=Path("unused"),
        pages=1,
        page_size=1,
        date_min="2020-01-01",
        api_mode="standard",
    )
    orch.run()
    expected = build_queries("FTC Section 5", company_file=companies)
    assert len(orch.queries_processed) == len(expected)
    for (_, q), q_expected in zip(orch.queries_processed, expected):
        assert q.strip() == q_expected.strip() 