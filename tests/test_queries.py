import csv
from pathlib import Path
import pytest
from corp_speech_risk_dataset.api.courtlistener.queries import build_queries, STATUTE_QUERIES

def test_build_queries_no_companies():
    for statute in STATUTE_QUERIES:
        qlist = build_queries(statute, company_file=None)
        assert isinstance(qlist, list)
        assert len(qlist) == 1
        assert STATUTE_QUERIES[statute].strip() in qlist[0]

def test_build_queries_with_companies(tmp_path):
    csv_path = tmp_path / "companies.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["official_name"])
        writer.writeheader()
        writer.writerow({"official_name": "Alpha"})
        writer.writerow({"official_name": "Beta"})
        writer.writerow({"official_name": "Gamma"})

    from corp_speech_risk_dataset.api.courtlistener import queries
    monkey_chunk = queries.CHUNK_SIZE
    queries.CHUNK_SIZE = 2

    qlist = build_queries("FTC Section 5", company_file=csv_path)
    assert len(qlist) == 2
    for q in qlist:
        assert '"Alpha"' in q or '"Beta"' in q or '"Gamma"' in q

    queries.CHUNK_SIZE = monkey_chunk 