"""
[DEPRECATED] Legacy CourtListener workflow.

This file is deprecated. The legacy workflow is now available as LegacyCourtListenerWorkflow in workflows/legacy_multistep.py.
All new orchestration should use the orchestrator and CLI entry point.
"""

from dotenv import load_dotenv
load_dotenv()
import os
import json
from pathlib import Path
from corp_speech_risk_dataset.cli import search_api, fetch, recap_fetch, docket_entries, documents
from corp_speech_risk_dataset.orchestrators.courtlistener_orchestrator import CourtListenerOrchestrator
from corp_speech_risk_dataset.config import load_config

# Organized queries (preserved from original)
QUERIES = [
    # Simplified FTC Section 5 query
    '(Section 5 OR "15 U.S.C. § 45") AND (tweet OR Facebook OR Instagram OR website OR blog) AND (deceptive OR misleading)'
]

# Utility: load JSON if exists, else None
def load_json(path):
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None

def orchestrate(queries, court: str = None, outdir: str = "CourtListener"):  # main orchestrator
    """Legacy orchestrate function - now uses the new orchestrator class."""
    token = os.environ.get("COURTLISTENER_API_TOKEN")
    config = load_config()

    # Convert legacy queries to statute format for the new orchestrator
    # For now, we'll use a custom approach that preserves the original logic
    for query in queries:
        # 1. Search for dockets using the original search_api approach
        search_dir = Path(outdir) / "search"
        search_dir.mkdir(parents=True, exist_ok=True)
        search_api(
            param=[f"q={query}", "type=d"],
            output_dir=search_dir,
            limit=500,
            show_url=False,
            token=token
        )
        search_results = load_json(search_dir / "search_api_results.json")
        if not search_results or "results" not in search_results:
            print(f"No dockets found for query: {query}")
            continue

        for dk in search_results["results"]:
            dk_id = dk["docket_id"]
            dk_num = dk.get("docketNumber")
            dk_court = dk.get("court_id") or court
            case_dir = Path(outdir) / f"{dk_num}_{dk_court}"
            case_dir.mkdir(parents=True, exist_ok=True)

            # 2. Fetch docket shell
            fetch(
                resource_type="dockets",
                output_dir=str(case_dir),
                param=[f"id={dk_id}"],
                limit=1,
                show_fields=False
            )
            docket_json = load_json(case_dir / "dockets" / "dockets_0.json")
            ia_url = docket_json.get("filepath_ia_json") if docket_json else None
            ia_path = case_dir / "ia_dump.json"
            if ia_url and not ia_path.exists():
                os.system(f"curl -L '{ia_url}' -o '{ia_path}'")
            ia_json = load_json(ia_path)

            # 3. Free attachment fetch if gaps
            if ia_json and any(e.get("recap_documents") == [] or not e.get("description") for e in ia_json.get("entries", [])):
                recap_fetch(
                    post_param=[
                        "request_type=3",
                        f"docket_number={dk_num}",
                        f"court={dk_court}",
                        f"pacer_username={os.environ.get('PACER_USER','')}",
                        f"pacer_password={os.environ.get('PACER_PASS','')}"
                    ],
                    show_url=False,
                    token=token
                )

            # 4. Docket entries
            entries_dir = case_dir / "entries"
            docket_entries(
                docket_id=dk_id,
                query=None,
                order_by="recap_sequence_number",
                pages=1,
                page_size=100,
                output_dir=entries_dir,
                api_mode="recap"
            )

            # 5. Documents (text extraction)
            for entry_file in entries_dir.glob("*.json"):
                entry = load_json(entry_file)
                for doc in entry.get("recap_documents", []):
                    if doc.get("filepath_local") and not doc.get("plain_text"):
                        url = f"https://www.courtlistener.com/{doc['filepath_local']}"
                        pdf_path = case_dir / "filings" / f"{doc['id']}.pdf"
                        pdf_path.parent.mkdir(exist_ok=True)
                        if not pdf_path.exists():
                            os.system(f"curl -L '{url}' -o '{pdf_path}'")
                        # Extraction step (pseudo, user to implement)
                        # text = pdfminer_extract(pdf_path)
                        # save_json(..., {"text": text})

            # 6. Opinions chain
            fetch(
                resource_type="clusters",
                output_dir=str(case_dir / "clusters"),
                param=[f"docket={dk_id}"],
                limit=100,
                show_fields=False
            )
            for cl_file in (case_dir / "clusters").glob("*.json"):
                cl = load_json(cl_file)
                cl_id = cl.get("id")
                if cl_id:
                    fetch(
                        resource_type="opinions",
                        output_dir=str(case_dir / "opinions"),
                        param=[f"cluster={cl_id}"],
                        limit=100,
                        show_fields=False
                    )

            # 7. Persist: all files are already in place
            print(f"Done: {case_dir}")

def orchestrate_with_new_class(statutes=None, company_file=None, outdir="CourtListener", **kwargs):
    """New orchestrate function using the CourtListenerOrchestrator class."""
    token = os.environ.get("COURTLISTENER_API_TOKEN")
    config = load_config()

    # Use default statutes if none provided
    if statutes is None:
        from corp_speech_risk_dataset.api.courtlistener import STATUTE_QUERIES
        statutes = list(STATUTE_QUERIES.keys())

    try:
        orchestrator = CourtListenerOrchestrator(
            config=config,
            statutes=statutes,
            company_file=Path(company_file) if company_file else None,
            outdir=Path(outdir),
            token=token,
            **kwargs
        )
        orchestrator.run()
    except Exception as e:
        print(f"Error during orchestration: {e}")
        raise

if __name__ == "__main__":
    # Run the legacy orchestrate function with the original QUERIES
    orchestrate(QUERIES)
