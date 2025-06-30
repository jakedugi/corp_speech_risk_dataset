# orchestrators/courtlistener_orchestrator.py
from pathlib import Path
import os
import json
from loguru import logger
from urllib.parse import urljoin

from corp_speech_risk_dataset.api.courtlistener.client import CourtListenerClient
from corp_speech_risk_dataset.api.courtlistener.queries import STATUTE_QUERIES, build_queries
from corp_speech_risk_dataset.api.courtlistener.core import process_full_docket, process_and_save, process_docket_entries, process_recap_fetch
from corp_speech_risk_dataset.utils.file_io import download, needs_recap_fetch, download_missing_pdfs, load_json, ensure_dir

"""
Main orchestration workflow for CourtListener multi-step process.

This class coordinates the multi-step download and processing of dockets, opinions, and related files.

Refactored to call the API client and process_* functions directly, avoiding any import from the CLI or cli_helpers.
"""
class CourtListenerOrchestrator:
    def __init__(
        self,
        config,
        statutes=None,
        company_file=None,
        outdir: Path | str = "CourtListener",
        token: str | None = None,
        pages: int = 1,
        page_size: int = 50,
        date_min: str | None = None,
        api_mode: str = "standard",
    ):
        self.config = config
        self.statutes = statutes or ["FTC Section 5"]
        self.company_file = company_file
        self.outdir = Path(outdir)
        self.token = token or os.getenv("COURTLISTENER_API_TOKEN")
        self.pages = pages
        self.page_size = page_size
        self.date_min = date_min
        self.api_mode = api_mode
        self.client = CourtListenerClient(config, api_mode=api_mode)

    def run(self):
        """
        Legacy-compatible orchestrator: for each statute/company chunk, run /search/ (type=d),
        then hydrate each docket with all related data (docket shell, IA dump, RECAP back-fill, entries, filings, clusters, opinions).
        """
        logger.info("Starting legacy-compatible CourtListener orchestration (search + full hydrate)")
        for statute in self.statutes:
            queries = build_queries(statute, self.company_file)
            for query in queries:
                search_dir = self.outdir / "search"
                self._search_and_hydrate(query, search_dir)
        logger.success("Legacy-compatible CourtListener orchestration finished")

    def _search_and_hydrate(self, query: str, search_dir: Path):
        """
        Run /search/ (type=d) for dockets, save results, and hydrate each docket found.
        Args:
            query: The search query string
            search_dir: Directory to save search results
        """
        params = {"q": query, "type": "d"}
        ensure_dir(search_dir)
        data = self.client._get(f"{self.client.BASE_URL}/search/", params)
        search_path = search_dir / "search_api_results.json"
        with open(search_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved search results to {search_path}")
        for dk in data.get("results", []):
            self._hydrate_docket(dk)

    def _hydrate_docket(self, dk: dict):
        """
        Run all legacy steps for a single docket: shell, IA dump, RECAP fetch, entries, filings, clusters, opinions.
        Args:
            dk: Docket JSON from /search/ results
        """
        dk_id = dk["docket_id"]
        dk_num = dk.get("docketNumber")
        court = dk.get("court_id")
        slug = f"{dk_num}_{court}"
        case_dir = self.outdir / slug
        ensure_dir(case_dir)

        # 2️⃣ Docket shell
        dockets_dir = case_dir / "dockets"
        process_and_save(self.client, "dockets", {"id": dk_id}, dockets_dir, limit=1)

        # 3️⃣ IA dump
        ia_json_path = dockets_dir / "dockets_0.json"
        ia_url = None
        if ia_json_path.exists():
            meta = load_json(ia_json_path)
            ia_url = meta.get("filepath_ia_json")
        if ia_url:
            ia_dump_path = case_dir / "ia_dump.json"
            if not ia_dump_path.exists():
                try:
                    download(ia_url, ia_dump_path)
                except Exception as e:
                    logger.warning(f"Failed to download IA dump for {slug}: {e}")

        # 4️⃣ Free-attachment back-fill if gaps
        ia_dump_path = case_dir / "ia_dump.json"
        if needs_recap_fetch(ia_dump_path):
            logger.info(f"Triggering RECAP free-attachment fetch for {slug}")
            payload = {
                "request_type": 3,
                "docket": str(dk_id),  # Use internal docket ID for v4
                "pacer_username": os.getenv("PACER_USER"),
                "pacer_password": os.getenv("PACER_PASS")
            }
            from httpx import HTTPStatusError
            try:
                process_recap_fetch(self.config, payload)
            except HTTPStatusError as e:
                if e.response.status_code == 400:
                    logger.warning(f"No free RECAP attachments for {slug}; skipping. ({e})")
                else:
                    raise

        # 5️⃣ Entries (RECAP mode gives nested docs)
        entries_dir = case_dir / "entries"
        process_docket_entries(
            self.config,
            docket_id=dk_id,
            order_by="recap_sequence_number",
            pages=1,
            page_size=100,
            output_dir=entries_dir,
            api_mode="recap"
        )

        # --- Patch: fetch recap-document metadata and PDFs for each entry ---
        filings_dir = case_dir / "filings"
        ensure_dir(filings_dir)
        for entry_file in entries_dir.glob("*.json"):
            entry = load_json(entry_file)
            for doc_meta in entry.get("recap_documents", []):
                resource = doc_meta.get("resource_uri")
                if resource:
                    doc = self.client._get(resource)
                    doc_path = entries_dir / f"doc_{doc['id']}.json"
                    with doc_path.open("w") as f:
                        json.dump(doc, f, indent=2)
                    # Download PDF if available
                    if doc.get("filepath_local"):
                        pdf_url = urljoin("https://www.courtlistener.com/", doc["filepath_local"])
                        pdf_dest = filings_dir / f"{doc['id']}.pdf"
                        if not pdf_dest.exists():
                            if not pdf_url.startswith("http"):
                                logger.warning(f"Skipping invalid URL: {pdf_url}")
                            else:
                                try:
                                    download(pdf_url, pdf_dest)
                                except Exception as e:
                                    logger.warning(f"Failed to download PDF for doc {doc['id']}: {e}")

        # 6️⃣ Clusters → Opinions
        clusters_dir = case_dir / "clusters"
        process_and_save(self.client, "clusters", {"docket": dk_id}, clusters_dir, limit=100)
        opinions_dir = case_dir / "opinions"
        ensure_dir(opinions_dir)
        for cluster_path in clusters_dir.glob("*.json"):
            cl_id = load_json(cluster_path)["id"]
            process_and_save(self.client, "opinions", {"cluster": cl_id}, opinions_dir, limit=100)

    # --- tiny helper for reading JSON dumps -------------------------------
    @staticmethod
    def _load_json(path: Path) -> dict | list | None:
        if path.exists():
            with open(path) as fh:
                return json.load(fh)
        return None