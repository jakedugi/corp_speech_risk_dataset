# orchestrators/courtlistener_orchestrator.py
from pathlib import Path
import os
import json
from loguru import logger
from urllib.parse import urljoin
from httpx import HTTPStatusError
import asyncio
import random

from corp_speech_risk_dataset.api.courtlistener.courtlistener_client import (
    CourtListenerClient,
    AsyncCourtListenerClient,
)
from corp_speech_risk_dataset.api.courtlistener.queries import (
    STATUTE_QUERIES,
    build_queries,
)
from corp_speech_risk_dataset.api.courtlistener.courtlistener_core import (
    process_full_docket,
    process_and_save,
    process_docket_entries,
    process_recap_fetch,
)
from corp_speech_risk_dataset.infrastructure.file_io import (
    download,
    needs_recap_fetch,
    download_missing_pdfs,
    load_json,
    ensure_dir,
)

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
        chunk_size: int = 10,
        async_mode: bool = False,
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
        self.chunk_size = chunk_size
        self.async_mode = async_mode
        self.client = CourtListenerClient(config, api_mode=api_mode)
        self.async_client = AsyncCourtListenerClient(
            self.token, max_concurrency=2, rate_limit=3.0
        )

    def run(self):
        """
        Legacy-compatible orchestrator: for each statute/company chunk, run /search/ (type=d),
        then hydrate each docket with all related data (docket shell, IA dump, RECAP back-fill, entries, filings, clusters, opinions).
        """
        if self.async_mode:
            asyncio.run(self.run_async())
            return
        logger.info(
            "Starting legacy-compatible CourtListener orchestration (search + full hydrate)"
        )
        for statute in self.statutes:
            queries = build_queries(
                statute, self.company_file, chunk_size=self.chunk_size
            )
            for query in queries:
                search_dir = self.outdir / "search"
                self._search_and_hydrate(query, search_dir)
        logger.success("Legacy-compatible CourtListener orchestration finished")

    async def run_async(self):
        """
        Async version: hydrates dockets in parallel using AsyncCourtListenerClient for doc fetching.
        """
        logger.info(
            "Starting ASYNC CourtListener orchestration (search + full hydrate)"
        )
        for statute in self.statutes:
            queries = build_queries(
                statute, self.company_file, chunk_size=self.chunk_size
            )
            for query in queries:
                search_dir = self.outdir / "search"
                await self._search_and_hydrate_async(query, search_dir)
        logger.success("ASYNC CourtListener orchestration finished")

    def _search_and_hydrate(self, query: str, search_dir: Path):
        """
        Run /search/ (type=d) for dockets, save results, and hydrate each docket found.
        Args:
            query: The search query string
            search_dir: Directory to save search results
        """
        params = {"q": query, "type": "d", "page_size": self.page_size}
        ensure_dir(search_dir)
        data = self.client._get(f"{self.client.BASE_URL}/search/", params)
        search_path = search_dir / "search_api_results.json"
        with open(search_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved search results to {search_path}")
        for dk in data.get("results", []):
            # Skip if already fetched
            dk_id = dk["docket_id"]
            slug = f"{dk.get('docketNumber')}_{dk.get('court_id')}"
            case_dir = self.outdir / slug
            if case_dir.exists():
                logger.info(f"Skipping existing case {slug}")
                continue
            self._hydrate_docket(dk)

    async def _search_and_hydrate_async(self, query: str, search_dir: Path):
        params = {"q": query, "type": "d", "page_size": self.page_size}
        ensure_dir(search_dir)
        # Use sync client for search (search endpoint is not bottleneck)
        data = self.client._get(f"{self.client.BASE_URL}/search/", params)
        search_path = search_dir / "search_api_results.json"
        with open(search_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved search results to {search_path}")
        # Fetch each docket, skipping ones already on disk
        tasks = []
        for dk in data.get("results", []):
            slug = f"{dk.get('docketNumber')}_{dk.get('court_id')}"
            case_dir = self.outdir / slug
            if case_dir.exists():
                logger.info(f"Skipping existing case {slug}")
                continue
            tasks.append(self._hydrate_docket_async(dk))
        if tasks:
            await asyncio.gather(*tasks)

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

        # Docket shell
        dockets_dir = case_dir / "dockets"
        process_and_save(self.client, "dockets", {"id": dk_id}, dockets_dir, limit=1)

        # IA dump
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

        # Free-attachment back-fill if gaps
        ia_dump_path = case_dir / "ia_dump.json"
        if needs_recap_fetch(ia_dump_path):
            logger.info(f"Triggering RECAP free-attachment fetch for {slug}")
            payload = {
                "request_type": 3,
                "docket": str(dk_id),  # Use internal docket ID for v4
                "pacer_username": os.getenv("PACER_USER"),
                "pacer_password": os.getenv("PACER_PASS"),
            }
            try:
                process_recap_fetch(self.config, payload)
            except HTTPStatusError as e:
                if e.response.status_code == 400:
                    logger.warning(
                        f"No free RECAP attachments for {slug}; skipping. ({e})"
                    )
                else:
                    raise

        # Entries (RECAP mode gives nested docs)
        entries_dir = case_dir / "entries"
        process_docket_entries(
            self.config,
            docket_id=dk_id,
            order_by="recap_sequence_number",
            pages=1,
            page_size=100,
            output_dir=entries_dir,
            api_mode="recap",
        )

        # --- Patch: fetch recap-document metadata and PDFs for each entry ---
        filings_dir = case_dir / "filings"
        ensure_dir(filings_dir)
        for entry_file in entries_dir.glob("*.json"):
            entry = load_json(entry_file)
            for doc_meta in entry.get("recap_documents", []):
                resource = doc_meta.get("resource_uri")
                if resource:
                    try:
                        doc = self.client._get(resource)
                    except HTTPStatusError as e:
                        if e.response.status_code == 503:
                            logger.warning(
                                f"Recap-document {resource} temporarily unavailable—skipping."
                            )
                            continue
                        else:
                            raise
                    doc_path = entries_dir / f"doc_{doc['id']}.json"
                    with doc_path.open("w") as f:
                        json.dump(doc, f, indent=2)
                    # Download PDF if available and allowed
                    if doc.get("filepath_local"):
                        if doc.get("is_available") is False:
                            logger.warning(
                                f"PDF {doc['id']} is marked unavailable—skipping."
                            )
                            continue
                        pdf_url = urljoin(
                            "https://www.courtlistener.com/", doc["filepath_local"]
                        )
                        pdf_dest = filings_dir / f"{doc['id']}.pdf"
                        if not pdf_dest.exists():
                            if not pdf_url.startswith("http"):
                                logger.warning(f"Skipping invalid URL: {pdf_url}")
                            else:
                                try:
                                    download(pdf_url, pdf_dest)
                                except HTTPStatusError as e:
                                    code = e.response.status_code
                                    if code in (403, 429):
                                        logger.warning(
                                            f"PDF {doc['id']} returned HTTP {code}—skipping."
                                        )
                                    else:
                                        raise
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to download PDF for doc {doc['id']}: {e}"
                                    )

        # Clusters → Opinions
        clusters_dir = case_dir / "clusters"
        process_and_save(
            self.client, "clusters", {"docket": dk_id}, clusters_dir, limit=100
        )
        opinions_dir = case_dir / "opinions"
        ensure_dir(opinions_dir)
        for cluster_path in clusters_dir.glob("*.json"):
            cl_id = load_json(cluster_path)["id"]
            process_and_save(
                self.client, "opinions", {"cluster": cl_id}, opinions_dir, limit=100
            )

    async def _hydrate_docket_async(self, dk: dict):
        """
        Async version: fetches recap_documents in parallel using AsyncCourtListenerClient.
        """
        dk_id = dk["docket_id"]
        dk_num = dk.get("docketNumber")
        court = dk.get("court_id")
        slug = f"{dk_num}_{court}"
        case_dir = self.outdir / slug
        ensure_dir(case_dir)
        # Docket shell
        dockets_dir = case_dir / "dockets"
        process_and_save(self.client, "dockets", {"id": dk_id}, dockets_dir, limit=1)
        # IA dump
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
        # Free-attachment back-fill if gaps
        ia_dump_path = case_dir / "ia_dump.json"
        if needs_recap_fetch(ia_dump_path):
            logger.info(f"Triggering RECAP free-attachment fetch for {slug}")
            payload = {
                "request_type": 3,
                "docket": str(dk_id),
                "pacer_username": os.getenv("PACER_USER"),
                "pacer_password": os.getenv("PACER_PASS"),
            }
            try:
                process_recap_fetch(self.config, payload)
            except HTTPStatusError as e:
                if e.response.status_code == 400:
                    logger.warning(
                        f"No free RECAP attachments for {slug}; skipping. ({e})"
                    )
                else:
                    raise
        # Entries (RECAP mode gives nested docs)
        entries_dir = case_dir / "entries"
        process_docket_entries(
            self.config,
            docket_id=dk_id,
            order_by="recap_sequence_number",
            pages=1,
            page_size=100,
            output_dir=entries_dir,
            api_mode="recap",
        )
        # --- Async: fetch recap-document metadata and PDFs in parallel ---
        filings_dir = case_dir / "filings"
        ensure_dir(filings_dir)
        entry_files = list(entries_dir.glob("*.json"))
        # Gather all doc resource URIs
        doc_uris = []
        entry_map = {}
        for entry_file in entry_files:
            entry = load_json(entry_file)
            for doc_meta in entry.get("recap_documents", []):
                resource = doc_meta.get("resource_uri")
                if resource:
                    doc_uris.append(resource)
                    entry_map[resource] = (entry_file, doc_meta)
        # Fetch all docs in batches of 5, with jittered sleep between batches
        results = []
        batch_size = 5
        for i in range(0, len(doc_uris), batch_size):
            batch = doc_uris[i : i + batch_size]
            res = await self.async_client.fetch_docs(batch)
            results.extend(res)
            # small sleep with jitter to avoid bursts
            await asyncio.sleep(self.async_client.rate_limit + random.random() * 0.5)
        for resource, doc in zip(doc_uris, results):
            if not isinstance(doc, dict) or "id" not in doc:
                logger.warning(
                    f"Skipping document without id for {resource!r}: {doc!r}"
                )
                continue
            entry_file, doc_meta = entry_map[resource]
            doc_path = entries_dir / f"doc_{doc['id']}.json"
            with doc_path.open("w") as f:
                json.dump(doc, f, indent=2)
            # Download PDF if available and allowed (sync for now)
            if doc.get("filepath_local"):
                if doc.get("is_available") is False:
                    logger.warning(f"PDF {doc['id']} is marked unavailable—skipping.")
                    continue
                pdf_url = urljoin(
                    "https://www.courtlistener.com/", doc["filepath_local"]
                )
                pdf_dest = filings_dir / f"{doc['id']}.pdf"
                if not pdf_dest.exists():
                    if not pdf_url.startswith("http"):
                        logger.warning(f"Skipping invalid URL: {pdf_url}")
                    else:
                        try:
                            download(pdf_url, pdf_dest)
                        except HTTPStatusError as e:
                            code = e.response.status_code
                            if code in (403, 429):
                                logger.warning(
                                    f"PDF {doc['id']} returned HTTP {code}—skipping."
                                )
                            else:
                                raise
                        except Exception as e:
                            logger.warning(
                                f"Failed to download PDF for doc {doc['id']}: {e}"
                            )
        # Clusters → Opinions (sync for now)
        clusters_dir = case_dir / "clusters"
        process_and_save(
            self.client, "clusters", {"docket": dk_id}, clusters_dir, limit=100
        )
        opinions_dir = case_dir / "opinions"
        ensure_dir(opinions_dir)
        for cluster_path in clusters_dir.glob("*.json"):
            cl_id = load_json(cluster_path)["id"]
            process_and_save(
                self.client, "opinions", {"cluster": cl_id}, opinions_dir, limit=100
            )

    # --- tiny helper for reading JSON dumps -------------------------------
    @staticmethod
    def _load_json(path: Path) -> dict | list | None:
        if path.exists():
            with open(path) as fh:
                return json.load(fh)
        return None
