import pytest
from pathlib import Path
from corp_speech_risk_dataset.workflows.legacy_multistep import (
    LegacyCourtListenerWorkflow,
)


class DummyWorkflow(LegacyCourtListenerWorkflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.called = []

    def _step1_search_dockets(self, query):
        self.called.append(("search", query))
        return [{"docket_id": 1, "docketNumber": "1-XYZ", "court_id": "CA"}]

    def _step2_fetch_docket_shell(self, dk_id, case_dir):
        self.called.append(("shell", dk_id, case_dir))

    def _step3_free_attachment_fetch(self, dk_num, dk_court):
        self.called.append(("free_attach", dk_num, dk_court))

    def _step4_docket_entries(self, dk_id, entries_dir):
        self.called.append(("entries", dk_id, entries_dir))

    def _step5_download_missing_pdfs(self, entries_dir, case_dir):
        self.called.append(("pdfs", entries_dir, case_dir))

    def _step6_opinion_chain(self, dk_id, case_dir):
        self.called.append(("opinions", dk_id, case_dir))


def test_legacy_workflow_steps(tmp_path):
    wf = DummyWorkflow(queries=["Q1"], court="CA", outdir=tmp_path / "out")
    wf.run()
    names = [c[0] for c in wf.called]
    assert names == ["search", "shell", "entries", "pdfs", "opinions"]
