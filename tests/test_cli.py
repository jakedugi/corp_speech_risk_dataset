import os
import pytest
from typer.testing import CliRunner
from pathlib import Path
from corp_speech_risk_dataset.cli import app

runner = CliRunner()

def test_legacy_command(tmp_path, monkeypatch):
    called = {}
    def fake_run(self):
        called["args"] = (self.queries, self.court, self.outdir)
    monkeypatch.setattr(
        "corp_speech_risk_dataset.workflows.legacy_multistep.LegacyCourtListenerWorkflow.run",
        fake_run,
    )
    result = runner.invoke(
        app,
        [
            "legacy",
            "FOO QUERY",
            "--court", "CA",
            "--outdir", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0
    assert called["args"] == (["FOO QUERY"], "CA", Path(str(tmp_path / "out")))

def test_orchestrate_command_minimal(monkeypatch):
    monkeypatch.setattr(
        "corp_speech_risk_dataset.orchestrators.courtlistener_orchestrator.CourtListenerOrchestrator.run",
        lambda self: setattr(self, "ran", True),
    )
    result = runner.invoke(app, ["orchestrate"])
    assert result.exit_code == 0 