"""Unit tests for /TUP Pikzels gate + live_result_workflow planner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.agent.live_result_workflow import plan_from_artifacts
from tests.e2e.helpers import pikzels_gate
from tests.e2e.helpers.config import ALL_UPLOAD_PLATFORMS, e2e_upload_platforms, e2e_use_persona


def test_pikzels_gate_once_per_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gate = tmp_path / "tup_pikzels_setup.json"
    monkeypatch.delenv("E2E_FORCE_PIKZELS", raising=False)
    monkeypatch.delenv("E2E_SKIP_PIKZELS", raising=False)

    assert pikzels_gate.should_use_pikzels(path=gate) is True
    assert pikzels_gate.consume_pikzels_slot(path=gate, note="first") is True
    assert pikzels_gate.should_use_pikzels(path=gate) is False
    assert pikzels_gate.consume_pikzels_slot(path=gate, note="second") is False

    pikzels_gate.reset_after_ship(path=gate)
    assert pikzels_gate.should_use_pikzels(path=gate) is True


def test_pikzels_force_skip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    gate = tmp_path / "g.json"
    monkeypatch.setenv("E2E_SKIP_PIKZELS", "1")
    assert pikzels_gate.should_use_pikzels(path=gate) is False
    monkeypatch.delenv("E2E_SKIP_PIKZELS")
    monkeypatch.setenv("E2E_FORCE_PIKZELS", "1")
    data = {"version": 1, "pikzels_used": True}
    gate.write_text(json.dumps(data), encoding="utf-8")
    assert pikzels_gate.should_use_pikzels(path=gate) is True


def test_e2e_upload_platforms_all_under_tup(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("E2E_TUP", "1")
    monkeypatch.delenv("E2E_UPLOAD_PLATFORMS", raising=False)
    assert e2e_upload_platforms() == ALL_UPLOAD_PLATFORMS
    monkeypatch.setenv("E2E_UPLOAD_PLATFORMS", "all")
    assert e2e_upload_platforms() == ALL_UPLOAD_PLATFORMS
    monkeypatch.setenv("E2E_UPLOAD_PLATFORMS", "tiktok,youtube")
    assert e2e_upload_platforms() == ("tiktok", "youtube")


def test_e2e_use_persona_default_under_tup(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("E2E_TUP", "1")
    monkeypatch.delenv("E2E_USE_PERSONA", raising=False)
    assert e2e_use_persona() is True
    monkeypatch.setenv("E2E_USE_PERSONA", "0")
    assert e2e_use_persona() is False


def test_live_result_workflow_green():
    plan = plan_from_artifacts(
        live_demo={"ok": True, "upload": {"status": "completed"}, "upload_ids": ["1"], "pages_visited": []},
        live_path=None,
        sentry={"cleared": True, "actionable_count": 0, "mcp_verified": True},
        checklist_failed=0,
        max_age_hours=12.0,
    )
    assert plan["ok"] is True
    assert plan["workflow"] == "tup-green"


def test_live_result_workflow_upload_failed():
    plan = plan_from_artifacts(
        live_demo={"ok": True, "upload": {"status": "failed"}, "pages_visited": []},
        live_path=None,
        sentry={"cleared": True, "actionable_count": 0, "mcp_verified": True},
        checklist_failed=0,
    )
    assert plan["ok"] is False
    assert any(s.startswith("upload_status:") for s in plan["signals"])


def test_live_result_workflow_api_5xx():
    plan = plan_from_artifacts(
        live_demo={
            "ok": True,
            "upload": {"status": "completed"},
            "pages_visited": [{"page": "dashboard.html", "api_5xx": ["/api/foo"]}],
        },
        live_path=None,
        sentry={"cleared": True, "actionable_count": 0, "mcp_verified": True},
        checklist_failed=0,
    )
    assert plan["ok"] is False
    assert "api_5xx:/api/foo" in plan["signals"]
    assert "router" in plan["eval_modes"]
