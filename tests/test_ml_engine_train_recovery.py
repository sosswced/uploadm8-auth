"""ML engine script launcher + train-fail recovery helpers."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from services.ml_engine import (
    _alert_ml_engine_failure,
    _is_transient_build_failure,
    _root_exception_line,
    _run_build_with_retries,
    _script_cmd,
    _seed_on_train_fail_enabled,
    _subprocess_detail,
)


def _cfg(**kwargs):
    base = dict(seed_bootstrap=False, seed_rows=60)
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_script_cmd_auto_prefers_sys_executable_when_sklearn_present(monkeypatch):
    monkeypatch.setenv("UM8_ML_USE_UV", "auto")
    with patch("services.ml_engine._which_uv", return_value="/usr/bin/uv"), patch(
        "services.ml_engine._sklearn_stack_available", return_value=True
    ), patch("services.ml_engine._datasets_available", return_value=False):
        cmd = _script_cmd("scripts/train_promo_uplift_baseline.py", needs_datasets=False)
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("train_promo_uplift_baseline.py")


def test_script_cmd_auto_uses_uv_when_datasets_needed_and_missing(monkeypatch):
    monkeypatch.setenv("UM8_ML_USE_UV", "auto")
    with patch("services.ml_engine._which_uv", return_value="/usr/bin/uv"), patch(
        "services.ml_engine._sklearn_stack_available", return_value=True
    ), patch("services.ml_engine._datasets_available", return_value=False):
        cmd = _script_cmd("scripts/build_promo_training_dataset.py", needs_datasets=True)
    assert cmd[:2] == ["/usr/bin/uv", "run"]


def test_script_cmd_force_off(monkeypatch):
    monkeypatch.setenv("UM8_ML_USE_UV", "0")
    with patch("services.ml_engine._which_uv", return_value="/usr/bin/uv"):
        cmd = _script_cmd("scripts/train_promo_uplift_baseline.py", needs_datasets=True)
    assert cmd[0] == sys.executable


def test_subprocess_detail_prefers_stderr():
    assert (
        _subprocess_detail({"stderr_tail": "boom", "stdout_tail": "ok", "error": "e"})
        == "boom"
    )


def test_seed_on_train_fail_default_on(monkeypatch):
    monkeypatch.delenv("UM8_ML_ENGINE_SEED_ON_TRAIN_FAIL", raising=False)
    assert _seed_on_train_fail_enabled(_cfg()) is True


def test_seed_on_train_fail_can_disable(monkeypatch):
    monkeypatch.setenv("UM8_ML_ENGINE_SEED_ON_TRAIN_FAIL", "0")
    assert _seed_on_train_fail_enabled(_cfg()) is False


def test_alert_ml_engine_failure_labels_api_lane_not_worker():
    rec = AsyncMock(return_value="inc-1")
    with patch("services.ops_incidents.record_operational_incident", new=rec):
        asyncio.run(
            _alert_ml_engine_failure(
                object(),
                {
                    "ok": False,
                    "error": "local training failed",
                    "cycle_status": "failed",
                    "steps": {"train_local": {"stderr_tail": "ModuleNotFoundError: sklearn"}},
                    "seeded": True,
                },
            )
        )
    assert rec.await_count == 1
    kw = rec.await_args.kwargs
    assert kw["source"] == "ml_engine"
    assert "not upload worker" in kw["subject"]
    assert kw["details"]["affects_upload_worker"] is False
    assert kw["details"]["service_lane"] == "api_ml_engine"


def test_alert_ml_engine_skips_blocked_on_data():
    rec = AsyncMock(return_value="inc-1")
    with patch("services.ops_incidents.record_operational_incident", new=rec):
        asyncio.run(
            _alert_ml_engine_failure(
                object(),
                {"ok": False, "cycle_status": "blocked_on_data", "error": "insufficient"},
            )
        )
    rec.assert_not_awaited()


def test_ops_webhook_resolve_rejects_non_discord():
    from services.ops_incidents import _allowed_ops_discord_webhook

    assert _allowed_ops_discord_webhook("https://evil.example/hook") == ""
    assert _allowed_ops_discord_webhook(
        "https://discord.com/api/webhooks/123/abc"
    ).startswith("https://discord.com/")


def test_root_exception_line_skips_traceback_noise():
    text = (
        '  File "C:\\\\Python\\\\thread.py", line 86, in run\n'
        "    result = ctx.run(self.task)\n"
        "             ^^^^^^^^^^^^^^^^^^^\n"
        "socket.gaierror: [Errno 11001] getaddrinfo failed"
    )
    assert "getaddrinfo failed" in _root_exception_line(text)


def test_is_transient_build_failure_detects_dns():
    assert _is_transient_build_failure(
        {"ok": False, "stderr_tail": "socket.gaierror: [Errno 11001] getaddrinfo failed"}
    )
    assert not _is_transient_build_failure(
        {"ok": False, "stderr_tail": "ValueError: bad parquet schema"}
    )
    assert not _is_transient_build_failure({"ok": True, "stderr_tail": "getaddrinfo"})


def test_run_build_with_retries_recovers_from_dns():
    calls = {"n": 0}

    def _step(*_a, **_k):
        calls["n"] += 1
        if calls["n"] < 2:
            return {
                "ok": False,
                "stderr_tail": "socket.gaierror: [Errno 11001] getaddrinfo failed",
            }
        return {"ok": True, "returncode": 0}

    with patch("services.ml_engine.asyncio.sleep", new=AsyncMock()):
        out = asyncio.run(_run_build_with_retries(_step, attempts=3, label="promo"))
    assert out["ok"] is True
    assert out.get("transient_retries") == 1
    assert calls["n"] == 2


def test_run_build_with_retries_does_not_retry_logic_errors():
    calls = {"n": 0}

    def _step(*_a, **_k):
        calls["n"] += 1
        return {"ok": False, "stderr_tail": "ValueError: bad schema"}

    out = asyncio.run(_run_build_with_retries(_step, attempts=3, label="promo"))
    assert out["ok"] is False
    assert calls["n"] == 1


def test_alert_includes_build_root():
    rec = AsyncMock(return_value="inc-1")
    with patch("services.ops_incidents.record_operational_incident", new=rec):
        asyncio.run(
            _alert_ml_engine_failure(
                object(),
                {
                    "ok": False,
                    "error": "dataset build failed",
                    "cycle_status": "failed",
                    "steps": {
                        "build_dataset": {
                            "stderr_tail": (
                                'File "x.py", line 1\n'
                                "socket.gaierror: [Errno 11001] getaddrinfo failed"
                            )
                        }
                    },
                    "seeded": False,
                },
            )
        )
    body = rec.await_args.kwargs["body"]
    assert "build_root=" in body
    assert "getaddrinfo failed" in body
    assert rec.await_args.kwargs["details"]["transient_build"] is True
