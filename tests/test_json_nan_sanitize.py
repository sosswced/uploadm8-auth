"""Starlette JSONResponse rejects NaN/Inf — ML rankings must sanitize before encode."""

from __future__ import annotations

import json
import math

from services.content_success_features import _coerce, _finite_round
from services.growth_intelligence import sanitize_coach_payload_for_json
from services.ml_engine import load_engine_state, save_engine_state


def test_sanitize_coach_payload_strips_nan_inf():
    payload = {
        "ok": True,
        "mean": float("nan"),
        "hot": float("inf"),
        "nested": [{"v": float("nan")}, 1.5],
    }
    clean = sanitize_coach_payload_for_json(payload)
    json.dumps(clean, allow_nan=False)
    assert clean["mean"] == 0.0
    assert clean["hot"] == 0.0
    assert clean["nested"][0]["v"] == 0.0
    assert clean["nested"][1] == 1.5


def test_finite_round_and_coerce():
    assert _finite_round(float("nan"), 4) == 0.0
    assert _finite_round(float("inf"), 2) == 0.0
    assert _finite_round(1.23456, 2) == 1.23
    assert _coerce(float("nan")) is None
    assert _coerce(float("-inf")) is None
    assert math.isfinite(_finite_round("nope", 1))


def test_engine_state_roundtrip_strips_nan(monkeypatch, tmp_path):
    from services import ml_engine as eng

    state_path = tmp_path / "engine_state.json"
    monkeypatch.setattr(eng, "_STATE_PATH", state_path)
    save_engine_state({"last_run": {"score": float("nan"), "ok": True}})
    loaded = load_engine_state()
    json.dumps(loaded, allow_nan=False)
    assert loaded["last_run"]["score"] == 0.0
    assert loaded["last_run"]["ok"] is True


def test_hub_rankings_payload_encodes_after_sanitize():
    from services.ai_insights_hub import ai_insights_hub_fallback

    hub = ai_insights_hub_fallback(error="unit")
    hub["content_rankings"] = sanitize_coach_payload_for_json(
        {"top_topics": [{"mean_engagement_pct": float("nan")}]}
    )
    json.dumps(hub, allow_nan=False)
