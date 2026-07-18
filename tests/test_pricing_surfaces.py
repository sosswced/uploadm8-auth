"""Pricing surfaces sync — generated JS stays in lockstep with SERVICE_WEIGHTS."""

from __future__ import annotations

from pathlib import Path

from services.pricing_surfaces import build_pricing_surfaces_snapshot, render_generated_js


def test_pricing_surfaces_snapshot_has_put_and_aic_estimates():
    snap = build_pricing_surfaces_snapshot()
    assert snap["put_cost_rules"]["base"] == 10
    assert snap["estimates"]["put_typical"] == 10
    assert snap["estimates"]["aic_light_60s"] >= 10
    assert snap["estimates"]["aic_full_60s"] >= snap["estimates"]["aic_light_60s"]
    assert snap["estimates"]["pikzels_recreate_aic"] >= 50
    assert "faq_put_aic" in snap["copy"]
    assert "caption_llm" in snap["service_weights"]


def test_generated_js_matches_repo_artifact():
    """CI: run ``python scripts/sync_pricing_surfaces.py`` after weight changes."""
    root = Path(__file__).resolve().parents[1]
    path = root / "frontend" / "js" / "pricing-surfaces.generated.js"
    assert path.is_file(), "missing pricing-surfaces.generated.js — run sync_pricing_surfaces.py"
    expected = render_generated_js(build_pricing_surfaces_snapshot())
    actual = path.read_text(encoding="utf-8")
    assert actual == expected, (
        "pricing-surfaces.generated.js is stale. "
        "Run: python scripts/sync_pricing_surfaces.py"
    )


def test_sync_script_check_mode_ok():
    from scripts.sync_pricing_surfaces import write_all

    report = write_all(check=True)
    assert report["ok"] is True, report
