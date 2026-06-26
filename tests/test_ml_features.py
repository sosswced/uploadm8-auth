"""Unit tests for ML feature registry and content-success feature engineering."""

from __future__ import annotations

import pandas as pd

from services.content_success_features import (
    FEATURES_CAT,
    FEATURES_NUM,
    TARGET,
    _base_features,
    build_records,
    label_hotness,
)
from services.ml_feature_registry import (
    active_cat,
    active_num,
    catalog,
    label,
    label_fallback,
)
from services.promo_targeting_features import FEATURES_CAT as PROMO_CAT
from services.promo_targeting_features import FEATURES_NUM as PROMO_NUM


def test_registry_derives_promo_columns():
    assert "channel" not in PROMO_CAT
    assert "delivery_status" not in PROMO_CAT
    assert "is_snapshot" in PROMO_NUM
    assert "prior_touchpoints" in PROMO_NUM
    assert label("promo") == "converted_7d"
    assert label_fallback("promo") == "engaged_7d"


def test_registry_derives_content_columns():
    assert "has_attribution" in FEATURES_NUM
    assert "primary_hashtag" not in FEATURES_CAT
    assert "views_per_day" in FEATURES_NUM
    assert label("content") == TARGET == "is_hot"


def test_catalog_includes_both_loops():
    rows = catalog()
    loops = {r["loop"] for r in rows}
    assert loops == {"promo", "content"}
    assert any(r["name"] == "views_per_day" and r["loop"] == "content" for r in rows)


def test_admin_ml_observability_routes_do_not_shadow_catalog():
    """Regression: route handler must not be named ``catalog`` — shadows registry import."""
    from core.deps import require_admin
    from app import app
    from fastapi.testclient import TestClient

    async def _fake_admin():
        return {"id": "test-admin", "role": "admin", "email": "admin@test.com"}

    app.dependency_overrides[require_admin] = _fake_admin
    try:
        with TestClient(app) as client:
            fc = client.get("/api/admin/ml/feature-catalog")
            assert fc.status_code == 200
            body = fc.json()
            assert isinstance(body.get("features"), list)
            assert len(body["features"]) >= 10

            ov = client.get("/api/admin/ml/observability-overview?mode=full")
            assert ov.status_code == 200
            assert ov.json().get("mode") == "full"
            assert ov.json().get("feature_catalog_count", 0) >= 10
    finally:
        app.dependency_overrides.pop(require_admin, None)


def test_base_features_null_not_fake_defaults():
    row = {"output_artifacts": {}, "hashtags": [], "created_at": None}
    feats = _base_features(row)
    assert feats["has_attribution"] == 0
    assert feats["content_category"] is None
    assert feats["caption_style"] is None


def test_label_hotness_cross_user_fallback_for_single_upload_users():
    # Six single-upload users on the same platform → cross-user percentile ranks.
    records = []
    for i in range(6):
        records.append(
            {
                "upload_id": f"u{i}",
                "user_id": f"user{i}",
                "platform": "tiktok",
                "views": 10 * (i + 1),
                "likes": i,
                "comments": 0,
                "shares": 0,
                "interactions": i,
                "engagement_rate_pct": float(i),
                "has_attribution": 1,
            }
        )
    df = label_hotness(pd.DataFrame.from_records(records))
    assert int(df["is_hot"].sum()) >= 1
    assert int(df["is_hot"].nunique()) == 2


def test_build_records_expands_platform_results():
    row = {
        "upload_id": "abc",
        "user_id": "u1",
        "created_at": pd.Timestamp("2026-01-15 12:00:00", tz="UTC"),
        "platforms": ["tiktok"],
        "hashtags": ["cars"],
        "platform_results": [
            {"platform": "tiktok", "status": "published", "views": 50, "likes": 5, "comments": 1, "shares": 0}
        ],
        "output_artifacts": {
            "content_attribution_v1": {
                "content_category": "cars",
                "caption_style": "story",
                "caption_tone": "authentic",
                "caption_voice": "default",
                "hashtag_style": "mixed",
                "thumbnail_selection_mode": "ai",
                "thumbnail_render_pipeline": "auto",
                "m8_engine": True,
                "ai_hashtags_enabled": True,
                "ai_hashtag_count": 3,
                "caption_frame_count": 6,
            }
        },
    }
    out = build_records([row])
    assert len(out) == 1
    assert out[0]["platform"] == "tiktok"
    assert out[0]["has_attribution"] == 1
    assert out[0]["content_category"] == "cars"
