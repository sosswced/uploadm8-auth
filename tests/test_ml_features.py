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
    rank_dimension,
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


def test_rank_dimension_skips_null_and_none_labels():
    """User-facing rankings must not lead with blank / sentinel buckets."""
    df = pd.DataFrame(
        [
            {"content_category": None, "engagement_rate_pct": 5.0, "views": 100, "interactions": 5, "is_hot": 1},
            {"content_category": None, "engagement_rate_pct": 4.0, "views": 90, "interactions": 4, "is_hot": 1},
            {"content_category": None, "engagement_rate_pct": 3.0, "views": 80, "interactions": 3, "is_hot": 0},
            {"content_category": "automotive", "engagement_rate_pct": 2.0, "views": 50, "interactions": 2, "is_hot": 0},
            {"content_category": "automotive", "engagement_rate_pct": 2.5, "views": 55, "interactions": 2, "is_hot": 1},
            {"content_category": "automotive", "engagement_rate_pct": 1.5, "views": 40, "interactions": 1, "is_hot": 0},
            {"primary_hashtag": "none", "engagement_rate_pct": 9.0, "views": 200, "interactions": 9, "is_hot": 1},
            {"primary_hashtag": "none", "engagement_rate_pct": 8.0, "views": 180, "interactions": 8, "is_hot": 1},
            {"primary_hashtag": "none", "engagement_rate_pct": 7.0, "views": 160, "interactions": 7, "is_hot": 1},
            {"primary_hashtag": "gloryboy", "engagement_rate_pct": 1.0, "views": 30, "interactions": 1, "is_hot": 0},
            {"primary_hashtag": "gloryboy", "engagement_rate_pct": 1.2, "views": 35, "interactions": 1, "is_hot": 0},
            {"primary_hashtag": "gloryboy", "engagement_rate_pct": 0.8, "views": 25, "interactions": 1, "is_hot": 0},
        ]
    )
    topics = rank_dimension(df, ["content_category"], min_samples=3, limit=10)
    assert topics and all(t.get("content_category") for t in topics)
    assert topics[0]["content_category"] == "automotive"

    tags = rank_dimension(df, ["primary_hashtag"], min_samples=3, limit=10)
    assert tags and all(t.get("primary_hashtag") not in (None, "none") for t in tags)
    assert tags[0]["primary_hashtag"] == "gloryboy"


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
                "thumbnail_studio_enabled": True,
                "thumbnail_studio_engine_enabled": True,
                "thumbnail_persona_enabled": True,
                "thumbnail_persona_strength": 70,
                "studio_variant_ctr_score": 0.62,
                "studio_pikzels_main_score": 0.58,
                "thumbnail_audience_niche": "cars",
                "thumbnail_engine_mode": "uploadm8_pikzels_v2_pipeline",
                "thumbnail_layout_pattern": "face_left_text_right",
            }
        },
    }
    out = build_records([row])
    assert len(out) == 1
    assert out[0]["platform"] == "tiktok"
    assert out[0]["has_attribution"] == 1
    assert out[0]["content_category"] == "cars"
    assert out[0]["thumbnail_persona_enabled"] == 1
    assert out[0]["studio_variant_ctr_score"] == 0.62
    assert out[0]["thumbnail_audience_niche"] == "cars"
    assert "thumbnail_studio_enabled" in FEATURES_NUM
    assert "thumbnail_audience_niche" in FEATURES_CAT


def test_registry_marks_leakage_columns_deprecated():
    from services.ml_feature_registry import PROMO_FEATURES

    by_name = {f.name: f for f in PROMO_FEATURES}
    assert by_name["channel"].status == "deprecated"
    assert by_name["delivery_status"].status == "deprecated"
    assert "channel" not in active_cat("promo")
    assert "delivery_status" not in active_cat("promo")
    assert "channel" not in active_num("promo")


def test_public_ml_hub_is_link_only_no_tokens():
    from services.ml_hub_config import build_ml_hub_public_response

    payload = build_ml_hub_public_response()
    blob = str(payload).lower()
    for needle in ("hf_token", "hugging_face_hub_token", "authorization", "bearer "):
        assert needle not in blob
    assert "huggingface" in payload
    assert "ecosystem" in payload

    def _walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                kl = str(k).lower()
                assert "token" not in kl or kl.endswith("_doc")
                _walk(v)
        elif isinstance(obj, list):
            for x in obj:
                _walk(x)

    _walk(payload)


def test_admin_ml_routes_require_admin():
    from app import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        for path in (
            "/api/admin/ml/feature-catalog",
            "/api/admin/ml/engine-status",
            "/api/admin/ml/observability-overview?mode=strip",
        ):
            r = client.get(path)
            assert r.status_code in (401, 403), path


def test_studio_estimate_debits_via_atomic_and_refs_service_weights():
    from stages.ai_service_costs import SERVICE_WEIGHTS
    from services.thumbnail_studio import estimate_studio_cost

    put, aic, breakdown = estimate_studio_cost(
        variant_count=2,
        has_persona=True,
        competitor_gap_mode=True,
        has_channel_memory=True,
    )
    assert put > 0 and aic > 0
    assert breakdown.get("debit_via") == "atomic_debit_tokens"
    assert breakdown["service_weight_refs"]["competitor_gap_aic"] == "thumbnail_competitor_gap"
    assert breakdown["components"]["competitor_gap_aic"] == SERVICE_WEIGHTS["thumbnail_competitor_gap"]


def test_dashcam_osd_included_in_presign_aic_when_vision_on():
    from stages.ai_service_costs import resolve_enabled_ai_services

    enabled = resolve_enabled_ai_services(
        can_ai=True,
        user_prefs={
            "autoCaptions": True,
            "autoThumbnails": True,
            "aiServiceDashcamOSD": True,
        },
        use_ai_request=True,
        has_telemetry=False,
        env={
            "VISION_STAGE_ENABLED": True,
            "AUDIO_STAGE_ENABLED": False,
            "TREND_INTEL_AVAILABLE": False,
        },
    )
    assert "dashcam_osd" in enabled
    assert "vision_google" in enabled

    disabled = resolve_enabled_ai_services(
        can_ai=True,
        user_prefs={
            "autoCaptions": True,
            "autoThumbnails": True,
            "aiServiceDashcamOSD": False,
        },
        use_ai_request=True,
        has_telemetry=False,
        env={"VISION_STAGE_ENABLED": True, "AUDIO_STAGE_ENABLED": False},
    )
    assert "dashcam_osd" not in disabled


def test_ml_engine_finalize_blocks_seeded_and_cold_data():
    import asyncio
    from unittest.mock import MagicMock

    from services.ml_engine import _finalize_publish
    from services.ml_engine_config import get_ml_engine_config

    async def _seeded():
        out: dict = {}
        cfg = get_ml_engine_config()
        await _finalize_publish(
            cfg,
            None,
            out,
            {
                "status": "ok",
                "train_rows": max(cfg.min_train_rows, 100),
                "roc_auc": 0.99,
                "n_user_groups": max(cfg.publish_min_user_groups, 200),
            },
            True,
            task="promo_targeting_uplift_baseline",
            push_step=MagicMock(return_value={"ok": True}),
            record_run=MagicMock(),
        )
        return out

    async def _cold():
        out: dict = {}
        cfg = get_ml_engine_config()
        await _finalize_publish(
            cfg,
            None,
            out,
            {"status": "insufficient_rows", "message": "need more rows"},
            False,
            task="promo_targeting_uplift_baseline",
            push_step=MagicMock(side_effect=AssertionError("must not push")),
            record_run=MagicMock(),
        )
        return out

    async def _low_roc():
        out: dict = {}
        cfg = get_ml_engine_config()
        await _finalize_publish(
            cfg,
            None,
            out,
            {
                "status": "ok",
                "train_rows": max(cfg.min_train_rows, 100),
                "roc_auc": max(0.0, cfg.publish_min_roc_auc - 0.2),
                "n_user_groups": max(cfg.publish_min_user_groups, 200),
            },
            False,
            task="promo_targeting_uplift_baseline",
            push_step=MagicMock(side_effect=AssertionError("must not push")),
            record_run=MagicMock(),
        )
        return out

    seeded = asyncio.run(_seeded())
    assert seeded["ok"] is True
    assert seeded["status"] == "trained_not_published"
    assert "seeded" in seeded["reason_not_published"]

    cold = asyncio.run(_cold())
    assert cold["ok"] is True
    assert cold["status"] == "blocked_on_data"

    low = asyncio.run(_low_roc())
    assert low["ok"] is True
    assert low["status"] == "trained_not_published"
    assert "roc_auc" in low["reason_not_published"]


def test_publish_quality_content_near_miss_and_promo_overfit():
    """Close the three watch gaps: content near-miss, promo small-n / overfit."""
    from services.ml_engine import _passes_publish_quality
    from services.ml_engine_config import get_ml_engine_config

    cfg = get_ml_engine_config()
    min_roc = cfg.publish_min_roc_auc

    # Content: 0.549-style near miss with healthy NDCG → promote
    ok, reason = _passes_publish_quality(
        cfg,
        {
            "roc_auc": min_roc - (cfg.publish_near_miss_epsilon * 0.5),
            "ndcg_at_10": max(cfg.publish_min_ndcg_at_10, 0.65),
            "train_rows": 50,
        },
        "content_success_hotness",
    )
    assert ok is True and reason is None

    # Content: near miss but weak rankings → withhold
    ok, reason = _passes_publish_quality(
        cfg,
        {
            "roc_auc": min_roc - (cfg.publish_near_miss_epsilon * 0.5),
            "ndcg_at_10": max(0.0, cfg.publish_min_ndcg_at_10 - 0.2),
            "train_rows": 50,
        },
        "content_success_hotness",
    )
    assert ok is False
    assert reason and "ndcg" in reason

    # Promo: 69 groups below min → withhold (small-n)
    ok, reason = _passes_publish_quality(
        cfg,
        {
            "roc_auc": 0.99,
            "n_user_groups": max(1, cfg.publish_min_user_groups - 1),
            "train_rows": 69,
        },
        "promo_targeting_uplift_baseline",
    )
    assert ok is False
    assert reason and "n_user_groups" in reason

    # Promo: enough for min groups but still under overfit floor with sky-high AUC
    mid_groups = max(cfg.publish_min_user_groups, 1)
    if mid_groups < cfg.publish_overfit_max_user_groups:
        ok, reason = _passes_publish_quality(
            cfg,
            {
                "roc_auc": max(cfg.publish_overfit_roc_auc, 0.99),
                "n_user_groups": mid_groups,
                "train_rows": 200,
            },
            "promo_targeting_uplift_baseline",
        )
        assert ok is False
        assert reason and "overfit" in reason

    # Promo: healthy sample + honest ROC → pass quality
    ok, reason = _passes_publish_quality(
        cfg,
        {
            "roc_auc": max(min_roc, 0.62),
            "n_user_groups": max(cfg.publish_overfit_max_user_groups, 200),
            "train_rows": 500,
        },
        "promo_targeting_uplift_baseline",
    )
    assert ok is True and reason is None


def test_m8_blend_hour_priors_mixes_static_and_learned():
    from core.scheduling import static_hour_prior_24
    from services.m8_publish_hour_model import M8_PRIOR_PLATFORMS, _blend_hour_priors

    learned = {
        "youtube": [0.0] * 23 + [1.0],
    }
    blended = _blend_hour_priors(learned, alpha=0.5, static_fn=static_hour_prior_24)
    assert set(blended.keys()) == set(M8_PRIOR_PLATFORMS)
    assert abs(sum(blended["youtube"]) - 1.0) < 1e-6
    # Peak hour should move toward 23 vs pure static
    static_yt = static_hour_prior_24("youtube")
    assert blended["youtube"][23] > static_yt[23]
    # Platforms without learned weights stay static
    assert blended["tiktok"] == list(static_hour_prior_24("tiktok"))


def test_smart_schedule_blend_prefers_user_when_enough_samples():
    from services.smart_schedule_insights import _blend_vectors, _normalize

    static_w = _normalize([1.0] * 24)
    global_w = _normalize([2.0 if i == 10 else 0.1 for i in range(24)])
    user_w = _normalize([5.0 if i == 18 else 0.05 for i in range(24)])
    momentum = [1.0] * 24

    low = _blend_vectors(static_w, global_w, user_w, user_sample_count=2, momentum=momentum)
    high = _blend_vectors(static_w, global_w, user_w, user_sample_count=20, momentum=momentum)
    assert abs(sum(low) - 1.0) < 1e-6
    assert abs(sum(high) - 1.0) < 1e-6
    assert high[18] > low[18]


def test_finalize_publishes_content_near_miss_with_ndcg():
    """Content ROC just under floor + healthy NDCG must reach Hub push."""
    import asyncio
    from unittest.mock import MagicMock

    from services.ml_engine import _finalize_publish
    from services.ml_engine_config import get_ml_engine_config

    async def _run():
        out: dict = {}
        cfg = get_ml_engine_config()
        push = MagicMock(return_value={"ok": True})
        await _finalize_publish(
            cfg,
            None,
            out,
            {
                "status": "ok",
                "train_rows": 50,
                "roc_auc": cfg.publish_min_roc_auc - (cfg.publish_near_miss_epsilon * 0.4),
                "ndcg_at_10": 0.65,
            },
            False,
            task="content_success_hotness",
            push_step=push,
            record_run=MagicMock(return_value="run-1"),
        )
        return out, push

    out, push = asyncio.run(_run())
    assert out.get("status") == "published"
    assert out.get("ok") is True
    push.assert_called_once()


def test_finalize_withholds_promo_small_n_overfit():
    import asyncio
    from unittest.mock import MagicMock

    from services.ml_engine import _finalize_publish
    from services.ml_engine_config import get_ml_engine_config

    async def _run():
        out: dict = {}
        cfg = get_ml_engine_config()
        push = MagicMock(side_effect=AssertionError("must not push overfit promo"))
        await _finalize_publish(
            cfg,
            None,
            out,
            {
                "status": "ok",
                "train_rows": 69,
                "roc_auc": 0.9949,
                "n_user_groups": 69,
            },
            False,
            task="promo_targeting_uplift_baseline",
            push_step=push,
            record_run=MagicMock(),
        )
        return out

    out = asyncio.run(_run())
    assert out.get("ok") is True
    assert out.get("status") == "trained_not_published"
    assert "n_user_groups" in (out.get("reason_not_published") or "")


def test_architecture_doc_restored():
    from pathlib import Path

    doc = Path(__file__).resolve().parents[1] / "docs" / "ml-ai-architecture.md"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    assert "ml_feature_registry" in text
    assert "atomic_debit_tokens" in text
    assert "trained_not_published" in text
    assert "require_admin" in text
