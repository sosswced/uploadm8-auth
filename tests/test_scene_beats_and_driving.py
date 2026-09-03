"""Scene beats, driving_evidence, clip-kind v2, hero window — glasses POV ladder.

Hashtag generic bans stay frozen. Dashcam filename / HUD / .map paths stay dashcam.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from core.content_identity import build_content_identity
from core.driving_evidence import DASHCAM_FILENAME_TOKENS, has_driving_evidence
from core.frame_quality import jpeg_is_unusable
from core.hero_window import (
    pick_second_pass_timestamps,
    pick_speech_second_pass_timestamps,
    select_hero_window,
)
from core.vision_labels import (
    is_generic_vision_label,
    is_junk_hashtag_body,
    prose_scene_beats_from_vi,
    resolve_ambient_profiles,
)
from core.visual_marks import collect_visual_marks
from core.wearable_source import is_meta_glasses_filename, source_needs_h264_proxy
from services.hydration_enforcer import collect_evidence
from services.multimodal_depth_router import classify_clip_kind
from services.place_evidence import extract_place_evidence
from stages.thumbnail_stage import (
    _concrete_thumbnail_headline,
    _hero_fact_headlines,
    effective_thumbnail_category,
)
from stages.vision_stage import _adaptive_vision_frame_count


def _ctx(**overrides) -> SimpleNamespace:
    base = dict(
        upload_id="beats-1",
        filename="walk.mp4",
        thumbnail_category="travel",
        telemetry=None,
        telemetry_data=None,
        dashcam_osd_context={},
        vision_context={},
        audio_context={},
        video_intelligence={},
        video_intelligence_context={},
        video_understanding={},
        ai_transcript="",
        output_artifacts={},
        duration_seconds=180,
        user_settings={},
        place_evidence={},
    )
    base.update(overrides)
    ns = SimpleNamespace(**base)
    ns.get_effective_title = lambda: str(getattr(ns, "ai_title", "") or "")
    ns.get_effective_caption = lambda: str(getattr(ns, "ai_caption", "") or "")
    return ns


def test_water_stays_generic_hashtag_ban():
    assert is_generic_vision_label("water") is True
    assert is_junk_hashtag_body("water") is True
    assert is_junk_hashtag_body("#water") is True


def test_vi_duration_beats_include_water_and_church_not_flicker():
    ctx = _ctx(
        video_intelligence={
            "object_tracks": [
                {"description": "water", "start_s": 10.0, "end_s": 50.0, "confidence": 0.9},
                {"description": "church", "start_s": 80.0, "end_s": 92.0, "confidence": 0.85},
                {"description": "water", "start_s": 4.0, "end_s": 5.0, "confidence": 0.95},
                {"description": "sky", "start_s": 0.0, "end_s": 40.0, "confidence": 0.99},
            ]
        }
    )
    beats = prose_scene_beats_from_vi(ctx)
    nouns = {str(b.get("noun") or "").lower() for b in beats}
    assert "water" in nouns
    assert "church" in nouns
    assert "sky" not in nouns
    for b in beats:
        assert float(b.get("duration_s") or 0) >= 2.0


def test_building_fallback_skipped_when_church_in_window():
    ctx = _ctx(
        video_intelligence={
            "object_tracks": [
                {"description": "building", "start_s": 70.0, "end_s": 90.0, "confidence": 0.8},
                {"description": "church", "start_s": 75.0, "end_s": 88.0, "confidence": 0.9},
            ]
        }
    )
    beats = prose_scene_beats_from_vi(ctx)
    nouns = [str(b.get("noun") or "").lower() for b in beats]
    assert "church" in nouns
    assert "building" not in nouns


def test_m8_and_escort_filenames_still_dashcam():
    for fname in ("M8_2024_DRIVE.MP4", "ESCORT_NIGHT.mp4"):
        ctx = _ctx(filename=fname, thumbnail_category="general", vision_context={})
        assert classify_clip_kind(ctx) == "dashcam"
        assert has_driving_evidence(ctx) is True


def test_peak_mph_without_driving_evidence_not_automotive():
    tel = SimpleNamespace(
        max_speed_mph=40.0,
        avg_speed_mph=28.0,
        points=[],
        location_city="Barcelona",
        location_state=None,
        location_road=None,
        location_display="Barcelona",
        location_start_display=None,
        gazetteer_place_name=None,
        padus_unit_name=None,
    )
    ctx = _ctx(
        filename="glasses_centro.mp4",
        thumbnail_category="travel",
        telemetry=tel,
        vision_context={"label_names": ["outdoor", "sky", "person"]},
    )
    assert has_driving_evidence(ctx) is False
    identity = build_content_identity(ctx)
    tags = [
        str(t.get("tag") or "").lower()
        for t in (identity.get("domain_tags") or [])
        if isinstance(t, dict)
    ]
    assert "automotive" not in tags
    assert not any(f.get("class") == "speed" for f in (identity.get("hero_facts") or []) if isinstance(f, dict))
    headlines = _hero_fact_headlines(ctx, "travel")
    assert not any("MPH" in str(c).upper() for c in headlines)


def test_map_points_are_driving_evidence():
    tel = SimpleNamespace(
        max_speed_mph=40.0,
        points=[{"lat": 34.0, "lon": -118.0}, {"lat": 34.01, "lon": -118.01}],
    )
    ctx = _ctx(filename="clip.mp4", telemetry=tel, vision_context={})
    assert has_driving_evidence(ctx) is True


def test_hero_window_prefers_late_landmark_over_early_sky():
    winner = select_hero_window(
        [
            {"t_seconds": 10.0, "kind": "outdoor", "label": "sky", "score": 10},
            {"t_seconds": 170.0, "kind": "landmark", "label": "Catedral", "score": 100},
        ],
        duration_s=180.0,
    )
    assert winner is not None
    assert winner["t_seconds"] == 170.0
    assert winner["kind"] == "landmark"


def test_second_pass_timestamps_cap_at_four():
    beats = [
        {"t": float(i * 20), "duration_s": 12.0, "noun": "church"}
        for i in range(8)
    ]
    ts = pick_second_pass_timestamps(beats, max_n=4, duration_s=180.0)
    assert len(ts) <= 4


def test_second_pass_exception_does_not_raise(monkeypatch):
    from stages import vision_stage

    async def boom(_ctx):
        raise RuntimeError("gcv down")

    monkeypatch.setattr(vision_stage, "_second_pass_should_run", lambda _ctx: (True, "test"))
    monkeypatch.setattr(vision_stage, "_vision_second_pass_impl", boom)
    ctx = _ctx()
    out = asyncio.run(vision_stage.maybe_run_vision_second_pass(ctx))
    assert out.get("ok") is False
    assert "gcv" in str(out.get("error") or "").lower()


def test_travel_ambient_is_not_automotive_without_windshield():
    profiles = resolve_ambient_profiles(
        category="travel",
        filename="centro_walk.mp4",
        vision_label_names=["outdoor", "sky", "person"],
    )
    assert "automotive" not in profiles
    assert "travel" in profiles


def test_windshield_stack_still_automotive_on_travel():
    profiles = resolve_ambient_profiles(
        category="travel",
        filename="centro_walk.mp4",
        vision_label_names=["windshield", "rear-view mirror", "hood"],
    )
    assert "automotive" in profiles


def test_clip_kind_wearable_for_travel_walk():
    ctx = _ctx(
        filename="vacation_clip.mp4",
        thumbnail_category="travel",
        vision_context={"label_names": ["outdoor", "sky", "person"], "has_faces": False},
        duration_seconds=45,
        ai_transcript="",
    )
    assert classify_clip_kind(ctx) == "wearable"


def test_clip_kind_vlog_from_long_transcript_without_faces():
    ctx = _ctx(
        filename="talk.mp4",
        thumbnail_category="general",
        vision_context={"label_names": ["indoor"], "has_faces": False},
        ai_transcript="x" * 80,
        duration_seconds=60,
    )
    assert classify_clip_kind(ctx) == "vlog"


def test_clip_kind_museum_from_ocr():
    ctx = _ctx(
        filename="gallery.mp4",
        thumbnail_category="general",
        vision_context={"ocr_text": "BRITISH MUSEUM exhibit hall", "label_names": ["indoor"]},
    )
    assert classify_clip_kind(ctx) == "museum"


def test_clip_kind_concert_from_labels():
    ctx = _ctx(
        filename="night.mp4",
        thumbnail_category="general",
        vision_context={"label_names": ["concert", "stage", "crowd"]},
    )
    assert classify_clip_kind(ctx) == "concert"


def test_place_evidence_harbor_plaza_fountain():
    ctx = _ctx(
        vision_context={
            "ocr_text": "Port Vell Harbor\nPlaça Catalunya fountain",
            "landmark_names": [],
            "logo_names": [],
        },
        video_intelligence={},
        audio_context={},
    )
    report = extract_place_evidence(ctx)
    blob = " ".join(report.get("places") or []).lower()
    harbors = [str(x).lower() for x in (report.get("harbors") or [])]
    plazas = [str(x).lower() for x in (report.get("plazas") or [])]
    assert harbors or "harbor" in blob or "port" in blob
    assert plazas or "plaza" in blob or "plaça" in blob or "fountain" in blob


def test_meta_glasses_filename_is_wearable_not_dashcam():
    for fname in (
        "mcp_video-45_singular_display.mov",
        "od_video-1_singular_display.mov",
    ):
        assert is_meta_glasses_filename(fname) is True
        for tok in DASHCAM_FILENAME_TOKENS:
            assert tok not in ("MCP_VIDEO", "OD_VIDEO", "SINGULAR_DISPLAY")
        ctx = _ctx(
            filename=fname,
            thumbnail_category="automotive",
            vision_context={},
            ai_transcript="x" * 90,
        )
        assert has_driving_evidence(ctx) is False
        assert classify_clip_kind(ctx) == "wearable"


def test_empty_vision_speech_chiefs_marks_and_headline_not_music():
    ctx = _ctx(
        filename="mcp_video-45_singular_display.mov",
        thumbnail_category="general",
        vision_context={"label_names": [], "logo_names": [], "landmark_names": [], "ocr_text": ""},
        audio_context={
            "music_title": "HandClap",
            "music_artist": "Fitz and The Tantrums",
            "transcript": "Touchback, the Chiefs will have 1st and 10. I'm on that side of the stadium.",
        },
        ai_transcript="Touchback, the Chiefs will have 1st and 10. I'm on that side of the stadium.",
    )
    marks = collect_visual_marks(ctx)
    blob = " ".join(str(m.get("text") or "") for m in marks).lower()
    assert "chiefs" in blob
    report = extract_place_evidence(ctx)
    teams = [str(t).lower() for t in (report.get("sports_teams") or [])]
    assert any("chiefs" in t for t in teams)
    headline = _concrete_thumbnail_headline(ctx, "general")
    assert "CHIEF" in headline.upper()
    assert "FITZ" not in headline.upper()
    assert "HANDCLAP" not in headline.upper()
    assert "VIBING" not in headline.upper()


def test_empty_vision_no_speech_does_not_invent_team():
    ctx = _ctx(
        filename="mcp_video-45_singular_display.mov",
        vision_context={"label_names": [], "logo_names": [], "ocr_text": ""},
        audio_context={},
        ai_transcript="",
    )
    marks = collect_visual_marks(ctx)
    blob = " ".join(str(m.get("text") or "") for m in marks).lower()
    assert "chiefs" not in blob
    report = extract_place_evidence(ctx)
    assert not (report.get("sports_teams") or [])


def test_garage_mustang_without_driving_not_in_hydration():
    ctx = _ctx(
        filename="mcp_video-45_singular_display.mov",
        vehicle_make_name="FORD",
        vehicle_model_name="Mustang",
        vision_context={},
        thumbnail_category="sports",
    )
    pool = collect_evidence(ctx)
    assert not pool.vehicle_make
    assert not pool.vehicle_model


def test_garage_mustang_active_when_automotive():
    ctx = _ctx(
        filename="mcp_video-45_singular_display.mov",
        vehicle_make_name="FORD",
        vehicle_model_name="Mustang",
        vision_context={},
        thumbnail_category="automotive",
    )
    pool = collect_evidence(ctx)
    assert pool.vehicle_make == "FORD"
    assert pool.vehicle_model == "Mustang"


def test_near_black_jpeg_rejected(tmp_path):
    from PIL import Image

    black = tmp_path / "black.jpg"
    Image.new("RGB", (640, 360), (0, 0, 0)).save(black, "JPEG", quality=85)
    assert jpeg_is_unusable(black) is True
    bright = tmp_path / "bright.jpg"
    Image.new("RGB", (640, 360), (200, 180, 40)).save(bright, "JPEG", quality=85)
    assert jpeg_is_unusable(bright) is False


def test_speech_second_pass_timestamps_from_stadium_segment():
    ctx = _ctx(
        filename="mcp_video-45_singular_display.mov",
        audio_context={
            "transcript_segments": [
                {"start": 10.0, "text": "photo tickets"},
                {"start": 43.0, "text": "I'm on that side of the stadium"},
                {"start": 90.0, "text": "Touchback, the Chiefs will have 1st and 10"},
            ]
        },
        duration_seconds=120,
    )
    ts = pick_speech_second_pass_timestamps(ctx, max_n=4, duration_s=120.0)
    assert ts
    assert any(abs(t - 43.0) < 2.5 or abs(t - 90.0) < 2.5 for t in ts)


def test_wearable_frame_floor_is_four():
    n_plain = _adaptive_vision_frame_count(0.0, wearable_floor=False)
    n_wear = _adaptive_vision_frame_count(0.0, wearable_floor=True)
    assert n_wear >= 4
    assert n_wear >= n_plain


def test_glasses_automotive_studio_niche_does_not_override():
    us = {"thumbnail_studio_default_strategy": {"audience_niche": "automotive"}}
    assert (
        effective_thumbnail_category(
            us, "sports", filename="mcp_video-45_singular_display.mov"
        )
        == "sports"
    )
    assert effective_thumbnail_category(us, "gardening", filename="walk.mp4") == "automotive"


def test_watchable_proxy_failure_skips_hevc_not_oversized():
    from stages.video_intelligence_stage import watchable_proxy_failure_reason

    assert watchable_proxy_failure_reason(
        needs_hevc_proxy=True, size=5_000_000, max_bytes=20_000_000, proxy_ok=True
    ) is None
    hevc_fail = watchable_proxy_failure_reason(
        needs_hevc_proxy=True, size=5_000_000, max_bytes=20_000_000, proxy_ok=False
    )
    assert hevc_fail and "HEVC" in hevc_fail
    oversized = watchable_proxy_failure_reason(
        needs_hevc_proxy=False, size=50_000_000, max_bytes=20_000_000, proxy_ok=False
    )
    assert oversized and "too large" in oversized.lower()


def test_source_needs_h264_proxy_unknown_mov_does_not_raise():
    ctx = _ctx(filename="clip.mov", video_info={"video_codec": ""})
    assert source_needs_h264_proxy(ctx, "clip.mov") is True
    ctx_hevc = _ctx(filename="walk.mp4", video_info={"video_codec": "hevc"})
    assert source_needs_h264_proxy(ctx_hevc, "walk.mp4") is True
    ctx_h264 = _ctx(filename="clip.mp4", video_info={"video_codec": "h264"})
    assert source_needs_h264_proxy(ctx_h264, "clip.mp4") is False
    ctx_meta = _ctx(filename="mcp_video-45_singular_display.mov", video_info={})
    assert source_needs_h264_proxy(ctx_meta, "mcp_video-45_singular_display.mov") is True


def test_vi_skip_status_is_persisted():
    from stages.video_intelligence_stage import record_video_intelligence_status

    ctx = _ctx()
    record_video_intelligence_status(ctx, status="skipped", reason="hevc timeout")
    raw = str((ctx.output_artifacts or {}).get("video_intelligence_status") or "")
    assert "skipped" in raw
    assert "hevc" in raw.lower()
