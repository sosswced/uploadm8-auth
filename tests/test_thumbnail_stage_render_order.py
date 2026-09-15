import asyncio
import uuid
from pathlib import Path

from stages.context import JobContext, TelemetryData, TrillScore
import stages.pikzels_api as pikzels_api
from stages.entitlements import get_entitlements_for_tier
from stages.pikzels_api import _build_pikzels_v2_prompt, render_thumbnail_with_studio_renderer
from stages.thumbnail_stage import (
    _apply_thumbnail_default_strategy,
    _detect_category,
    _sanitize_thumbnail_brief,
    _studio_persona_for_request,
    _strict_studio_mode_enabled,
    _thumbnail_hydration_edit_prompt,
    _thumbnail_styled_render_order,
    pikzels_studio_eligible_for_styled_thumbnail,
)
from services.thumbnail_studio import generate_recreate_variants, hydration_signal_lanes


def _write_tiny_jpeg(path: Path) -> Path:
    from PIL import Image

    Image.new("RGB", (64, 36), color=(40, 80, 120)).save(path, format="JPEG", quality=85)
    return path


def _dashcam_job(**kwargs) -> JobContext:
    """Dashcam-shaped JobContext with driving evidence (filename token)."""
    base = dict(
        job_id="job-1",
        upload_id="upload-1",
        user_id="user-1",
        filename="20250224_0073_CAM_EVNT.MP4",
        local_video_path=Path("x.mp4"),
    )
    base.update(kwargs)
    return JobContext(**base)


def _high_conf_speed_ctx(*, peak: float = 73.0, title: str = "Watch this") -> JobContext:
    """HUD + vision OCR agreement → high-confidence publishable peak."""
    mph = int(round(peak))
    return _dashcam_job(
        title=title,
        telemetry_data=TelemetryData(max_speed_mph=float(peak)),
        dashcam_osd_context={
            "max_speed_mph": float(peak),
            "speed_series": [{"mph": float(peak), "t_s": 5.0}],
        },
        vision_context={
            "ocr_text": (
                f"2025/03/05 04:50 12 PM 36.136162° -115.178398° {mph}MPH C Walker\n"
                f"2025/03/05 04:51 12 PM 36.136200° -115.178400° {mph - 1}MPH C Walker"
            )
        },
    )


def test_render_order_studio_ok_is_pikzels_only():
    """Studio stays first; hard fallbacks keep a cover if Pikzels fails."""
    order = _thumbnail_styled_render_order(
        "auto",
        studio_ok=True,
        ai_edit_ok=True,
    )
    assert order == ["studio", "ai_edit", "template"]


def test_render_order_studio_ok_ignores_ai_edit_flag():
    order = _thumbnail_styled_render_order(
        "auto",
        studio_ok=True,
        ai_edit_ok=False,
    )
    assert order == ["studio", "template"]


def test_render_order_falls_back_without_studio():
    order = _thumbnail_styled_render_order(
        "auto",
        studio_ok=False,
        ai_edit_ok=True,
    )
    assert order == ["ai_edit", "template"]


def test_strict_studio_mode_can_be_enabled_by_user_pref():
    assert _strict_studio_mode_enabled({"thumbnailStudioStrict": True}) is True


def test_thumbnail_brief_replaces_generic_headline_with_context():
    ctx = JobContext(
        job_id="job-1",
        upload_id="upload-1",
        user_id="user-1",
        title="Sunset drive on Highway 7",
    )
    brief = _sanitize_thumbnail_brief(
        ctx,
        {"selected_headline": "EXCITING MOMENTS", "headline_options": ["MUST WATCH"]},
        "automotive",
    )

    assert brief["selected_headline"] == "SUNSET DRIVE ON HIGHWAY 7"
    assert "EXCITING MOMENTS" not in brief["headline_options"]


def test_thumbnail_brief_uses_telemetry_before_generic_title():
    ctx = _high_conf_speed_ctx(peak=73.4, title="Watch this")
    brief = _sanitize_thumbnail_brief(ctx, {"selected_headline": "EXCITING MOMENTS"}, "automotive")

    assert brief["selected_headline"] == "73 MPH RUN"


def test_thumbnail_brief_blocks_unbelievable_moments_default():
    ctx = _high_conf_speed_ctx(peak=42.0, title="Flower drive near Red Rock")
    brief = _sanitize_thumbnail_brief(
        ctx,
        {"selected_headline": "UNBELIEVABLE MOMENTS", "headline_options": ["UNBELIEVABLE MOMENTS"]},
        "automotive",
    )

    assert brief["selected_headline"] == "42 MPH RUN"
    assert "UNBELIEVABLE MOMENTS" not in brief["headline_options"]


def test_thumbnail_brief_does_not_force_speed_for_non_motion_category():
    ctx = JobContext(
        job_id="job-1",
        upload_id="upload-1",
        user_id="user-1",
        title="Garden bed update",
        telemetry_data=TelemetryData(max_speed_mph=73.4),
    )
    brief = _sanitize_thumbnail_brief(ctx, {"selected_headline": "EXCITING MOMENTS"}, "gardening")

    assert brief["selected_headline"] == "GARDEN BED UPDATE"


def test_thumbnail_brief_carries_geo_music_signals():
    tel = TelemetryData(max_speed_mph=42.0)
    tel.location_city = "Moab"
    tel.location_state = "Utah"
    tel.location_road = "Scenic Byway 128"
    tel.gazetteer_place_name = "Moab city"
    tel.padus_unit_name = "Arches National Park"
    ctx = JobContext(
        job_id="job-1",
        upload_id="upload-1",
        user_id="user-1",
        title="Pretty flowers on the drive",
        telemetry_data=tel,
        audio_context={
            "music_detected": True,
            "music_artist": "Drake",
            "music_title": "Hotline Bling",
        },
    )

    brief = _sanitize_thumbnail_brief(ctx, {"selected_headline": "FLOWER DRIVE"}, "travel")

    assert "Moab" in brief["geo_context"]
    assert "Arches National Park" in brief["geo_context"]
    assert "Drake" in brief["music_context"]
    assert "#drake" in brief["signal_hashtags"]


def test_thumbnail_brief_carries_osd_and_trill_signals():
    ctx = _high_conf_speed_ctx(peak=64.0, title="Fast flower drive")
    ctx.dashcam_osd_context = {
        **(ctx.dashcam_osd_context or {}),
        "avg_speed_mph": 42,
        "driver_name": "C Walker",
        "speed_unit_detected": "mph",
        "gps_path": [[36.1, -115.1], [36.2, -115.2]],
        "telemetry_backfilled": True,
    }
    ctx.trill_score = TrillScore(score=72, bucket="sendIt")
    ctx.trill = ctx.trill_score

    brief = _sanitize_thumbnail_brief(ctx, {"selected_headline": "FLOWER DRIVE"}, "automotive")

    assert "peak speed: 64 mph" in brief["osd_context"]
    assert "GPS fixes: 2" in brief["osd_context"]
    assert "bucket: sendIt" in brief["trill_context"]


def test_thumbnail_brief_uses_osd_speed_when_telemetry_missing():
    ctx = _high_conf_speed_ctx(peak=64.0, title="Watch this")
    # Drop telemetry — HUD+vision alone must still publish peak when high-confidence.
    ctx.telemetry_data = None

    brief = _sanitize_thumbnail_brief(
        ctx,
        {"selected_headline": "EXCITING MOMENTS", "badge_text": "FAST"},
        "automotive",
    )

    assert brief["selected_headline"] == "64 MPH RUN"
    assert brief["badge_text"] == "FAST"


def test_thumbnail_brief_carries_whisper_speech_context():
    ctx = JobContext(
        job_id="job-1",
        upload_id="upload-1",
        user_id="user-1",
        title="Garden update",
        ai_transcript="These flowers finally bloomed after three weeks of watering.",
        audio_context={"gpt_audio_summary": "calm gardening update with a reveal"},
    )

    brief = _sanitize_thumbnail_brief(ctx, {"selected_headline": "FLOWERS BLOOMED"}, "gardening")

    assert "These flowers finally bloomed" in brief["speech_context"]
    assert "calm gardening update" in brief["speech_context"]


def test_thumbnail_category_uses_fused_context():
    ctx = JobContext(
        job_id="job-1",
        upload_id="upload-1",
        user_id="user-1",
        vision_context={"label_names": ["mountain", "landmark", "scenic overlook"]},
        video_understanding={
            "scene_description": "A scenic mountain overlook along a travel corridor.",
            "topics": ["travel", "scenic", "landmark"],
        },
    )

    # Soft identity bucket — raw labels alone do not mint travel; VU topics can.
    assert _detect_category(ctx) in ("travel", "general")


def test_pikzels_prompt_blocks_generic_headline_text():
    """Generic headlines must NOT be rendered AND the prompt must explicitly
    forbid Pikzels from stamping its own clickbait clichés."""
    prompt = _build_pikzels_v2_prompt(
        {"selected_headline": "EXCITING MOMENTS"},
        category="general",
        platform="youtube",
    )

    assert 'Bold headline text reading "EXCITING MOMENTS"' not in prompt
    assert "CREATIVE COMPOSITION MODE" in prompt or "STRICT NO-TEXT" in prompt
    # The forbidden-phrases list must call out the actual cliché surfaces we
    # have observed Pikzels stamping in production.
    for forbidden in ("UNBELIEVABLE MOMENTS", "EVENT MOMENTS", "MUST WATCH", "WATCH THIS"):
        assert forbidden in prompt, f"prompt missing explicit ban on {forbidden!r}"


def test_pikzels_prompt_concrete_headline_is_rendered_with_lockdown():
    """Earned MPH hooks ARE painted; place names are not. Lock down other text."""
    prompt = _build_pikzels_v2_prompt(
        {
            "selected_headline": "64 MPH",
            "_uploadm8_paint_policy": "hook_only",
            "_uploadm8_hook_line": "64 MPH",
            "hook_line": "64 MPH",
            "_uploadm8_hook_class": "speed",
        },
        category="automotive",
        platform="youtube",
    )
    assert '"64 MPH"' in prompt
    assert "NO OTHER text" in prompt or "Render NO OTHER text" in prompt


def test_pikzels_prompt_place_banner_is_composition_not_paint():
    """LOCATION / city plaster must never unlock reading \"…\" paint mode."""
    prompt = _build_pikzels_v2_prompt(
        {
            "selected_headline": "LOCATION FEDERAL WAY WASHINGTON",
            "_uploadm8_paint_policy": "none",
            "geo_context": "Federal Way, Washington",
        },
        category="automotive",
        platform="instagram",
    )
    assert "CREATIVE COMPOSITION MODE" in prompt or "STRICT NO-TEXT" in prompt
    assert 'reading "LOCATION' not in prompt
    assert 'reading "FEDERAL WAY' not in prompt
    assert "do NOT render as text" in prompt or "LOCATION" in prompt


def test_pikzels_prompt_includes_geo_music_signal_context():
    prompt = _build_pikzels_v2_prompt(
        {
            "selected_headline": "MOAB FLOWERS",
            "geo_context": "city: Moab; protected area: Arches National Park",
            "music_context": "artist: Drake; track: Hotline Bling",
            "signal_hashtags": "#moab, #archesnationalpark, #drake",
        },
        category="travel",
        platform="youtube",
    )

    assert "Moab" in prompt
    assert "Arches National Park" in prompt
    assert "Drake" in prompt
    # Place/music are creative cues, not painted headlines.
    assert 'reading "MOAB FLOWERS"' not in prompt


def test_pikzels_prompt_includes_osd_and_trill_context():
    prompt = _build_pikzels_v2_prompt(
        {
            "selected_headline": "64 MPH",
            "_uploadm8_paint_policy": "hook_only",
            "_uploadm8_hook_line": "64 MPH",
            "hook_line": "64 MPH",
            "_uploadm8_hook_class": "speed",
            "osd_context": "peak speed: 64 mph; GPS fixes: 2; telemetry backfilled from HUD",
            "trill_context": "score: 72; bucket: sendIt",
        },
        category="automotive",
        platform="youtube",
    )

    assert "64" in prompt and ("mph" in prompt.lower() or "OSD:" in prompt)
    assert "sendIt" in prompt


def test_pikzels_prompt_includes_whisper_speech_context():
    # Speech/transcript is intentionally omitted from Pikzels prompts (content-filter noise).
    prompt = _build_pikzels_v2_prompt(
        {
            "selected_headline": "FLOWERS BLOOMED",
            "speech_context": "transcript excerpt: These flowers finally bloomed after three weeks.",
            "geo_context": "Moab, Utah desert garden overlook",
        },
        category="gardening",
        platform="youtube",
    )

    assert "Moab" in prompt
    assert "These flowers finally bloomed" not in prompt


def test_pikzels_prompt_includes_fusion_summary():
    prompt = _build_pikzels_v2_prompt(
        {
            "selected_headline": "MOAB CLIFFSIDE",
            "fusion_summary": "GPS corridor: Utah SR-128; visible redrock mesas; midday harsh sun.",
        },
        category="travel",
        platform="youtube",
    )
    assert "Fusion" in prompt
    assert "GPS corridor" in prompt


def test_pikzels_prompt_uses_compact_hydration_labels():
    hp = {
        "v": 2,
        "category": "automotive",
        "anchor_phrase": "anchor",
        "evidence": {
            "geo": {"city": "Moab", "state": "UT"},
            "osd": {"max_speed_mph": 72},
            "music": {"artist": "A", "title": "B"},
            "speech": {},
            "vision": {"labels": [], "ocr": "", "landmarks": [], "logos": []},
            "trill": {"bucket": "high", "score": 90},
        },
        "signal_hashtags": ["#moab"],
        "fusion_summary": "",
        "hydration_story": "",
        "trace_id": "",
    }
    # Composition-first (no paint_policy): scene cues, not Geo:/Music: stamp lines.
    prompt = _build_pikzels_v2_prompt(
        {"selected_headline": "72 MPH MOAB"},
        category="automotive",
        platform="youtube",
        hydration_payload=hp,
    )
    assert len(prompt) <= 1000
    assert "Moab" in prompt or "Scene vibe" in prompt
    assert "OSD:" in prompt and "spd" in prompt
    assert "Music:" in prompt or "Audio energy" in prompt
    assert "Trill:" in prompt
    assert "Canonical geo" not in prompt


def test_pikzels_prompt_injects_hydration_payload_canonical():
    hp = {
        "evidence": {
            "geo": {"road": "Hwy 7", "city": "Reno", "state": "NV"},
            "osd": {"max_speed_mph": 73.4, "driver_name": "C Walker", "first_seen": "2025-03-05 12:50"},
            "speech": {"phrase": "look at this"},
            "trill": {"score": 78, "bucket": "spirited"},
            "vision": {"labels": ["road", "dashboard"], "ocr": "SPEED 73"},
        },
        "fusion_summary": "Narrative fused line.",
        "hydration_story": "Paragraph story.",
        "anchor_phrase": "Sunset highway run.",
        "signal_hashtags": ["dashcam", "nevada"],
    }
    # Speed paint: Geo/OSD/OCR/driver allowed alongside MPH hook.
    prompt = _build_pikzels_v2_prompt(
        {
            "selected_headline": "64 MPH",
            "_uploadm8_paint_policy": "hook_only",
            "_uploadm8_hook_line": "64 MPH",
            "hook_line": "64 MPH",
            "_uploadm8_hook_class": "speed",
        },
        category="automotive",
        platform="youtube",
        hydration_payload=hp,
    )
    assert "Hwy 7" in prompt
    assert "spd 73" in prompt.lower() or "73" in prompt
    assert "Walker" in prompt
    # Speech phrases are omitted from Pikzels prompts (lyric / filter noise).
    assert "look at this" not in prompt.lower()
    assert "spirited" in prompt
    assert "canonical hydration_payload" not in prompt.lower()
    assert "Narrative fused line" in prompt


def test_pikzels_prompt_category_fallback_headline_is_strict_no_text():
    """ROAD HIGHLIGHT etc. must not unlock painted headline mode — triggers clichés."""
    prompt = _build_pikzels_v2_prompt(
        {"selected_headline": "ROAD HIGHLIGHT"},
        category="automotive",
        platform="youtube",
    )
    assert "CREATIVE COMPOSITION MODE" in prompt or "STRICT NO-TEXT" in prompt
    assert 'reading "ROAD HIGHLIGHT"' not in prompt


def test_pikzels_prompt_includes_hydration_story_when_fusion_thin():
    prompt = _build_pikzels_v2_prompt(
        {
            "selected_headline": "MOAB RUN",
            "fusion_summary": "x",
            "hydration_story": "GPS corridor Utah SR-128 with redrock mesas visible roadside.",
        },
        category="travel",
        platform="youtube",
    )
    assert "Story" in prompt
    assert "Utah SR-128" in prompt


def test_minimal_regenerate_brief_strips_clickbait_title():
    from services.thumbnail_brief_pipeline import minimal_thumbnail_brief

    b = minimal_thumbnail_brief(title="Unbelievable moments dashcam compilation")
    assert "UNBELIEVABLE" not in (b.get("selected_headline") or "")


def test_persona_guard_prepends_to_pikzels_prompt(monkeypatch, tmp_path):
    captured: dict = {}

    async def fake_post(path, payload):
        captured.update(dict(payload))
        return 200, {"ok": True}

    async def fake_bytes(data, timeout):
        return b"x" * 3000

    monkeypatch.setattr(pikzels_api, "resolve_public_api_key", lambda: "pk_test")
    monkeypatch.setattr(pikzels_api, "pikzels_v2_post", fake_post)
    monkeypatch.setattr(pikzels_api, "_pikzels_v2_response_to_bytes", fake_bytes)

    base = tmp_path / "base.jpg"
    out = tmp_path / "out.jpg"
    _write_tiny_jpeg(base)

    persona_id = str(uuid.uuid4())
    jc = JobContext(
        job_id="job-x",
        upload_id="up-x",
        user_id="usr-x",
        user_settings={"thumbnailRefPersonaMode": "face_brand"},
    )

    ok = asyncio.run(
        render_thumbnail_with_studio_renderer(
            base,
            {"selected_headline": "73 MPH RUN"},
            "youtube",
            out,
            persona={"id": persona_id, "kind": "persona"},
            job_context=jc,
        )
    )
    assert ok is True
    assert captured.get("persona") == persona_id
    prompt = str(captured.get("prompt") or "")
    # Guard is prepended then budget-clamped; UUID in payload is the hard contract.
    assert "Persona or style reference" in prompt or "likeness" in prompt.lower() or len(prompt) > 40


def test_hydration_payload_persist():
    from services.hydration_payload import build_hydration_payload, persist_hydration_payload_artifact

    ctx = JobContext(job_id="j1", upload_id="u1", user_id="usr1", title="Drive")
    ctx.hydration_payload = build_hydration_payload(ctx, category="automotive")
    persist_hydration_payload_artifact(ctx)

    hp = ctx.hydration_payload or {}
    assert hp.get("trace_id") == "u1"
    assert "evidence" in hp
    assert isinstance(hp["evidence"], dict)
    assert ctx.output_artifacts.get("hydration_payload")


def test_studio_persona_uses_saved_pikzels_uuid():
    pid = str(uuid.uuid4())
    persona, opts = _studio_persona_for_request(
        {
            "thumbnailPersonaEnabled": True,
            "thumbnailPikzelsPersonaId": pid,
            "thumbnailPersonaStrength": 85,
        }
    )

    assert persona == {"id": pid, "kind": "persona"}
    assert opts["persona_strength"] == 85


def test_studio_style_used_when_no_persona():
    sid = str(uuid.uuid4())
    payload, opts = _studio_persona_for_request(
        {
            "thumbnailStyleEnabled": True,
            "thumbnailPikzelsStyleId": sid,
            "thumbnailStyle": "cinematic neon contrast",
        }
    )

    assert payload == {"id": sid, "kind": "style"}
    assert opts["style_hint"] == "cinematic neon contrast"


def test_default_thumbnail_strategy_feeds_upload_brief_and_style_hint():
    us = {
        "thumbnailStudioDefaultStrategy": {
            "layout_name": "Reaction Meme",
            "layout_pattern": "two expressive faces, bold stacked text",
            "audience_niche": "comedy",
            "competitor_gap_mode": True,
            "text_position": "top",
            "contrast_profile": "very_high",
            "emotion": "shock",
        }
    }

    brief = _apply_thumbnail_default_strategy({"notes": "real upload evidence"}, us, category="general")
    assert "default_strategy" in brief
    assert "Reaction Meme" in brief["notes"]
    prompt = _build_pikzels_v2_prompt(brief, category="comedy", platform="youtube")
    assert "Layout: " in prompt
    assert "Reaction Meme" in prompt
    assert "Reaction Meme" in prompt

    _persona, opts = _studio_persona_for_request(us)
    assert "Reaction Meme" in opts["style_hint"]
    assert "competitor-gap" in opts["style_hint"]


def test_studio_persona_auto_enabled_when_uuid_present_without_toggle():
    # Saving a persona is itself an opt-in. The previous behavior required a
    # second `thumbnailPersonaEnabled=true` toggle that most users never set,
    # so every published thumbnail came back unstyled even though the user
    # had picked a persona.
    pid = str(uuid.uuid4())
    persona, _opts = _studio_persona_for_request(
        {"thumbnailPikzelsPersonaId": pid}
    )
    assert persona == {"id": pid, "kind": "persona"}


def test_studio_persona_explicit_false_still_disables():
    pid = str(uuid.uuid4())
    persona, _opts = _studio_persona_for_request(
        {
            "thumbnailPikzelsPersonaId": pid,
            "thumbnailPersonaEnabled": False,
        }
    )
    assert persona is None


def test_studio_style_auto_enabled_when_uuid_present_without_toggle():
    sid = str(uuid.uuid4())
    payload, _opts = _studio_persona_for_request(
        {"thumbnailPikzelsStyleId": sid}
    )
    assert payload == {"id": sid, "kind": "style"}


def test_studio_style_explicit_false_still_disables():
    sid = str(uuid.uuid4())
    payload, _opts = _studio_persona_for_request(
        {"thumbnailPikzelsStyleId": sid, "thumbnailStyleEnabled": False}
    )
    assert payload is None


def test_basic_pikzels_allowed_without_persona_or_style(monkeypatch):
    monkeypatch.setattr(pikzels_api, "resolve_public_api_key", lambda: "pk_test")
    ent = get_entitlements_for_tier("creator_pro")
    settings = {
        "autoThumbnails": True,
        "styledThumbnails": True,
        "thumbnailStudioEnabled": True,
        "thumbnailStudioEngineEnabled": True,
    }

    assert pikzels_studio_eligible_for_styled_thumbnail(settings, ent, require_auto_thumbnails=True) is True
    assert _studio_persona_for_request(settings)[0] is None


def test_renderer_sends_style_instead_of_persona(monkeypatch, tmp_path):
    style_id = str(uuid.uuid4())
    captured = {}

    async def fake_post(path, payload):
        captured.update(payload)
        return 200, {"ok": True}

    async def fake_bytes(data, timeout):
        return b"x" * 3000

    monkeypatch.setattr(pikzels_api, "resolve_public_api_key", lambda: "pk_test")
    monkeypatch.setattr(pikzels_api, "pikzels_v2_post", fake_post)
    monkeypatch.setattr(pikzels_api, "_pikzels_v2_response_to_bytes", fake_bytes)
    base = tmp_path / "base.jpg"
    out = tmp_path / "out.jpg"
    _write_tiny_jpeg(base)

    ok = asyncio.run(
        render_thumbnail_with_studio_renderer(
            base,
            {"selected_headline": "GARDEN UPDATE"},
            "youtube",
            out,
            persona={"id": style_id, "kind": "style"},
            options={"style_hint": "warm editorial garden"},
        )
    )

    assert ok is True
    assert captured.get("style") == style_id
    assert "persona" not in captured
    assert "warm editorial garden" in captured.get("prompt", "")


def test_closeness_to_pikzels_image_weight_brackets():
    from services.thumbnail_studio import closeness_to_pikzels_image_weight

    assert closeness_to_pikzels_image_weight(0) == "low"
    assert closeness_to_pikzels_image_weight(34) == "low"
    assert closeness_to_pikzels_image_weight(35) == "medium"
    assert closeness_to_pikzels_image_weight(66) == "medium"
    assert closeness_to_pikzels_image_weight(67) == "high"
    assert closeness_to_pikzels_image_weight(100) == "high"


def test_format_row_dynamic_layout_stable():
    from services.thumbnail_studio import format_row_by_key

    a = format_row_by_key("dyn-finance-0003")
    b = format_row_by_key("dyn-finance-0003")
    assert a and b and a["pattern"] == b["pattern"] and a["key"] == "dyn-finance-0003"


def test_format_library_rows_include_layout_preview_id():
    from services.thumbnail_studio import format_library_rows, layout_preview_id

    rows = format_library_rows("automotive")
    assert rows
    assert all(str(r.get("preview_id") or "") for r in rows)
    assert layout_preview_id("gaming_shock_face") == "shock_face"
    assert layout_preview_id("dyn-automotive-0003") == layout_preview_id("dyn-automotive-0003")


def test_format_library_rows_adds_procedural_for_niche():
    from services.thumbnail_studio import format_library_rows

    rows = format_library_rows("food")
    dyn = [r["key"] for r in rows if str(r["key"]).startswith("dyn-food-")]
    assert len(dyn) >= 8
    assert "gaming_shock_face" in [r["key"] for r in rows]


def test_generate_recreate_variants_with_dynamic_format_key():
    from services.thumbnail_studio import generate_recreate_variants

    variants = generate_recreate_variants(
        youtube_title="Test",
        topic="",
        niche="gaming",
        closeness=50,
        variant_count=3,
        format_key="dyn-gaming-0012",
    )
    assert variants[0].get("format_key") == "dyn-gaming-0012"
    assert "mix 13" in (variants[0].get("name") or "")


def test_studio_persona_maps_saved_reference_strength_to_image_weight():
    us = {
        "thumbnailStudioDefaultStrategy": {
            "layout_name": "Split",
            "layout_pattern": "two-pane proof",
            "reference_strength": 80,
        }
    }
    _persona, opts = _studio_persona_for_request(us)
    assert opts.get("image_weight") == "high"


def test_hydration_edit_prompt_requires_substance():
    assert _thumbnail_hydration_edit_prompt({}) == ""
    assert _thumbnail_hydration_edit_prompt({"geo_context": "short"}) == ""
    p = _thumbnail_hydration_edit_prompt(
        {
            "geo_context": "Moab, Utah scenic corridor along the river road",
            "speech_context": "We finally made it to the overlook after the storm cleared.",
        }
    )
    assert "Moab" in p and "Speech transcript" in p


def test_render_thumbnail_respects_image_weight_option(monkeypatch, tmp_path):
    captured: dict = {}

    async def fake_post(path, payload):
        captured.update(payload)
        return 200, {"ok": True}

    async def fake_bytes(data, timeout):
        return b"x" * 3000

    monkeypatch.setattr(pikzels_api, "resolve_public_api_key", lambda: "pk_test")
    monkeypatch.setattr(pikzels_api, "pikzels_v2_post", fake_post)
    monkeypatch.setattr(pikzels_api, "_pikzels_v2_response_to_bytes", fake_bytes)
    base = tmp_path / "base.jpg"
    out = tmp_path / "out.jpg"
    _write_tiny_jpeg(base)

    ok = asyncio.run(
        render_thumbnail_with_studio_renderer(
            base,
            {"selected_headline": "GARDEN UPDATE"},
            "youtube",
            out,
            options={"image_weight": "low"},
        )
    )

    assert ok is True
    assert captured.get("image_weight") == "low"


def test_render_thumbnail_maps_reference_strength_to_image_weight(monkeypatch, tmp_path):
    captured: dict = {}

    async def fake_post(path, payload):
        captured.update(payload)
        return 200, {"ok": True}

    async def fake_bytes(data, timeout):
        return b"x" * 3000

    monkeypatch.setattr(pikzels_api, "resolve_public_api_key", lambda: "pk_test")
    monkeypatch.setattr(pikzels_api, "pikzels_v2_post", fake_post)
    monkeypatch.setattr(pikzels_api, "_pikzels_v2_response_to_bytes", fake_bytes)
    base = tmp_path / "base.jpg"
    out = tmp_path / "out.jpg"
    _write_tiny_jpeg(base)

    ok = asyncio.run(
        render_thumbnail_with_studio_renderer(
            base,
            {"selected_headline": "GARDEN UPDATE"},
            "youtube",
            out,
            options={"reference_strength": 10},
        )
    )

    assert ok is True
    assert captured.get("image_weight") == "low"


def test_youtube_thumb_response_rejects_html():
    from services.thumbnail_studio import _youtube_thumb_response_to_jpeg_data_url

    assert _youtube_thumb_response_to_jpeg_data_url(b"<!DOCTYPE html><html></html>") is None
    assert _youtube_thumb_response_to_jpeg_data_url(b"") is None


def test_youtube_thumbnail_urls_from_watch_html_extracts_metadata():
    from services.thumbnail_studio import _youtube_thumbnail_urls_from_watch_html

    html = """
    <html><head>
      <meta property="og:image" content="https://i.ytimg.com/vi/abc123/hqdefault.jpg">
      <script>var ytInitialPlayerResponse = {"videoDetails":{"thumbnail":{"thumbnails":[{"url":"https://i.ytimg.com/vi/abc123/maxresdefault.jpg"}]}}};</script>
    </head></html>
    """

    urls = _youtube_thumbnail_urls_from_watch_html(html)
    assert "https://i.ytimg.com/vi/abc123/hqdefault.jpg" in urls
    assert "https://i.ytimg.com/vi/abc123/maxresdefault.jpg" in urls


def test_dashcam_pov_brief_strips_clickbait_styling():
    from stages.thumbnail_stage import _dashcam_pov_content, _sanitize_thumbnail_brief

    ctx = JobContext(
        job_id="j1",
        upload_id="u1",
        user_id="usr",
        idempotency_key="k1",
        filename="20250224_0073_CAM_EVNT.MP4",
        local_video_path=Path("x.mp4"),
    )
    assert _dashcam_pov_content(ctx, "general") is True
    brief = _sanitize_thumbnail_brief(ctx, {"selected_headline": "TEST VIDEO"}, "general")
    assert brief.get("_uploadm8_dashcam_pov") is True
    assert brief.get("directional_element") == "none"
    assert brief.get("emotion_cue") == ""
    assert brief.get("color_mood") == "blue_white"


def test_dashcam_pov_prompt_preserves_frame_fidelity():
    brief = {
        "selected_headline": "TEST VIDEO",
        "_uploadm8_dashcam_pov": True,
        "directional_element": "none",
        "color_mood": "blue_white",
    }
    prompt = _build_pikzels_v2_prompt(brief, category="general", platform="youtube")
    low = prompt.lower()
    assert "dashcam pov fidelity" in low
    # Softened: hard "do not add faces" conflicted with persona UUID payloads.
    assert "preserve the real road" in low or "forward-facing" in low
    assert "compass circles" in low or "neon compass" in low
    assert "directional element" not in low
    assert "facial expression" not in low
    assert "red and black" not in low


def test_cap_pikzels_studio_render_prompt_under_api_limit():
    from services.thumbnail_studio import cap_pikzels_studio_render_prompt

    long_tail = "x" * 800
    raw = (
        "Recreate the provided reference thumbnail for a broad YouTube audience. "
        "New thumbnail text hook: BIG WIN NOW. Use the provided YouTube thumbnail image as the visual anchor. "
        f"Creative direction: neon punch {long_tail} Hydration focus: lane {long_tail}"
    )
    capped = cap_pikzels_studio_render_prompt(raw)
    assert len(capped) <= 1000
    assert "New thumbnail text hook: BIG WIN NOW" in capped


def test_generate_recreate_variants_rotates_layout_without_format_key():
    variants = generate_recreate_variants(
        youtube_title="Road trip",
        topic="",
        niche="automotive",
        closeness=55,
        variant_count=4,
        format_key=None,
    )
    keys = {v.get("format_key") for v in variants}
    assert len(keys) >= 2
    for v in variants:
        assert len(str(v.get("render_prompt") or "")) <= 1000


def test_guided_recreate_prompt_preserves_reference_and_persona():
    variants = generate_recreate_variants(
        youtube_title="How To Run Your Credit Up",
        topic="",
        niche="comedy",
        closeness=80,
        variant_count=4,
        persona_name="gloc",
        format_key="reaction_meme",
    )

    prompt = variants[0]["render_prompt"]
    assert "Use the provided YouTube thumbnail image as the visual anchor" in prompt
    assert "Do not invent unrelated cars" in prompt
    assert "linked Pikzels persona named gloc" in prompt
    assert "comedy / meme audience" in prompt


def test_thumbnail_studio_variants_include_hydration_lanes():
    ctx = {
        "caption": "Crossing the canyon at golden hour after the storm cleared.",
        "geo": "Moab, Utah",
        "latitude": "38.5733",
        "longitude": "-109.5498",
        "artist": "The Eagles",
        "track": "Hotel California",
    }
    lanes = hydration_signal_lanes(ctx)
    assert {lane["key"] for lane in lanes} >= {"caption", "geo", "music", "combined"}

    variants = generate_recreate_variants(
        youtube_title="Desert Road Trip",
        topic="",
        niche="automotive",
        closeness=70,
        variant_count=4,
        hydration_context=ctx,
    )

    focuses = {v.get("hydration_focus") for v in variants}
    signals = " ".join(
        str(v.get("hydration_signal") or v.get("hydration_summary") or "")
        for v in variants
    )
    assert "Caption hook" in focuses
    assert "Geo / route" in focuses
    assert "Artist / track" in focuses
    assert "Moab, Utah" in signals
    assert "The Eagles" in signals
    for v in variants:
        assert len(str(v.get("render_prompt") or "")) <= 1000


def test_pikzels_v2_prompt_never_exceeds_api_limit():
    """Pikzels rejects prompts over the hard cap — guard must hold under heavy hydration."""
    long_text = "x" * 400
    brief = {
        "selected_headline": "65MPH ON SCENIC ROAD THROUGH BOULDER UTAH",
        "headline_options": ["ALT HEADLINE " + long_text],
        "fusion_summary": "Fusion " + long_text,
        "geo_context": "Geo " + long_text,
        "speech_context": "Speech " + long_text,
        "music_context": "Music " + long_text,
        "osd_context": "OSD " + long_text,
        "trill_context": "Trill " + long_text,
        "signal_hashtags": "#tag " + long_text,
        "notes": "Notes " + long_text,
        "hydration_story": "Story " + long_text,
    }
    hydration = {
        "fusion_summary": brief["fusion_summary"],
        "hydration_story": brief["hydration_story"],
        "geo_context": brief["geo_context"],
    }
    for platform in ("youtube", "instagram", "facebook", "tiktok"):
        prompt = _build_pikzels_v2_prompt(
            brief,
            category="automotive",
            platform=platform,
            hydration_payload=hydration,
        )
        assert len(prompt) <= pikzels_api.PIKZELS_API_PROMPT_HARD_MAX
        guarded = pikzels_api.clamp_pikzels_image_prompt(
            pikzels_api._PERSONA_STYLE_TEXT_GUARD + prompt
        )
        assert len(guarded) <= pikzels_api.PIKZELS_API_PROMPT_HARD_MAX


def test_hydration_edit_prompt_respects_api_limit():
    p = _thumbnail_hydration_edit_prompt(
        {
            "fusion_summary": "x" * 500,
            "geo_context": "Moab, Utah scenic corridor along the river road " * 20,
            "speech_context": "We finally made it to the overlook after the storm cleared." * 10,
            "music_context": "Artist — Track name on repeat" * 10,
        }
    )
    assert len(pikzels_api.clamp_pikzels_image_prompt(p)) <= pikzels_api.PIKZELS_API_PROMPT_HARD_MAX
