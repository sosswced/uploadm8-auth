"""Soccer / kit-color identity must reach hydration titles and thumbnails."""

from __future__ import annotations

from types import SimpleNamespace

from core.sports_identity import (
    infer_sports_identity,
    sports_story_clause,
    sports_title_phrase,
)
from services.place_evidence import extract_place_evidence
from stages.context import build_hydration_story_text
from stages.thumbnail_stage import _concrete_thumbnail_headline


def _ctx(**overrides) -> SimpleNamespace:
    base = dict(
        filename="IMG_5135.MOV",
        title="IMG_5135.MOV",
        caption="",
        ai_title="",
        ai_caption="",
        ai_transcript="",
        vision_context={},
        audio_context={},
        telemetry=None,
        telemetry_data=None,
        dashcam_osd_context={},
        video_intelligence={},
        video_intelligence_context={},
        video_understanding={},
        hydration_payload={},
        thumbnail_category="general",
        output_artifacts={},
        content_identity={},
        place_evidence={},
    )
    base.update(overrides)
    ns = SimpleNamespace(**base)
    ns.get_effective_title = lambda: ""
    ns.get_effective_caption = lambda: ""
    return ns


def test_kit_colors_plus_soccer_stadium_name_barca():
    ctx = _ctx(
        vision_context={
            "label_names": ["Soccer", "Stadium", "Sports", "Jersey"],
            "logo_names": [],
            "landmark_names": [],
            "web_entities": [],
            "ocr_text": "",
            "dominant_colors": [
                {"name": "red", "rgb": [190, 20, 40], "score": 0.22},
                {"name": "blue", "rgb": [20, 40, 170], "score": 0.18},
            ],
        }
    )
    ident = infer_sports_identity(ctx)
    assert ident["sport_kind"] == "soccer"
    assert "FC Barcelona" in ident["sports_teams"]
    assert ident["kit_match"] == "FC Barcelona"
    report = extract_place_evidence(ctx)
    assert "FC Barcelona" in (report.get("sports_teams") or [])
    story = build_hydration_story_text(ctx)
    assert "soccer" in story.lower()
    assert "barcelona" in story.lower()
    headline = _concrete_thumbnail_headline(ctx, "general")
    assert "BARCELONA" in headline.upper()
    assert "IMG" not in headline.upper()
    assert "HYDRATION" not in headline.upper()


def test_red_blue_without_soccer_does_not_invent_barca():
    ctx = _ctx(
        vision_context={
            "label_names": ["Person", "Night", "Clothing"],
            "dominant_colors": [
                {"name": "red", "rgb": [190, 20, 40], "score": 0.3},
                {"name": "blue", "rgb": [20, 40, 170], "score": 0.2},
            ],
        }
    )
    ident = infer_sports_identity(ctx)
    assert "FC Barcelona" not in (ident.get("sports_teams") or [])
    report = extract_place_evidence(ctx)
    assert not (report.get("sports_teams") or [])


def test_camp_nou_web_entity_names_venue_and_club():
    ctx = _ctx(
        vision_context={
            "label_names": ["Stadium", "Person"],
            "web_entities": ["Camp Nou", "FC Barcelona"],
            "landmark_names": [],
            "logo_names": [],
            "ocr_text": "",
        }
    )
    ident = infer_sports_identity(ctx)
    assert "Camp Nou" in ident["stadiums"]
    assert "FC Barcelona" in ident["sports_teams"]
    assert sports_title_phrase(ident) == "FC Barcelona at Camp Nou"
    report = extract_place_evidence(ctx)
    assert any("camp nou" in str(s).lower() for s in (report.get("stadiums") or []))


def test_service_web_entity_titles_unknown_club_without_enum():
    """Vision web names must title the clip even if we never listed that club."""
    from core.upload_domain_plan import compose_service_title, detect_planned_domain

    ctx = _ctx(
        thumbnail_category="sports",
        vision_context={
            "label_names": ["Soccer", "Stadium"],
            "web_entities": ["Sporting CP", "Estádio José Alvalade"],
            "logo_names": [],
            "landmark_names": [],
            "ocr_text": "",
        },
    )
    assert detect_planned_domain(ctx) == "sports"
    title = compose_service_title(ctx)
    assert "Sporting" in title
    assert "IMG" not in title


def test_food_domain_uses_vision_web_name():
    from core.upload_domain_plan import compose_service_title, detect_planned_domain

    ctx = _ctx(
        thumbnail_category="food",
        vision_context={
            "label_names": ["Food", "Restaurant"],
            "web_entities": ["In-N-Out Burger"],
            "logo_names": ["In-N-Out Burger"],
        },
    )
    assert detect_planned_domain(ctx) == "food"
    assert "In-N-Out" in compose_service_title(ctx)


def test_sports_story_clause_mentions_kit_colors():
    clause = sports_story_clause(
        {
            "sport_kind": "soccer",
            "sports_teams": ["FC Barcelona"],
            "stadiums": ["Camp Nou"],
            "kit_colors": ["red", "blue"],
        }
    )
    assert "soccer" in clause.lower()
    assert "FC Barcelona" in clause
    assert "red" in clause and "blue" in clause


def test_invented_person_and_filename_hashtags_are_junk():
    from core.vision_labels import is_invented_person_hashtag, is_junk_hashtag_body

    assert is_invented_person_hashtag("LuisMontosChuty")
    assert is_junk_hashtag_body("LuisMontosChuty")
    assert is_junk_hashtag_body("img5135")
    assert is_junk_hashtag_body("IMG_5135")
    assert not is_invented_person_hashtag("FCBarcelona")
    assert not is_invented_person_hashtag("CampNou")
    assert not is_invented_person_hashtag("Barca")


def test_soccer_discovery_hashtags_fill_to_max_without_name_mash():
    from core.upload_domain_plan import discovery_hashtags_for_upload
    from services.hydration_enforcer import enforce_hydration
    from stages.context import JobContext

    ctx = JobContext(
        job_id="j-soccer",
        upload_id="u-soccer",
        user_id="u-soccer",
        filename="IMG_5135.MOV",
        title="IMG_5135.MOV",
        ai_title="Celebrating Barça's Representation",
        ai_caption=(
            "Join the electric atmosphere as Luis Montos Chuty proudly stands "
            "as Barça's representative. ⚽️ #LuisMontosChuty #Barça"
        ),
        ai_hashtags=["LuisMontosChuty", "Barça"],
        m8_platform_hashtags={"instagram": ["LuisMontosChuty", "Barça"]},
        m8_platform_captions={
            "instagram": (
                "Join the electric atmosphere as Luis Montos Chuty proudly stands "
                "as Barça's representative. ⚽️ #LuisMontosChuty #Barça"
            )
        },
        user_settings={"maxHashtags": 15},
        vision_context={
            "label_names": ["Soccer", "Stadium", "Sports", "Jersey"],
            "web_entities": ["FC Barcelona"],
            "logo_names": ["FC Barcelona"],
            "ocr_text": "FC BARCELONA HYDRATION POINT REPRESENTANTE CHUTY",
            "dominant_colors": [
                {"name": "red", "rgb": [190, 20, 40], "score": 0.22},
                {"name": "blue", "rgb": [20, 40, 170], "score": 0.18},
            ],
        },
    )
    ctx.thumbnail_category = "sports"
    discovered = discovery_hashtags_for_upload(ctx, limit=15)
    low = {t.lower() for t in discovered}
    assert len(discovered) >= 10
    assert "soccer" in low
    assert "football" in low
    assert "fcbarcelona" in low or "barca" in low
    assert "luismontoschuty" not in low
    assert "img5135" not in low

    enforce_hydration(ctx)
    final = [
        str(t).lower().lstrip("#")
        for t in (ctx.m8_platform_hashtags.get("instagram") or ctx.ai_hashtags or [])
    ]
    assert len(final) == 15
    assert "soccer" in final
    assert "luismontoschuty" not in final
    assert "img5135" not in final
    caption = ctx.get_effective_caption("instagram")
    assert "#LuisMontosChuty" not in caption
    assert "5135" not in caption
