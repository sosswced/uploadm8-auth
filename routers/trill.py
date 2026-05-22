"""
UploadM8 Trill Telemetry routes -- extracted from app.py.

Handles trill analysis, places lookup, and AI content preview generation.

**Architecture (important):**

- **Full upload jobs (worker.py)** — Authoritative path for publish-ready titles,
  captions, and hashtags: runs audio (Whisper/ACR/YAMNet), Google Vision, Twelve Labs,
  Video Intelligence, dashcam OSD (burned-in HUD read; GPS backfill only without .map),
  telemetry + PADUS/gazetteer on .map routes,
  then ``run_caption_stage`` (M8 scene graph) and ``merge_signal_hashtags_into_ctx``.

- **Trill HTTP API (this router)** — Fast preview for Drive / map + video: uses
  ``telemetry_trill.safe_analyze_video`` (Trill score + **US Census gazetteer** when
  ``GAZETTEER_PLACES_PATH`` exists, plus **PAD-US** from PostGIS when the DB has
  ``padus_protected_areas``) and optional
  ``generate_trill_content`` (OpenAI from those metrics only). It does **not** run
  Vision, Whisper, Twelve Labs, or Video Intelligence (avoid duplicate cost and long
  requests on a preview endpoint). For full multimodal copy, process the upload through
  the worker so captions use M8 + all signals.
"""

import hashlib
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from math import radians, sin, cos, sqrt, atan2
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

import core.state
from core.config import (
    COST_PER_OPENAI_TOKEN,
    GAZETTEER_PLACES_PATH,
    OPENAI_API_KEY,
    R2_BUCKET_NAME,
    TRILL_SYSTEM_PROMPT,
    ADMIN_DISCORD_WEBHOOK_URL,
)
from core.deps import get_current_user, get_verified_user_id, require_verified_user_on_conn
from core.r2 import get_s3_client
from core.notifications import discord_notify
from services.ops_incidents import record_operational_incident
from core.wallet import credit_wallet
from services.trill_access import TRILL_ROUTE_PREDICATE
from services.trill_engagement import (
    add_rival,
    archive_due_seasons,
    award_badge,
    check_challenge_for_user,
    compute_chase_targets,
    ensure_badge_definitions,
    ensure_current_season,
    ensure_weekly_challenge,
    evaluate_and_award_badges,
    fetch_badge_catalog,
    fetch_badges_for_users,
    fetch_hall_of_fame,
    fetch_recent_scores_batch,
    fetch_region_options,
    fetch_rivals,
    fetch_user_badge_collection,
    process_rival_rank_changes,
    public_driver_id,
    remove_rival,
    resolve_user_from_public_id,
)
from services.trill_vehicle_filter import trill_vehicle_where_fragment
from services.vehicle_catalog import fetch_vehicle_labels

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/trill", tags=["trill"])


# ============================================================
# Trill Helpers
# ============================================================

def generate_trill_content(trill_metadata: dict, user_prefs: dict = None) -> dict:
    """
    Use OpenAI to generate titles, captions, and hashtags from **Trill telemetry analysis**.

    ``trill_metadata`` comes from ``telemetry_trill.safe_analyze_video`` (includes
    Census nearest place, PAD-US protected-lands hit (PostGIS), speed, curvature, etc.
    when configured). This is **not** the full upload worker pipeline
    (no Vision / audio stack / M8). See module docstring above.

    Args:
        trill_metadata: ``telemetry_trill.safe_analyze_video(...)["data"]`` dict
        user_prefs: User preferences for generation style

    Returns:
        {
            "title": "Generated title",
            "caption": "Generated caption",
            "hashtags": ["tag1", "tag2", ...],
            "tokens_used": 150,
            "model": "gpt-4o-mini"
        }
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not configured")

    import openai
    openai.api_key = OPENAI_API_KEY

    # Extract key metrics
    score = trill_metadata.get("trill_score", 0)
    bucket = trill_metadata.get("speed_bucket", "CRUISE MODE")
    place = trill_metadata.get("place_name", "")
    state = trill_metadata.get("state", "")
    protected_name = trill_metadata.get("protected_name")
    near_protected = trill_metadata.get("near_protected", False)
    elev_gain = trill_metadata.get("elev_gain_m", 0)
    curv_score = trill_metadata.get("curv_score", 0)
    dyn_score = trill_metadata.get("dyn_score", 0)
    turny = trill_metadata.get("turny", False)
    spirited = trill_metadata.get("spirited", False)
    state_usps = (trill_metadata.get("state_usps") or "").strip()
    max_mph = trill_metadata.get("max_speed_mph")
    avg_mph = trill_metadata.get("avg_speed_mph")
    dist_mi = trill_metadata.get("distance_miles")
    dur_s = trill_metadata.get("duration_seconds")

    # Build context (place_name = US Census gazetteer nearest place when gazetteer path was used)
    location = f"near {place}, {state}" if place and state else state if state else "the open road"
    if state_usps and state_usps not in location:
        location = f"{location} (USPS {state_usps})".strip()
    scene = f"{protected_name} (verified protected lands)" if near_protected and protected_name else "public lands" if near_protected else "backroads"

    # User preferences
    prefs = user_prefs or {}
    model = prefs.get("trill_openai_model", "gpt-4o-mini")

    # Build prompt
    user_prompt = f"""Write title, caption, and hashtags for a driving clip using only these facts (do not invent
scenes or stunts). Sound like a real person posting their footage — no emojis anywhere.

TRILL SCORE: {score}/100 (higher = more thrilling)
SPEED BUCKET: {bucket}
CENSUS NEAREST PLACE (when gazetteer configured on server): {place or "(not resolved)"}
STATE / REGION: {state or "(unknown)"}
LOCATION LINE: {location}
PADUS / PROTECTED LANDS: {scene} (near_protected={near_protected})
ROUTE STATS: peak_speed_mph={max_mph} avg_speed_mph={avg_mph} distance_miles={dist_mi} duration_s={dur_s}
ELEVATION GAIN: {elev_gain}m
CURVATURE: {curv_score}/10 (higher = more twisty/switchbacks)
DYNAMICS: {dyn_score}/10 (higher = more spirited cornering)
MOTION FLAGS: {"Turny roads" if turny else ""} {"Spirited driving" if spirited else ""}

GENERATE:
1. TITLE (max 80 chars)
   - Specific to location/scene/motion flags above; avoid generic AI patterns ("POV:", "Wait for it", "Nobody talks about")
   - No emojis or emoticons; normal capitalization (not Title Case Every Word)

2. CAPTION (max 200 chars)
   - First-person or direct address only if it fits the tone; conversational, not salesy
   - Optional one short question; no FOMO clichés; no emojis

3. HASHTAGS (exactly 15 tags)
   - 3-4 mega viral: #fyp #foryou #viral #trending
   - 4-5 niche community: #roadtrip #driving #explore
   - 3-4 location: #{state} #{place} (if available)
   - 2-3 motion: #curvyroads #spiriteddrive (if applicable)
   - 1-2 protected lands: #publiclands #nationalpark (ONLY if verified: {near_protected})

RETURN ONLY THIS JSON (no markdown, no backticks):
{{
  "title": "your title here",
  "caption": "your caption here",
  "hashtags": ["tag1", "tag2", ...]
}}
"""

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": TRILL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.85,
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        # Parse response
        result = json.loads(content)

        # Validate and clean hashtags
        hashtags = result.get("hashtags", [])
        hashtags = [h.lower().replace("#", "").replace(" ", "") for h in hashtags]
        hashtags = [h for h in hashtags if h and len(h) <= 30][:15]

        return {
            "title": result.get("title", "")[:100],
            "caption": result.get("caption", "")[:250],
            "hashtags": hashtags,
            "tokens_used": tokens_used,
            "model": model,
            "trill_score": score,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"OpenAI generation failed: {e}")
        return {
            "title": trill_metadata.get("title", ""),
            "caption": trill_metadata.get("caption", ""),
            "hashtags": trill_metadata.get("hashtags", []),
            "tokens_used": 0,
            "model": "fallback",
            "error": str(e)
        }


async def seed_trill_places(conn):
    """Seed database with popular driving locations"""
    places = [
        {"name": "Moab", "state": "UT", "lat": 38.5733, "lon": -109.5498, "popularity": 95, "protected_name": "Arches National Park"},
        {"name": "Zion", "state": "UT", "lat": 37.2982, "lon": -113.0263, "popularity": 90, "protected_name": "Zion National Park"},
        {"name": "Big Sur", "state": "CA", "lat": 36.2704, "lon": -121.8081, "popularity": 92, "protected_name": "Los Padres National Forest"},
        {"name": "Malibu", "state": "CA", "lat": 34.0259, "lon": -118.7798, "popularity": 85},
        {"name": "Yosemite", "state": "CA", "lat": 37.8651, "lon": -119.5383, "popularity": 88, "protected_name": "Yosemite National Park"},
        {"name": "Rocky Mountain NP", "state": "CO", "lat": 40.3428, "lon": -105.6836, "popularity": 87, "protected_name": "Rocky Mountain National Park"},
        {"name": "Sedona", "state": "AZ", "lat": 34.8697, "lon": -111.7610, "popularity": 88, "protected_name": "Coconino National Forest"},
        {"name": "Grand Canyon", "state": "AZ", "lat": 36.1069, "lon": -112.1129, "popularity": 95, "protected_name": "Grand Canyon National Park"},
        {"name": "Glacier National Park", "state": "MT", "lat": 48.7596, "lon": -113.7870, "popularity": 85, "protected_name": "Glacier National Park"},
        {"name": "Yellowstone", "state": "WY", "lat": 44.4280, "lon": -110.5885, "popularity": 92, "protected_name": "Yellowstone National Park"},
        {"name": "Blue Ridge Parkway", "state": "NC", "lat": 35.5951, "lon": -82.5515, "popularity": 82, "protected_name": "Blue Ridge Parkway"},
        {"name": "Tail of the Dragon", "state": "NC", "lat": 35.5159, "lon": -83.9293, "popularity": 90},
    ]

    for p in places:
        hashtags = [p["name"].lower().replace(" ", ""), f"{p['state'].lower()}roadtrip"]
        if p.get("protected_name"):
            hashtags.extend(["publiclands", "nationalpark"])

        try:
            await conn.execute("""
                INSERT INTO trill_places (name, state, lat, lon, popularity_score, is_protected, protected_name, hashtags)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (name, state) DO UPDATE SET
                    popularity_score = EXCLUDED.popularity_score,
                    is_protected = EXCLUDED.is_protected,
                    protected_name = EXCLUDED.protected_name,
                    hashtags = EXCLUDED.hashtags,
                    updated_at = NOW()
            """, p["name"], p["state"], p["lat"], p["lon"], p.get("popularity", 50),
                 bool(p.get("protected_name")), p.get("protected_name"), hashtags)
        except Exception as e:
            logger.error(f"Failed to seed trill place {p['name']}: {e}")


async def get_nearby_trill_place(conn, lat: float, lon: float, max_distance_km: float = 50) -> Optional[dict]:
    """Find nearest popular trill place for geo-targeting"""
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in km
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        return 2 * R * atan2(sqrt(a), sqrt(1-a))

    places = await conn.fetch("SELECT * FROM trill_places")

    best = None
    min_dist = float('inf')

    for p in places:
        dist = haversine(lat, lon, float(p["lat"]), float(p["lon"]))
        if dist < min_dist and dist <= max_distance_km:
            min_dist = dist
            best = {**dict(p), "distance_km": dist}

    return best


async def process_telemetry(conn, upload_id: str, user_id: str, video_path: str, map_path: str, user_prefs: dict) -> dict:
    """
    Process telemetry data and generate content.

    Returns:
        {
            "trill_metadata": {...},
            "ai_content": {...},
        }
    """
    # Import trill module dynamically
    try:
        import telemetry_trill as tt
    except ImportError:
        raise HTTPException(503, "Telemetry processing not available - telemetry_trill.py not found")

    try:
        # Run trill analysis
        result = tt.safe_analyze_video(
            video_path,
            map_path,
            gaz_places_path=GAZETTEER_PLACES_PATH if (GAZETTEER_PLACES_PATH and os.path.exists(GAZETTEER_PLACES_PATH)) else None,
            padus_path=None,
            padus_layer=None,
        )

        if not result.get("ok"):
            raise Exception(result.get("error", "Analysis failed"))

        trill_data = result["data"]

        mid_lat = trill_data.get("place_lat")
        mid_lon = trill_data.get("place_lon")
        if mid_lat is not None and mid_lon is not None:
            from services.padus_db import merge_padus_enrichment_into_mapping, padus_hit_dict_from_db

            try:
                pad_extra = await padus_hit_dict_from_db(conn, float(mid_lat), float(mid_lon))
                merge_padus_enrichment_into_mapping(trill_data, pad_extra)
            except Exception as e:
                logger.debug("Trill PADUS DB enrichment skipped: %s", e)

        # Check if score meets minimum threshold
        trill_score = trill_data.get("trill_score", 0)
        min_score = user_prefs.get("trill_min_score", 60)

        # Enrich with nearby trill place if available
        if mid_lat is not None and mid_lon is not None:
            trill_place = await get_nearby_trill_place(conn, mid_lat, mid_lon)
            if trill_place:
                trill_data["trill_place"] = trill_place["name"]
                trill_data["trill_place_hashtags"] = trill_place.get("hashtags", [])

        # Generate AI content if enabled and score is high enough
        ai_content = None
        if user_prefs.get("trill_ai_enhance", True) and trill_score >= min_score:
            try:
                ai_content = generate_trill_content(trill_data, user_prefs)
            except Exception as e:
                logger.error(f"AI generation failed: {e}")
                ai_content = {
                    "title": trill_data.get("title"),
                    "caption": trill_data.get("caption"),
                    "hashtags": trill_data.get("hashtags", []),
                    "model": "trill_fallback"
                }

        # Store in database
        await conn.execute("""
            UPDATE uploads SET
                trill_score = $1,
                speed_bucket = $2,
                trill_metadata = $3,
                ai_generated_title = $4,
                ai_generated_caption = $5,
                ai_generated_hashtags = $6,
                updated_at = NOW()
            WHERE id = $7
        """,
            trill_score,
            trill_data.get("speed_bucket"),
            json.dumps(trill_data),
            ai_content.get("title") if ai_content else None,
            ai_content.get("caption") if ai_content else None,
            ai_content.get("hashtags") if ai_content else None,
            upload_id
        )

        # Track OpenAI costs if used
        if ai_content and ai_content.get("tokens_used"):
            cost = ai_content["tokens_used"] * COST_PER_OPENAI_TOKEN
            await conn.execute("""
                INSERT INTO cost_tracking (user_id, category, operation, tokens, cost_usd)
                VALUES ($1, 'openai', 'trill_generation', $2, $3)
            """, user_id, ai_content["tokens_used"], cost)

        return {
            "trill_metadata": trill_data,
            "ai_content": ai_content,
        }

    except Exception as e:
        logger.error(f"Telemetry processing failed: {e}")
        raise HTTPException(500, f"Telemetry processing failed: {str(e)}")


# ============================================================
# Trill Routes
# ============================================================

@router.post("/analyze/{upload_id}")
async def analyze_telemetry(upload_id: str, user: dict = Depends(get_current_user)):
    """Manually trigger trill analysis on an upload with telemetry data"""
    async with core.state.db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            "SELECT * FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"]
        )

        if not upload:
            raise HTTPException(404, "Upload not found")

        if not upload.get("telemetry_r2_key"):
            raise HTTPException(400, "No telemetry data for this upload")

        # Download files from R2
        s3 = get_s3_client()

        video_obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=upload["r2_key"])
        video_data = video_obj["Body"].read()

        telem_obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=upload["telemetry_r2_key"])
        telem_data = telem_obj["Body"].read()

        # Write to temp files
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vf:
            vf.write(video_data)
            video_path = vf.name

        with tempfile.NamedTemporaryFile(suffix=".map", delete=False) as tf:
            tf.write(telem_data)
            map_path = tf.name

        try:
            # Get user preferences
            prefs = await conn.fetchrow(
                "SELECT * FROM user_preferences WHERE user_id = $1",
                user["id"]
            )
            user_prefs = dict(prefs) if prefs else {}

            # Process
            result = await process_telemetry(
                conn, upload_id, user["id"],
                video_path, map_path, user_prefs
            )

            return {
                "success": True,
                "trill_score": result["trill_metadata"].get("trill_score"),
                "speed_bucket": result["trill_metadata"].get("speed_bucket"),
                "ai_enhanced": bool(result.get("ai_content")),
                "ai_content": result.get("ai_content")
            }
        finally:
            os.unlink(video_path)
            os.unlink(map_path)


@router.get("/places")
async def get_trill_places(
    state: Optional[str] = Query(None),
    limit: int = Query(20, le=100),
    user: dict = Depends(get_current_user)
):
    """Get popular trill places for targeting"""
    async with core.state.db_pool.acquire() as conn:
        if state:
            places = await conn.fetch(
                "SELECT * FROM trill_places WHERE state = $1 ORDER BY popularity_score DESC LIMIT $2",
                state.upper(), limit
            )
        else:
            places = await conn.fetch(
                "SELECT * FROM trill_places ORDER BY popularity_score DESC LIMIT $1",
                limit
            )

    return [dict(p) for p in places]


@router.post("/generate-preview")
async def generate_trill_preview(
    data: dict = Body(...),
    user: dict = Depends(get_current_user)
):
    """Preview AI-generated content without saving"""
    trill_metadata = data.get("trill_metadata", {})

    # Get user preferences
    async with core.state.db_pool.acquire() as conn:
        prefs = await conn.fetchrow(
            "SELECT * FROM user_preferences WHERE user_id = $1",
            user["id"]
        )
        user_prefs = dict(prefs) if prefs else {}

    result = generate_trill_content(trill_metadata, user_prefs)
    return result


# ── Map unlock: completed upload with Trill score + route evidence (.map telemetry) ─────────────
def _trill_since_dt(range_key: str) -> datetime:
    minutes = {"7d": 10080, "30d": 43200, "90d": 262800, "1y": 525600, "all": 525600}.get(
        (range_key or "30d").strip(), 43200
    )
    return datetime.now(timezone.utc) - timedelta(minutes=minutes)


_TRILL_DISPLAY_NAME_RE = re.compile(r"^[A-Za-z0-9 _\-]{2,32}$")


class TrillDisplayNameIn(BaseModel):
    name: str = Field(..., min_length=2, max_length=32)


@router.post("/display-name")
async def submit_trill_display_name(body: TrillDisplayNameIn, user: dict = Depends(get_current_user)):
    """Request a public Trill leaderboard display name (requires master admin approval)."""
    raw = (body.name or "").strip()
    if not _TRILL_DISPLAY_NAME_RE.match(raw):
        raise HTTPException(
            status_code=400,
            detail="Name must be 2-32 characters: letters, numbers, spaces, hyphen, underscore only.",
        )
    uid = user["id"]
    pool = core.state.db_pool
    prior_approved = None
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO user_preferences (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
            uid,
        )
        row = await conn.fetchrow(
            "SELECT trill_public_name, trill_public_name_pending, trill_public_name_status "
            "FROM user_preferences WHERE user_id = $1",
            uid,
        )
        if row:
            prior_approved = row.get("trill_public_name")
        await conn.execute(
            """
            UPDATE user_preferences
            SET trill_public_name_pending = $2,
                trill_public_name_status = 'pending',
                trill_public_name_rejection_reason = NULL,
                updated_at = NOW()
            WHERE user_id = $1
            """,
            uid,
            raw,
        )
        await conn.execute(
            """
            INSERT INTO support_messages (user_id, name, email, subject, message)
            VALUES ($1, $2, $3, $4, $5)
            """,
            uid,
            (user.get("name") or "").strip() or None,
            (user.get("email") or "").strip() or None,
            "[Trill leaderboard name] Pending approval",
            f"Proposed public name: {raw}\nUser id: {uid}\nPrior approved name: {prior_approved or '(none)'}",
        )

    await record_operational_incident(
        pool,
        source="app",
        incident_type="trill_leaderboard_name_request",
        subject=f"Trill leaderboard name request: {raw}",
        details={
            "user_id": str(uid),
            "email": user.get("email"),
            "proposed_name": raw,
            "prior_approved": prior_approved,
        },
        user_id=uid,
        bypass_dedupe=True,
    )
    if ADMIN_DISCORD_WEBHOOK_URL:
        try:
            await discord_notify(
                ADMIN_DISCORD_WEBHOOK_URL,
                embeds=[
                    {
                        "title": "Trill leaderboard name request",
                        "color": 0xA855F7,
                        "fields": [
                            {"name": "User", "value": f"{user.get('email','')} `{uid}`"},
                            {"name": "Proposed", "value": raw[:256]},
                        ],
                    }
                ],
            )
        except Exception as e:
            logger.debug("trill display-name discord: %s", e)

    return {"ok": True, "status": "pending", "pending": raw}


@router.get("/onboarding-state")
async def trill_onboarding_state(user: dict = Depends(get_current_user)):
    """First-success Trill welcome + sidebar unlock (map-backed telemetry)."""
    uid = user["id"]
    map_unlocked = False
    prefs = None
    try:
        async with core.state.db_pool.acquire() as conn:
            map_unlocked = bool(
                await conn.fetchval(
                    f"""
                    SELECT EXISTS (
                        SELECT 1 FROM uploads u
                        WHERE u.user_id = $1
                        AND {TRILL_ROUTE_PREDICATE.strip()}
                    )
                    """,
                    uid,
                )
            )
            try:
                prefs = await conn.fetchrow(
                    "SELECT * FROM user_preferences WHERE user_id = $1",
                    uid,
                )
            except Exception:
                prefs = None
    except Exception as e:
        logger.warning("trill onboarding-state: %s", e)
        return {
            "map_unlocked": False,
            "show_first_trill_welcome": False,
            "trill_leaderboard_opt_in": False,
            "trill_map_sharing_opt_in": False,
            "trill_public_name": None,
            "trill_public_name_pending": None,
            "trill_public_name_status": "none",
            "trill_public_name_rejection_reason": None,
        }

    pd = dict(prefs) if prefs else {}
    seen_at = pd.get("trill_welcome_modal_seen_at")
    show_first = bool(map_unlocked) and seen_at is None
    return {
        "map_unlocked": map_unlocked,
        "show_first_trill_welcome": show_first,
        "trill_leaderboard_opt_in": bool(pd.get("trill_leaderboard_opt_in")),
        "trill_map_sharing_opt_in": bool(pd.get("trill_map_sharing_opt_in")),
        "trill_public_name": (str(pd.get("trill_public_name")).strip() if pd.get("trill_public_name") else None),
        "trill_public_name_pending": (
            str(pd.get("trill_public_name_pending")).strip() if pd.get("trill_public_name_pending") else None
        ),
        "trill_public_name_status": str(pd.get("trill_public_name_status") or "none"),
        "trill_public_name_rejection_reason": pd.get("trill_public_name_rejection_reason"),
    }


_LEADERBOARD_SORTS = frozenset({"best_trill", "speed", "runs", "distance", "avg_trill"})


def _format_leaderboard_region(place: Optional[str], state: Optional[str]) -> Optional[str]:
    """Human-readable region from place + state (worker telemetry blob or flat Trill API metadata)."""
    p = (place or "").strip()
    s = (state or "").strip()
    if p and s:
        if s.upper() in p.upper():
            return p
        return f"{p}, {s}"
    return p or s or None


def _lb_meta_state_sql(alias: str = "u") -> str:
    """USPS state from nested telemetry or legacy flat trill_metadata keys."""
    t = f"{alias}.trill_metadata"
    flat_state = "{state}"
    return f"""NULLIF(TRIM(COALESCE(
        NULLIF(btrim({t}#>>'{{telemetry,gazetteer_state_usps}}'), ''),
        NULLIF(btrim({t}#>>'{{telemetry,location_state}}'), ''),
        NULLIF(btrim({t}#>>'{{state_usps}}'), ''),
        NULLIF(btrim({t}#>>'{flat_state}'), '')
    )), ''), '')"""


def _lb_meta_place_sql(alias: str = "u") -> str:
    t = f"{alias}.trill_metadata"
    return f"""NULLIF(TRIM(COALESCE(
        NULLIF(btrim({t}#>>'{{telemetry,gazetteer_place_name}}'), ''),
        NULLIF(btrim({t}#>>'{{telemetry,location_display}}'), ''),
        NULLIF(btrim({t}#>>'{{place_name}}'), ''),
        NULLIF(btrim({t}#>>'{{telemetry,location_city}}'), '')
    )), ''), '')"""


def _lb_meta_lat_sql(alias: str = "u") -> str:
    t = f"{alias}.trill_metadata"
    return f"""COALESCE(
        NULLIF(btrim({t}#>>'{{telemetry,mid_lat}}'), '')::double precision,
        NULLIF(btrim({t}#>>'{{telemetry,start_lat}}'), '')::double precision,
        NULLIF(btrim({t}#>>'{{place_lat}}'), '')::double precision
    )"""


def _lb_meta_lon_sql(alias: str = "u") -> str:
    t = f"{alias}.trill_metadata"
    return f"""COALESCE(
        NULLIF(btrim({t}#>>'{{telemetry,mid_lon}}'), '')::double precision,
        NULLIF(btrim({t}#>>'{{telemetry,start_lon}}'), '')::double precision,
        NULLIF(btrim({t}#>>'{{place_lon}}'), '')::double precision
    )"""


def _leaderboard_sort_sql(sort: str) -> str:
    key = (sort or "best_trill").strip()
    if key not in _LEADERBOARD_SORTS:
        key = "best_trill"
    return {
        "best_trill": "a.best_trill DESC NULLS LAST, a.best_speed DESC NULLS LAST",
        "speed": "a.best_speed DESC NULLS LAST, a.best_trill DESC NULLS LAST",
        "runs": "a.run_count DESC NULLS LAST, a.best_trill DESC NULLS LAST",
        "distance": "a.total_distance DESC NULLS LAST, a.best_trill DESC NULLS LAST",
        "avg_trill": "a.avg_trill DESC NULLS LAST, a.best_trill DESC NULLS LAST",
    }[key]


def _leaderboard_driver_handle(user_id: str, trill_public_name, status: str) -> tuple:
    uid_s = str(user_id)
    anon = f"Driver-{hashlib.md5(uid_s.encode('utf-8')).hexdigest()[:6]}"
    pub = (trill_public_name or "").strip()
    st = (status or "").strip().lower()
    approved = bool(pub and st == "approved")
    handle = pub if approved else anon
    return handle, anon, approved


def _leaderboard_speed_expr(alias: str = "u") -> str:
    a = alias
    return f"""COALESCE(
        {a}.max_speed_mph,
        NULLIF(btrim({a}.trill_metadata#>>'{{telemetry,max_speed_mph}}'), '')::double precision,
        0.0
    )"""


def _leaderboard_distance_expr(alias: str = "u") -> str:
    a = alias
    return f"""COALESCE(
        {a}.distance_miles,
        NULLIF(btrim({a}.trill_metadata#>>'{{telemetry,total_distance_miles}}'), '')::double precision,
        0.0
    )"""


@router.get("/leaderboard/regions")
async def trill_leaderboard_regions(
    range: str = Query("30d"),
    user: dict = Depends(get_current_user),
):
    _ = user
    since = _trill_since_dt(range)
    try:
        async with core.state.db_pool.acquire() as conn:
            regions = await fetch_region_options(conn, since)
    except Exception as e:
        logger.warning("trill leaderboard regions: %s", e)
        regions = []
    return {"range": range, "regions": regions}


@router.get("/leaderboard")
async def trill_leaderboard(
    range: str = Query("30d"),
    sort: str = Query("best_trill"),
    limit: int = Query(50, ge=1, le=100),
    region: Optional[str] = Query(None, max_length=8),
    user_id: str = Depends(get_verified_user_id),
):
    """Community leaderboard — opted-in users only; aggregates (no route geometry)."""
    uid = user_id
    since = _trill_since_dt(range)
    sort_key = (sort or "best_trill").strip()
    if sort_key not in _LEADERBOARD_SORTS:
        sort_key = "best_trill"
    order_sql = _leaderboard_sort_sql(sort_key)
    speed_x = _leaderboard_speed_expr("u")
    dist_x = _leaderboard_distance_expr("u")
    place_x = _lb_meta_place_sql("u")
    state_x = _lb_meta_state_sql("u")
    lat_x = _lb_meta_lat_sql("u")
    lon_x = _lb_meta_lon_sql("u")
    region_code = (region or "").strip().upper() or None
    viewer_unlocked = False
    bucket_rows = []
    rows = []
    rival_ids: set = set()
    rival_alerts: list = []
    rival_handles: dict = {}
    rank_by_uid: dict = {}
    hall_of_fame: list = []
    challenge_row = None
    challenge_completion = None
    pref_row = None
    opted_in = False
    viewer_row = None
    viewer_rank = None
    scores_map: dict = {}
    badges_map: dict = {}
    viewer_badges: list = []
    viewer_badge_collection: list = []
    try:
        async with core.state.db_pool.acquire() as conn:
            await require_verified_user_on_conn(conn, uid)
            await ensure_badge_definitions(conn)
            await ensure_current_season(conn)
            viewer_unlocked = bool(
                await conn.fetchval(
                    f"""
                    SELECT EXISTS (
                        SELECT 1 FROM uploads u
                        WHERE u.user_id = $1
                        AND {TRILL_ROUTE_PREDICATE.strip()}
                    )
                    """,
                    uid,
                )
            )
            if not viewer_unlocked:
                pass
            else:
                pref_row = await conn.fetchrow(
                    """
                    SELECT
                        COALESCE(trill_leaderboard_opt_in, FALSE) AS opted_in,
                        NULLIF(btrim(trill_public_name), '') AS trill_public_name,
                        trill_public_name_status
                    FROM user_preferences
                    WHERE user_id = $1
                    """,
                    uid,
                )
                opted_in = bool(pref_row and pref_row["opted_in"])

                viewer_row = await conn.fetchrow(
                    f"""
                    SELECT
                        COUNT(*)::int AS run_count,
                        COALESCE(MAX(u.trill_score), 0)::float AS best_trill,
                        COALESCE(AVG(u.trill_score), 0)::float AS avg_trill,
                        COALESCE(MAX({speed_x}), 0)::float AS best_speed,
                        COALESCE(SUM({dist_x}), 0)::float AS total_distance
                    FROM uploads u
                    WHERE u.user_id = $1
                      AND u.created_at >= $2
                      AND {TRILL_ROUTE_PREDICATE.strip()}
                    """,
                    uid,
                    since,
                )

                bucket_rows = await conn.fetch(
                    f"""
                    SELECT COALESCE(NULLIF(btrim(u.speed_bucket), ''), 'unknown') AS bucket, COUNT(*)::int AS cnt
                    FROM uploads u
                    INNER JOIN user_preferences pref ON pref.user_id = u.user_id
                        AND COALESCE(pref.trill_leaderboard_opt_in, FALSE) = TRUE
                    WHERE u.created_at >= $1
                      AND {TRILL_ROUTE_PREDICATE.strip()}
                    GROUP BY 1
                    """,
                    since,
                )

                rows = await conn.fetch(
                    f"""
                    WITH per_upload AS (
                        SELECT
                            u.user_id,
                            u.trill_score::float AS trill_score,
                            u.speed_bucket,
                            {speed_x}::float AS speed_mph,
                            {dist_x}::float AS dist_mi,
                            {place_x} AS g_place,
                            {state_x} AS g_state,
                            {lat_x} AS g_lat,
                            {lon_x} AS g_lon
                        FROM uploads u
                        INNER JOIN user_preferences pref ON pref.user_id = u.user_id
                            AND COALESCE(pref.trill_leaderboard_opt_in, FALSE) = TRUE
                        WHERE u.created_at >= $1
                          AND {TRILL_ROUTE_PREDICATE.strip()}
                          AND ($3::text IS NULL OR UPPER(TRIM({state_x})) = $3)
                    ),
                    agg AS (
                        SELECT
                            user_id::text AS user_id,
                            COUNT(*)::int AS run_count,
                            MAX(trill_score)::float AS best_trill,
                            AVG(trill_score)::float AS avg_trill,
                            MAX(speed_mph)::float AS best_speed,
                            SUM(COALESCE(dist_mi, 0))::float AS total_distance
                        FROM per_upload
                        GROUP BY user_id
                    ),
                    best_run AS (
                        SELECT DISTINCT ON (user_id)
                            user_id::text AS user_id,
                            speed_bucket,
                            g_place,
                            g_state,
                            g_lat,
                            g_lon
                        FROM per_upload
                        ORDER BY user_id, trill_score DESC NULLS LAST
                    )
                    SELECT
                        a.user_id,
                        a.run_count,
                        a.best_trill,
                        a.avg_trill,
                        a.best_speed,
                        a.total_distance,
                        br.speed_bucket AS best_bucket,
                        br.g_place AS best_run_place,
                        br.g_state AS best_run_state,
                        br.g_lat AS best_run_lat,
                        br.g_lon AS best_run_lon,
                        NULLIF(btrim(pub.trill_public_name), '') AS trill_public_name,
                        pub.trill_public_name_status,
                        ROW_NUMBER() OVER (ORDER BY {order_sql})::int AS rank
                    FROM agg a
                    LEFT JOIN best_run br ON br.user_id = a.user_id
                    LEFT JOIN user_preferences pub ON pub.user_id = a.user_id::uuid
                    ORDER BY rank
                    LIMIT $2
                    """,
                    since,
                    limit,
                    region_code,
                )

                rival_ids = set(await fetch_rivals(conn, uid))
                challenge_row = await ensure_weekly_challenge(conn)
                challenge_completion = None
                if challenge_row and opted_in:
                    challenge_completion = await check_challenge_for_user(conn, uid, challenge_row)
                    if challenge_completion:
                        rp = int(challenge_completion.get("reward_put") or 0)
                        ra = int(challenge_completion.get("reward_aic") or 0)
                        if rp > 0:
                            await credit_wallet(conn, uid, "put", rp, "trill_weekly_challenge")
                        if ra > 0:
                            await credit_wallet(conn, uid, "aic", ra, "trill_weekly_challenge")
                        await award_badge(conn, uid, "challenge_champ", presented_by="UploadM8 Challenge Desk")
                hall_of_fame = await fetch_hall_of_fame(conn)
                rank_by_uid = {str(r["user_id"]): int(r["rank"]) for r in rows}
                rival_handles = {}
                for r in rows:
                    uid_row = str(r["user_id"])
                    if uid_row in rival_ids:
                        h, _, _ = _leaderboard_driver_handle(
                            uid_row, r.get("trill_public_name"), r.get("trill_public_name_status")
                        )
                        rival_handles[uid_row] = h

                uid_str = str(uid)
                user_ids = [str(r["user_id"]) for r in rows]
                if user_ids:
                    scores_map = await fetch_recent_scores_batch(conn, user_ids, since)
                    badges_map = await fetch_badges_for_users(conn, user_ids)

                viewer_rank = None
                for r in rows:
                    if str(r["user_id"]) == uid_str:
                        viewer_rank = int(r["rank"])
                        break

                v_best = float(viewer_row["best_trill"] or 0) if viewer_row else 0.0
                v_runs = int(viewer_row["run_count"] or 0) if viewer_row else 0
                v_speed = round(float(viewer_row["best_speed"] or 0), 1) if viewer_row else 0.0
                try:
                    await evaluate_and_award_badges(
                        conn,
                        uid_str,
                        viewer_stats={
                            "best_trill": v_best,
                            "rank": viewer_rank,
                            "run_count": v_runs,
                            "best_speed": v_speed,
                        },
                        since=since,
                    )
                    viewer_badges = (await fetch_badges_for_users(conn, [uid_str])).get(uid_str, [])
                    viewer_badge_collection = await fetch_user_badge_collection(conn, uid_str)
                    if opted_in and viewer_rank is not None and rival_handles:
                        rival_alerts = await process_rival_rank_changes(
                            conn,
                            uid_str,
                            sort_key,
                            rank_by_uid,
                            viewer_rank,
                            rival_handles,
                            db_pool=core.state.db_pool,
                        )
                except Exception:
                    viewer_badges = []
                    viewer_badge_collection = []
    except Exception as e:
        logger.warning("trill leaderboard: %s", e)
        return {
            "unlocked": viewer_unlocked,
            "map_unlocked": viewer_unlocked,
            "range": range,
            "sort": sort_key,
            "rows": [],
            "summary": {},
            "viewer": {},
            "highlights": {},
            "bucket_distribution": {},
            "error": str(e),
        }

    if not viewer_unlocked:
        return {
            "unlocked": False,
            "map_unlocked": False,
            "range": range,
            "sort": sort_key,
            "rows": [],
            "summary": {},
            "viewer": {},
            "highlights": {},
            "bucket_distribution": {},
            "message": "Complete a map-backed Trill upload to unlock the community leaderboard.",
        }

    bucket_dist = {b: 0 for b in ("gloryBoy", "euphoric", "sendIt", "spirited", "chill")}
    for br in bucket_rows:
        b = (br["bucket"] or "unknown").strip()
        if b in bucket_dist:
            bucket_dist[b] = int(br["cnt"] or 0)

    out_rows = []
    viewer_rank = None
    uid_str = str(uid)
    for r in rows:
        handle, anon, approved = _leaderboard_driver_handle(
            r["user_id"], r.get("trill_public_name"), r.get("trill_public_name_status")
        )
        rank = int(r["rank"])
        is_you = str(r["user_id"]) == uid_str
        if is_you:
            viewer_rank = rank
        br_place = (r.get("best_run_place") or "").strip() or None
        br_state = (r.get("best_run_state") or "").strip() or None
        br_lat = r.get("best_run_lat")
        br_lon = r.get("best_run_lon")
        try:
            br_lat_f = round(float(br_lat), 4) if br_lat is not None else None
        except (TypeError, ValueError):
            br_lat_f = None
        try:
            br_lon_f = round(float(br_lon), 4) if br_lon is not None else None
        except (TypeError, ValueError):
            br_lon_f = None
        uid_row = str(r["user_id"])
        out_rows.append(
            {
                "rank": rank,
                "public_id": public_driver_id(uid_row),
                "driver_handle": handle,
                "driver_handle_anonymous": anon,
                "has_public_name": approved,
                "is_you": is_you,
                "is_rival": uid_row in rival_ids,
                "run_count": int(r["run_count"] or 0),
                "best_trill_score": float(r["best_trill"] or 0),
                "avg_trill_score": round(float(r["avg_trill"] or 0), 1),
                "best_speed_mph": float(r["best_speed"] or 0),
                "total_distance_miles": round(float(r["total_distance"] or 0), 1),
                "best_bucket": (r.get("best_bucket") or "").strip() or None,
                "best_run_place": br_place,
                "best_run_state": br_state,
                "best_run_region": _format_leaderboard_region(br_place, br_state),
                "best_run_lat": br_lat_f,
                "best_run_lon": br_lon_f,
                "recent_scores": [],
                "badges": [],
            }
        )

    for i, r in enumerate(rows):
        if i < len(out_rows):
            uid_row = str(r["user_id"])
            out_rows[i]["recent_scores"] = scores_map.get(uid_row, [])
            out_rows[i]["badges"] = badges_map.get(uid_row, [])

    driver_count = len(out_rows)
    total_runs = sum(int(r["run_count"] or 0) for r in out_rows)
    best_scores = [float(r["best_trill_score"] or 0) for r in out_rows]
    summary = {
        "driver_count": driver_count,
        "total_runs": total_runs,
        "avg_best_trill": round(sum(best_scores) / driver_count, 1) if driver_count else 0.0,
        "max_best_trill": round(max(best_scores), 1) if best_scores else 0.0,
    }

    highlights = {}
    if out_rows:
        top_trill = max(out_rows, key=lambda x: x["best_trill_score"])
        highlights["best_trill"] = {
            "driver_handle": top_trill["driver_handle"],
            "is_you": top_trill.get("is_you"),
            "best_trill_score": top_trill["best_trill_score"],
        }
        top_speed = max(out_rows, key=lambda x: x["best_speed_mph"])
        highlights["top_speed"] = {
            "driver_handle": top_speed["driver_handle"],
            "is_you": top_speed.get("is_you"),
            "best_speed_mph": top_speed["best_speed_mph"],
        }
        top_runs = max(out_rows, key=lambda x: x["run_count"])
        highlights["most_runs"] = {
            "driver_handle": top_runs["driver_handle"],
            "is_you": top_runs.get("is_you"),
            "run_count": top_runs["run_count"],
        }
        top_dist = max(out_rows, key=lambda x: x["total_distance_miles"])
        highlights["most_distance"] = {
            "driver_handle": top_dist["driver_handle"],
            "is_you": top_dist.get("is_you"),
            "total_distance_miles": top_dist["total_distance_miles"],
        }

    v_best = float(viewer_row["best_trill"] or 0) if viewer_row else 0.0
    v_runs = int(viewer_row["run_count"] or 0) if viewer_row else 0
    percentile = None
    if opted_in and viewer_rank and driver_count > 0:
        percentile = max(0, min(100, int(round(100 * (1 - (viewer_rank - 1) / driver_count)))))

    viewer = {
        "opted_in": opted_in,
        "on_board": bool(opted_in and viewer_rank),
        "rank": viewer_rank,
        "run_count": v_runs,
        "best_trill_score": round(v_best, 1),
        "avg_trill_score": round(float(viewer_row["avg_trill"] or 0), 1) if viewer_row else 0.0,
        "best_speed_mph": round(float(viewer_row["best_speed"] or 0), 1) if viewer_row else 0.0,
        "total_distance_miles": round(float(viewer_row["total_distance"] or 0), 1) if viewer_row else 0.0,
        "percentile": percentile,
        "display_name_status": str((pref_row or {}).get("trill_public_name_status") or "none"),
        "public_name": (pref_row or {}).get("trill_public_name"),
    }
    viewer["chase_targets"] = compute_chase_targets(out_rows, viewer, sort_key)
    viewer["badges"] = viewer_badges
    viewer["badge_collection"] = viewer_badge_collection

    active_challenge = None
    if challenge_row:
        active_challenge = {
            "id": int(challenge_row["id"]),
            "title": challenge_row.get("title"),
            "description": challenge_row.get("description"),
            "challenge_type": challenge_row.get("challenge_type"),
            "target_value": float(challenge_row.get("target_value") or 0),
            "reward_put": int(challenge_row.get("reward_put") or 0),
            "reward_aic": int(challenge_row.get("reward_aic") or 0),
            "completed": bool(challenge_completion),
        }

    return {
        "unlocked": True,
        "map_unlocked": True,
        "range": range,
        "sort": sort_key,
        "region": region_code,
        "summary": summary,
        "viewer": viewer,
        "highlights": highlights,
        "bucket_distribution": bucket_dist,
        "hall_of_fame": hall_of_fame,
        "active_challenge": active_challenge,
        "rival_alerts": rival_alerts,
        "rival_count": len(rival_ids),
        "rival_max": 3,
        "rows": out_rows,
    }


@router.get("/map-feed")
async def trill_map_feed(
    range: str = Query("30d"),
    trill_vehicle_make: Optional[str] = Query(None, max_length=120),
    trill_vehicle_model: Optional[str] = Query(None, max_length=120),
    user: dict = Depends(get_current_user),
):
    """Pins for the signed-in user only (no full GPS arrays)."""
    uid = user["id"]
    since = _trill_since_dt(range)
    vf_sql, vf_vals = trill_vehicle_where_fragment(3, trill_vehicle_make, trill_vehicle_model)
    try:
        async with core.state.db_pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    u.id,
                    COALESCE(NULLIF(btrim(u.title), ''), u.filename) AS title,
                    u.trill_score,
                    u.speed_bucket,
                    u.status,
                    COALESCE(
                        NULLIF(btrim(u.trill_metadata#>>'{{telemetry,mid_lat}}'), '')::double precision,
                        NULLIF(btrim(u.trill_metadata#>>'{{telemetry,start_lat}}'), '')::double precision
                    ) AS lat,
                    COALESCE(
                        NULLIF(btrim(u.trill_metadata#>>'{{telemetry,mid_lon}}'), '')::double precision,
                        NULLIF(btrim(u.trill_metadata#>>'{{telemetry,start_lon}}'), '')::double precision
                    ) AS lon,
                    NULLIF(btrim(u.trill_metadata#>>'{{telemetry,gazetteer_place_name}}'), '') AS gazetteer_place,
                    NULLIF(btrim(u.trill_metadata#>>'{{telemetry,padus_unit_name}}'), '') AS padus_unit,
                    COALESCE(
                        (NULLIF(btrim(u.trill_metadata#>>'{{telemetry,near_padus}}'), ''))::boolean,
                        FALSE
                    ) AS near_padus
                FROM uploads u
                WHERE u.user_id = $1
                  AND u.created_at >= $2
                  AND u.trill_score IS NOT NULL
                  AND (
                    COALESCE(
                        NULLIF(btrim(u.trill_metadata#>>'{{telemetry,mid_lat}}'), '')::double precision,
                        NULLIF(btrim(u.trill_metadata#>>'{{telemetry,start_lat}}'), '')::double precision
                    ) IS NOT NULL
                  )
                  {vf_sql}
                ORDER BY u.created_at DESC
                LIMIT 200
                """,
                uid,
                since,
                *vf_vals,
            )
    except Exception as e:
        logger.warning("trill map-feed: %s", e)
        return {"pins": []}

    pins = []
    for r in rows:
        if r["lat"] is None or r["lon"] is None:
            continue
        pins.append(
            {
                "id": str(r["id"]),
                "title": r["title"] or "Untitled",
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "trill_score": float(r["trill_score"]) if r["trill_score"] is not None else None,
                "speed_bucket": r["speed_bucket"],
                "status": r["status"],
                "gazetteer_place": r["gazetteer_place"],
                "padus_unit": r["padus_unit"],
                "near_padus": bool(r["near_padus"]),
            }
        )
    return {"pins": pins}


@router.get("/route/{upload_id}")
async def trill_route_downsample(upload_id: str, user: dict = Depends(get_current_user), max_points: int = Query(400, ge=10, le=800)):
    """Downsampled route for map polyline (owner only)."""
    uid = user["id"]
    try:
        import uuid as _uuid

        _uuid.UUID(upload_id)
    except Exception:
        raise HTTPException(400, "Invalid upload id")

    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT trill_metadata FROM uploads
            WHERE id = $1::uuid AND user_id = $2::uuid
            """,
            upload_id,
            uid,
        )
    if not row or not row["trill_metadata"]:
        raise HTTPException(404, "No telemetry for this upload")

    meta = row["trill_metadata"]
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    tel = (meta or {}).get("telemetry") or {}
    pts = tel.get("points") or []
    if not isinstance(pts, list) or len(pts) < 2:
        raise HTTPException(404, "No route points")

    step = max(1, len(pts) // max_points)
    sampled = []
    for i in range(0, len(pts), step):
        p = pts[i]
        if not isinstance(p, dict):
            continue
        try:
            la = float(p.get("lat"))
            lo = float(p.get("lon"))
        except (TypeError, ValueError):
            continue
        spd = p.get("speed_mph")
        try:
            spd_f = float(spd) if spd is not None else None
        except (TypeError, ValueError):
            spd_f = None
        sampled.append({"lat": la, "lon": lo, "speed_mph": spd_f})
    if len(sampled) < 2:
        raise HTTPException(404, "Could not sample route")
    return {"upload_id": upload_id, "points": sampled}


class TrillUploadVehicleBody(BaseModel):
    """Assign catalog vehicle to a completed Trill upload (analytics / garage)."""

    model_config = ConfigDict(populate_by_name=True)

    vehicle_make_id: Optional[int] = Field(None, alias="vehicleMakeId")
    vehicle_model_id: Optional[int] = Field(None, alias="vehicleModelId")


@router.put("/uploads/{upload_id}/vehicle")
async def set_trill_upload_vehicle(
    upload_id: str,
    body: TrillUploadVehicleBody,
    user: dict = Depends(get_current_user),
):
    """Set make/model on a Trill-scored upload; updates columns and ``trill_metadata.vehicle``."""
    uid = user["id"]
    vm_id = body.vehicle_make_id
    vmd_id = body.vehicle_model_id
    if vm_id is not None and vmd_id is not None:
        async with core.state.db_pool.acquire() as conn:
            ok = await conn.fetchrow(
                "SELECT 1 FROM vehicle_models WHERE id = $1 AND make_id = $2",
                vmd_id,
                vm_id,
            )
            if not ok:
                raise HTTPException(400, "Invalid vehicle model for selected make")
    elif vmd_id is not None and vm_id is None:
        raise HTTPException(400, "vehicle_make_id required when vehicle_model_id is set")

    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, trill_score, trill_metadata
            FROM uploads
            WHERE id = $1 AND user_id = $2
            """,
            upload_id,
            uid,
        )
        if not row:
            raise HTTPException(404, "Upload not found")
        if row["trill_score"] is None:
            raise HTTPException(400, "Upload has no Trill score")

        lab = await fetch_vehicle_labels(conn, vm_id, vmd_id)
        vehicle_meta = {
            "make_id": vm_id,
            "model_id": vmd_id,
            "make_name": lab.get("make_name"),
            "model_name": lab.get("model_name"),
        }
        vehicle_meta = {k: v for k, v in vehicle_meta.items() if v is not None}
        meta_patch = json.dumps({"vehicle": vehicle_meta})

        await conn.execute(
            """
            UPDATE uploads
            SET vehicle_make_id = $3,
                vehicle_model_id = $4,
                trill_metadata = COALESCE(trill_metadata, '{}'::jsonb) || $5::jsonb,
                updated_at = NOW()
            WHERE id = $1 AND user_id = $2
            """,
            upload_id,
            uid,
            vm_id,
            vmd_id,
            meta_patch,
        )

    return {
        "ok": True,
        "upload_id": upload_id,
        "vehicle_make_id": vm_id,
        "vehicle_model_id": vmd_id,
        "make_name": lab.get("make_name"),
        "model_name": lab.get("model_name"),
    }


class TrillRivalBody(BaseModel):
    rival_user_id: Optional[str] = Field(None, min_length=8, max_length=64)
    rival_public_id: Optional[str] = Field(None, min_length=8, max_length=32)


@router.get("/rivals")
async def list_trill_rivals(user: dict = Depends(get_current_user)):
    uid = user["id"]
    async with core.state.db_pool.acquire() as conn:
        rival_uids = await fetch_rivals(conn, uid)
        if not rival_uids:
            return {"rivals": []}
        rows = await conn.fetch(
            """
            SELECT user_id::text,
                   NULLIF(btrim(trill_public_name), '') AS trill_public_name,
                   trill_public_name_status
            FROM user_preferences
            WHERE user_id = ANY($1::uuid[])
            """,
            rival_uids,
        )
    out = []
    for r in rows:
        h, anon, _ = _leaderboard_driver_handle(r["user_id"], r.get("trill_public_name"), r.get("trill_public_name_status"))
        out.append({"user_id": str(r["user_id"]), "public_id": public_driver_id(str(r["user_id"])), "driver_handle": h})
    return {"rivals": out, "count": len(out), "max": 3}


@router.post("/rivals")
async def add_trill_rival(body: TrillRivalBody, user: dict = Depends(get_current_user)):
    uid = user["id"]
    rid = (body.rival_user_id or "").strip()
    if body.rival_public_id:
        async with core.state.db_pool.acquire() as conn:
            resolved = await resolve_user_from_public_id(conn, body.rival_public_id.strip().lower())
        if not resolved:
            raise HTTPException(404, "Driver not found")
        rid = resolved
    if not rid:
        raise HTTPException(400, "rival_user_id or rival_public_id required")
    if rid == str(uid):
        raise HTTPException(400, "Cannot rival yourself")
    async with core.state.db_pool.acquire() as conn:
        ok = await conn.fetchval(
            """
            SELECT 1 FROM user_preferences
            WHERE user_id = $1::uuid AND COALESCE(trill_leaderboard_opt_in, FALSE) = TRUE
            """,
            rid,
        )
        if not ok:
            raise HTTPException(404, "Driver not on leaderboard")
        added = await add_rival(conn, uid, rid)
        if not added:
            raise HTTPException(400, "Maximum 3 rivals")
    return {"ok": True}


@router.delete("/rivals/{rival_user_id}")
async def delete_trill_rival(rival_user_id: str, user: dict = Depends(get_current_user)):
    rid = rival_user_id.strip()
    if len(rid) <= 20:
        async with core.state.db_pool.acquire() as conn:
            resolved = await resolve_user_from_public_id(conn, rid.lower())
        if resolved:
            rid = resolved
    async with core.state.db_pool.acquire() as conn:
        await remove_rival(conn, user["id"], rid)
    return {"ok": True}


@router.post("/jobs/archive-seasons")
async def trill_archive_seasons_job(user: dict = Depends(get_current_user)):
    """Manual trigger for season → hall of fame archive (also runs hourly in API lifespan)."""
    role = str(user.get("role") or "")
    if role not in ("admin", "master_admin"):
        raise HTTPException(403, "Admin only")
    async with core.state.db_pool.acquire() as conn:
        n = await archive_due_seasons(conn)
    return {"ok": True, "archived": n}


@router.get("/badges/catalog")
async def trill_badge_catalog(user: dict = Depends(get_current_user)):
    """All badge definitions for showcase (locked + earned states merged client-side)."""
    _ = user
    async with core.state.db_pool.acquire() as conn:
        catalog = await fetch_badge_catalog(conn)
        earned = await fetch_user_badge_collection(conn, user["id"])
    earned_slugs = {b["slug"] for b in earned}
    return {
        "catalog": catalog,
        "earned": earned,
        "earned_count": len(earned),
        "total_count": len(catalog),
        "earned_slugs": sorted(earned_slugs),
    }


@router.get("/badges/me")
async def trill_my_badges(user: dict = Depends(get_current_user)):
    """Full earned badge collection with presentation metadata."""
    async with core.state.db_pool.acquire() as conn:
        await evaluate_and_award_badges(conn, user["id"], since=_trill_since_dt("1y"))
        collection = await fetch_user_badge_collection(conn, user["id"])
    return {"badges": collection, "count": len(collection)}


@router.get("/driver/{public_id}")
async def trill_driver_profile(public_id: str, range: str = Query("30d"), user: dict = Depends(get_current_user)):
    _ = user
    since = _trill_since_dt(range)
    async with core.state.db_pool.acquire() as conn:
        target_uid = await resolve_user_from_public_id(conn, public_id.strip().lower())
        if not target_uid:
            raise HTTPException(404, "Driver not found")
        opted = await conn.fetchval(
            "SELECT COALESCE(trill_leaderboard_opt_in, FALSE) FROM user_preferences WHERE user_id = $1",
            target_uid,
        )
        if not opted:
            raise HTTPException(404, "Driver not on leaderboard")
        pref = await conn.fetchrow(
            "SELECT trill_public_name, trill_public_name_status FROM user_preferences WHERE user_id = $1",
            target_uid,
        )
        handle, anon, approved = _leaderboard_driver_handle(
            target_uid, pref.get("trill_public_name") if pref else None, pref.get("trill_public_name_status") if pref else None
        )
        stats = await conn.fetchrow(
            f"""
            SELECT COUNT(*)::int AS run_count,
                   COALESCE(MAX(trill_score), 0)::float AS best_trill,
                   COALESCE(AVG(trill_score), 0)::float AS avg_trill,
                   COALESCE(MAX(max_speed_mph), 0)::float AS best_speed,
                   COALESCE(SUM(distance_miles), 0)::float AS total_distance
            FROM uploads u
            WHERE u.user_id = $1 AND u.created_at >= $2 AND {TRILL_ROUTE_PREDICATE.strip()}
            """,
            target_uid,
            since,
        )
        uploads = await conn.fetch(
            f"""
            SELECT id, COALESCE(NULLIF(btrim(title), ''), filename) AS title,
                   trill_score::float, max_speed_mph::float, created_at
            FROM uploads u
            WHERE u.user_id = $1 AND u.created_at >= $2 AND u.trill_score IS NOT NULL
              AND {TRILL_ROUTE_PREDICATE.strip()}
            ORDER BY trill_score DESC NULLS LAST
            LIMIT 8
            """,
            target_uid,
            since,
        )
        badges = await fetch_user_badge_collection(conn, target_uid)
        recent = (await fetch_recent_scores_batch(conn, [target_uid], since)).get(target_uid, [])
    return {
        "public_id": public_id,
        "driver_handle": handle,
        "has_public_name": approved,
        "stats": {
            "run_count": int(stats["run_count"] or 0) if stats else 0,
            "best_trill_score": round(float(stats["best_trill"] or 0), 1) if stats else 0,
            "avg_trill_score": round(float(stats["avg_trill"] or 0), 1) if stats else 0,
            "best_speed_mph": round(float(stats["best_speed"] or 0), 1) if stats else 0,
            "total_distance_miles": round(float(stats["total_distance"] or 0), 1) if stats else 0,
        },
        "recent_scores": recent,
        "badges": badges,
        "top_uploads": [
            {
                "id": str(u["id"]),
                "title": u["title"],
                "trill_score": float(u["trill_score"] or 0),
                "max_speed_mph": float(u["max_speed_mph"] or 0) if u["max_speed_mph"] else None,
                "created_at": u["created_at"].isoformat() if u.get("created_at") else None,
            }
            for u in uploads
        ],
    }
