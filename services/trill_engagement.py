"""Trill leaderboard engagement: badges, chase targets, sparklines, seasons, rivals, challenges."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from services.trill_access import TRILL_ROUTE_PREDICATE

logger = logging.getLogger(__name__)

PRESENTED_BY_DEFAULT = "UploadM8 Trill League"

BADGE_SEED: List[Dict[str, Any]] = [
    {
        "slug": "map_unlocked",
        "title": "Cartographer",
        "description": "Logged your first map-backed Trill run — the open road is yours.",
        "icon": "fa-map-location-dot",
        "tier": "bronze",
        "category": "onboarding",
        "sort_order": 10,
    },
    {
        "slug": "first_send_it",
        "title": "Send It Initiate",
        "description": "Crossed 60 Trill on a single run. The send has begun.",
        "icon": "fa-rocket",
        "tier": "bronze",
        "category": "performance",
        "sort_order": 20,
    },
    {
        "slug": "glory_boy",
        "title": "Glory Boy Crown",
        "description": "Hit 90+ Trill — peak vibes, peak glory.",
        "icon": "fa-crown",
        "tier": "gold",
        "category": "performance",
        "sort_order": 30,
    },
    {
        "slug": "century_club",
        "title": "Century Club",
        "description": "Touched 100 mph on a verified map run.",
        "icon": "fa-gauge-high",
        "tier": "silver",
        "category": "speed",
        "sort_order": 40,
    },
    {
        "slug": "speed_demon",
        "title": "Speed Demon",
        "description": "Blazed past 110 mph — hold on tight.",
        "icon": "fa-bolt",
        "tier": "gold",
        "category": "speed",
        "sort_order": 50,
    },
    {
        "slug": "road_warrior",
        "title": "Road Warrior",
        "description": "1,000+ miles of map-backed driving logged.",
        "icon": "fa-road",
        "tier": "gold",
        "category": "volume",
        "sort_order": 60,
    },
    {
        "slug": "century_driver",
        "title": "Century Driver",
        "description": "100 map-backed runs — a true road veteran.",
        "icon": "fa-flag-checkered",
        "tier": "silver",
        "category": "volume",
        "sort_order": 70,
    },
    {
        "slug": "weekly_grinder",
        "title": "Weekly Grinder",
        "description": "Seven map runs in seven days. Consistency wins.",
        "icon": "fa-calendar-week",
        "tier": "bronze",
        "category": "volume",
        "sort_order": 80,
    },
    {
        "slug": "on_the_board",
        "title": "On the Board",
        "description": "Opted in and ranked on the community leaderboard.",
        "icon": "fa-trophy",
        "tier": "bronze",
        "category": "leaderboard",
        "sort_order": 90,
    },
    {
        "slug": "top_10",
        "title": "Top Ten Ace",
        "description": "Cracked the top 10 on the Trill leaderboard.",
        "icon": "fa-medal",
        "tier": "silver",
        "category": "leaderboard",
        "sort_order": 100,
    },
    {
        "slug": "podium",
        "title": "Podium Finisher",
        "description": "Top 3 on the board — champagne energy.",
        "icon": "fa-award",
        "tier": "gold",
        "category": "leaderboard",
        "sort_order": 110,
    },
    {
        "slug": "name_on_the_door",
        "title": "Name on the Door",
        "description": "Approved public display name — you’re not anonymous anymore.",
        "icon": "fa-id-card",
        "tier": "bronze",
        "category": "community",
        "sort_order": 120,
    },
    {
        "slug": "challenge_champ",
        "title": "Challenge Champion",
        "description": "Completed the weekly Trill challenge.",
        "icon": "fa-star",
        "tier": "silver",
        "category": "challenges",
        "sort_order": 130,
        "presented_by": "UploadM8 Challenge Desk",
    },
    # ── 20 US landmark / scenic place badges ─────────────────────────────
    {
        "slug": "place_grand_canyon",
        "title": "Canyon Sentinel",
        "description": "Drove near Grand Canyon — one mile deep, infinite awe.",
        "icon": "fa-mountain-sun",
        "tier": "gold",
        "category": "landmarks",
        "sort_order": 200,
    },
    {
        "slug": "place_yosemite",
        "title": "Yosemite Skyward",
        "description": "Ran routes through Yosemite — granite cathedrals and sky.",
        "icon": "fa-tree",
        "tier": "gold",
        "category": "landmarks",
        "sort_order": 201,
    },
    {
        "slug": "place_yellowstone",
        "title": "Yellowstone Wildlands",
        "description": "Explored Yellowstone country — geysers, bison, and wild miles.",
        "icon": "fa-fire-flame-curved",
        "tier": "gold",
        "category": "landmarks",
        "sort_order": 202,
    },
    {
        "slug": "place_rushmore",
        "title": "Rushmore Heritage",
        "description": "Passed Mount Rushmore — carved history on the Black Hills.",
        "icon": "fa-monument",
        "tier": "silver",
        "category": "landmarks",
        "sort_order": 203,
    },
    {
        "slug": "place_golden_gate",
        "title": "Golden Gate Voyager",
        "description": "Crossed Golden Gate territory — fog, bridge, Pacific dreams.",
        "icon": "fa-bridge",
        "tier": "gold",
        "category": "landmarks",
        "sort_order": 204,
    },
    {
        "slug": "place_niagara",
        "title": "Niagara Thunder",
        "description": "Ran near Niagara Falls — 750,000 gallons per second of power.",
        "icon": "fa-water",
        "tier": "silver",
        "category": "landmarks",
        "sort_order": 205,
    },
    {
        "slug": "place_great_lakes",
        "title": "Great Lakes Navigator",
        "description": "Cruised Great Lakes shoreline — inland seas, endless horizon.",
        "icon": "fa-ship",
        "tier": "silver",
        "category": "landmarks",
        "sort_order": 206,
    },
    {
        "slug": "place_rockies",
        "title": "Rockies Crest",
        "description": "Climbed Rocky Mountain routes — thin air, big views.",
        "icon": "fa-mountain",
        "tier": "gold",
        "category": "landmarks",
        "sort_order": 207,
    },
    {
        "slug": "place_monument_valley",
        "title": "Monument Valley Spirit",
        "description": "Rolled through Monument Valley — red mesas and cinematic dust.",
        "icon": "fa-sun",
        "tier": "gold",
        "category": "landmarks",
        "sort_order": 208,
    },
    {
        "slug": "place_death_valley",
        "title": "Death Valley Extremist",
        "description": "Survived Death Valley miles — hottest, lowest, boldest.",
        "icon": "fa-temperature-high",
        "tier": "silver",
        "category": "landmarks",
        "sort_order": 209,
    },
    {
        "slug": "place_everglades",
        "title": "Everglades Pathfinder",
        "description": "Explored Everglades country — river of grass, wild Florida.",
        "icon": "fa-frog",
        "tier": "silver",
        "category": "landmarks",
        "sort_order": 210,
    },
    {
        "slug": "place_blue_ridge",
        "title": "Blue Ridge Runner",
        "description": "Drove Blue Ridge Parkway — 469 miles of Appalachian sky.",
        "icon": "fa-road",
        "tier": "gold",
        "category": "landmarks",
        "sort_order": 211,
    },
    {
        "slug": "place_route_66",
        "title": "Route 66 Wayfarer",
        "description": "Touched Historic Route 66 — get your kicks on the mother road.",
        "icon": "fa-route",
        "tier": "gold",
        "category": "landmarks",
        "sort_order": 212,
    },
    {
        "slug": "place_arches",
        "title": "Arches Stone Wanderer",
        "description": "Ran Arches country — natural stone windows and desert light.",
        "icon": "fa-archway",
        "tier": "silver",
        "category": "landmarks",
        "sort_order": 213,
    },
    {
        "slug": "place_glacier",
        "title": "Glacier Ice Crown",
        "description": "Explored Glacier National Park — going-to-the-sun energy.",
        "icon": "fa-snowflake",
        "tier": "gold",
        "category": "landmarks",
        "sort_order": 214,
    },
    {
        "slug": "place_pacific_coast",
        "title": "Pacific Coastal Cruiser",
        "description": "Cruised Big Sur / Pacific Coast — cliffs, surf, and Highway 1.",
        "icon": "fa-umbrella-beach",
        "tier": "gold",
        "category": "landmarks",
        "sort_order": 215,
    },
    {
        "slug": "place_hoover_dam",
        "title": "Hoover Dam Iron Bridge",
        "description": "Passed Hoover Dam — Art Deco engineering on the Colorado.",
        "icon": "fa-industry",
        "tier": "silver",
        "category": "landmarks",
        "sort_order": 216,
    },
    {
        "slug": "place_great_smoky",
        "title": "Smoky Summit",
        "description": "Drove Great Smoky Mountains — mist, peaks, and old growth.",
        "icon": "fa-cloud",
        "tier": "gold",
        "category": "landmarks",
        "sort_order": 217,
    },
    {
        "slug": "place_mississippi",
        "title": "Mississippi Mighty",
        "description": "Ran along the Mississippi — America’s great river corridor.",
        "icon": "fa-water",
        "tier": "bronze",
        "category": "landmarks",
        "sort_order": 218,
    },
    {
        "slug": "place_crater_lake",
        "title": "Crater Lake Mirror",
        "description": "Explored Crater Lake country — deepest blue in the Cascades.",
        "icon": "fa-circle",
        "tier": "silver",
        "category": "landmarks",
        "sort_order": 219,
    },
]

# Geo + text matching for landmark badge awards (slug must match BADGE_SEED).
LANDMARK_MATCHERS: List[Dict[str, Any]] = [
    {"slug": "place_grand_canyon", "lat": 36.1069, "lon": -112.1129, "radius_km": 80, "keywords": ("grand canyon", "south rim", "north rim")},
    {"slug": "place_yosemite", "lat": 37.8651, "lon": -119.5383, "radius_km": 70, "keywords": ("yosemite", "el capitan", "half dome")},
    {"slug": "place_yellowstone", "lat": 44.4280, "lon": -110.5885, "radius_km": 90, "keywords": ("yellowstone", "old faithful", "geyser")},
    {"slug": "place_rushmore", "lat": 43.8791, "lon": -103.4591, "radius_km": 45, "keywords": ("mount rushmore", "rushmore", "black hills")},
    {"slug": "place_golden_gate", "lat": 37.8199, "lon": -122.4783, "radius_km": 35, "keywords": ("golden gate", "marin headlands", "presidio")},
    {"slug": "place_niagara", "lat": 43.0962, "lon": -79.0377, "radius_km": 40, "keywords": ("niagara", "niagara falls")},
    {"slug": "place_great_lakes", "lat": 43.0, "lon": -87.0, "radius_km": 120, "keywords": ("lake michigan", "lake superior", "lake erie", "lake huron", "great lakes")},
    {"slug": "place_rockies", "lat": 40.3428, "lon": -105.6836, "radius_km": 75, "keywords": ("rocky mountain", "estes park", "trail ridge")},
    {"slug": "place_monument_valley", "lat": 36.9980, "lon": -110.0985, "radius_km": 60, "keywords": ("monument valley", "mitten butte", "navajo nation")},
    {"slug": "place_death_valley", "lat": 36.5054, "lon": -117.0794, "radius_km": 80, "keywords": ("death valley", "badwater", "furnace creek")},
    {"slug": "place_everglades", "lat": 25.2866, "lon": -80.8987, "radius_km": 70, "keywords": ("everglades", "florida bay", "ten thousand islands")},
    {"slug": "place_blue_ridge", "lat": 35.5951, "lon": -82.5515, "radius_km": 100, "keywords": ("blue ridge parkway", "blue ridge", "linville gorge")},
    {"slug": "place_route_66", "lat": 35.2220, "lon": -101.8313, "radius_km": 150, "keywords": ("route 66", "route66", "historic route", "mother road")},
    {"slug": "place_arches", "lat": 38.7331, "lon": -109.5925, "radius_km": 55, "keywords": ("arches national", "delicate arch", "moab")},
    {"slug": "place_glacier", "lat": 48.7596, "lon": -113.7870, "radius_km": 75, "keywords": ("glacier national", "going-to-the-sun", "going to the sun")},
    {"slug": "place_pacific_coast", "lat": 36.2704, "lon": -121.8081, "radius_km": 80, "keywords": ("big sur", "pacific coast", "highway 1", "pacific coast highway", "17-mile")},
    {"slug": "place_hoover_dam", "lat": 36.0161, "lon": -114.7377, "radius_km": 40, "keywords": ("hoover dam", "boulder dam", "lake mead")},
    {"slug": "place_great_smoky", "lat": 35.6532, "lon": -83.5070, "radius_km": 80, "keywords": ("great smoky", "smoky mountain", "smokies", "clingmans dome")},
    {"slug": "place_mississippi", "lat": 29.9511, "lon": -90.0715, "radius_km": 100, "keywords": ("mississippi river", "mississippi", "river road")},
    {"slug": "place_crater_lake", "lat": 42.9446, "lon": -122.1090, "radius_km": 55, "keywords": ("crater lake", "rim drive", "wizard island")},
]

_BADGE_PRESENTED_BY: Dict[str, str] = {
    b["slug"]: b.get("presented_by", PRESENTED_BY_DEFAULT) for b in BADGE_SEED
}


def public_driver_id(user_id: str) -> str:
    return hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:16]


async def resolve_user_from_public_id(conn: Any, public_id: str) -> Optional[str]:
    """Resolve opaque public id to user_id by scanning opted-in users (small community)."""
    pid = (public_id or "").strip().lower()
    if not pid or len(pid) < 8:
        return None
    rows = await conn.fetch(
        """
        SELECT user_id::text FROM user_preferences
        WHERE COALESCE(trill_leaderboard_opt_in, FALSE) = TRUE
        """
    )
    for r in rows:
        uid = str(r["user_id"])
        if public_driver_id(uid) == pid:
            return uid
    return None


_badge_defs_synced_at: float = 0.0
_badge_defs_sync_interval_sec = 300.0
_current_season_cached_slug: Optional[str] = None
_current_season_cached_id: int = 0


async def ensure_badge_definitions(conn: Any) -> None:
    global _badge_defs_synced_at
    now = time.time()
    if now - _badge_defs_synced_at < _badge_defs_sync_interval_sec:
        return
    for b in BADGE_SEED:
        await conn.execute(
            """
            INSERT INTO trill_badge_definitions (slug, title, description, icon, tier, category, sort_order)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (slug) DO UPDATE SET
                title = EXCLUDED.title,
                description = EXCLUDED.description,
                icon = EXCLUDED.icon,
                tier = EXCLUDED.tier,
                category = EXCLUDED.category,
                sort_order = EXCLUDED.sort_order
            """,
            b["slug"],
            b["title"],
            b.get("description") or b["title"],
            b["icon"],
            b["tier"],
            b["category"],
            b["sort_order"],
        )
    _badge_defs_synced_at = now


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _badge_row_to_dict(r: Any) -> Dict[str, Any]:
    meta = r.get("meta") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    earned = r.get("earned_at")
    earned_iso = earned.isoformat() if earned else None
    earned_date = None
    earned_time = None
    if earned:
        if earned.tzinfo is None:
            earned = earned.replace(tzinfo=timezone.utc)
        earned_date = earned.strftime("%b %d, %Y")
        earned_time = earned.strftime("%I:%M %p UTC")
    return {
        "slug": r["slug"],
        "title": r["title"],
        "description": r.get("description") or r["title"],
        "icon": r["icon"],
        "tier": r["tier"],
        "category": r.get("category") or "general",
        "earned_at": earned_iso,
        "earned_date": earned_date,
        "earned_time": earned_time,
        "presented_to": meta.get("presented_to") or "",
        "presented_by": meta.get("presented_by") or PRESENTED_BY_DEFAULT,
        "place_name": meta.get("place_name"),
    }


async def _driver_handle_for_user(conn: Any, user_id: str) -> str:
    pref = await conn.fetchrow(
        """
        SELECT trill_public_name, trill_public_name_status
        FROM user_preferences WHERE user_id = $1
        """,
        user_id,
    )
    return _driver_handle_from_pref(
        user_id,
        pref.get("trill_public_name") if pref else None,
        str(pref.get("trill_public_name_status") or "none") if pref else "none",
    )


async def award_badge(
    conn: Any,
    user_id: str,
    slug: str,
    *,
    presented_by: Optional[str] = None,
    presented_to: Optional[str] = None,
    place_name: Optional[str] = None,
    upload_id: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> bool:
    """Award a badge once; returns True if newly inserted."""
    await ensure_badge_definitions(conn)
    handle = presented_to or await _driver_handle_for_user(conn, user_id)
    meta: Dict[str, Any] = {
        "presented_to": handle,
        "presented_by": presented_by or _BADGE_PRESENTED_BY.get(slug, PRESENTED_BY_DEFAULT),
    }
    if place_name:
        meta["place_name"] = place_name
    if upload_id:
        meta["proof_upload_id"] = upload_id
    if extra_meta:
        meta.update(extra_meta)
    row = await conn.fetchrow(
        """
        INSERT INTO trill_user_badges (user_id, badge_id, meta)
        SELECT $1::uuid, id, $3::jsonb FROM trill_badge_definitions WHERE slug = $2
        ON CONFLICT (user_id, badge_id) DO NOTHING
        RETURNING user_id
        """,
        user_id,
        slug,
        json.dumps(meta),
    )
    return row is not None


async def fetch_badges_for_users(conn: Any, user_ids: Sequence[str], limit_per_user: int = 3) -> Dict[str, List[Dict]]:
    if not user_ids:
        return {}
    out: Dict[str, List[Dict]] = {uid: [] for uid in user_ids}
    rows = await conn.fetch(
        """
        SELECT ub.user_id::text AS user_id, d.slug, d.title, d.description, d.icon, d.tier, d.category,
               ub.earned_at, ub.meta,
               ROW_NUMBER() OVER (PARTITION BY ub.user_id ORDER BY d.sort_order) AS rn
        FROM trill_user_badges ub
        INNER JOIN trill_badge_definitions d ON d.id = ub.badge_id AND d.is_active = TRUE
        WHERE ub.user_id = ANY($1::uuid[])
        ORDER BY ub.user_id, d.sort_order
        """,
        list(user_ids),
    )
    for r in rows:
        if int(r["rn"]) > limit_per_user:
            continue
        uid = str(r["user_id"])
        out.setdefault(uid, []).append(_badge_row_to_dict(dict(r)))
    return out


async def fetch_user_badge_collection(conn: Any, user_id: str) -> List[Dict]:
    """Full earned badge list for profile / showcase."""
    rows = await conn.fetch(
        """
        SELECT d.slug, d.title, d.description, d.icon, d.tier, d.category,
               ub.earned_at, ub.meta
        FROM trill_user_badges ub
        INNER JOIN trill_badge_definitions d ON d.id = ub.badge_id AND d.is_active = TRUE
        WHERE ub.user_id = $1::uuid
        ORDER BY d.sort_order, ub.earned_at DESC
        """,
        user_id,
    )
    return [_badge_row_to_dict(dict(r)) for r in rows]


async def fetch_badge_catalog(conn: Any) -> List[Dict]:
    """All active badge definitions (earned or not)."""
    await ensure_badge_definitions(conn)
    rows = await conn.fetch(
        """
        SELECT slug, title, description, icon, tier, category, sort_order
        FROM trill_badge_definitions
        WHERE is_active = TRUE
        ORDER BY sort_order
        """
    )
    return [dict(r) for r in rows]


async def evaluate_and_award_badges(
    conn: Any,
    user_id: str,
    *,
    viewer_stats: Optional[Dict] = None,
    since: Optional[datetime] = None,
) -> None:
    """Lightweight badge evaluation for one user (leaderboard load / upload hook)."""
    await ensure_badge_definitions(conn)
    uid = user_id
    stats = viewer_stats
    if stats is None and since is not None:
        row = await conn.fetchrow(
            f"""
            SELECT
                COUNT(*)::int AS run_count,
                COALESCE(MAX(trill_score), 0)::float AS best_trill,
                COALESCE(MAX(max_speed_mph), 0)::float AS best_speed,
                COALESCE(SUM(distance_miles), 0)::float AS total_distance
            FROM uploads u
            WHERE u.user_id = $1 AND u.created_at >= $2 AND {TRILL_ROUTE_PREDICATE.strip()}
            """,
            uid,
            since,
        )
        stats = dict(row) if row else {}
    if not stats:
        return

    pref = await conn.fetchrow(
        """
        SELECT trill_leaderboard_opt_in, trill_public_name_status,
               trill_public_name
        FROM user_preferences WHERE user_id = $1
        """,
        uid,
    )
    opted = bool(pref and pref["trill_leaderboard_opt_in"])
    best_trill = float(stats.get("best_trill") or stats.get("best_trill_score") or 0)
    best_speed = float(stats.get("best_speed") or stats.get("best_speed_mph") or 0)
    run_count = int(stats.get("run_count") or 0)
    total_dist = float(stats.get("total_distance") or stats.get("total_distance_miles") or 0)
    rank = stats.get("rank")

    candidates: List[str] = []
    if run_count >= 1:
        candidates.append("map_unlocked")
    if best_trill >= 60:
        candidates.append("first_send_it")
    if best_trill >= 90:
        candidates.append("glory_boy")
    if best_speed >= 100:
        candidates.append("century_club")
    if best_speed >= 110:
        candidates.append("speed_demon")
    if total_dist >= 1000:
        candidates.append("road_warrior")
    if run_count >= 100:
        candidates.append("century_driver")
    if opted:
        candidates.append("on_the_board")
    if rank is not None and int(rank) <= 10:
        candidates.append("top_10")
    if rank is not None and int(rank) <= 3:
        candidates.append("podium")
    if pref and str(pref.get("trill_public_name_status") or "").lower() == "approved":
        candidates.append("name_on_the_door")

    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    wk = await conn.fetchval(
        f"""
        SELECT COUNT(*)::int FROM uploads u
        WHERE u.user_id = $1 AND u.created_at >= $2 AND {TRILL_ROUTE_PREDICATE.strip()}
        """,
        uid,
        week_ago,
    )
    if (wk or 0) >= 7:
        candidates.append("weekly_grinder")

    for slug in candidates:
        try:
            await award_badge(conn, uid, slug)
        except Exception as e:
            logger.debug("badge award %s: %s", slug, e)

    try:
        await evaluate_landmark_badges(conn, uid, since=since)
    except Exception as e:
        logger.debug("landmark badges: %s", e)


def _extract_upload_coords(meta: Any) -> Tuple[Optional[float], Optional[float]]:
    if not meta:
        return None, None
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            return None, None
    if not isinstance(meta, dict):
        return None, None
    tel = meta.get("telemetry") or {}
    for lat_k, lon_k in (
        ("mid_lat", "mid_lon"),
        ("place_lat", "place_lon"),
        ("start_lat", "start_lon"),
    ):
        try:
            lat = tel.get(lat_k) if lat_k in tel else meta.get(lat_k)
            lon = tel.get(lon_k) if lon_k in tel else meta.get(lon_k)
            if lat is not None and lon is not None:
                return float(lat), float(lon)
        except (TypeError, ValueError):
            continue
    return None, None


def _upload_text_blob(meta: Any) -> str:
    if not meta:
        return ""
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            return meta.lower()
    parts: List[str] = []
    if isinstance(meta, dict):
        for key in ("place_name", "protected_name", "trill_place", "state", "state_usps"):
            v = meta.get(key)
            if v:
                parts.append(str(v))
        tel = meta.get("telemetry") or {}
        for key in (
            "gazetteer_place_name",
            "gazetteer_state_usps",
            "padus_unit_name",
            "location_state",
        ):
            v = tel.get(key)
            if v:
                parts.append(str(v))
        vc = meta.get("vision_context") or tel.get("vision_context") or {}
        for lm in vc.get("landmark_names") or []:
            parts.append(str(lm))
        scenic = meta.get("scenic_boost") or tel.get("scenic_boost") or {}
        for f in scenic.get("factors") or []:
            parts.append(str(f))
    return " ".join(parts).lower()


def _upload_matches_landmark(lat: Optional[float], lon: Optional[float], text: str, matcher: Dict) -> bool:
    if lat is not None and lon is not None:
        dist = _haversine_km(lat, lon, matcher["lat"], matcher["lon"])
        if dist <= float(matcher.get("radius_km") or 50):
            return True
    for kw in matcher.get("keywords") or ():
        if kw in text:
            return True
    return False


async def evaluate_landmark_badges(conn: Any, user_id: str, *, since: Optional[datetime] = None) -> None:
    """Award place/landmark badges when map runs pass near iconic US locations."""
    since_clause = "AND u.created_at >= $2" if since else ""
    params: List[Any] = [user_id]
    if since:
        params.append(since)
    rows = await conn.fetch(
        f"""
        SELECT u.id::text AS upload_id, u.trill_metadata
        FROM uploads u
        WHERE u.user_id = $1::uuid {since_clause}
          AND {TRILL_ROUTE_PREDICATE.strip()}
        ORDER BY u.created_at DESC
        LIMIT 200
        """,
        *params,
    )
    awarded: Set[str] = set()
    badge_titles = {b["slug"]: b["title"] for b in BADGE_SEED}
    for row in rows:
        meta = row["trill_metadata"]
        lat, lon = _extract_upload_coords(meta)
        text = _upload_text_blob(meta)
        for matcher in LANDMARK_MATCHERS:
            slug = matcher["slug"]
            if slug in awarded:
                continue
            if _upload_matches_landmark(lat, lon, text, matcher):
                ok = await award_badge(
                    conn,
                    user_id,
                    slug,
                    place_name=badge_titles.get(slug),
                    upload_id=row["upload_id"],
                    presented_by="UploadM8 Scenic Routes",
                )
                if ok:
                    awarded.add(slug)


def compute_chase_targets(
    rows: List[Dict],
    viewer: Dict,
    sort_key: str = "best_trill",
) -> List[Dict]:
    """Drivers immediately above viewer on the active sort metric."""
    if not viewer or not viewer.get("rank") or not rows:
        return []
    vr = int(viewer["rank"])
    if vr <= 1:
        return []

    def metric(r: Dict, key: str) -> float:
        m = {
            "best_trill": "best_trill_score",
            "speed": "best_speed_mph",
            "runs": "run_count",
            "distance": "total_distance_miles",
            "avg_trill": "avg_trill_score",
        }
        return float(r.get(m.get(key, "best_trill_score")) or 0)

    my_val = metric(
        next((r for r in rows if r.get("is_you")), viewer),
        sort_key,
    )
    above = [r for r in rows if int(r.get("rank") or 999) < vr]
    above.sort(key=lambda r: int(r.get("rank") or 0), reverse=True)
    targets = []
    label = {
        "best_trill": "Trill",
        "speed": "mph",
        "runs": "runs",
        "distance": "mi",
        "avg_trill": "avg Trill",
    }.get(sort_key, "Trill")
    for r in above[:2]:
        their = metric(r, sort_key)
        delta = round(their - my_val, 1)
        if delta <= 0:
            continue
        targets.append(
            {
                "rank": int(r["rank"]),
                "driver_handle": r.get("driver_handle"),
                "public_id": r.get("public_id"),
                "delta": delta,
                "delta_label": label,
                "sort_key": sort_key,
            }
        )
    return targets


async def fetch_recent_scores_batch(
    conn: Any,
    user_ids: Sequence[str],
    since: datetime,
    limit_per_user: int = 7,
) -> Dict[str, List[float]]:
    if not user_ids:
        return {}
    rows = await conn.fetch(
        f"""
        SELECT user_id::text, trill_score::float AS score, created_at
        FROM (
            SELECT u.user_id, u.trill_score, u.created_at,
                   ROW_NUMBER() OVER (PARTITION BY u.user_id ORDER BY u.created_at DESC) AS rn
            FROM uploads u
            WHERE u.user_id = ANY($1::uuid[])
              AND u.created_at >= $2
              AND {TRILL_ROUTE_PREDICATE.strip()}
        ) sub
        WHERE rn <= $3
        ORDER BY user_id, created_at ASC
        """,
        list(user_ids),
        since,
        limit_per_user,
    )
    out: Dict[str, List[float]] = {}
    for r in rows:
        uid = str(r["user_id"])
        out.setdefault(uid, []).append(float(r["score"] or 0))
    return out


async def fetch_region_options(conn: Any, since: datetime) -> List[Dict]:
    state_sql = """NULLIF(TRIM(COALESCE(
        NULLIF(btrim(u.trill_metadata#>>'{telemetry,gazetteer_state_usps}}'), ''),
        NULLIF(btrim(u.trill_metadata#>>'{telemetry,location_state}}'), ''),
        NULLIF(btrim(u.trill_metadata#>>'{state_usps}}'), ''),
        NULLIF(btrim(u.trill_metadata#>>'{state}}'), '')
    )), '')"""
    rows = await conn.fetch(
        f"""
        SELECT {state_sql} AS state, COUNT(DISTINCT u.user_id)::int AS drivers
        FROM uploads u
        INNER JOIN user_preferences pref ON pref.user_id = u.user_id
            AND COALESCE(pref.trill_leaderboard_opt_in, FALSE) = TRUE
        WHERE u.created_at >= $1 AND {TRILL_ROUTE_PREDICATE.strip()}
          AND {state_sql} IS NOT NULL
        GROUP BY 1
        HAVING COUNT(DISTINCT u.user_id) >= 1
        ORDER BY drivers DESC, state ASC
        LIMIT 60
        """,
        since,
    )
    return [{"code": r["state"], "label": r["state"], "driver_count": int(r["drivers"])} for r in rows]


def current_season_slug(dt: Optional[datetime] = None) -> str:
    d = dt or datetime.now(timezone.utc)
    return d.strftime("%Y-%m")


async def ensure_current_season(conn: Any) -> int:
    global _current_season_cached_slug, _current_season_cached_id
    slug = current_season_slug()
    if _current_season_cached_slug == slug and _current_season_cached_id:
        return _current_season_cached_id
    start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    row = await conn.fetchrow(
        """
        INSERT INTO trill_seasons (slug, starts_at, ends_at, status)
        VALUES ($1, $2, $3, 'active')
        ON CONFLICT (slug) DO UPDATE SET status = 'active'
        RETURNING id
        """,
        slug,
        start,
        end,
    )
    season_id = int(row["id"]) if row else 0
    if season_id:
        _current_season_cached_slug = slug
        _current_season_cached_id = season_id
    return season_id


async def fetch_hall_of_fame(conn: Any, limit: int = 24) -> List[Dict]:
    rows = await conn.fetch(
        """
        SELECT h.rank, h.driver_handle, h.best_trill_score, s.slug AS season_slug
        FROM trill_hall_of_fame h
        INNER JOIN trill_seasons s ON s.id = h.season_id
        ORDER BY s.starts_at DESC, h.rank ASC
        LIMIT $1
        """,
        limit,
    )
    return [
        {
            "rank": int(r["rank"]),
            "driver_handle": r["driver_handle"],
            "best_trill_score": float(r["best_trill_score"] or 0),
            "season_slug": r["season_slug"],
        }
        for r in rows
    ]


async def fetch_rivals(conn: Any, user_id: str) -> List[str]:
    rows = await conn.fetch(
        "SELECT rival_user_id::text FROM trill_rivals WHERE user_id = $1 ORDER BY created_at",
        user_id,
    )
    return [str(r["rival_user_id"]) for r in rows]


async def add_rival(conn: Any, user_id: str, rival_user_id: str) -> bool:
    cnt = await conn.fetchval("SELECT COUNT(*)::int FROM trill_rivals WHERE user_id = $1", user_id)
    if (cnt or 0) >= 3:
        return False
    await conn.execute(
        """
        INSERT INTO trill_rivals (user_id, rival_user_id)
        VALUES ($1, $2)
        ON CONFLICT DO NOTHING
        """,
        user_id,
        rival_user_id,
    )
    return True


async def remove_rival(conn: Any, user_id: str, rival_user_id: str) -> None:
    await conn.execute(
        "DELETE FROM trill_rivals WHERE user_id = $1 AND rival_user_id = $2",
        user_id,
        rival_user_id,
    )


async def ensure_weekly_challenge(conn: Any) -> Optional[Dict]:
    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    row = await conn.fetchrow(
        "SELECT * FROM trill_weekly_challenges WHERE week_start = $1",
        week_start,
    )
    if not row:
        # Rotate challenge types by ISO week number
        wn = week_start.isocalendar()[1]
        kinds = [
            ("max_speed", 90.0, 25, 0, "Speed Week", "Hit 90 mph on a map-backed run."),
            ("trill_score", 75.0, 0, 50, "Trill Week", "Score 75+ Trill on one run."),
            ("run_count", 3.0, 25, 0, "Volume Week", "Log 3 map-backed runs this week."),
        ]
        k = kinds[wn % len(kinds)]
        row = await conn.fetchrow(
            """
            INSERT INTO trill_weekly_challenges
                (week_start, challenge_type, target_value, reward_put, reward_aic, title, description)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING *
            """,
            week_start,
            k[0],
            k[1],
            k[2],
            k[3],
            k[4],
            k[5],
        )
    if not row:
        return None
    return dict(row)


async def check_challenge_for_user(conn: Any, user_id: str, challenge: Dict) -> Optional[Dict]:
    """Return completion dict if user just qualified; awards badge slug separately."""
    cid = int(challenge["id"])
    done = await conn.fetchval(
        "SELECT 1 FROM trill_challenge_completions WHERE user_id = $1 AND challenge_id = $2",
        user_id,
        cid,
    )
    if done:
        return None
    week_start = challenge["week_start"]
    week_end = week_start + timedelta(days=7)
    ctype = challenge["challenge_type"]
    target = float(challenge["target_value"])
    row = None
    if ctype == "max_speed":
        row = await conn.fetchrow(
            f"""
            SELECT id FROM uploads u
            WHERE u.user_id = $1 AND u.created_at >= $2 AND u.created_at < $3
              AND {TRILL_ROUTE_PREDICATE.strip()}
              AND COALESCE(u.max_speed_mph, 0) >= $4
            LIMIT 1
            """,
            user_id,
            week_start,
            week_end,
            target,
        )
    elif ctype == "trill_score":
        row = await conn.fetchrow(
            f"""
            SELECT id FROM uploads u
            WHERE u.user_id = $1 AND u.created_at >= $2 AND u.created_at < $3
              AND {TRILL_ROUTE_PREDICATE.strip()}
              AND u.trill_score >= $4
            LIMIT 1
            """,
            user_id,
            week_start,
            week_end,
            target,
        )
    elif ctype == "run_count":
        cnt = await conn.fetchval(
            f"""
            SELECT COUNT(*)::int FROM uploads u
            WHERE u.user_id = $1 AND u.created_at >= $2 AND u.created_at < $3
              AND {TRILL_ROUTE_PREDICATE.strip()}
            """,
            user_id,
            week_start,
            week_end,
        )
        row = {"id": None} if (cnt or 0) >= int(target) else None
    if not row:
        return None
    proof = str(row["id"]) if row.get("id") else None
    await conn.execute(
        """
        INSERT INTO trill_challenge_completions (user_id, challenge_id, proof_upload_id)
        VALUES ($1, $2, $3::uuid)
        ON CONFLICT DO NOTHING
        """,
        user_id,
        cid,
        proof,
    )
    return {
        "challenge_id": cid,
        "title": challenge.get("title"),
        "reward_put": int(challenge.get("reward_put") or 0),
        "reward_aic": int(challenge.get("reward_aic") or 0),
    }


SORT_LABELS = {
    "best_trill": "best Trill",
    "speed": "top speed",
    "runs": "most runs",
    "distance": "total distance",
    "avg_trill": "avg Trill",
}

OVERTAKE_COOLDOWN = timedelta(hours=24)


def _driver_handle_from_pref(user_id: str, trill_public_name, status: str) -> str:
    approved = str(status or "none").strip().lower() == "approved"
    pub = (trill_public_name or "").strip()
    if approved and pub:
        return pub
    return f"Driver-{hashlib.md5(str(user_id).encode('utf-8')).hexdigest()[:6]}"


async def archive_due_seasons(conn: Any, top_n: int = 10) -> int:
    """Archive seasons whose window has ended; returns count archived."""
    due = await conn.fetch(
        """
        SELECT id, slug, starts_at, ends_at
        FROM trill_seasons
        WHERE status = 'active' AND ends_at <= NOW()
        ORDER BY ends_at ASC
        """
    )
    archived = 0
    for season in due:
        sid = int(season["id"])
        since = season["starts_at"]
        until = season["ends_at"]
        rows = await conn.fetch(
            f"""
            WITH agg AS (
                SELECT u.user_id,
                       MAX(u.trill_score)::float AS best_trill
                FROM uploads u
                INNER JOIN user_preferences pref ON pref.user_id = u.user_id
                    AND COALESCE(pref.trill_leaderboard_opt_in, FALSE) = TRUE
                WHERE u.created_at >= $1 AND u.created_at < $2
                  AND {TRILL_ROUTE_PREDICATE.strip()}
                GROUP BY u.user_id
            ),
            ranked AS (
                SELECT user_id, best_trill,
                       ROW_NUMBER() OVER (ORDER BY best_trill DESC NULLS LAST)::int AS rank
                FROM agg
            )
            SELECT r.user_id::text, r.best_trill, r.rank,
                   NULLIF(btrim(pub.trill_public_name), '') AS trill_public_name,
                   pub.trill_public_name_status
            FROM ranked r
            LEFT JOIN user_preferences pub ON pub.user_id = r.user_id
            WHERE r.rank <= $3
            ORDER BY r.rank
            """,
            since,
            until,
            top_n,
        )
        for r in rows:
            handle = _driver_handle_from_pref(
                r["user_id"], r.get("trill_public_name"), r.get("trill_public_name_status")
            )
            await conn.execute(
                """
                INSERT INTO trill_hall_of_fame
                    (season_id, rank, user_id, driver_handle, best_trill_score, category)
                VALUES ($1, $2, $3::uuid, $4, $5, 'overall')
                ON CONFLICT (season_id, rank, category) DO UPDATE SET
                    driver_handle = EXCLUDED.driver_handle,
                    best_trill_score = EXCLUDED.best_trill_score,
                    user_id = EXCLUDED.user_id
                """,
                sid,
                int(r["rank"]),
                r["user_id"],
                handle[:64],
                round(float(r["best_trill"] or 0), 1),
            )
        await conn.execute(
            "UPDATE trill_seasons SET status = 'archived' WHERE id = $1",
            sid,
        )
        archived += 1
    if archived:
        await ensure_current_season(conn)
    return archived


async def _recent_overtake_notify(
    conn: Any, user_id: str, rival_user_id: str, sort_key: str
) -> bool:
    return bool(
        await conn.fetchval(
            """
            SELECT 1 FROM trill_notifications
            WHERE user_id = $1 AND kind = 'rival_overtake'
              AND rival_user_id = $2::uuid AND sort_key = $3
              AND created_at > NOW() - INTERVAL '24 hours'
            LIMIT 1
            """,
            user_id,
            rival_user_id,
            sort_key,
        )
    )


async def _write_in_app_rival_alert(
    conn: Any, user_id: str, title: str, body: str, meta: Dict[str, Any]
) -> None:
    await conn.execute(
        """
        INSERT INTO marketing_touchpoint_deliveries
            (id, user_id, channel, subject, body_text, status, meta)
        VALUES ($1::uuid, $2::uuid, 'in_app', $3, $4, 'pending', $5::jsonb)
        """,
        uuid.uuid4(),
        user_id,
        title[:500],
        body[:500],
        json.dumps({**meta, "source": "trill_rival_overtake"}),
    )


async def process_rival_rank_changes(
    conn: Any,
    user_id: str,
    sort_key: str,
    rank_by_uid: Dict[str, int],
    viewer_rank: Optional[int],
    rival_handles: Dict[str, str],
    *,
    db_pool: Any = None,
) -> List[Dict[str, Any]]:
    """
    Compare current ranks to trill_lb_snapshot; notify when a tracked rival passes the viewer.
    Returns alerts for the current request (toast in UI).
    """
    if viewer_rank is None:
        return []
    uid = str(user_id)
    sort_key = (sort_key or "best_trill").strip()
    if sort_key not in SORT_LABELS:
        sort_key = "best_trill"

    pref = await conn.fetchrow(
        "SELECT trill_lb_snapshot FROM user_preferences WHERE user_id = $1",
        uid,
    )
    snapshot = pref.get("trill_lb_snapshot") if pref else {}
    if isinstance(snapshot, str):
        try:
            snapshot = json.loads(snapshot)
        except Exception:
            snapshot = {}
    if not isinstance(snapshot, dict):
        snapshot = {}

    prev = snapshot.get(sort_key) or {}
    if not prev.get("my_rank"):
        rival_ranks = {rid: rank_by_uid.get(rid) for rid in rival_handles if rank_by_uid.get(rid)}
        snapshot[sort_key] = {"my_rank": int(viewer_rank), "rivals": rival_ranks}
        await conn.execute(
            "UPDATE user_preferences SET trill_lb_snapshot = $2::jsonb WHERE user_id = $1",
            uid,
            json.dumps(snapshot),
        )
        return []

    rival_ids = list(rival_handles.keys())
    alerts: List[Dict[str, Any]] = []
    old_my = int(prev.get("my_rank") or 999)
    old_rivals = prev.get("rivals") or {}
    new_my = int(viewer_rank)

    user_row = await conn.fetchrow(
        "SELECT email, name FROM users WHERE id = $1::uuid",
        uid,
    )

    for rid in rival_ids:
        new_r = rank_by_uid.get(rid)
        if new_r is None:
            continue
        old_r = old_rivals.get(rid)
        if new_r >= new_my:
            continue
        if old_r is not None and old_r <= old_my:
            continue
        if await _recent_overtake_notify(conn, uid, rid, sort_key):
            continue

        handle = rival_handles.get(rid) or "A rival"
        sort_label = SORT_LABELS.get(sort_key, sort_key)
        title = f"{handle} passed you"
        body = (
            f"{handle} is now #{new_r} on the board (you are #{new_my}, {sort_label}). "
            "Climb back on the Trill leaderboard."
        )
        meta = {
            "rival_user_id": rid,
            "rival_rank": new_r,
            "your_rank": new_my,
            "sort_key": sort_key,
            "rival_handle": handle,
        }
        await conn.execute(
            """
            INSERT INTO trill_notifications (user_id, kind, rival_user_id, sort_key, meta)
            VALUES ($1::uuid, 'rival_overtake', $2::uuid, $3, $4::jsonb)
            """,
            uid,
            rid,
            sort_key,
            json.dumps(meta),
        )
        await _write_in_app_rival_alert(conn, uid, title, body, meta)
        alerts.append(
            {
                "rival_handle": handle,
                "rival_rank": new_r,
                "your_rank": new_my,
                "sort_key": sort_key,
                "rival_public_id": public_driver_id(rid),
            }
        )

        if user_row and user_row.get("email"):
            try:
                from stages.emails.trill import send_trill_rival_overtake_email

                await send_trill_rival_overtake_email(
                    user_row["email"],
                    user_row.get("name") or "there",
                    handle,
                    new_r,
                    new_my,
                    sort_label,
                )
            except Exception as ex:
                logger.debug("rival overtake email: %s", ex)

        if db_pool:
            try:
                import httpx
                from stages.notify_stage import fetch_user_discord_webhook_from_db

                wh = await fetch_user_discord_webhook_from_db(db_pool, uid)
                if wh:
                    msg = f"⚔ **Trill leaderboard** — {handle} passed you (#{new_r} vs your #{new_my}, {sort_label})."
                    async with httpx.AsyncClient(timeout=12) as client:
                        await client.post(wh, json={"content": msg[:1800]})
            except Exception as ex:
                logger.debug("rival overtake discord: %s", ex)

    rival_ranks = {rid: rank_by_uid.get(rid) for rid in rival_ids if rank_by_uid.get(rid)}
    snapshot[sort_key] = {"my_rank": new_my, "rivals": rival_ranks}
    await conn.execute(
        "UPDATE user_preferences SET trill_lb_snapshot = $2::jsonb WHERE user_id = $1",
        uid,
        json.dumps(snapshot),
    )
    return alerts
