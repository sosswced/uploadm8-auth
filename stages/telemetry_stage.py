"""
UploadM8 Telemetry Stage
========================
Parses .map telemetry files, calculates Trill score,
and performs reverse geocoding to extract location name
for use in AI caption/title/hashtag generation.
"""

import logging
import asyncio
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from core.config import GAZETTEER_PLACES_PATH

from .errors import TelemetryError, SkipStage, ErrorCode
from .context import JobContext, TelemetryData, TrillScore


logger = logging.getLogger("uploadm8-worker")


# Default thresholds
DEFAULT_SPEEDING_MPH = 80
DEFAULT_EUPHORIA_MPH = 100

# Nominatim reverse geocoding (free, no API key required)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
GEOCODE_TIMEOUT = 8  # seconds


def _parse_timestamp(raw: Any, fallback: float) -> float:
    """Parse telemetry timestamp (seconds/epoch/ISO)."""
    if raw is None:
        return fallback
    s = str(raw).strip()
    if not s:
        return fallback
    try:
        return float(s)
    except ValueError:
        pass
    try:
        # Accept ISO-8601 with optional trailing Z.
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return fallback


def _to_float(raw: Any) -> Optional[float]:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _looks_like_header(parts: List[str]) -> bool:
    lower = [p.strip().lower() for p in parts]
    wanted = {"lat", "latitude", "lon", "lng", "long", "longitude", "speed", "speed_mph", "mph", "kph", "kmh"}
    return any(x in wanted for x in lower)


def _normalize_header(parts: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i, p in enumerate(parts):
        key = p.strip().lower()
        if key:
            out[key] = i
    return out


def _pick_col(row: List[str], hdr: Dict[str, int], *aliases: str) -> Optional[str]:
    for a in aliases:
        idx = hdr.get(a)
        if idx is not None and 0 <= idx < len(row):
            return row[idx]
    return None


def _parse_point_from_row(
    row: List[str],
    *,
    fallback_ts: float,
    hdr: Optional[Dict[str, int]] = None,
) -> Optional[Dict[str, float]]:
    """Parse one telemetry line from either header-aware or positional formats."""
    if hdr:
        lat = _to_float(_pick_col(row, hdr, "lat", "latitude"))
        lon = _to_float(_pick_col(row, hdr, "lon", "lng", "long", "longitude"))
        speed = _to_float(_pick_col(row, hdr, "speed_mph", "mph", "speed", "kph", "kmh"))
        if speed is None:
            return None
        # Convert kph/kmh to mph if that's the mapped speed source.
        speed_src = None
        for cand in ("speed_mph", "mph", "speed", "kph", "kmh"):
            if hdr.get(cand) is not None:
                speed_src = cand
                break
        if speed_src in {"kph", "kmh"}:
            speed *= 0.621371

        ts_raw = _pick_col(row, hdr, "timestamp", "time", "ts", "seconds", "epoch")
        ts = _parse_timestamp(ts_raw, fallback=fallback_ts)
        alt = _to_float(_pick_col(row, hdr, "altitude", "alt", "elevation", "elev")) or 0.0
    else:
        if len(row) < 4:
            return None
        lat = _to_float(row[1])
        lon = _to_float(row[2])
        speed = _to_float(row[3])
        if speed is None:
            return None
        ts = _parse_timestamp(row[0], fallback=fallback_ts)
        alt = _to_float(row[4]) if len(row) > 4 else 0.0
        alt = alt or 0.0

    if lat is None or lon is None:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return {
        "timestamp": float(ts),
        "lat": float(lat),
        "lon": float(lon),
        "speed_mph": float(speed),
        "altitude": float(alt),
    }


def parse_map_file(map_path: Path) -> TelemetryData:
    """
    Parse .map telemetry file.

    Expected CSV format: timestamp,lat,lon,speed_mph,altitude
    Lines starting with # are comments.

    Args:
        map_path: Path to .map file

    Returns:
        TelemetryData with parsed points

    Raises:
        TelemetryError: If parsing fails
    """
    data_points: List[Dict] = []

    try:
        hdr: Optional[Dict[str, int]] = None
        fallback_ts = 0.0
        with open(map_path, 'r', encoding='utf-8', errors='replace') as f:
            for _line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Prefer CSV; fall back to whitespace-delimited records.
                parts = [p.strip() for p in (line.split(',') if ',' in line else line.split())]
                if len(parts) < 3:
                    continue

                if hdr is None and _looks_like_header(parts):
                    hdr = _normalize_header(parts)
                    continue

                point = _parse_point_from_row(parts, fallback_ts=fallback_ts, hdr=hdr)
                if point:
                    data_points.append(point)
                    fallback_ts = float(point["timestamp"]) + 1.0

    except FileNotFoundError:
        raise TelemetryError(
            f"Telemetry file not found: {map_path}",
            code=ErrorCode.TELEMETRY_PARSE_FAILED
        )
    except Exception as e:
        raise TelemetryError(
            f"Failed to parse telemetry file: {e}",
            code=ErrorCode.TELEMETRY_PARSE_FAILED,
            detail=str(e)
        )

    if not data_points:
        raise TelemetryError(
            "No valid telemetry data found",
            code=ErrorCode.TELEMETRY_EMPTY
        )

    speeds = [p['speed_mph'] for p in data_points]
    total_duration = (
        data_points[-1]['timestamp'] - data_points[0]['timestamp']
        if len(data_points) > 1 else 0
    )

    # Estimate distance via time x avg speed
    total_distance = 0.0
    if len(data_points) > 1:
        for i in range(1, len(data_points)):
            dt = data_points[i]['timestamp'] - data_points[i - 1]['timestamp']
            avg_spd = (data_points[i]['speed_mph'] + data_points[i - 1]['speed_mph']) / 2
            total_distance += (avg_spd * dt) / 3600  # convert mph*seconds -> miles

    telemetry = TelemetryData(
        points=data_points,
        max_speed_mph=max(speeds),
        avg_speed_mph=sum(speeds) / len(speeds),
        total_distance_miles=total_distance,
        duration_seconds=total_duration,
        max_altitude_ft=max(p['altitude'] for p in data_points),
        speeding_seconds=0.0,
        euphoria_seconds=0.0,
    )

    logger.info(
        f"Parsed telemetry: {len(data_points)} points, "
        f"max_speed={telemetry.max_speed_mph:.1f} mph, "
        f"distance={telemetry.total_distance_miles:.2f} mi"
    )
    return telemetry


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in miles between two GPS points."""
    R = 3958.8  # Earth radius in miles
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(d_lon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_representative_coords(data_points: List[Dict]) -> Optional[Tuple[float, float]]:
    """
    Pick the most representative GPS coordinate for geocoding.
    Uses the midpoint of the route by time.
    """
    if not data_points:
        return None
    mid_idx = len(data_points) // 2
    pt = data_points[mid_idx]
    lat, lon = pt.get('lat'), pt.get('lon')
    if lat is None or lon is None:
        return None
    # Sanity check: valid globe coords
    if -90 <= lat <= 90 and -180 <= lon <= 180 and not (lat == 0 and lon == 0):
        return lat, lon
    return None


def _nominatim_address_parts(data: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Normalize Nominatim JSON into fields we persist on TelemetryData."""
    address = data.get("address") or {}
    if not isinstance(address, dict):
        address = {}
    city = (
        address.get("city")
        or address.get("town")
        or address.get("village")
        or address.get("suburb")
        or address.get("hamlet")
        or address.get("county")
        or ""
    )
    state = address.get("state") or address.get("region") or ""
    country = (address.get("country") or "").strip() or None
    country_code = (address.get("country_code") or "").strip().upper() or None
    road = (
        address.get("road")
        or address.get("pedestrian")
        or address.get("path")
        or address.get("motorway")
        or address.get("trunk")
        or address.get("primary")
        or ""
    )
    road = str(road).strip() or None
    display = ", ".join(p for p in [str(city).strip(), str(state).strip()] if p)
    if not display and country_code:
        display = country_code
    full_display = (data.get("display_name") or "").strip() or None
    return {
        "location_display": display or full_display,
        "location_city": str(city).strip() or None,
        "location_state": str(state).strip() or None,
        "location_country": country or country_code,
        "location_road": road,
        "nominatim_display_name": full_display,
    }


async def reverse_geocode_details(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Reverse geocode via Nominatim; return structured fields for TelemetryData + M8 scene graph.
    """
    try:
        async with httpx.AsyncClient(timeout=GEOCODE_TIMEOUT) as client:
            resp = await client.get(
                NOMINATIM_URL,
                params={
                    "lat": lat,
                    "lon": lon,
                    "format": "json",
                    "zoom": 18,
                    "addressdetails": 1,
                },
                headers={"User-Agent": "UploadM8/1.0 (uploadm8.com)"},
            )

            if resp.status_code != 200:
                logger.warning("Geocode HTTP %s for (%.4f, %.4f)", resp.status_code, lat, lon)
                return None

            data = resp.json()
            if not isinstance(data, dict):
                return None
            parts = _nominatim_address_parts(data)
            if parts.get("location_display"):
                logger.info(
                    "Geocoded (%.4f, %.4f) → %s",
                    lat,
                    lon,
                    parts["location_display"],
                )
            return parts

    except asyncio.TimeoutError:
        logger.warning("Geocode timeout for (%s, %s)", lat, lon)
        return None
    except Exception as e:
        logger.warning("Geocode error for (%s, %s): %s", lat, lon, e)
        return None


async def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """
    Reverse geocode coordinates to a human-readable location string.
    Uses OpenStreetMap Nominatim (free, no API key).

    Returns:
        Location string like "Los Angeles, California" or "Las Vegas, Nevada"
        None if geocoding fails.
    """
    details = await reverse_geocode_details(lat, lon)
    if not details:
        return None
    return details.get("location_display")


def calculate_trill_score(
    telemetry: TelemetryData,
    speeding_mph: int = DEFAULT_SPEEDING_MPH,
    euphoria_mph: int = DEFAULT_EUPHORIA_MPH,
) -> TrillScore:
    """
    Calculate Trill score from telemetry data.

    Score Components (0-100):
    - Speed score (0-40): Based on max speed vs euphoria threshold
    - Speeding bonus (0-30): % of time above speeding threshold
    - Euphoria bonus (0-20): % of time above euphoria threshold
    - Consistency bonus (0-10): Inverse of speed variance

    Buckets:
    - 0-39:  chill
    - 40-59: spirited
    - 60-79: sendIt
    - 80-89: euphoric
    - 90-100: gloryBoy
    """
    if not telemetry.points:
        return TrillScore()

    speeds = [p['speed_mph'] for p in telemetry.points]
    total_points = len(speeds)

    # Speed score (0-40)
    speed_score = min(40.0, (telemetry.max_speed_mph / euphoria_mph) * 40)

    # Speeding time bonus (0-30)
    speeding_count = sum(1 for s in speeds if s >= speeding_mph)
    speeding_ratio = speeding_count / total_points
    speeding_score = speeding_ratio * 30

    # Euphoria bonus (0-20)
    euphoria_count = sum(1 for s in speeds if s >= euphoria_mph)
    euphoria_ratio = euphoria_count / total_points
    euphoria_score = euphoria_ratio * 20

    # Consistency bonus (0-10)
    if total_points > 1:
        variance = sum((s - telemetry.avg_speed_mph) ** 2 for s in speeds) / total_points
        std_dev = variance ** 0.5
        consistency_score = max(0.0, 10.0 - (std_dev / 10.0))
    else:
        consistency_score = 5.0

    total = int(min(100, speed_score + speeding_score + euphoria_score + consistency_score))

    if total >= 90:
        bucket = "gloryBoy"
    elif total >= 80:
        bucket = "euphoric"
    elif total >= 60:
        bucket = "sendIt"
    elif total >= 40:
        bucket = "spirited"
    else:
        bucket = "chill"

    return TrillScore(
        score=int(total),
        bucket=bucket,
        speed_score=float(speed_score),
        speeding_score=float(speeding_score),
        euphoria_score=float(euphoria_score),
        consistency_score=float(consistency_score),
        excessive_speed=bool(telemetry.max_speed_mph >= float(euphoria_mph)),
    )


def get_trill_modifiers(score: int, max_speed: float, bucket: str) -> tuple:
    """Get title modifier and hashtags based on Trill score."""
    if bucket == "gloryBoy":
        return " - GLORY BOY 🏆", ["GloryBoyTour", "TrillScore100", "SendIt", "DashCam", "CarLife"]
    elif bucket == "euphoric":
        return " - Euphoric Run 🔥", ["Euphoric", "TrillScore", "SpeedDemon", "DashCam"]
    elif bucket == "sendIt":
        return " - Send It 🚀", ["SendIt", "TrillScore", "Spirited", "DashCam"]
    elif bucket == "spirited":
        return " - Spirited Drive 🎯", ["SpiritedDrive", "TrillScore", "DashCam"]
    elif max_speed >= 100:
        return " 🛣️", ["TrillScore", "RoadTrip", "DashCam"]
    else:
        return " 🚗", ["TrillScore", "CruiseControl", "DashCam"]


async def apply_padus_gazetteer_to_telemetry(
    telemetry: TelemetryData,
    *,
    db_pool: Any = None,
) -> None:
    """US Census gazetteer (local file) + PAD-US (PostGIS via ``db_pool``).

    Mutates ``telemetry`` in place. Safe no-op when there are no route points and
    no representative coordinates, or when neither gazetteer nor DB pool is
    available. Gazetteer work runs in ``asyncio.to_thread``; PAD-US uses
    ``ST_Contains`` on ``padus_protected_areas`` (see ``services.padus_db``).

    Called after **both** ``.map`` parse (``run_telemetry_stage``) and HUD backfill
    (``dashcam_osd_stage``) so captions see the same geo signals whether GPS came
    from a companion map file or burned-in OSD.
    """
    gaz = GAZETTEER_PLACES_PATH if (GAZETTEER_PLACES_PATH and os.path.isfile(GAZETTEER_PLACES_PATH)) else None
    if not telemetry.points and not (telemetry.mid_lat is not None and telemetry.mid_lon is not None):
        return
    if not (gaz or db_pool):
        return
    try:
        from telemetry_trill import enrich_route_padus_gazetteer

        extra = await asyncio.to_thread(
            enrich_route_padus_gazetteer,
            telemetry.points,
            telemetry.mid_lat,
            telemetry.mid_lon,
            gaz_places_path=gaz,
            padus_path=None,
            padus_layer=None,
        )
        if not isinstance(extra, dict):
            extra = {}
        if extra.get("gazetteer_place_name"):
            telemetry.gazetteer_place_name = str(extra["gazetteer_place_name"]).strip() or None
        if extra.get("gazetteer_state_usps"):
            telemetry.gazetteer_state_usps = str(extra["gazetteer_state_usps"]).strip() or None

        if db_pool is not None:
            from services.padus_db import padus_hit_dict_from_db

            async with db_pool.acquire() as conn:
                lat_f: Optional[float] = None
                lon_f: Optional[float] = None
                if telemetry.mid_lat is not None and telemetry.mid_lon is not None:
                    try:
                        lat_f = float(telemetry.mid_lat)
                        lon_f = float(telemetry.mid_lon)
                    except (TypeError, ValueError):
                        lat_f, lon_f = None, None
                if lat_f is None or lon_f is None:
                    mid = get_representative_coords(telemetry.points)
                    if mid:
                        lat_f, lon_f = float(mid[0]), float(mid[1])
                if lat_f is not None and lon_f is not None:
                    db_extra = await padus_hit_dict_from_db(conn, lat_f, lon_f)
                    extra.update(db_extra)

        telemetry.near_padus = bool(extra.get("near_padus"))
        if extra.get("padus_unit_name"):
            telemetry.padus_unit_name = str(extra["padus_unit_name"]).strip() or None
        elif not telemetry.near_padus:
            telemetry.padus_unit_name = None
        logger.info(
            "Geo enrichment: gazetteer=%s padus=%s",
            telemetry.gazetteer_place_name or "—",
            telemetry.padus_unit_name or ("near" if telemetry.near_padus else "—"),
        )
    except Exception as e:
        logger.debug("PADUS/gazetteer enrichment skipped: %s", e)


async def run_telemetry_stage(ctx: JobContext) -> JobContext:
    """
    Execute telemetry processing stage.

    - Parses .map file
    - Calculates Trill score
    - Reverse geocodes starting/mid coordinates to get location name
    - Stores location_name on ctx for caption_stage to use

    Args:
        ctx: Job context

    Returns:
        Updated context with telemetry, trill, and location data

    Raises:
        SkipStage: If no telemetry file
        TelemetryError: If processing fails critically
    """
    if not ctx.local_telemetry_path or not ctx.local_telemetry_path.exists():
        raise SkipStage("No telemetry file available")

    logger.info(f"Processing telemetry for upload {ctx.upload_id}")

    # Parse telemetry
    try:
        telemetry = parse_map_file(ctx.local_telemetry_path)
    except TelemetryError as e:
        logger.error(f"Telemetry parse failed: {e}")
        raise

    ctx.telemetry_data = telemetry

    # Get user thresholds from settings
    speeding_mph = int((ctx.user_settings or {}).get("speeding_mph", DEFAULT_SPEEDING_MPH))
    euphoria_mph = int((ctx.user_settings or {}).get("euphoria_mph", DEFAULT_EUPHORIA_MPH))

    # Calculate speeding/euphoria seconds
    if telemetry.points and len(telemetry.points) > 1:
        speeding_secs = 0.0
        euphoria_secs = 0.0
        for i in range(1, len(telemetry.points)):
            dt = telemetry.points[i]['timestamp'] - telemetry.points[i - 1]['timestamp']
            mid_speed = (telemetry.points[i]['speed_mph'] + telemetry.points[i - 1]['speed_mph']) / 2
            if mid_speed >= speeding_mph:
                speeding_secs += dt
            if mid_speed >= euphoria_mph:
                euphoria_secs += dt
        telemetry.speeding_seconds = speeding_secs
        telemetry.euphoria_seconds = euphoria_secs

    # Calculate Trill score
    trill = calculate_trill_score(telemetry, speeding_mph, euphoria_mph)
    modifier, hashtags = get_trill_modifiers(trill.score, telemetry.max_speed_mph, trill.bucket)
    trill.title_modifier = modifier
    trill.hashtags = hashtags
    ctx.trill_score = trill
    ctx.trill = trill

    logger.info(f"Trill score: {trill.total} (bucket derived from score)")

    # ─── GPS coordinates from .map (start + midpoint for geocode / scene graph) ─
    pts = telemetry.points
    if pts:
        try:
            la = float(pts[0]["lat"])
            lo = float(pts[0]["lon"])
            if -90 <= la <= 90 and -180 <= lo <= 180 and not (abs(la) < 1e-9 and abs(lo) < 1e-9):
                telemetry.start_lat = la
                telemetry.start_lon = lo
        except (KeyError, TypeError, ValueError):
            pass
    mid = get_representative_coords(telemetry.points)
    if mid:
        telemetry.mid_lat, telemetry.mid_lon = mid[0], mid[1]

    # ─── REVERSE GEOCODING (persist on TelemetryData for M8 / Trill / captions) ─
    coords = mid
    if coords:
        lat, lon = coords
        logger.info("Reverse geocoding telemetry midpoint (%.5f, %.5f)...", lat, lon)
        details = await reverse_geocode_details(lat, lon)
        if details:
            telemetry.location_display = details.get("location_display")
            telemetry.location_city = details.get("location_city")
            telemetry.location_state = details.get("location_state")
            telemetry.location_country = details.get("location_country")
            telemetry.location_road = details.get("location_road")
            # NOTE: ctx.location_name is a read-only @property that proxies
            # telemetry.location_display — assigning to it raises AttributeError
            # and silently kills this stage. Callers read ctx.location_name and
            # always get the fresh telemetry.location_display value.
            if telemetry.location_display:
                logger.info("Location resolved: %s", telemetry.location_display)
        else:
            logger.warning(
                "Reverse geocode returned no result — location will be omitted from captions"
            )
    else:
        logger.warning("No valid GPS coordinates found in .map file")

    # ─── Second geocode at route start when far from midpoint (OSM etiquette: pause) ─
    try:
        min_sep = float(os.environ.get("TELEMETRY_START_GEOCODE_MIN_SEPARATION_MILES", "5") or 5)
    except (TypeError, ValueError):
        min_sep = 5.0
    min_sep = max(1.0, min(min_sep, 200.0))
    if (
        telemetry.start_lat is not None
        and telemetry.start_lon is not None
        and telemetry.mid_lat is not None
        and telemetry.mid_lon is not None
    ):
        sep = _haversine_miles(
            float(telemetry.start_lat),
            float(telemetry.start_lon),
            float(telemetry.mid_lat),
            float(telemetry.mid_lon),
        )
        if sep >= min_sep:
            await asyncio.sleep(1.15)
            start_details = await reverse_geocode_details(
                float(telemetry.start_lat), float(telemetry.start_lon)
            )
            if start_details and start_details.get("location_display"):
                telemetry.location_start_display = start_details["location_display"]
                logger.info(
                    "Geocoded route start (%.2f mi from mid): %s",
                    sep,
                    telemetry.location_start_display,
                )
    # ─────────────────────────────────────────────────────────────────────────

    ctx.telemetry_data = telemetry
    ctx.telemetry = telemetry

    await apply_padus_gazetteer_to_telemetry(
        telemetry, db_pool=getattr(ctx, "_db_pool", None)
    )
    ctx.telemetry_data = telemetry
    ctx.telemetry = telemetry

    return ctx
