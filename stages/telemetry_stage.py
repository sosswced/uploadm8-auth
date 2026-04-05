"""
UploadM8 Telemetry Stage
========================
Parses .map telemetry files, calculates Trill score,
and performs reverse geocoding to extract location name
for use in AI caption/title/hashtag generation.
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone

import httpx

from .errors import TelemetryError, SkipStage, ErrorCode
from .context import JobContext, TelemetryData, TrillScore


logger = logging.getLogger("uploadm8-worker")


# Default thresholds
DEFAULT_SPEEDING_MPH = 80
DEFAULT_EUPHORIA_MPH = 100

# Nominatim reverse geocoding (free, no API key required)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
GEOCODE_TIMEOUT = 8  # seconds


def _parse_dm_coord(raw: str, hemi: str) -> Optional[float]:
    """
    Parse DMM.mmmm coordinate format used by many dashcam/GPS .map files.
    Example: 3759.6885 N -> 37.9948083
    """
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    deg = int(v // 100)
    mins = v - (deg * 100.0)
    dec = float(deg) + (mins / 60.0)
    h = (hemi or "").strip().upper()
    if h in ("S", "W"):
        dec = -dec
    if not (-90.0 <= dec <= 90.0 and h in ("N", "S")) and not (-180.0 <= dec <= 180.0 and h in ("E", "W")):
        return None
    return dec


def _parse_ymd_hms_epoch(yyMMdd: str, hhmmss: str) -> Optional[float]:
    """Convert YYMMDD + HHMMSS -> unix epoch seconds (UTC)."""
    y = (yyMMdd or "").strip()
    t = (hhmmss or "").strip()
    if len(y) != 6 or len(t) != 6:
        return None
    try:
        dt = datetime.strptime(y + t, "%y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    except ValueError:
        return None


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

    fallback_ts = 0.0
    try:
        with open(map_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split(',')
                # Format A: timestamp,lat,lon,speed_mph,altitude
                if len(parts) >= 4:
                    try:
                        point = {
                            'timestamp': float(parts[0]),
                            'lat': float(parts[1]),
                            'lon': float(parts[2]),
                            'speed_mph': float(parts[3]),
                            'altitude': float(parts[4]) if len(parts) > 4 else 0.0,
                        }
                        data_points.append(point)
                        continue
                    except ValueError:
                        pass

                # Format B (dashcam style):
                # A,YYMMDD,HHMMSS,lat_dm,N|S,lon_dm,E|W,speed,ax,ay,az;
                if len(parts) >= 8 and parts[0].strip().upper() == "A":
                    date_token = parts[1].strip()
                    time_token = parts[2].strip()
                    lat_dm = parts[3].strip()
                    lat_hemi = parts[4].strip()
                    lon_dm = parts[5].strip()
                    lon_hemi = parts[6].strip()
                    speed_token = parts[7].strip().rstrip(";")
                    altitude_token = parts[10].strip().rstrip(";") if len(parts) > 10 else "0"

                    lat = _parse_dm_coord(lat_dm, lat_hemi)
                    lon = _parse_dm_coord(lon_dm, lon_hemi)
                    if lat is None or lon is None:
                        continue
                    ts = _parse_ymd_hms_epoch(date_token, time_token)
                    if ts is None:
                        ts = fallback_ts
                    fallback_ts = max(fallback_ts + 1.0, ts)
                    try:
                        speed = float(speed_token)
                    except ValueError:
                        continue
                    try:
                        altitude = float(altitude_token)
                    except ValueError:
                        altitude = 0.0
                    data_points.append({
                        "timestamp": ts,
                        "lat": lat,
                        "lon": lon,
                        "speed_mph": speed,
                        "altitude": altitude,
                    })
                    continue

    except FileNotFoundError:
        raise TelemetryError(
            f"Telemetry file not found: {map_path}",
            code=ErrorCode.TELEMETRY_PARSE_FAILED
        )
    except (OSError, UnicodeDecodeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
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


async def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """
    Reverse geocode coordinates to a human-readable location string.
    Uses OpenStreetMap Nominatim (free, no API key).

    Returns:
        Location string like "Los Angeles, California" or "Las Vegas, Nevada"
        None if geocoding fails.
    """
    try:
        async with httpx.AsyncClient(timeout=GEOCODE_TIMEOUT) as client:
            resp = await client.get(
                NOMINATIM_URL,
                params={
                    "lat": lat,
                    "lon": lon,
                    "format": "json",
                    "zoom": 10,  # city level
                    "addressdetails": 1,
                },
                headers={"User-Agent": "UploadM8/1.0 (uploadm8.com)"},
            )

            if resp.status_code != 200:
                logger.warning(f"Geocode HTTP {resp.status_code} for ({lat}, {lon})")
                return None

            data = resp.json()
            address = data.get("address", {})

            # Build location string: prefer city/town, fallback to county
            city = (
                address.get("city") or
                address.get("town") or
                address.get("village") or
                address.get("suburb") or
                address.get("county") or
                ""
            )
            state = (
                address.get("state") or
                address.get("region") or
                ""
            )
            country = address.get("country_code", "").upper()

            parts = [p for p in [city, state] if p]
            if not parts and country:
                return country

            location = ", ".join(parts)
            logger.info(f"Geocoded ({lat:.4f}, {lon:.4f}) → {location}")
            return location or None

    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        logger.warning("Geocode timeout for (%s, %s)", lat, lon)
        return None
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.warning("Geocode error for (%s, %s): %s", lat, lon, e)
        return None


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
        score=total,
        bucket=bucket,
        speed_score=round(speed_score, 2),
        speeding_score=round(speeding_score, 2),
        euphoria_score=round(euphoria_score, 2),
        consistency_score=round(consistency_score, 2),
        excessive_speed=telemetry.max_speed_mph >= float(euphoria_mph),
    )


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
    # HUD stage reads ctx.telemetry; __post_init__ only runs at context creation.
    ctx.telemetry = telemetry

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
    ctx.trill_score = trill

    logger.info(f"Trill score: {trill.total} (bucket derived from score)")

    # ─── REVERSE GEOCODING ────────────────────────────────────────────────────
    coords = get_representative_coords(telemetry.points)
    if coords:
        lat, lon = coords
        logger.info(f"Reverse geocoding telemetry location ({lat:.5f}, {lon:.5f})...")
        location_name = await reverse_geocode(lat, lon)
        if location_name:
            telemetry.location_display = location_name
            logger.info(f"Location resolved: {location_name}")
        else:
            logger.warning("Reverse geocode returned no result — location will be omitted from captions")
            telemetry.location_display = None
    else:
        logger.warning("No valid GPS coordinates found in .map file")
        telemetry.location_display = None
    # ─────────────────────────────────────────────────────────────────────────

    return ctx
