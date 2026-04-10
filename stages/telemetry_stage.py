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
from pathlib import Path
from typing import List, Dict, Optional, Tuple

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
        with open(map_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split(',')
                if len(parts) < 4:
                    continue

                try:
                    point = {
                        'timestamp': float(parts[0]),
                        'lat': float(parts[1]),
                        'lon': float(parts[2]),
                        'speed_mph': float(parts[3]),
                        'altitude': float(parts[4]) if len(parts) > 4 else 0.0,
                    }
                    data_points.append(point)
                except ValueError:
                    continue

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

    except asyncio.TimeoutError:
        logger.warning(f"Geocode timeout for ({lat}, {lon})")
        return None
    except Exception as e:
        logger.warning(f"Geocode error for ({lat}, {lon}): {e}")
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
        total=total,
        speed_score=int(speed_score),
        distance_score=0,
        duration_score=0,
        altitude_score=0,
        thrill_factor=round(telemetry.max_speed_mph / max(telemetry.avg_speed_mph, 1), 2),
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
    ctx.trill_score = trill

    logger.info(f"Trill score: {trill.total} (bucket derived from score)")

    # ─── REVERSE GEOCODING ────────────────────────────────────────────────────
    coords = get_representative_coords(telemetry.points)
    if coords:
        lat, lon = coords
        logger.info(f"Reverse geocoding telemetry location ({lat:.5f}, {lon:.5f})...")
        location_name = await reverse_geocode(lat, lon)
        if location_name:
            ctx.location_name = location_name
            logger.info(f"Location resolved: {location_name}")
        else:
            logger.warning("Reverse geocode returned no result — location will be omitted from captions")
            ctx.location_name = None
    else:
        logger.warning("No valid GPS coordinates found in .map file")
        ctx.location_name = None
    # ─────────────────────────────────────────────────────────────────────────

    return ctx
