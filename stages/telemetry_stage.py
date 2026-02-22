"""
UploadM8 Telemetry Stage
========================
Parse .map telemetry files and calculate Trill scores.
Also extracts GPS coordinates and reverse-geocodes them to city/state
so the caption stage can ground AI content in a real location.
"""

import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Optional

import httpx

from .errors import TelemetryError, SkipStage, ErrorCode
from .context import JobContext, TelemetryData, TrillScore


logger = logging.getLogger("uploadm8-worker")


# Default thresholds
DEFAULT_SPEEDING_MPH = 80
DEFAULT_EUPHORIA_MPH = 100


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
    data_points: List[Dict[str, float]] = []

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
                        'altitude': float(parts[4]) if len(parts) > 4 else 0
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

    # Calculate aggregate stats
    speeds = [p['speed_mph'] for p in data_points]
    telemetry = TelemetryData()
    telemetry.points = data_points
    telemetry.max_speed_mph = max(speeds)
    telemetry.avg_speed_mph = sum(speeds) / len(speeds)
    telemetry.duration_seconds = (data_points[-1]['timestamp'] - data_points[0]['timestamp']) if len(data_points) > 1 else 0.0

    # Estimate distance (rough approximation)
    if len(data_points) > 1:
        total_distance = 0
        for i in range(1, len(data_points)):
            dt = data_points[i]['timestamp'] - data_points[i-1]['timestamp']
            avg_speed = (data_points[i]['speed_mph'] + data_points[i-1]['speed_mph']) / 2
            total_distance += (avg_speed * dt) / 3600  # miles
        telemetry.distance_miles = total_distance

    # Max altitude
    altitudes = [p.get('altitude', 0) for p in data_points]
    if altitudes:
        telemetry.max_altitude_ft = max(altitudes)

    logger.info(f"Parsed telemetry: {len(data_points)} points, max_speed={telemetry.max_speed:.1f} mph")
    return telemetry


def _extract_representative_gps(data_points: List[Dict]) -> Optional[tuple]:
    """
    Pick the best single GPS point to represent the clip's location.
    Uses the midpoint of the route rather than start/end — midpoint tends
    to be the most "interesting" part of a dashcam clip and avoids
    driveway/parking lot coordinates at the very start.

    Returns (lat, lon) tuple or None if no valid GPS points exist.
    """
    valid = [
        p for p in data_points
        if p.get("lat") and p.get("lon")
        and abs(p["lat"]) > 0.001   # filter out null-island / zeroed GPS
        and abs(p["lon"]) > 0.001
    ]
    if not valid:
        return None

    # Use the midpoint of the clip
    mid_idx = len(valid) // 2
    p = valid[mid_idx]
    return (p["lat"], p["lon"])


async def reverse_geocode(lat: float, lon: float) -> Optional[Dict[str, str]]:
    """
    Reverse geocode a lat/lon to city, state, country using the free
    OpenStreetMap Nominatim API. No API key required.

    Returns a dict with keys: city, state, country, road, display
    Returns None on any failure — geocoding is always non-fatal.
    """
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": str(lat),
            "lon": str(lon),
            "format": "json",
            "addressdetails": "1",
            "zoom": "12",          # city-level precision
        }
        headers = {
            # Nominatim requires a User-Agent identifying your app
            "User-Agent": "UploadM8/1.0 (video-upload-platform; contact@uploadm8.com)"
        }

        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(url, params=params, headers=headers)

        if resp.status_code != 200:
            logger.warning(f"Nominatim returned {resp.status_code}")
            return None

        data = resp.json()
        addr = data.get("address", {})

        # Extract city — Nominatim uses different keys depending on location type
        city = (
            addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or addr.get("hamlet")
            or addr.get("suburb")
            or addr.get("county")
            or ""
        )

        # State — "state" for US, "county" for UK, etc.
        state = addr.get("state") or addr.get("region") or ""

        # US: abbreviate state name if possible
        state_display = _abbreviate_us_state(state) if state else ""

        country = addr.get("country_code", "").upper()

        # Road / highway name
        road = (
            addr.get("road")
            or addr.get("motorway")
            or addr.get("trunk")
            or addr.get("primary")
            or ""
        )

        # Build clean display string
        if city and state_display:
            if country == "US":
                display = f"{city}, {state_display}"
            else:
                display = f"{city}, {state_display}, {country}"
        elif city:
            display = city
        elif state:
            display = state
        else:
            display = data.get("display_name", "").split(",")[0]

        result = {
            "city": city,
            "state": state,
            "state_display": state_display,
            "country": country,
            "road": road,
            "display": display,
        }
        logger.info(f"Reverse geocoded ({lat:.4f}, {lon:.4f}) → {display}")
        return result

    except Exception as e:
        logger.warning(f"Reverse geocode failed (non-fatal): {e}")
        return None


# US state name → abbreviation lookup
_US_STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
}


def _abbreviate_us_state(state_name: str) -> str:
    """Return 2-letter abbreviation for US state, or original string if not found."""
    return _US_STATES.get(state_name, state_name)


def calculate_trill_score(
    telemetry: TelemetryData,
    speeding_mph: int = DEFAULT_SPEEDING_MPH,
    euphoria_mph: int = DEFAULT_EUPHORIA_MPH
) -> TrillScore:
    """
    Calculate Trill score from telemetry data.

    Score Components (0-100):
    - Speed score (0-40): Based on max speed vs euphoria threshold
    - Speeding bonus (0-30): Percentage of time above speeding threshold
    - Euphoria bonus (0-20): Percentage of time above euphoria threshold
    - Consistency bonus (0-10): Inverse of speed variance

    Buckets:
    - 0-39: chill
    - 40-59: spirited
    - 60-79: sendIt
    - 80-89: euphoric
    - 90-100: gloryBoy
    """
    if not telemetry.data_points:
        return TrillScore()

    speeds = [p['speed_mph'] for p in telemetry.data_points]
    total_points = len(speeds)

    # Speed score (0-40)
    speed_score = min(40, (telemetry.max_speed / euphoria_mph) * 40)

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
        variance = sum((s - telemetry.avg_speed) ** 2 for s in speeds) / total_points
        std_dev = variance ** 0.5
        consistency_score = max(0, 10 - (std_dev / 10))
    else:
        consistency_score = 5

    # Total score
    total = int(min(100, speed_score + speeding_score + euphoria_score + consistency_score))

    # Determine bucket
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

    # Check for excessive speed (safety flag)
    excessive_speed = telemetry.max_speed >= 120

    # Generate title modifier and hashtags
    title_modifier, hashtags = get_trill_modifiers(total, telemetry.max_speed, bucket)

    return TrillScore(
        score=total,
        bucket=bucket,
        speed_score=speed_score,
        speeding_score=speeding_score,
        euphoria_score=euphoria_score,
        consistency_score=consistency_score,
        excessive_speed=excessive_speed,
        title_modifier=title_modifier,
        hashtags=hashtags,
    )


def get_trill_modifiers(score: int, max_speed: float, bucket: str) -> tuple:
    """Get title modifier and hashtags based on Trill score."""
    if bucket == "gloryBoy":
        return " - GLORY BOY", ["#GloryBoyTour", "#TrillScore100", "#SendIt"]
    elif bucket == "euphoric":
        return " - Euphoric", ["#Euphoric", "#TrillScore", "#SpeedDemon"]
    elif bucket == "sendIt":
        return " - Send It", ["#SendIt", "#TrillScore", "#Spirited"]
    elif bucket == "spirited":
        return " - Spirited Drive", ["#SpiritedDrive", "#TrillScore"]
    elif max_speed >= 100:
        return "", ["#TrillScore", "#RoadTrip"]
    else:
        return "", ["#TrillScore"]


async def run_telemetry_stage(ctx: JobContext) -> JobContext:
    """
    Execute telemetry processing stage.

    Args:
        ctx: Job context

    Returns:
        Updated context with telemetry, trill data, and location

    Raises:
        SkipStage: If no telemetry file
        TelemetryError: If processing fails
    """
    ctx.mark_stage("telemetry")

    if not ctx.local_telemetry_path or not ctx.local_telemetry_path.exists():
        # Ensure downstream stages never crash on missing attributes
        ctx.telemetry = None
        ctx.telemetry_data = None
        ctx.trill = None
        ctx.trill_score = None
        raise SkipStage("No telemetry file available")

    logger.info(f"Processing telemetry for upload {ctx.upload_id}")

    # Parse telemetry
    telemetry = parse_map_file(ctx.local_telemetry_path)
    ctx.telemetry = telemetry
    ctx.telemetry_data = telemetry

    # ------------------------------------------------------------------ #
    # GPS → Location                                                       #
    # Extract a representative GPS point and reverse geocode to city/state #
    # so the caption stage can write location-aware content.              #
    # ------------------------------------------------------------------ #
    gps_point = _extract_representative_gps(telemetry.points)

    if gps_point:
        lat, lon = gps_point
        telemetry.mid_lat = lat
        telemetry.mid_lon = lon

        # Store start point too for reference
        if telemetry.points:
            telemetry.start_lat = telemetry.points[0].get("lat")
            telemetry.start_lon = telemetry.points[0].get("lon")

        # Reverse geocode — non-fatal, just skips location on failure
        geo = await reverse_geocode(lat, lon)
        if geo:
            telemetry.location_city = geo["city"]
            telemetry.location_state = geo["state"]
            telemetry.location_country = geo["country"]
            telemetry.location_display = geo["display"]
            telemetry.location_road = geo["road"]
            logger.info(f"Location resolved: {geo['display']}")
        else:
            logger.info(f"GPS coords available ({lat:.4f}, {lon:.4f}) but geocoding failed")
    else:
        logger.info("No valid GPS coordinates found in telemetry data")

    # Get user thresholds from settings
    speeding_mph = ctx.user_settings.get("speeding_mph", DEFAULT_SPEEDING_MPH)
    euphoria_mph = ctx.user_settings.get("euphoria_mph", DEFAULT_EUPHORIA_MPH)

    # Calculate Trill score
    trill = calculate_trill_score(telemetry, speeding_mph, euphoria_mph)
    ctx.trill = trill
    ctx.trill_score = trill

    logger.info(f"Trill score: {trill.score} ({trill.bucket})")

    return ctx
