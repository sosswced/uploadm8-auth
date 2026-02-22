"""
UploadM8 Telemetry Stage
========================
Parse .map telemetry files and calculate Trill scores.
Also extracts GPS coordinates and reverse-geocodes them to city/state
so the caption stage can ground AI content in a real location.

PANDAS-POWERED:
  - Uses pandas DataFrame for statistical analysis of speed data
  - Calculates speeding_seconds and euphoria_seconds precisely
  - Computes acceleration events, speed percentiles, variance
  - All metrics are fed into caption stage for richer AI-generated content
"""

import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from io import StringIO

import httpx

# pandas is used for statistical telemetry analysis
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None
    np = None

from .errors import TelemetryError, SkipStage, ErrorCode
from .context import JobContext, TelemetryData, TrillScore


logger = logging.getLogger("uploadm8-worker")


# Default thresholds
DEFAULT_SPEEDING_MPH = 80
DEFAULT_EUPHORIA_MPH = 100


def parse_map_file(map_path: Path) -> TelemetryData:
    """
    Parse .map telemetry file using pandas when available,
    falling back to manual CSV parsing.

    Expected CSV format: timestamp,lat,lon,speed_mph,altitude
    Lines starting with # are comments.

    Returns:
        TelemetryData with parsed points and aggregate statistics
    """
    data_points: List[Dict[str, float]] = []

    try:
        with open(map_path, "r") as f:
            raw = f.read()
    except FileNotFoundError:
        raise TelemetryError(
            f"Telemetry file not found: {map_path}",
            code=ErrorCode.TELEMETRY_PARSE_FAILED,
        )
    except Exception as e:
        raise TelemetryError(
            f"Failed to read telemetry file: {e}",
            code=ErrorCode.TELEMETRY_PARSE_FAILED,
            detail=str(e),
        )

    # Strip comment lines before parsing
    clean_lines = [
        line for line in raw.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not clean_lines:
        raise TelemetryError("No valid telemetry data found", code=ErrorCode.TELEMETRY_EMPTY)

    # ------------------------------------------------------------------ #
    # PANDAS PATH — richer stats when available                           #
    # ------------------------------------------------------------------ #
    if HAS_PANDAS:
        try:
            csv_text = "\n".join(clean_lines)
            df = pd.read_csv(
                StringIO(csv_text),
                header=None,
                names=["timestamp", "lat", "lon", "speed_mph", "altitude"],
                dtype=float,
                on_bad_lines="skip",
            )

            # Drop rows with null or zero lat/lon (GPS dropout)
            df = df.dropna(subset=["timestamp", "lat", "lon", "speed_mph"])
            df = df[
                (df["speed_mph"] >= 0)
                & (df["speed_mph"] <= 300)  # sanity clamp
            ]

            if df.empty:
                raise TelemetryError("No valid rows after filtering", code=ErrorCode.TELEMETRY_EMPTY)

            # Sort by timestamp ascending
            df = df.sort_values("timestamp").reset_index(drop=True)

            data_points = df.to_dict(orient="records")

            telemetry = TelemetryData()
            telemetry.points = data_points

            # Core stats via pandas
            telemetry.max_speed_mph = float(df["speed_mph"].max())
            telemetry.avg_speed_mph = float(df["speed_mph"].mean())

            # Duration
            if len(df) > 1:
                telemetry.duration_seconds = float(
                    df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
                )

            # Altitude
            if "altitude" in df.columns:
                alt_valid = df["altitude"].dropna()
                if not alt_valid.empty:
                    telemetry.max_altitude_ft = float(alt_valid.max())

            # Distance — integrate speed over time
            if len(df) > 1:
                dt = df["timestamp"].diff().fillna(0)
                avg_v = (df["speed_mph"] + df["speed_mph"].shift(1).fillna(df["speed_mph"].iloc[0])) / 2
                distance_increments = (avg_v * dt) / 3600.0  # miles
                telemetry.total_distance_miles = float(distance_increments.sum())

            # Store pandas stats for caption enrichment
            telemetry._df = df
            telemetry._pandas_loaded = True

            logger.info(
                f"[pandas] Parsed {len(df)} telemetry points — "
                f"max={telemetry.max_speed_mph:.1f} mph, "
                f"avg={telemetry.avg_speed_mph:.1f} mph, "
                f"dist={telemetry.total_distance_miles:.2f} mi"
            )
            return telemetry

        except TelemetryError:
            raise
        except Exception as e:
            logger.warning(f"Pandas parse failed, falling back to manual: {e}")
            data_points = []

    # ------------------------------------------------------------------ #
    # MANUAL FALLBACK                                                      #
    # ------------------------------------------------------------------ #
    for line in clean_lines:
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            point = {
                "timestamp": float(parts[0]),
                "lat": float(parts[1]),
                "lon": float(parts[2]),
                "speed_mph": float(parts[3]),
                "altitude": float(parts[4]) if len(parts) > 4 else 0.0,
            }
            data_points.append(point)
        except ValueError:
            continue

    if not data_points:
        raise TelemetryError("No valid telemetry data found", code=ErrorCode.TELEMETRY_EMPTY)

    speeds = [p["speed_mph"] for p in data_points]
    telemetry = TelemetryData()
    telemetry.points = data_points
    telemetry.max_speed_mph = max(speeds)
    telemetry.avg_speed_mph = sum(speeds) / len(speeds)

    if len(data_points) > 1:
        telemetry.duration_seconds = data_points[-1]["timestamp"] - data_points[0]["timestamp"]
        total_distance = 0.0
        for i in range(1, len(data_points)):
            dt = data_points[i]["timestamp"] - data_points[i - 1]["timestamp"]
            avg_speed = (data_points[i]["speed_mph"] + data_points[i - 1]["speed_mph"]) / 2
            total_distance += (avg_speed * dt) / 3600.0
        telemetry.total_distance_miles = total_distance

    altitudes = [p.get("altitude", 0.0) for p in data_points]
    if altitudes:
        telemetry.max_altitude_ft = max(altitudes)

    telemetry._pandas_loaded = False
    logger.info(f"[manual] Parsed {len(data_points)} telemetry points")
    return telemetry


def _compute_pandas_stats(
    telemetry: TelemetryData,
    speeding_mph: float,
    euphoria_mph: float,
) -> Dict:
    """
    Use pandas DataFrame (if available) to compute richer stats:
      - speeding_seconds: time spent above speeding threshold
      - euphoria_seconds: time spent above euphoria threshold
      - speed_p95: 95th percentile speed
      - accel_events: count of hard acceleration events (Δspeed > 15 mph/s)
      - speed_std: standard deviation of speed
    """
    stats = {
        "speeding_seconds": 0.0,
        "euphoria_seconds": 0.0,
        "speed_p95": telemetry.max_speed_mph * 0.9,
        "accel_events": 0,
        "speed_std": 0.0,
        "speed_p50": telemetry.avg_speed_mph,
    }

    df = getattr(telemetry, "_df", None)

    if df is not None and HAS_PANDAS and len(df) > 1:
        try:
            # Time deltas (seconds between GPS points)
            dt = df["timestamp"].diff().fillna(0).clip(upper=10)  # cap gaps at 10s

            # Speeding and euphoria seconds
            stats["speeding_seconds"] = float(dt[df["speed_mph"] >= speeding_mph].sum())
            stats["euphoria_seconds"] = float(dt[df["speed_mph"] >= euphoria_mph].sum())

            # Speed distribution
            stats["speed_p95"] = float(df["speed_mph"].quantile(0.95))
            stats["speed_p50"] = float(df["speed_mph"].quantile(0.50))
            stats["speed_std"] = float(df["speed_mph"].std())

            # Acceleration events: where speed increases by > 10 mph between consecutive points
            speed_delta = df["speed_mph"].diff().fillna(0)
            stats["accel_events"] = int((speed_delta > 10).sum())

            logger.info(
                f"[pandas stats] speeding={stats['speeding_seconds']:.0f}s, "
                f"euphoria={stats['euphoria_seconds']:.0f}s, "
                f"p95={stats['speed_p95']:.1f} mph, "
                f"accel_events={stats['accel_events']}"
            )
        except Exception as e:
            logger.warning(f"Pandas stats computation failed: {e}")
    else:
        # Manual fallback for speeding/euphoria seconds
        if len(telemetry.points) > 1:
            speeding_s = 0.0
            euphoria_s = 0.0
            for i in range(1, len(telemetry.points)):
                dt = telemetry.points[i]["timestamp"] - telemetry.points[i - 1]["timestamp"]
                dt = min(dt, 10.0)  # cap gaps
                avg_speed = (
                    telemetry.points[i]["speed_mph"] + telemetry.points[i - 1]["speed_mph"]
                ) / 2
                if avg_speed >= speeding_mph:
                    speeding_s += dt
                if avg_speed >= euphoria_mph:
                    euphoria_s += dt
            stats["speeding_seconds"] = speeding_s
            stats["euphoria_seconds"] = euphoria_s

    return stats


def _extract_representative_gps(data_points: List[Dict]) -> Optional[tuple]:
    """
    Pick the best single GPS point to represent the clip's location.
    Uses the midpoint of the route — avoids driveway/parking lot coords.

    Returns (lat, lon) tuple or None.
    """
    valid = [
        p for p in data_points
        if p.get("lat") and p.get("lon")
        and abs(p["lat"]) > 0.001
        and abs(p["lon"]) > 0.001
    ]
    if not valid:
        return None

    mid_idx = len(valid) // 2
    p = valid[mid_idx]
    return (p["lat"], p["lon"])


async def reverse_geocode(lat: float, lon: float) -> Optional[Dict[str, str]]:
    """
    Reverse geocode a lat/lon to city, state, country using
    OpenStreetMap Nominatim API (no API key required).

    Returns dict with: city, state, country, road, display
    Returns None on any failure — always non-fatal.
    """
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": str(lat),
            "lon": str(lon),
            "format": "json",
            "addressdetails": "1",
            "zoom": "12",
        }
        headers = {
            "User-Agent": "UploadM8/1.0 (video-upload-platform; contact@uploadm8.com)"
        }

        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(url, params=params, headers=headers)

        if resp.status_code != 200:
            logger.warning(f"Nominatim returned {resp.status_code}")
            return None

        data = resp.json()
        addr = data.get("address", {})

        city = (
            addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or addr.get("hamlet")
            or addr.get("suburb")
            or addr.get("county")
            or ""
        )

        state = addr.get("state") or addr.get("region") or ""
        state_display = _abbreviate_us_state(state) if state else ""
        country = addr.get("country_code", "").upper()

        road = (
            addr.get("road")
            or addr.get("motorway")
            or addr.get("trunk")
            or addr.get("primary")
            or ""
        )

        if city and state_display:
            display = f"{city}, {state_display}" if country == "US" else f"{city}, {state_display}, {country}"
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
    euphoria_mph: int = DEFAULT_EUPHORIA_MPH,
    pandas_stats: Optional[Dict] = None,
) -> TrillScore:
    """
    Calculate Trill score from telemetry data.

    Score Components (0-100):
    - Speed score (0-40): Based on max speed vs euphoria threshold
    - Speeding bonus (0-30): Percentage of time above speeding threshold
    - Euphoria bonus (0-20): Percentage of time above euphoria threshold
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

    stats = pandas_stats or {}
    speeds = [p["speed_mph"] for p in telemetry.points]
    total_points = len(speeds)

    # Speed score (0-40)
    speed_score = min(40.0, (telemetry.max_speed_mph / max(euphoria_mph, 1)) * 40.0)

    # Speeding time bonus (0-30)
    if stats.get("speeding_seconds") and telemetry.duration_seconds > 0:
        speeding_ratio = min(stats["speeding_seconds"] / telemetry.duration_seconds, 1.0)
    else:
        speeding_count = sum(1 for s in speeds if s >= speeding_mph)
        speeding_ratio = speeding_count / max(total_points, 1)
    speeding_score = speeding_ratio * 30.0

    # Euphoria bonus (0-20)
    if stats.get("euphoria_seconds") and telemetry.duration_seconds > 0:
        euphoria_ratio = min(stats["euphoria_seconds"] / telemetry.duration_seconds, 1.0)
    else:
        euphoria_count = sum(1 for s in speeds if s >= euphoria_mph)
        euphoria_ratio = euphoria_count / max(total_points, 1)
    euphoria_score = euphoria_ratio * 20.0

    # Consistency bonus (0-10) — uses pandas std if available
    if stats.get("speed_std") is not None and stats["speed_std"] > 0:
        consistency_score = max(0.0, 10.0 - (stats["speed_std"] / 10.0))
    elif total_points > 1:
        avg = sum(speeds) / total_points
        variance = sum((s - avg) ** 2 for s in speeds) / total_points
        std_dev = variance ** 0.5
        consistency_score = max(0.0, 10.0 - (std_dev / 10.0))
    else:
        consistency_score = 5.0

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

    excessive_speed = telemetry.max_speed_mph >= 120

    title_modifier, hashtags = get_trill_modifiers(total, telemetry.max_speed_mph, bucket)

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
        return " - GLORY BOY", ["#GloryBoyTour", "#TrillScore100", "#SendIt", "#dashcam"]
    elif bucket == "euphoric":
        return " - Euphoric", ["#Euphoric", "#TrillScore", "#SpeedDemon", "#dashcam"]
    elif bucket == "sendIt":
        return " - Send It", ["#SendIt", "#TrillScore", "#Spirited", "#dashcam"]
    elif bucket == "spirited":
        return " - Spirited Drive", ["#SpiritedDrive", "#TrillScore", "#dashcam"]
    elif max_speed >= 100:
        return "", ["#TrillScore", "#RoadTrip", "#dashcam"]
    else:
        return "", ["#TrillScore", "#dashcam"]


async def run_telemetry_stage(ctx: JobContext) -> JobContext:
    """
    Execute telemetry processing stage.

    - Parses .map file using pandas (or fallback)
    - Computes speeding_seconds, euphoria_seconds, p95 speed, accel events
    - Reverse geocodes midpoint GPS to city/state for caption grounding
    - Calculates Trill score
    - Stores all data on ctx for downstream stages
    """
    ctx.mark_stage("telemetry")

    if not ctx.local_telemetry_path or not ctx.local_telemetry_path.exists():
        ctx.telemetry = None
        ctx.telemetry_data = None
        ctx.trill = None
        ctx.trill_score = None
        raise SkipStage("No telemetry file available")

    logger.info(
        f"Processing telemetry for upload {ctx.upload_id} "
        f"[pandas={'available' if HAS_PANDAS else 'unavailable'}]"
    )

    # Parse the .map file
    telemetry = parse_map_file(ctx.local_telemetry_path)
    ctx.telemetry = telemetry
    ctx.telemetry_data = telemetry

    # Get user thresholds
    speeding_mph = float(ctx.user_settings.get("speeding_mph", DEFAULT_SPEEDING_MPH))
    euphoria_mph = float(ctx.user_settings.get("euphoria_mph", DEFAULT_EUPHORIA_MPH))

    # Compute richer pandas statistics
    pandas_stats = _compute_pandas_stats(telemetry, speeding_mph, euphoria_mph)

    # Write computed stats back to telemetry object
    telemetry.speeding_seconds = pandas_stats["speeding_seconds"]
    telemetry.euphoria_seconds = pandas_stats["euphoria_seconds"]

    # Store extra pandas stats on the telemetry object for caption stage
    telemetry._speed_p95 = pandas_stats.get("speed_p95", telemetry.max_speed_mph)
    telemetry._speed_p50 = pandas_stats.get("speed_p50", telemetry.avg_speed_mph)
    telemetry._accel_events = pandas_stats.get("accel_events", 0)
    telemetry._speed_std = pandas_stats.get("speed_std", 0.0)

    # ------------------------------------------------------------------ #
    # GPS → Location                                                       #
    # ------------------------------------------------------------------ #
    gps_point = _extract_representative_gps(telemetry.points)

    if gps_point:
        lat, lon = gps_point
        telemetry.mid_lat = lat
        telemetry.mid_lon = lon

        if telemetry.points:
            telemetry.start_lat = telemetry.points[0].get("lat")
            telemetry.start_lon = telemetry.points[0].get("lon")

        geo = await reverse_geocode(lat, lon)
        if geo:
            telemetry.location_city = geo["city"]
            telemetry.location_state = geo["state"]
            telemetry.location_country = geo["country"]
            telemetry.location_display = geo["display"]
            telemetry.location_road = geo["road"]
            logger.info(f"Location resolved: {geo['display']}")
        else:
            logger.info(
                f"GPS available ({lat:.4f}, {lon:.4f}) but geocoding failed — "
                "will use raw coordinates in captions"
            )
    else:
        logger.info("No valid GPS coordinates in telemetry data")

    # Calculate Trill score (pandas-enhanced)
    trill = calculate_trill_score(telemetry, int(speeding_mph), int(euphoria_mph), pandas_stats)
    ctx.trill = trill
    ctx.trill_score = trill

    logger.info(
        f"Trill score: {trill.score}/100 ({trill.bucket}) | "
        f"speeding={telemetry.speeding_seconds:.0f}s | "
        f"euphoria={telemetry.euphoria_seconds:.0f}s | "
        f"location={getattr(telemetry, 'location_display', 'unknown')}"
    )

    return ctx
