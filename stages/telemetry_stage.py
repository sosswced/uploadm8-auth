"""
UploadM8 Telemetry Stage
========================
Parse .map telemetry files and calculate Trill scores.
"""

import logging
from pathlib import Path
from typing import List, Dict

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
        Updated context with telemetry and trill data

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

    # Get user thresholds from settings
    speeding_mph = ctx.user_settings.get("speeding_mph", DEFAULT_SPEEDING_MPH)
    euphoria_mph = ctx.user_settings.get("euphoria_mph", DEFAULT_EUPHORIA_MPH)

    # Calculate Trill score
    trill = calculate_trill_score(telemetry, speeding_mph, euphoria_mph)
    ctx.trill = trill
    ctx.trill_score = trill

    logger.info(f"Trill score: {trill.score} ({trill.bucket})")

    return ctx
