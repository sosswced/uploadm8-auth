"""
UploadM8 Dashcam OSD Stage
==========================
Reads the burned-in HUD overlay (date / time / GPS / speed / heading / driver
name) from dashcam footage across the *entire* clip and turns it into a
structured time-series the rest of the pipeline can use.

Tuned for the M8 / Escort dashcam HUD layout, which renders one line at the
bottom of the visible video region, e.g.::

    2025/03/05 04:50 12 PM 36.136162° -115.178398° 46MPH C Walker      ESCORT.

Pipeline:

1. Sample frames at ``DASHCAM_OSD_FPS`` (default 1.0) up to
   ``DASHCAM_OSD_MAX_FRAMES`` (default 240).
2. Crop the bottom strip (``DASHCAM_OSD_BOTTOM_PCT``, default 12 %) — the OSD
   sits in the bottom of the active video, and cropping cuts both Cloud Vision
   cost and OCR noise from the road scene above.
3. Run Cloud Vision ``batch_annotate_images`` (``TEXT_DETECTION``) over the
   cropped strips in chunks of 16.
4. Parse each strip with M8/Escort-tuned regexes for date, time, lat/lon,
   speed (mph/kph), heading and driver name. The ``ESCORT.`` watermark is
   filtered.
5. Aggregate into ``ctx.dashcam_osd_context`` (``samples``, ``first_seen``,
   ``last_seen``, ``max_speed_mph``, ``gps_path``, ``coverage_pct`` …).
6. **Backfill** ``ctx.telemetry_data`` when the user did not upload a .map file
   so reverse-geocoding, Trill scoring and the existing scene digest light up
   from OSD GPS alone.

Env:
  ``DASHCAM_OSD_STAGE_ENABLED`` — master toggle (default ``true``).
  ``DASHCAM_OSD_FPS``           — frame sampling rate (default ``1.0``).
  ``DASHCAM_OSD_MAX_FRAMES``    — hard cap on sampled frames (default ``240``).
  ``DASHCAM_OSD_BOTTOM_PCT``    — bottom-strip crop fraction (default ``12``).
  ``DASHCAM_OSD_BATCH_SIZE``    — Cloud Vision batch size (default ``16``).
  ``DASHCAM_OSD_SKIP_WHEN_MAP_TELEMETRY`` — ``true`` to skip the OCR sweep when
      a Trill ``.map`` already has route points (saves GCV cost). Default ``false``:
      always read burned-in HUD into ``dashcam_osd_context``; ``.map`` remains the
      source of truth for Trill / route, and PADUS+gazetteer stay on that route
      (``telemetry_stage``). Backfill from OSD only when there are no ``.map`` points.
  ``DASHCAM_OSD_FORCE_WHEN_MAP_TELEMETRY`` — ``true`` to **always** run the OCR sweep
      even when ``DASHCAM_OSD_SKIP_WHEN_MAP_TELEMETRY=true`` (evidence-first ops; higher GCV cost).
  ``DASHCAM_OSD_BACKFILL_TELEMETRY`` — ``true`` to backfill ctx.telemetry_data
                                       when no .map file (default ``true``).
  ``DASHCAM_OSD_MAX_PLAUSIBLE_MPH`` — reject OCR speeds above this (default ``200``).
  ``DASHCAM_OSD_SPEED_SPIKE_DELTA_MPH`` — drop singleton spikes this many mph
      above the median (default ``45``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .ai_service_costs import user_pref_ai_service_enabled
from .context import JobContext, TelemetryData
from .errors import SkipStage
from .ffmpeg_env import resolve_ffmpeg_executable

logger = logging.getLogger("uploadm8-worker")

DASHCAM_OSD_STAGE_ENABLED = os.environ.get("DASHCAM_OSD_STAGE_ENABLED", "true").lower() == "true"

_DASHCAM_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dashcam-osd")


# ───────────────────────────── env parsers ──────────────────────────────


def _env_float(key: str, default: float, lo: float, hi: float) -> float:
    raw = (os.environ.get(key) or "").strip()
    if not raw:
        return default
    try:
        return max(lo, min(float(raw), hi))
    except ValueError:
        return default


def _env_int(key: str, default: int, lo: int, hi: int) -> int:
    raw = (os.environ.get(key) or "").strip()
    if not raw:
        return default
    try:
        return max(lo, min(int(raw, 10), hi))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = (os.environ.get(key) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


# ───────────────────────────── HUD parsers ──────────────────────────────
#
# M8/Escort HUD example line (one strip, one moment in time):
#   "2025/03/05 04:50 12 PM 36.136162° -115.178398° 46MPH C Walker"
#
# OCR quirks we tolerate:
#   • degree symbol may render as "°", "º", "o", "0" or be dropped entirely
#   • the seconds delimiter on M8 is a SPACE, not a colon ("04:50 12 PM")
#   • watermark "ESCORT." may bleed onto the same line — must be scrubbed
#   • minus sign on longitude may be picked up as a hyphen, en-dash or em-dash

_DEG = r"[°ºo*?\u00B0\u00BA]?"

# 2025/03/05  or 2025-03-05
_DATE_RE = re.compile(r"\b(20\d{2})[/\-](\d{1,2})[/\-](\d{1,2})\b")

# 04:50:12 PM  /  04:50 12 PM  /  04:50:12  /  4:50 PM
# H:MM is always colon-separated; MM/SS may be space (M8 quirk) or colon.
_TIME_RE = re.compile(
    r"\b(\d{1,2}):(\d{2})(?:[:\.\s](\d{2}))?\s*([AaPp][Mm])?\b"
)

# Decimal-degree GPS pair (lat then lon). Accepts optional leading sign,
# tolerates a missing minus on longitude and OCR-mangled degree marks.
# Min 2 fractional digits — real dashcams emit 5–7, but synthetic short forms
# (3–4 chars) still pass the range validators above.
_LATLON_RE = re.compile(
    r"([+\-\u2010-\u2015]?\d{1,2}\.\d{2,7})" + _DEG
    + r"\s+"
    + r"([+\-\u2010-\u2015]?\d{1,3}\.\d{2,7})" + _DEG
)

_SPEED_RE = re.compile(r"\b(\d{1,3})\s*(MPH|KMH|KPH|KM/H)\b", re.IGNORECASE)

# Roadside / signage context — never treat these as vehicle HUD speed.
_SPEED_LIMIT_CONTEXT_RE = re.compile(
    r"(?:speed\s*limit|limit\s*(?:is|:)?\s*\d|maximum\s*speed|max\.?\s*speed|"
    r"\bzone\s*\d{2,3}\b|\bmin(?:imum)?\s*speed\b|\badvised\s*speed\b|"
    r"\bschool\s*zone\b|\bwork\s*zone\b)",
    re.IGNORECASE,
)

# Absolute sanity bounds for passenger/dashcam HUD (mph after unit conversion).
_SPEED_MPH_MIN = 0.0
_SPEED_MPH_MAX_DEFAULT = 200.0

# Heading: cardinal letters (N/NE/E/SE/S/SW/W/NW) or numeric bearing.
_HEADING_RE = re.compile(
    r"\b(N|NE|E|SE|S|SW|W|NW)\b|\b(\d{1,3})\s*°\s*(?:HDG|BRG)?\b",
    re.IGNORECASE,
)

# Trailing driver name (1–3 capitalized words after speed/heading, before
# the ESCORT watermark). Heuristic — only kept when matched cleanly.
_DRIVER_TAIL_RE = re.compile(
    r"(?:MPH|KPH|KMH|KM/H)\s+([A-Z][A-Za-z\.]{0,15}(?:\s[A-Z][A-Za-z\.]{0,15}){0,2})",
)

# Brand watermarks that must never be parsed as driver names.
_BRAND_WORDS = frozenset(
    {"escort", "garmin", "viofo", "blackvue", "nextbase", "thinkware", "rove", "vantrue"}
)


def _norm_minus(s: str) -> str:
    """Normalize unicode dashes to ASCII minus so float() accepts them."""
    return (
        s.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2015", "-")
    )


def _scrub(line: str) -> str:
    """Strip dashcam brand watermarks and collapse whitespace."""
    s = line
    for word in _BRAND_WORDS:
        s = re.sub(rf"\b{word}\.?\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_date(line: str) -> Optional[str]:
    m = _DATE_RE.search(line)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}"


def _parse_time(line: str) -> Optional[str]:
    """Return ``HH:MM:SS`` (24-hour). Tolerates space-as-colon between MM/SS."""
    for m in _TIME_RE.finditer(line):
        try:
            hh = int(m.group(1))
            mm = int(m.group(2))
            ss = int(m.group(3) or 0)
        except (TypeError, ValueError):
            continue
        ampm = (m.group(4) or "").upper()
        if not (0 <= mm < 60 and 0 <= ss < 60):
            continue
        if ampm:
            if not (1 <= hh <= 12):
                continue
            if ampm == "PM" and hh != 12:
                hh += 12
            elif ampm == "AM" and hh == 12:
                hh = 0
        else:
            if not (0 <= hh < 24):
                continue
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return None


def _parse_latlon(line: str) -> Tuple[Optional[float], Optional[float]]:
    m = _LATLON_RE.search(line)
    if not m:
        return None, None
    try:
        lat = float(_norm_minus(m.group(1)))
        lon = float(_norm_minus(m.group(2)))
    except ValueError:
        return None, None
    # Dashcam OCR sometimes drops the minus on western-US longitude.
    if 15.0 <= lat <= 75.0 and 60.0 <= lon <= 180.0:
        lon = -lon
    if not (-90.0 <= lat <= 90.0):
        return None, None
    if not (-180.0 <= lon <= 180.0):
        return None, None
    if abs(lat) < 1e-6 and abs(lon) < 1e-6:
        return None, None
    return lat, lon


def _max_plausible_mph() -> float:
    return _env_float("DASHCAM_OSD_MAX_PLAUSIBLE_MPH", _SPEED_MPH_MAX_DEFAULT, 80.0, 300.0)


def _speed_match_to_mph(m: re.Match) -> Tuple[Optional[float], Optional[str]]:
    try:
        v = float(m.group(1))
    except ValueError:
        return None, None
    unit = m.group(2).upper().replace("/", "")
    if unit in ("MPH",):
        return v, "mph"
    if unit in ("KMH", "KPH"):
        return v * 0.621371, "kph"
    return None, None


def _latlon_span(line: str) -> Optional[Tuple[int, int]]:
    m = _LATLON_RE.search(line)
    if not m:
        return None
    return m.start(), m.end()


def _parse_speed(line: str) -> Tuple[Optional[float], Optional[str]]:
    """Return (mph, unit_string). KPH is converted to MPH for caller.

    Prefers the HUD speed token that follows a lat/lon pair (Escort/M8 order:
    date → time → GPS → speed → driver). Rejects roadside speed-limit copy and
    values outside a plausible dashcam range so OCR of signs / noise cannot
    become the published peak.
    """
    if not line or _SPEED_LIMIT_CONTEXT_RE.search(line):
        return None, None

    matches = list(_SPEED_RE.finditer(line))
    if not matches:
        return None, None

    gps_span = _latlon_span(line)
    ranked: List[Tuple[int, float, str, re.Match]] = []
    hi = _max_plausible_mph()
    for m in matches:
        mph, unit = _speed_match_to_mph(m)
        if mph is None or unit is None:
            continue
        if not (_SPEED_MPH_MIN <= mph <= hi):
            continue
        # Prefer tokens after GPS (true HUD). Deprioritize tokens before GPS
        # (often signage OCR that landed on the same strip/frame text blob).
        if gps_span is not None:
            priority = 0 if m.start() >= gps_span[1] else 2
        else:
            priority = 1
        ranked.append((priority, mph, unit, m))

    if not ranked:
        return None, None
    ranked.sort(key=lambda row: (row[0], -row[3].start()))
    _prio, mph, unit, _m = ranked[0]
    return mph, unit


def _line_has_hud_anchor(line: str, *, lat: Optional[float], date: Optional[str], time: Optional[str]) -> bool:
    """True when the OCR line looks like a dashcam HUD, not a roadside sign."""
    if lat is not None:
        return True
    if date and time:
        return True
    # Escort-style compact line without a successful lat parse still often has
    # a degree-marked coordinate token next to the speed.
    if _LATLON_RE.search(line) or re.search(r"\d+\.\d{3,7}\s*" + _DEG, line):
        return True
    return False


def _parse_heading(line: str) -> Optional[str]:
    m = _HEADING_RE.search(line)
    if not m:
        return None
    if m.group(1):
        return m.group(1).upper()
    if m.group(2):
        try:
            deg = int(m.group(2))
            if 0 <= deg <= 360:
                return f"{deg}°"
        except ValueError:
            return None
    return None


def _parse_driver(line: str) -> Optional[str]:
    m = _DRIVER_TAIL_RE.search(line)
    if not m:
        return None
    name = m.group(1).strip()
    low = name.lower()
    for brand in _BRAND_WORDS:
        if brand in low:
            return None
    if len(name) < 2 or len(name) > 40:
        return None
    return name


def parse_osd_line(
    raw_text: str,
    *,
    t_s: Optional[float] = None,
    require_hud_anchor_for_speed: bool = False,
) -> Dict[str, Any]:
    """Parse one OCR'd HUD strip into a structured record.

    When ``require_hud_anchor_for_speed`` is True (Vision full-frame OCR),
    speed is kept only if the same line also has GPS/date+time HUD anchors.
    Bottom-strip OCR already crops to the HUD band, so anchoring is optional
    there — still recorded for aggregation quality.
    """
    line = _scrub(_norm_minus(raw_text or "").replace("\n", " "))
    if not line:
        return {
            "t_s": t_s,
            "raw": "",
            "date": None,
            "time": None,
            "lat": None,
            "lon": None,
            "speed_mph": None,
            "speed_unit": None,
            "speed_hud_anchored": False,
            "heading": None,
            "driver": None,
            "has_signal": False,
        }
    lat, lon = _parse_latlon(line)
    date = _parse_date(line)
    time = _parse_time(line)
    speed_mph, unit = _parse_speed(line)
    hud_anchored = _line_has_hud_anchor(line, lat=lat, date=date, time=time)
    if speed_mph is not None and require_hud_anchor_for_speed and not hud_anchored:
        speed_mph, unit = None, None
    rec: Dict[str, Any] = {
        "t_s": t_s,
        "raw": line[:300],
        "date": date,
        "time": time,
        "lat": lat,
        "lon": lon,
        "speed_mph": round(speed_mph, 1) if speed_mph is not None else None,
        "speed_unit": unit,
        "speed_hud_anchored": bool(speed_mph is not None and hud_anchored),
        "heading": _parse_heading(line),
        "driver": _parse_driver(line),
    }
    rec["has_signal"] = any(
        rec[k] is not None for k in ("date", "time", "lat", "speed_mph")
    )
    return rec


# ───────────────────────────── frame sampling ───────────────────────────


async def _ffprobe_duration(video_path: Path) -> float:
    ffprobe = resolve_ffmpeg_executable("ffprobe") or "ffprobe"
    proc = await asyncio.create_subprocess_exec(
        ffprobe, "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0 or not stdout:
        return 0.0
    try:
        data = json.loads(stdout.decode("utf-8", errors="replace"))
        return float(data.get("format", {}).get("duration") or 0.0)
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0


async def _sample_bottom_strips(
    video_path: Path,
    out_dir: Path,
    fps: float,
    max_frames: int,
    bottom_pct: int,
) -> List[Tuple[float, Path]]:
    """
    Use one ffmpeg invocation to crop the bottom strip of the active video and
    emit JPEGs at ``fps``. Returns list of (timestamp_s, jpeg_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "osd_%05d.jpg")
    bottom_pct = max(5, min(int(bottom_pct), 40))

    # crop=w:h:x:y — crop the bottom `bottom_pct` of the source.
    vf = f"fps={fps:.3f},crop=iw:ih*{bottom_pct}/100:0:ih*(100-{bottom_pct})/100"

    ffmpeg = resolve_ffmpeg_executable() or "ffmpeg"
    cmd = [
        ffmpeg, "-y",
        "-i", str(video_path),
        "-vf", vf,
        "-vsync", "vfr",
        "-frames:v", str(max_frames),
        "-q:v", "2",
        pattern,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.warning(
            "[dashcam_osd] ffmpeg sampling failed: %s",
            (stderr or b"").decode("utf-8", errors="replace")[:300],
        )
        return []

    frames = sorted(out_dir.glob("osd_*.jpg"))
    interval = 1.0 / max(fps, 0.001)
    pairs: List[Tuple[float, Path]] = []
    for idx, p in enumerate(frames):
        if p.stat().st_size < 400:
            continue
        pairs.append((round(idx * interval, 3), p))
    return pairs


# ───────────────────────────── OCR (Cloud Vision) ───────────────────────


def _ocr_batch_sync(image_bytes_list: List[bytes]) -> List[str]:
    """Run GCV TEXT_DETECTION on a batch of JPEGs. Returns text per request."""
    from . import vision_stage as _vs  # type: ignore

    client = _vs._get_gcv_client()  # sets _vs._vision_module as side effect
    v = _vs._vision_module           # read after init so we get the real module
    if client is None or v is None:
        return [""] * len(image_bytes_list)

    requests = [
        v.AnnotateImageRequest(
            image=v.Image(content=blob),
            features=[v.Feature(type_=v.Feature.Type.TEXT_DETECTION)],
        )
        for blob in image_bytes_list
    ]
    try:
        batch = client.batch_annotate_images(requests=requests)
    except Exception as e:  # noqa: BLE001 — vendor SDK raises a wide range
        logger.warning("[dashcam_osd] batch OCR failed: %s", e)
        return [""] * len(image_bytes_list)

    out: List[str] = []
    for resp in batch.responses:
        err = getattr(resp, "error", None)
        if err and getattr(err, "message", None):
            out.append("")
            continue
        anns = getattr(resp, "text_annotations", None) or []
        out.append(anns[0].description.strip() if anns else "")
    return out


async def _ocr_all_strips(
    pairs: List[Tuple[float, Path]],
    batch_size: int,
) -> List[Tuple[float, str]]:
    if not pairs:
        return []
    loop = asyncio.get_running_loop()
    out: List[Tuple[float, str]] = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        blobs = [p.read_bytes() for _, p in chunk]
        texts = await loop.run_in_executor(
            _DASHCAM_EXECUTOR, partial(_ocr_batch_sync, blobs)
        )
        for (t_s, _), text in zip(chunk, texts):
            out.append((t_s, text))
    return out


# ───────────────────────────── aggregation ──────────────────────────────


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.8
    dl = math.radians(lat2 - lat1)
    do = math.radians(lon2 - lon1)
    a = (
        math.sin(dl / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(do / 2) ** 2
    )
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _gps_implied_peak_mph(samples: List[Dict[str, Any]]) -> Optional[float]:
    """Peak ground speed implied by successive GPS fixes (miles / hours)."""
    pts: List[Tuple[float, float, float]] = []
    for s in samples:
        la, lo, t = s.get("lat"), s.get("lon"), s.get("t_s")
        if la is None or lo is None or t is None:
            continue
        try:
            pts.append((float(la), float(lo), float(t)))
        except (TypeError, ValueError):
            continue
    if len(pts) < 2:
        return None
    pts.sort(key=lambda p: p[2])
    peak = 0.0
    for a, b in zip(pts, pts[1:]):
        dt = b[2] - a[2]
        if dt < 0.4:
            continue
        if dt > 30.0:
            # Large gaps are unreliable for instantaneous speed.
            continue
        dist = _haversine_miles(a[0], a[1], b[0], b[1])
        mph = dist / (dt / 3600.0)
        if mph > peak:
            peak = mph
    return round(peak, 1) if peak > 0 else None


def _select_trusted_speeds(
    with_signal: List[Dict[str, Any]],
) -> Tuple[List[float], List[Dict[str, Any]], Dict[str, Any]]:
    """Pick speeds safe to publish; drop OCR outliers / signage ghosts.

    Rules (in order):
      1. Prefer HUD-anchored samples when any exist.
      2. Drop values above plausible max.
      3. Drop singleton spikes far above the median of the trusted set.
      4. Require peak confirmation (2+ samples near peak) unless GPS motion
         corroborates a similar peak.
    """
    meta: Dict[str, Any] = {
        "speed_samples_raw": 0,
        "speed_samples_trusted": 0,
        "speed_rejected_unanchored": 0,
        "speed_rejected_outlier": 0,
        "speed_rejected_unconfirmed_peak": 0,
        "gps_implied_peak_mph": None,
    }
    candidates = [s for s in with_signal if s.get("speed_mph") is not None]
    meta["speed_samples_raw"] = len(candidates)
    if not candidates:
        return [], [], meta

    anchored = [s for s in candidates if s.get("speed_hud_anchored")]
    pool = anchored if anchored else candidates
    if anchored and len(anchored) < len(candidates):
        meta["speed_rejected_unanchored"] = len(candidates) - len(anchored)

    hi = _max_plausible_mph()
    pool = [
        s for s in pool
        if _SPEED_MPH_MIN <= float(s["speed_mph"]) <= hi
    ]
    if not pool:
        return [], [], meta

    speeds = sorted(float(s["speed_mph"]) for s in pool)
    mid = speeds[len(speeds) // 2]
    # Spike gate: a lone OCR digitation error (e.g. 146 vs 46) sits far above
    # the median; keep values within a generous but finite band.
    spike_delta = _env_float("DASHCAM_OSD_SPEED_SPIKE_DELTA_MPH", 45.0, 20.0, 100.0)
    filtered: List[Dict[str, Any]] = []
    for s in pool:
        v = float(s["speed_mph"])
        if len(speeds) >= 3 and v > mid + spike_delta:
            meta["speed_rejected_outlier"] += 1
            continue
        filtered.append(s)
    if not filtered:
        # Fall back to the median sample rather than publishing nothing / a spike.
        nearest = min(pool, key=lambda s: abs(float(s["speed_mph"]) - mid))
        filtered = [nearest]

    gps_peak = _gps_implied_peak_mph(with_signal)
    meta["gps_implied_peak_mph"] = gps_peak

    vals = [float(s["speed_mph"]) for s in filtered]
    peak = max(vals)
    mid_trusted = sorted(vals)[len(vals) // 2]
    near = [v for v in vals if v >= peak - 8.0]
    confirmed = len(near) >= 2
    # Normal clip max (not a wild OCR tip): peak sits near the distribution.
    if not confirmed and len(vals) >= 2 and peak <= mid_trusted + 20.0:
        confirmed = True
    if not confirmed and gps_peak is not None and peak <= gps_peak * 1.35 + 8.0:
        confirmed = True
    if not confirmed and len(vals) == 1 and gps_peak is None:
        # Single read is publishable only when the same OCR line carried HUD
        # anchors (GPS/date+time). Bare "65 MPH" signage stays rejected.
        confirmed = bool(filtered[0].get("speed_hud_anchored"))
    if not confirmed and len(near) < 2:
        # Demote unconfirmed peak to the next-best confirmed cluster, or drop
        # entirely when the only reads are uncorroborated (signage / OCR noise).
        meta["speed_rejected_unconfirmed_peak"] += 1
        demoted = [v for v in vals if v < peak - 0.01]
        if demoted:
            peak2 = max(demoted)
            filtered = [s for s in filtered if float(s["speed_mph"]) <= peak2 + 0.01]
            vals = [float(s["speed_mph"]) for s in filtered]
        else:
            filtered = []
            vals = []

    meta["speed_samples_trusted"] = len(filtered)
    return vals, filtered, meta


def _speed_series_from_trusted(speed_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compact time-series of trusted HUD speeds for timeline / caption prompts."""
    series: List[Dict[str, Any]] = []
    for s in speed_samples:
        try:
            mph = float(s.get("speed_mph"))
        except (TypeError, ValueError):
            continue
        if mph < 0:
            continue
        try:
            t_s = float(s.get("t_s") if s.get("t_s") is not None else 0.0)
        except (TypeError, ValueError):
            t_s = 0.0
        series.append({
            "t_s": round(t_s, 3),
            "mph": round(mph, 1),
            "speed_mph": round(mph, 1),
            "hud_anchored": bool(s.get("speed_hud_anchored")),
        })
    series.sort(key=lambda row: row["t_s"])
    return series


def _aggregate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reduce per-frame parses to a clip-level summary."""
    with_signal = [s for s in samples if s.get("has_signal")]
    coverage = round(len(with_signal) / max(len(samples), 1), 3)

    def _first(field: str) -> Optional[Any]:
        for s in with_signal:
            v = s.get(field)
            if v is not None:
                return v
        return None

    def _last(field: str) -> Optional[Any]:
        for s in reversed(with_signal):
            v = s.get(field)
            if v is not None:
                return v
        return None

    speeds, speed_samples, speed_meta = _select_trusted_speeds(with_signal)
    max_speed = max(speeds) if speeds else 0.0
    max_at_s: Optional[float] = None
    for s in speed_samples:
        if s.get("speed_mph") is not None and float(s["speed_mph"]) >= max_speed - 0.01:
            max_at_s = s.get("t_s")
            break
    avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else 0.0

    # Speed unit: report what the dashcam itself displayed (we always store mph)
    units = [s.get("speed_unit") for s in speed_samples if s.get("speed_unit")]
    if not units:
        units = [s.get("speed_unit") for s in with_signal if s.get("speed_unit")]
    unit_detected = max(set(units), key=units.count) if units else None

    # Per-fix speed on the path: use trusted value when the frame was kept,
    # else nearest trusted sample by time (never re-inject rejected spikes).
    trusted_by_id = {id(s): float(s["speed_mph"]) for s in speed_samples}
    trusted_timeline = [
        (float(s.get("t_s") or 0.0), float(s["speed_mph"]))
        for s in speed_samples
        if s.get("speed_mph") is not None
    ]

    def _trusted_speed_for(sample: Dict[str, Any]) -> float:
        if id(sample) in trusted_by_id:
            return trusted_by_id[id(sample)]
        if not trusted_timeline:
            return 0.0
        t = float(sample.get("t_s") or 0.0)
        return min(trusted_timeline, key=lambda row: abs(row[0] - t))[1]

    gps_path: List[List[float]] = []
    last_lat: Optional[float] = None
    last_lon: Optional[float] = None
    for s in with_signal:
        la, lo = s.get("lat"), s.get("lon")
        if la is None or lo is None:
            continue
        # Drop noisy duplicates (~< 30 ft from previous fix).
        if last_lat is not None and last_lon is not None:
            if _haversine_miles(last_lat, last_lon, la, lo) < 0.005:
                continue
        gps_path.append([
            round(la, 6),
            round(lo, 6),
            round(_trusted_speed_for(s), 1),
            float(s.get("t_s") or 0.0),
        ])
        last_lat, last_lon = la, lo

    drivers = [s.get("driver") for s in with_signal if s.get("driver")]
    driver_name = max(set(drivers), key=drivers.count) if drivers else None

    return {
        "samples": samples,
        "frames_sampled": len(samples),
        "frames_with_signal": len(with_signal),
        "coverage_pct": coverage,
        "first_seen": {
            "date": _first("date"),
            "time": _first("time"),
            "lat": _first("lat"),
            "lon": _first("lon"),
            "t_s": with_signal[0].get("t_s") if with_signal else None,
        },
        "last_seen": {
            "date": _last("date"),
            "time": _last("time"),
            "lat": _last("lat"),
            "lon": _last("lon"),
            "t_s": with_signal[-1].get("t_s") if with_signal else None,
        },
        "max_speed_mph": round(max_speed, 1),
        "max_speed_at_s": max_at_s,
        # Alias used by timeline builder (historical key name).
        "max_speed_t_s": max_at_s,
        "avg_speed_mph": avg_speed,
        "speed_unit_detected": unit_detected,
        "speed_quality": speed_meta,
        "speed_series": _speed_series_from_trusted(speed_samples),
        "gps_path": gps_path,
        "driver_name": driver_name,
        "engine": "gcv",
    }


def _backfill_telemetry(ctx: JobContext, osd: Dict[str, Any]) -> bool:
    """When no .map telemetry, populate ctx.telemetry_data from OSD GPS.

    Returns True if backfill happened. Downstream stages (telemetry trill,
    digest, reverse-geocoding) will then treat the OSD-derived points as if
    they came from a .map file.
    """
    existing = ctx.telemetry_data or ctx.telemetry
    has_existing_points = bool(existing and getattr(existing, "points", None))
    if has_existing_points:
        return False

    path = osd.get("gps_path") or []
    if not path:
        return False

    points: List[Dict[str, Any]] = []
    for row in path:
        try:
            lat, lon, sp_mph, t_s = row[0], row[1], row[2], row[3]
        except (IndexError, TypeError):
            continue
        points.append({
            "lat": float(lat),
            "lon": float(lon),
            "speed_mph": float(sp_mph or 0.0),
            "t_s": float(t_s or 0.0),
        })
    if not points:
        return False

    tel = TelemetryData()
    tel.points = points
    tel.max_speed_mph = float(osd.get("max_speed_mph") or 0.0)
    tel.avg_speed_mph = float(osd.get("avg_speed_mph") or 0.0)
    if len(points) >= 2:
        # Cumulative miles along the OSD-sampled path.
        total = 0.0
        for a, b in zip(points, points[1:]):
            total += _haversine_miles(a["lat"], a["lon"], b["lat"], b["lon"])
        tel.total_distance_miles = round(total, 2)
    last_seen = osd.get("last_seen") or {}
    first_seen = osd.get("first_seen") or {}
    try:
        t0 = float(first_seen.get("t_s") or 0.0)
        t1 = float(last_seen.get("t_s") or 0.0)
        tel.duration_seconds = max(0.0, t1 - t0)
    except (TypeError, ValueError):
        tel.duration_seconds = 0.0
    tel.start_lat = points[0]["lat"]
    tel.start_lon = points[0]["lon"]
    mid = points[len(points) // 2]
    tel.mid_lat = mid["lat"]
    tel.mid_lon = mid["lon"]

    ctx.telemetry_data = tel
    ctx.telemetry = tel
    return True


def _apply_trill_from_backfilled_telemetry(ctx: JobContext) -> None:
    """OSD-only routes skip telemetry_stage, so compute Trill here after HUD backfill."""
    tel = ctx.telemetry_data or ctx.telemetry
    if not tel or not getattr(tel, "points", None):
        return
    try:
        from .telemetry_stage import (
            DEFAULT_EUPHORIA_MPH,
            DEFAULT_SPEEDING_MPH,
            calculate_trill_score,
            get_trill_modifiers,
        )

        us = ctx.user_settings or {}
        speeding_mph = int(us.get("speeding_mph") or us.get("speedingMph") or DEFAULT_SPEEDING_MPH)
        euphoria_mph = int(us.get("euphoria_mph") or us.get("euphoriaMph") or DEFAULT_EUPHORIA_MPH)
        trill = calculate_trill_score(tel, speeding_mph, euphoria_mph)
        modifier, hashtags = get_trill_modifiers(trill.score, tel.max_speed_mph, trill.bucket)
        trill.title_modifier = modifier
        trill.hashtags = hashtags
        ctx.trill_score = trill
        ctx.trill = trill
    except Exception as e:  # pragma: no cover - defensive, Trill is enrichment only
        logger.debug("[dashcam_osd] OSD Trill calculation skipped: %s", e)


def _vision_osd_context(ctx: JobContext) -> Dict[str, Any]:
    """
    Parse OSD-like GPS/speed text from the regular Vision OCR result.

    This is a fallback for clips where the general Vision frame scan sees the
    HUD clearly but the dedicated bottom-strip OSD sweep is disabled, skipped,
    or fails to sample usable frames.
    """
    vc = getattr(ctx, "vision_context", None) or {}
    if not isinstance(vc, dict):
        return {}
    ocr = str(vc.get("ocr_text") or "").strip()
    if not ocr:
        return {}

    chunks = [c.strip() for c in re.split(r"\n-{3,}\n", ocr) if c.strip()]
    if not chunks:
        chunks = [line.strip() for line in ocr.splitlines() if line.strip()]
    if not chunks:
        return {}

    fractions = vc.get("vision_sample_fractions")
    if not isinstance(fractions, list):
        fractions = []
    duration = 0.0
    try:
        duration = float((getattr(ctx, "video_info", None) or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0

    samples: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        t_s: Optional[float]
        if idx < len(fractions):
            try:
                frac = max(0.0, min(1.0, float(fractions[idx])))
                t_s = round(frac * duration, 3) if duration > 0 else round(frac, 3)
            except (TypeError, ValueError):
                t_s = float(idx)
        else:
            t_s = float(idx)
        # Full-frame Vision OCR includes road signs; require HUD anchors so
        # "SPEED LIMIT 65 MPH" never becomes the published peak.
        samples.append(
            parse_osd_line(chunk, t_s=t_s, require_hud_anchor_for_speed=True)
        )

    osd = _aggregate(samples)
    if osd.get("gps_path"):
        osd["engine"] = "vision_gcv_ocr"
        osd["source"] = "vision_stage"
        osd["vision_ocr_chars"] = len(ocr)
        return osd
    # Speed-only path: Vision saw MPH/KMH on HUD-anchored lines but no GPS.
    # Require at least one hud-anchored speed sample (already enforced above).
    peak = float(osd.get("max_speed_mph") or 0)
    anchored_speeds = [
        s for s in (osd.get("samples") or [])
        if s.get("speed_mph") is not None and s.get("speed_hud_anchored")
    ]
    if peak > 0 and anchored_speeds:
        osd["engine"] = "vision_gcv_ocr"
        osd["source"] = "vision_stage_speed_only"
        osd["vision_ocr_chars"] = len(ocr)
        return osd
    return {}


async def backfill_telemetry_from_vision_osd(ctx: JobContext, *, enrich_geo: bool = True) -> bool:
    """
    Use regular Vision OCR as an OSD GPS fallback, then optionally run the same
    reverse-geocode + PAD-US/gazetteer enrichment as the dedicated OSD stage.
    """
    existing = ctx.telemetry_data or ctx.telemetry
    if existing and getattr(existing, "points", None):
        return False

    osd = _vision_osd_context(ctx)
    if not osd:
        return False

    did = _backfill_telemetry(ctx, osd)
    if not did and float(osd.get("max_speed_mph") or 0) > 0:
        cur = getattr(ctx, "dashcam_osd_context", None)
        if not isinstance(cur, dict):
            cur = {}
        cur = dict(cur)
        prev_peak = float(cur.get("max_speed_mph") or 0)
        od_peak = float(osd.get("max_speed_mph") or 0)
        # Vision speed-only merge: fill empty peaks, or replace only when the
        # Vision peak agrees with an existing read (within 8 mph). Never let a
        # higher uncorroborated OCR value overwrite a lower trusted peak.
        if prev_peak <= 0 and od_peak > 0:
            cur["max_speed_mph"] = od_peak
        elif od_peak > 0 and prev_peak > 0 and abs(od_peak - prev_peak) <= 8.0:
            cur["max_speed_mph"] = max(prev_peak, od_peak)
        cur["vision_osd_speed_fallback"] = osd
        ctx.dashcam_osd_context = cur
        logger.info(
            "[dashcam_osd] merged Vision OCR peak speed (no GPS path): %.1f mph (prev=%.1f)",
            float(cur.get("max_speed_mph") or 0),
            prev_peak,
        )
        return True
    if not did:
        return False
    _apply_trill_from_backfilled_telemetry(ctx)

    current_osd = getattr(ctx, "dashcam_osd_context", None) or {}
    if isinstance(current_osd, dict) and current_osd and not current_osd.get("gps_path"):
        current_osd["vision_osd_fallback"] = osd
        ctx.dashcam_osd_context = current_osd
    else:
        ctx.dashcam_osd_context = osd

    if enrich_geo:
        await _enrich_backfilled_telemetry_with_geocode(ctx)
    logger.info(
        "[dashcam_osd] backfilled telemetry from Vision OCR: points=%d peak=%smph",
        len((ctx.telemetry_data.points if ctx.telemetry_data else []) or []),
        osd.get("max_speed_mph"),
    )
    return True


async def _enrich_backfilled_telemetry_with_geocode(ctx: JobContext) -> None:
    """Reverse-geocode HUD-derived GPS so city/state/road land on TelemetryData.

    Runs when ``_backfill_telemetry`` built ``ctx.telemetry_data`` from on-screen
    GPS (no ``.map`` was uploaded). Matches ``telemetry_stage`` Nominatim flow
    plus **PADUS / US gazetteer** enrichment so the same geo signals exist whether
    GPS came from a companion ``.map`` file or burned-in OSD.
    """
    tel = ctx.telemetry_data or ctx.telemetry
    if not tel:
        return

    mid_lat = getattr(tel, "mid_lat", None)
    mid_lon = getattr(tel, "mid_lon", None)

    # Late import keeps dashcam_osd_stage importable without httpx at module scope.
    from .telemetry_stage import (
        apply_padus_gazetteer_to_telemetry,
        reverse_geocode_details,
        _haversine_miles,
    )

    if not getattr(tel, "location_display", None) and mid_lat is not None and mid_lon is not None:
        details = await reverse_geocode_details(float(mid_lat), float(mid_lon))
        if details:
            tel.location_display = details.get("location_display")
            tel.location_city = details.get("location_city")
            tel.location_state = details.get("location_state")
            tel.location_country = details.get("location_country")
            tel.location_road = details.get("location_road")
            if tel.location_display:
                logger.info(
                    "[dashcam_osd] geocoded HUD midpoint (%.5f, %.5f) → %s",
                    float(mid_lat),
                    float(mid_lon),
                    tel.location_display,
                )

    # Second geocode at route start when far from midpoint (same as telemetry_stage).
    try:
        min_sep = float(os.environ.get("TELEMETRY_START_GEOCODE_MIN_SEPARATION_MILES", "5") or 5)
    except (TypeError, ValueError):
        min_sep = 5.0
    min_sep = max(1.0, min(min_sep, 200.0))

    start_lat = getattr(tel, "start_lat", None)
    start_lon = getattr(tel, "start_lon", None)
    if (
        start_lat is not None
        and start_lon is not None
        and mid_lat is not None
        and mid_lon is not None
    ):
        sep = _haversine_miles(
            float(start_lat), float(start_lon), float(mid_lat), float(mid_lon)
        )
        if sep >= min_sep:
            await asyncio.sleep(1.15)
            start_details = await reverse_geocode_details(float(start_lat), float(start_lon))
            if start_details and start_details.get("location_display"):
                tel.location_start_display = start_details["location_display"]
                logger.info(
                    "[dashcam_osd] geocoded HUD route start (%.2f mi from mid): %s",
                    sep,
                    tel.location_start_display,
                )

    await apply_padus_gazetteer_to_telemetry(
        tel, db_pool=getattr(ctx, "_db_pool", None)
    )
    ctx.telemetry = tel
    ctx.telemetry_data = tel


# ───────────────────────────── stage entrypoint ─────────────────────────


async def run_dashcam_osd_stage(ctx: JobContext) -> JobContext:
    """Populate ``ctx.dashcam_osd_context`` and (optionally) backfill telemetry.

    With a Trill ``.map``, route + Trill + PADUS/gazetteer stay driven by
    ``telemetry_stage``; this stage still OCRs the burned-in HUD by default so
    speed/driver/HUD timeline merge into fusion/captions without replacing ``.map`` points.
    """
    ctx.mark_stage("dashcam_osd")

    if not DASHCAM_OSD_STAGE_ENABLED:
        raise SkipStage("Dashcam OSD stage disabled via env")

    tier_allowed = getattr(ctx.entitlements, "allowed_ai_services", None) if ctx.entitlements else None
    tier_allowed_set = set(tier_allowed) if tier_allowed is not None else None
    if not user_pref_ai_service_enabled(
        ctx.user_settings or {}, "dashcam_osd", default=False, allowed_services=tier_allowed_set
    ):
        raise SkipStage("Dashcam HUD Reader disabled in upload preferences (aiServiceDashcamOSD)")

    # Legacy / cost opt-out: skip full-strip OCR when Trill .map already populated.
    # Default is to always run so captions + fusion see HUD speed/driver/timeline
    # alongside .map GPS; PADUS/gazetteer still apply only to the .map route in
    # telemetry_stage (backfill path unchanged when no .map).
    if (
        _env_bool("DASHCAM_OSD_SKIP_WHEN_MAP_TELEMETRY", False)
        and not _env_bool("DASHCAM_OSD_FORCE_WHEN_MAP_TELEMETRY", False)
    ):
        has_telemetry = bool(
            (ctx.telemetry_data and ctx.telemetry_data.points)
            or (ctx.telemetry and ctx.telemetry.points)
        )
        if has_telemetry:
            raise SkipStage(
                "Dashcam OSD skipped: .map telemetry present and "
                "DASHCAM_OSD_SKIP_WHEN_MAP_TELEMETRY=true"
            )

    # GCV must be configured (we share credentials with vision_stage).
    from .vision_stage import gcp_vision_credentials_configured
    if not gcp_vision_credentials_configured():
        raise SkipStage("Dashcam OSD skipped: GCP credentials not configured")

    video_path: Optional[Path] = None
    for c in (ctx.processed_video_path, ctx.local_video_path):
        if c and Path(c).exists():
            video_path = Path(c)
            break
    if not video_path or not ctx.temp_dir:
        raise SkipStage("Dashcam OSD skipped: no local video available")

    fps = _env_float("DASHCAM_OSD_FPS", 1.0, 0.1, 4.0)
    max_frames = _env_int("DASHCAM_OSD_MAX_FRAMES", 240, 8, 1200)
    bottom_pct = _env_int("DASHCAM_OSD_BOTTOM_PCT", 12, 5, 40)
    batch_size = _env_int("DASHCAM_OSD_BATCH_SIZE", 16, 1, 32)
    backfill_enabled = _env_bool("DASHCAM_OSD_BACKFILL_TELEMETRY", True)

    duration = await _ffprobe_duration(video_path)
    if duration > 0:
        # Don't oversample: cap frames to ceil(duration * fps).
        max_frames = min(max_frames, max(2, int(math.ceil(duration * fps)) + 2))

    out_dir = Path(ctx.temp_dir) / "dashcam_osd"
    try:
        pairs = await _sample_bottom_strips(
            video_path, out_dir, fps=fps, max_frames=max_frames, bottom_pct=bottom_pct
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("[dashcam_osd] frame sampling error: %s", e)
        pairs = []

    if not pairs:
        ctx.dashcam_osd_context = {
            "engine": "gcv",
            "frames_sampled": 0,
            "coverage_pct": 0.0,
            "skipped": "no_frames_sampled",
        }
        return ctx

    ocr_pairs = await _ocr_all_strips(pairs, batch_size=batch_size)
    samples: List[Dict[str, Any]] = [
        parse_osd_line(text, t_s=t_s) for t_s, text in ocr_pairs
    ]
    osd = _aggregate(samples)
    osd["fps"] = fps
    osd["bottom_pct"] = bottom_pct
    osd["video_duration_s"] = round(duration, 2) if duration else None

    # Full-frame Vision OCR often captures ESCORT-style footer HUD (lat/lon/speed) that
    # bottom-strip sampling misses when the crop excludes the overlay.
    strip_gps = bool(osd.get("gps_path"))
    if not strip_gps:
        vision_osd = _vision_osd_context(ctx)
        path_v = (vision_osd or {}).get("gps_path") or []
        if path_v:
            osd["gps_path"] = path_v
            osd["vision_strip_fallback"] = True
            osd["vision_osd_source"] = str(vision_osd.get("source") or "vision_stage")
            for _key in ("first_seen", "last_seen"):
                vs = vision_osd.get(_key)
                if isinstance(vs, dict) and vs.get("lat") is not None and vs.get("lon") is not None:
                    cur = osd.get(_key)
                    if not isinstance(cur, dict) or cur.get("lat") is None:
                        osd[_key] = dict(vs)
            try:
                v_peak = float(vision_osd.get("max_speed_mph") or 0.0)
                s_peak = float(osd.get("max_speed_mph") or 0.0)
                # Only adopt Vision peak when strip OCR had no trusted speed,
                # or Vision peak is within a small band of strip peak (agreeing
                # reads). Never inflate strip peak with a lone higher OCR hit.
                adopt_vision_peak = False
                if s_peak <= 0 and v_peak > 0:
                    adopt_vision_peak = True
                elif v_peak > 0 and abs(v_peak - s_peak) <= 8.0 and v_peak > s_peak:
                    adopt_vision_peak = True
                if adopt_vision_peak:
                    osd["max_speed_mph"] = vision_osd.get("max_speed_mph")
                    osd["max_speed_at_s"] = vision_osd.get("max_speed_at_s")
                    osd["avg_speed_mph"] = vision_osd.get("avg_speed_mph", osd.get("avg_speed_mph"))
            except (TypeError, ValueError):
                pass
            if not osd.get("driver_name") and vision_osd.get("driver_name"):
                osd["driver_name"] = vision_osd.get("driver_name")
            osd["engine"] = "gcv+vision_fallback"
            logger.info(
                "[dashcam_osd] merged Vision OCR HUD path: fixes=%d source=%s",
                len(path_v),
                osd.get("vision_osd_source"),
            )

    backfilled = False
    if backfill_enabled:
        try:
            backfilled = _backfill_telemetry(ctx, osd)
            if backfilled:
                _apply_trill_from_backfilled_telemetry(ctx)
        except (TypeError, ValueError, AttributeError) as e:
            logger.warning("[dashcam_osd] telemetry backfill failed (non-fatal): %s", e)
    osd["telemetry_backfilled"] = backfilled

    # When we backfilled telemetry from the HUD (no .map uploaded) the
    # telemetry_stage has already run (or skipped), so we reverse-geocode here
    # and run the same PADUS / gazetteer enrichment as ``run_telemetry_stage``.
    if backfilled:
        try:
            await _enrich_backfilled_telemetry_with_geocode(ctx)
        except Exception as e:  # pragma: no cover - defensive, never fatal
            logger.warning("[dashcam_osd] reverse-geocode after backfill failed (non-fatal): %s", e)

    ctx.dashcam_osd_context = osd

    fs = osd.get("first_seen") or {}
    ls = osd.get("last_seen") or {}
    logger.info(
        "[dashcam_osd] frames=%d coverage=%.0f%% peak=%smph driver=%s "
        "first=%s %s @%s,%s last=%s %s @%s,%s backfill=%s",
        osd.get("frames_sampled", 0),
        100.0 * float(osd.get("coverage_pct") or 0.0),
        osd.get("max_speed_mph"),
        osd.get("driver_name"),
        fs.get("date"), fs.get("time"), fs.get("lat"), fs.get("lon"),
        ls.get("date"), ls.get("time"), ls.get("lat"), ls.get("lon"),
        backfilled,
    )
    return ctx


__all__ = [
    "DASHCAM_OSD_STAGE_ENABLED",
    "parse_osd_line",
    "run_dashcam_osd_stage",
    "backfill_telemetry_from_vision_osd",
]
