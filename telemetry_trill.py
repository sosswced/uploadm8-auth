"""Telemetry Trill compatibility module.

This local module provides the small API surface the worker and Trill preview
router expect:

- ``safe_analyze_video(...)`` for .map smoke analysis.
- ``enrich_route_padus_gazetteer(...)`` for Census gazetteer + PAD-US lookups.

It intentionally uses ``pyogrio`` + ``shapely`` directly so Windows/Python 3.14
does not need Fiona/GeoPandas compiled from source.
"""

from __future__ import annotations

from functools import lru_cache
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("uploadm8-worker.telemetry_trill")


def _valid_lat_lon(lat: Any, lon: Any) -> bool:
    try:
        la = float(lat)
        lo = float(lon)
    except (TypeError, ValueError):
        return False
    return -90 <= la <= 90 and -180 <= lo <= 180 and not (abs(la) < 1e-9 and abs(lo) < 1e-9)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_km = 6371.0088
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return r_km * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _representative_lat_lon(
    points: Sequence[Dict[str, Any]],
    mid_lat: Optional[float],
    mid_lon: Optional[float],
) -> Optional[Tuple[float, float]]:
    if _valid_lat_lon(mid_lat, mid_lon):
        return float(mid_lat), float(mid_lon)
    clean = [p for p in points if _valid_lat_lon(p.get("lat"), p.get("lon"))]
    if not clean:
        return None
    p = clean[len(clean) // 2]
    return float(p["lat"]), float(p["lon"])


@lru_cache(maxsize=4)
def _load_gazetteer(path: str):
    import pandas as pd

    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    needed = {"NAME", "USPS", "INTPTLAT", "INTPTLONG"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"gazetteer missing columns: {sorted(missing)}")
    out = df[["NAME", "USPS", "INTPTLAT", "INTPTLONG"]].copy()
    out["lat"] = pd.to_numeric(out["INTPTLAT"], errors="coerce")
    out["lon"] = pd.to_numeric(out["INTPTLONG"], errors="coerce")
    return out.dropna(subset=["lat", "lon"])


def _nearest_gazetteer_place(path: str, lat: float, lon: float) -> Dict[str, Any]:
    df = _load_gazetteer(str(Path(path)))
    # Fast enough for 32k rows; avoids requiring sklearn/scipy.
    best: Optional[Dict[str, Any]] = None
    best_km = float("inf")
    for row in df.itertuples(index=False):
        d = _haversine_km(lat, lon, float(row.lat), float(row.lon))
        if d < best_km:
            best_km = d
            best = {
                "gazetteer_place_name": str(row.NAME),
                "gazetteer_state_usps": str(row.USPS),
                "gazetteer_distance_km": round(d, 3),
            }
    return best or {}


def _padus_layers(path: str) -> List[Tuple[str, Optional[str]]]:
    import pyogrio

    rows = pyogrio.list_layers(path)
    out: List[Tuple[str, Optional[str]]] = []
    for row in rows:
        name = str(row[0])
        geom = str(row[1]) if len(row) > 1 and row[1] is not None else None
        out.append((name, geom))
    return out


def _choose_padus_layer(path: str, requested: Optional[str]) -> Optional[str]:
    if requested:
        return requested
    layers = _padus_layers(path)
    polygon_layers = [name for name, geom in layers if geom and "polygon" in geom.lower()]
    preferred = [
        "PADUS4_1Combined_Proclamation_Marine_Fee_Designation_Easement",
        "PADUS4_1Fee",
        "PADUS4_1Designation",
        "PADUS4_1Easement",
        "PADUS4_1Proclamation",
    ]
    for name in preferred:
        if name in polygon_layers:
            return name
    if polygon_layers:
        return polygon_layers[0]
    logger.warning(
        "PADUS file at %s has no polygon layers — check PADUS_LAYER env var or file version. "
        "Available layers: %s",
        path,
        [name for name, _ in layers],
    )
    return None


def _padus_hit(
    path: str,
    layer: Optional[str],
    points: Sequence[Dict[str, Any]],
    lat: float,
    lon: float,
) -> Dict[str, Any]:
    import pyogrio
    from pyproj import CRS, Transformer
    from shapely import wkb
    from shapely.geometry import Point

    chosen = _choose_padus_layer(path, layer)
    if not chosen:
        return {}

    info = pyogrio.read_info(path, layer=chosen)
    crs_raw = info.get("crs")
    crs = CRS.from_user_input(crs_raw) if crs_raw else CRS.from_epsg(4326)
    to_layer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    clean = [p for p in points if _valid_lat_lon(p.get("lat"), p.get("lon"))]
    if not clean:
        clean = [{"lat": lat, "lon": lon}]
    xy = [to_layer.transform(float(p["lon"]), float(p["lat"])) for p in clean]
    xs = [p[0] for p in xy]
    ys = [p[1] for p in xy]
    # PAD-US is in meters for the official GDB. Give route points a practical
    # near-public-lands buffer so close road-adjacent parks still surface.
    buffer_m = 1609.34
    bbox = (min(xs) - buffer_m, min(ys) - buffer_m, max(xs) + buffer_m, max(ys) + buffer_m)

    meta, table = pyogrio.read_arrow(
        path,
        layer=chosen,
        columns=["Unit_Nm", "Loc_Nm", "Mang_Name", "Own_Name"],
        bbox=bbox,
    )
    if table.num_rows <= 0:
        return {"near_padus": False, "padus_layer": chosen}

    geom_col = meta.get("geometry_name") or "SHAPE"
    names = {name: table[name].to_pylist() for name in table.column_names if name != geom_col}
    geoms = table[geom_col].to_pylist() if geom_col in table.column_names else []
    route_points = [Point(x, y) for x, y in xy]

    best_name = ""
    best_dist = float("inf")
    for idx, raw_geom in enumerate(geoms):
        if not raw_geom:
            continue
        geom = wkb.loads(bytes(raw_geom))
        for pt in route_points:
            dist = geom.distance(pt)
            if dist < best_dist:
                best_dist = dist
                for col in ("Unit_Nm", "Loc_Nm", "Mang_Name", "Own_Name"):
                    vals = names.get(col) or []
                    val = vals[idx] if idx < len(vals) else None
                    if val and str(val).strip():
                        best_name = str(val).strip()
                        break
    return {
        "near_padus": bool(best_dist <= buffer_m),
        "padus_unit_name": best_name if best_name and best_dist <= buffer_m else None,
        "padus_distance_m": round(best_dist, 2) if best_dist != float("inf") else None,
        "padus_layer": chosen,
    }


def enrich_route_padus_gazetteer(
    points: Sequence[Dict[str, Any]],
    mid_lat: Optional[float] = None,
    mid_lon: Optional[float] = None,
    *,
    gaz_places_path: Optional[str] = None,
    padus_path: Optional[str] = None,
    padus_layer: Optional[str] = None,
) -> Dict[str, Any]:
    """Return gazetteer/PAD-US enrichment for route points."""
    rep = _representative_lat_lon(points, mid_lat, mid_lon)
    if not rep:
        return {}
    lat, lon = rep
    out: Dict[str, Any] = {}

    if gaz_places_path and Path(gaz_places_path).is_file():
        out.update(_nearest_gazetteer_place(str(gaz_places_path), lat, lon))

    if padus_path and Path(padus_path).exists():
        try:
            out.update(_padus_hit(str(padus_path), padus_layer, points, lat, lon))
        except Exception as e:
            out["padus_error"] = str(e)[:300]

    return out


def safe_analyze_video(
    video_path: str,
    map_path: str,
    *,
    gaz_places_path: Optional[str] = None,
    padus_path: Optional[str] = None,
    padus_layer: Optional[str] = None,
    hud_enabled: bool = False,
) -> Dict[str, Any]:
    """Analyze a .map route and return the historical UploadM8 result shape."""
    try:
        from stages.telemetry_stage import (
            calculate_trill_score,
            get_representative_coords,
            get_trill_modifiers,
            parse_map_file,
        )

        telemetry = parse_map_file(Path(map_path))
        mid = get_representative_coords(telemetry.points)
        if mid:
            telemetry.mid_lat, telemetry.mid_lon = mid
        if telemetry.points:
            telemetry.start_lat = float(telemetry.points[0]["lat"])
            telemetry.start_lon = float(telemetry.points[0]["lon"])
        trill = calculate_trill_score(telemetry)
        modifier, hashtags = get_trill_modifiers(trill.score, telemetry.max_speed_mph, trill.bucket)
        extra = enrich_route_padus_gazetteer(
            telemetry.points,
            telemetry.mid_lat,
            telemetry.mid_lon,
            gaz_places_path=gaz_places_path,
            padus_path=padus_path,
            padus_layer=padus_layer,
        )
        data: Dict[str, Any] = {
            "trill_score": trill.score,
            "score": trill.score,
            "speed_bucket": trill.bucket,
            "speed_bucket_key": trill.bucket,
            "max_speed_mph": telemetry.max_speed_mph,
            "avg_speed_mph": telemetry.avg_speed_mph,
            "distance_miles": telemetry.total_distance_miles,
            "duration_seconds": telemetry.duration_seconds,
            "title_modifier": modifier,
            "hashtags": hashtags,
            "place_lat": telemetry.mid_lat,
            "place_lon": telemetry.mid_lon,
            "hud_enabled": bool(hud_enabled),
        }
        data.update(extra)
        if extra.get("gazetteer_place_name"):
            data.setdefault("place_name", extra["gazetteer_place_name"])
        return {"ok": True, "data": data}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def ensure_hud_mp4(video_path: str, map_path: str) -> str:
    """Compatibility shim for preview router; full worker uses hud_stage."""
    return video_path
