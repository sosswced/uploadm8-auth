"""
UploadM8 Vision Stage
=====================
Run Google Cloud Vision API multi-feature analysis on one or more sampled frames (merged for captions).
Extracts:
  - Face detection (bounding boxes, emotion likelihoods)
  - OCR text (scoreboards, signs, labels, product names)
  - Scene labels (people, activities, objects)
  - Logo detection (brands / marks when visible)
  - Landmark detection (named places / structures when recognizable)

Outputs stored in ctx.vision_context, used by:
  - thumbnail_stage: face-priority frame crop, face bounding for overlay
  - caption_stage: OCR text injected into prompt (scoreboards → sports recap)
  - m8_engine / fusion digest: labels + logos + landmarks for grounded captions

Cost scales with enabled Vision features (see Google Cloud Vision pricing).
Free tier: 1,000 units/month per feature (varies by feature).
"""

import asyncio
import base64
import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .context import JobContext
from .errors import SkipStage
from .ai_service_costs import user_pref_ai_service_enabled
from .ffmpeg_env import resolve_ffmpeg_executable
from services.provider_error_trace import append_provider_error

logger = logging.getLogger("uploadm8-worker")

try:
    from google.auth.exceptions import GoogleAuthError as _GoogleAuthError

    _GOOGLE_AUTH_ERRS = (_GoogleAuthError,)
except ImportError:  # pragma: no cover
    _GOOGLE_AUTH_ERRS = ()

try:
    from google.api_core import exceptions as _google_api_core_exc

    _GOOGLE_API_CORE_ERRS = (_google_api_core_exc.GoogleAPIError,)
except ImportError:  # pragma: no cover
    _GOOGLE_API_CORE_ERRS = ()

_VISION_RUN_NONFATAL = (
    OSError,
    asyncio.TimeoutError,
    ValueError,
    TypeError,
    RuntimeError,
) + _GOOGLE_AUTH_ERRS + _GOOGLE_API_CORE_ERRS

VISION_STAGE_ENABLED = os.environ.get("VISION_STAGE_ENABLED", "true").lower() == "true"

# Inline JSON (Render: paste service account JSON as a secret env var if file mount path mismatches)
_GCP_JSON_ENVS = ("GCP_SERVICE_ACCOUNT_JSON", "GOOGLE_CREDENTIALS_JSON")

_RENDER_SECRET_NAMES = (
    "gcp-sa.json",
    "gcp-service-account.json",
    "google-credentials.json",
    "service-account.json",
    "credentials.json",
)


def _path_looks_like_gcp_service_account(path: Path) -> bool:
    """True if JSON file is a Google service account key (any filename, e.g. social-media-up-….json)."""
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return (
            isinstance(data, dict)
            and data.get("type") == "service_account"
            and bool(data.get("client_email"))
            and bool(data.get("private_key"))
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def _pick_repo_root_gcp_json() -> Optional[Path]:
    """
    Local dev: if GOOGLE_APPLICATION_CREDENTIALS is unset, use a single
    social-media-up-*.json service-account file in the repo root (e.g. checked-in key name).
    """
    root = Path(__file__).resolve().parents[1]
    matches = sorted(root.glob("social-media-up-*.json"))
    sa = [p for p in matches if _path_looks_like_gcp_service_account(p)]
    if len(sa) == 1:
        return sa[0].resolve()
    if len(sa) > 1:
        logger.warning(
            "[vision] Multiple social-media-up-*.json files in repo root; "
            "set GOOGLE_APPLICATION_CREDENTIALS to pick one"
        )
    return None


def _pick_gcp_json_under_secrets(secrets_dir: Path) -> Optional[Path]:
    """
    Prefer explicit GOOGLE_APPLICATION_CREDENTIALS basename if it exists;
    else the only *.json that parses as a GCP service account;
    else sole *.json file (legacy).
    """
    raw = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "") or "").strip()
    if raw:
        hint = Path(raw).name
        hinted = secrets_dir / hint
        if hinted.is_file() and _path_looks_like_gcp_service_account(hinted):
            logger.info("[vision] Using GCP credentials from /etc/secrets (env basename): %s", hint)
            return hinted.resolve()

    json_files = sorted(secrets_dir.glob("*.json"))
    sa_candidates = [p for p in json_files if _path_looks_like_gcp_service_account(p)]
    if len(sa_candidates) == 1:
        logger.info("[vision] Using GCP service account JSON under /etc/secrets: %s", sa_candidates[0].name)
        return sa_candidates[0].resolve()
    if len(sa_candidates) > 1:
        # Deterministic: prefer env basename match among SA files, else shortest name (often the GCP key)
        if raw:
            bn = Path(raw).name
            for p in sa_candidates:
                if p.name == bn:
                    logger.info("[vision] Using GCP credentials (matched env name): %s", p.name)
                    return p.resolve()
        chosen = sorted(sa_candidates, key=lambda x: (len(x.name), x.name))[0]
        logger.warning(
            "[vision] Multiple service-account JSON files under /etc/secrets; using %s "
            "(set GOOGLE_APPLICATION_CREDENTIALS=/etc/secrets/<file> to pick another)",
            chosen.name,
        )
        return chosen.resolve()

    if len(json_files) == 1:
        if _path_looks_like_gcp_service_account(json_files[0]):
            logger.info("[vision] Using sole JSON secret under /etc/secrets: %s", json_files[0].name)
            return json_files[0].resolve()
        logger.warning(
            "[vision] One JSON file under /etc/secrets but it is not a GCP service_account key: %s",
            json_files[0].name,
        )
        return None
    if len(json_files) > 1:
        logger.warning(
            "[vision] Multiple JSON files under /etc/secrets; none look like a GCP service_account key. "
            "Set GOOGLE_APPLICATION_CREDENTIALS to the full path (e.g. /etc/secrets/social-media-up-….json)."
        )
    return None

_gcv_client      = None
_vision_module   = None
_gcv_executor    = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gcv")

# Set VISION_LOGO_LANDMARK=false to skip LOGO_DETECTION + LANDMARK_DETECTION
# features per frame (saves ~$0.0005/image when brand/landmark data is not
# needed). Default: true — both features run when GCV is enabled.
VISION_LOGO_LANDMARK = (
    os.environ.get("VISION_LOGO_LANDMARK", "true").lower() in ("1", "true", "yes", "on")
)
VISION_WEB_DETECTION = os.environ.get("VISION_WEB_DETECTION", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
VISION_OBJECT_LOCALIZATION = os.environ.get("VISION_OBJECT_LOCALIZATION", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
VISION_IMAGE_PROPERTIES = os.environ.get("VISION_IMAGE_PROPERTIES", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

LIKELIHOOD_MAP = {
    0: "UNKNOWN",
    1: "VERY_UNLIKELY",
    2: "UNLIKELY",
    3: "POSSIBLE",
    4: "LIKELY",
    5: "VERY_LIKELY",
}

POSITIVE_EMOTIONS = {"LIKELY", "VERY_LIKELY"}


def _raw_inline_gcp_json() -> str:
    for key in _GCP_JSON_ENVS:
        raw = (os.environ.get(key) or "").strip()
        if raw:
            return raw
    return ""


def _resolve_gcp_credentials_path() -> Optional[Path]:
    """
    Resolve GOOGLE_APPLICATION_CREDENTIALS robustly:
    - absolute path as-is
    - relative to current working directory
    - relative to repo root (one level above stages/)
    - Render secret files: /etc/secrets/<name> when the env path is missing or wrong
    """
    raw = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "") or "").strip()
    if not raw:
        # Alternate env names (Render / Docker templates often use these)
        for alt in (
            "GCP_SERVICE_ACCOUNT_FILE",
            "GCP_CREDENTIALS_PATH",
            "GOOGLE_SERVICE_ACCOUNT_FILE",
        ):
            v = (os.environ.get(alt) or "").strip()
            if v:
                raw = v
                break
    checked: List[Path] = []

    def _try_candidates(paths: List[Path]) -> Optional[Path]:
        for p in paths:
            if p.exists():
                return p.resolve()
            checked.append(p)
        return None

    if raw:
        p = Path(raw)
        candidates = [p]
        if not p.is_absolute():
            candidates.append(Path.cwd() / p)
            repo_root = Path(__file__).resolve().parents[1]
            candidates.append(repo_root / p)
        hit = _try_candidates(candidates)
        if hit:
            return hit

    if not raw:
        repo_json = _pick_repo_root_gcp_json()
        if repo_json:
            logger.info(
                "[vision] Using GCP credentials from repo root (optional: set "
                "GOOGLE_APPLICATION_CREDENTIALS=%s)",
                repo_json,
            )
            return repo_json

    secrets_dir = Path("/etc/secrets")
    if secrets_dir.is_dir():
        for name in _RENDER_SECRET_NAMES:
            p = secrets_dir / name
            if p.is_file():
                logger.info("[vision] Using GCP credentials file from /etc/secrets: %s", name)
                return p.resolve()
        picked = _pick_gcp_json_under_secrets(secrets_dir)
        if picked:
            return picked

    if raw and checked:
        logger.warning(
            "[vision] GOOGLE_APPLICATION_CREDENTIALS path not found: %s (tried %s)",
            raw,
            ", ".join(str(x) for x in checked[:5]),
        )
    return None


def load_gcp_service_account_credentials() -> Optional[Any]:
    """
    Service account credentials for Vision / Video Intelligence.
    Prefers inline JSON env (GCP_SERVICE_ACCOUNT_JSON or GOOGLE_CREDENTIALS_JSON), else JSON file path.
    """
    raw = _raw_inline_gcp_json()
    if raw:
        try:
            info = json.loads(raw)
            if not isinstance(info, dict) or not info.get("client_email"):
                logger.warning("[vision] Inline GCP JSON missing client_email")
                return None
            from google.oauth2 import service_account

            return service_account.Credentials.from_service_account_info(info)
        except json.JSONDecodeError as e:
            logger.warning("[vision] Inline GCP JSON is not valid JSON: %s", e)
            return None
        except (ImportError, ValueError, TypeError, KeyError, OSError) + _GOOGLE_AUTH_ERRS as e:
            logger.warning("[vision] Inline GCP credentials failed: %s", e)
            return None
    path = _resolve_gcp_credentials_path()
    if not path:
        return None
    try:
        from google.oauth2 import service_account

        return service_account.Credentials.from_service_account_file(str(path))
    except (ImportError, OSError, ValueError, TypeError) + _GOOGLE_AUTH_ERRS as e:
        logger.warning("[vision] GCP credentials file failed: %s", e)
        return None


def gcp_vision_credentials_configured() -> bool:
    return bool(_raw_inline_gcp_json()) or _resolve_gcp_credentials_path() is not None


def _get_gcv_client():
    """Lazy-load GCV client (avoids import errors when GCP not configured)."""
    global _gcv_client, _vision_module
    if _gcv_client is not None:
        return _gcv_client

    try:
        creds_obj = load_gcp_service_account_credentials()
        if not creds_obj:
            logger.warning(
                "[vision] No GCP credentials (set GOOGLE_APPLICATION_CREDENTIALS to a mounted file, "
                "or GCP_SERVICE_ACCOUNT_JSON, or place gcp-sa.json under /etc/secrets)"
            )
            return None
        path = _resolve_gcp_credentials_path()
        if path and not _raw_inline_gcp_json():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)

        from google.cloud import vision as v

        _vision_module = v
        _gcv_client = v.ImageAnnotatorClient(credentials=creds_obj)
        logger.info("[vision] Google Cloud Vision client initialized")
        return _gcv_client
    except (ImportError, OSError, ValueError, TypeError, AttributeError, RuntimeError) + _GOOGLE_AUTH_ERRS as e:
        logger.warning("[vision] GCV client init failed: %s", e)
        return None


def _vision_label_params() -> Tuple[int, float]:
    label_max = int(os.environ.get("VISION_LABEL_MAX_RESULTS", "32") or 32)
    label_max = max(8, min(label_max, 50))
    label_min_score = float(os.environ.get("VISION_LABEL_MIN_SCORE", "0.55") or 0.55)
    label_min_score = max(0.35, min(label_min_score, 0.95))
    return label_max, label_min_score


def _gcv_feature_list(v: Any, label_max: int) -> List[Any]:
    features = [
        v.Feature(type_=v.Feature.Type.FACE_DETECTION, max_results=10),
        v.Feature(type_=v.Feature.Type.TEXT_DETECTION),
        v.Feature(type_=v.Feature.Type.LABEL_DETECTION, max_results=label_max),
    ]
    if VISION_LOGO_LANDMARK:
        features.append(v.Feature(type_=v.Feature.Type.LOGO_DETECTION, max_results=10))
        features.append(v.Feature(type_=v.Feature.Type.LANDMARK_DETECTION, max_results=10))
    if VISION_WEB_DETECTION:
        features.append(v.Feature(type_=v.Feature.Type.WEB_DETECTION, max_results=20))
    if VISION_OBJECT_LOCALIZATION:
        features.append(v.Feature(type_=v.Feature.Type.OBJECT_LOCALIZATION, max_results=20))
    if VISION_IMAGE_PROPERTIES:
        features.append(v.Feature(type_=v.Feature.Type.IMAGE_PROPERTIES))
    return features


def _single_response_to_dict(response: Any, label_min_score: float) -> Dict[str, Any]:
    """Turn one AnnotateImageResponse into our vision_context dict."""
    faces: List[Dict[str, Any]] = []
    for face in response.face_annotations:
        vertices = [(v_pt.x, v_pt.y) for v_pt in face.bounding_poly.vertices]
        joy_score = LIKELIHOOD_MAP.get(face.joy_likelihood, "UNKNOWN")
        surprise = LIKELIHOOD_MAP.get(face.surprise_likelihood, "UNKNOWN")
        sorrow = LIKELIHOOD_MAP.get(face.sorrow_likelihood, "UNKNOWN")
        anger = LIKELIHOOD_MAP.get(face.anger_likelihood, "UNKNOWN")

        expressive_score = 0
        if joy_score in POSITIVE_EMOTIONS:
            expressive_score += 3
        if surprise in POSITIVE_EMOTIONS:
            expressive_score += 3
        if sorrow in POSITIVE_EMOTIONS:
            expressive_score += 1
        if anger in POSITIVE_EMOTIONS:
            expressive_score += 1

        faces.append(
            {
                "confidence": face.detection_confidence,
                "bounding_poly": vertices,
                "center_x": sum(v[0] for v in vertices) // 4,
                "center_y": sum(v[1] for v in vertices) // 4,
                "joy": joy_score,
                "surprise": surprise,
                "sorrow": sorrow,
                "anger": anger,
                "roll_angle": face.roll_angle,
                "pan_angle": face.pan_angle,
                "tilt_angle": face.tilt_angle,
                "expressive_score": expressive_score,
            }
        )

    ocr_text = ""
    if response.text_annotations:
        ocr_text = response.text_annotations[0].description.strip()

    labels = [
        {"description": lbl.description, "score": round(lbl.score, 3)}
        for lbl in response.label_annotations
        if lbl.score >= label_min_score
    ]

    logos: List[Dict[str, Any]] = []
    for lo in response.logo_annotations:
        desc = (getattr(lo, "description", None) or "").strip()
        if not desc:
            continue
        sc = float(getattr(lo, "score", 0.0) or 0.0)
        logos.append({"description": desc, "score": round(sc, 3)})

    landmarks: List[Dict[str, Any]] = []
    for lm in response.landmark_annotations:
        desc = (getattr(lm, "description", None) or "").strip()
        if not desc:
            continue
        sc = float(getattr(lm, "score", 0.0) or 0.0)
        # Pull the landmark's geographic coordinates so downstream stages can
        # reverse-geocode (city/state) or seed geo-based hashtags when no
        # dashcam telemetry exists. Vision usually returns one location per
        # landmark; we keep the first valid lat/lng.
        lat: Optional[float] = None
        lon: Optional[float] = None
        try:
            for loc in (getattr(lm, "locations", None) or []):
                ll = getattr(loc, "lat_lng", None) or getattr(loc, "latLng", None)
                if ll is None:
                    continue
                la = getattr(ll, "latitude", None)
                lo = getattr(ll, "longitude", None)
                if la is None or lo is None:
                    continue
                la_f = float(la)
                lo_f = float(lo)
                if -90.0 <= la_f <= 90.0 and -180.0 <= lo_f <= 180.0 and not (la_f == 0.0 and lo_f == 0.0):
                    lat, lon = la_f, lo_f
                    break
        except (TypeError, ValueError, AttributeError):
            lat = lon = None
        entry: Dict[str, Any] = {"description": desc, "score": round(sc, 3)}
        if lat is not None and lon is not None:
            entry["lat"] = lat
            entry["lon"] = lon
        landmarks.append(entry)

    logo_names = [x["description"] for x in logos if x.get("description")]
    landmark_names = [x["description"] for x in landmarks if x.get("description")]

    web_entities: List[Dict[str, Any]] = []
    web_best_guess: List[str] = []
    wd = getattr(response, "web_detection", None)
    if wd is not None:
        for ent in getattr(wd, "web_entities", None) or []:
            desc = (getattr(ent, "description", None) or "").strip()
            sc = float(getattr(ent, "score", 0.0) or 0.0)
            web_min = float(os.environ.get("VISION_WEB_ENTITY_MIN_SCORE", "0.35") or 0.35)
            if desc and sc >= web_min:
                web_entities.append({"description": desc, "score": round(sc, 3)})
        for lbl in getattr(wd, "best_guess_labels", None) or []:
            g = (getattr(lbl, "label", None) or str(lbl) or "").strip()
            if g:
                web_best_guess.append(g)

    localized_objects: List[Dict[str, Any]] = []
    for obj in getattr(response, "localized_object_annotations", None) or []:
        name = (getattr(obj, "name", None) or "").strip()
        sc = float(getattr(obj, "score", 0.0) or 0.0)
        obj_min = float(os.environ.get("VISION_LOCALIZED_OBJECT_MIN_SCORE", "0.45") or 0.45)
        if name and sc >= obj_min:
            localized_objects.append({"name": name, "score": round(sc, 3)})

    dominant_colors: List[Dict[str, Any]] = []
    ipa = getattr(response, "image_properties_annotation", None)
    if ipa is not None:
        dc = getattr(ipa, "dominant_colors", None)
        for col in getattr(dc, "colors", None) or []:
            try:
                color = getattr(col, "color", None)
                if color is None:
                    continue
                r = int(getattr(color, "red", 0) or 0)
                g = int(getattr(color, "green", 0) or 0)
                b = int(getattr(color, "blue", 0) or 0)
                frac = float(getattr(col, "pixel_fraction", 0) or 0)
                if frac < 0.04:
                    continue
                from services.google_visual_recognition import _rgb_color_name

                dominant_colors.append(
                    {
                        "name": _rgb_color_name(r, g, b),
                        "rgb": [r, g, b],
                        "score": round(frac, 3),
                    }
                )
            except (TypeError, ValueError, AttributeError):
                continue

    return {
        "faces": faces,
        "face_count": len(faces),
        "has_faces": len(faces) > 0,
        "best_face": max(faces, key=lambda f: f["expressive_score"]) if faces else None,
        "expressive": any(f["expressive_score"] >= 3 for f in faces),
        "ocr_text": ocr_text,
        "labels": labels,
        "label_names": [lbl["description"] for lbl in labels],
        "logos": logos,
        "landmarks": landmarks,
        "logo_names": logo_names,
        "landmark_names": landmark_names,
        "web_entities": web_entities,
        "web_best_guess": web_best_guess,
        "localized_objects": localized_objects,
        "dominant_colors": dominant_colors[:8],
    }


def _merge_vision_dicts(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multi-frame Vision results: union labels/logos/landmarks, best face bucket, OCR stitched."""
    if not parts:
        return {}
    if len(parts) == 1:
        return dict(parts[0])

    def _merge_scored(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            desc = (row.get("description") or "").strip()
            if not desc:
                continue
            key = desc.lower()
            sc = float(row.get("score") or 0.0)
            cur = best.get(key)
            if cur is None or sc > float(cur.get("score") or 0.0):
                entry: Dict[str, Any] = {"description": desc, "score": round(sc, 3)}
                if "lat" in row and "lon" in row:
                    entry["lat"] = row["lat"]
                    entry["lon"] = row["lon"]
                best[key] = entry
        out = list(best.values())
        out.sort(key=lambda x: -float(x.get("score") or 0.0))
        return out

    all_labels: List[Dict[str, Any]] = []
    all_logos: List[Dict[str, Any]] = []
    all_landmarks: List[Dict[str, Any]] = []
    all_web: List[Dict[str, Any]] = []
    all_localized: List[Dict[str, Any]] = []
    all_colors: List[Dict[str, Any]] = []
    web_guess_seen: set[str] = set()
    web_best_guess: List[str] = []
    for p in parts:
        all_labels.extend(p.get("labels") or [])
        all_logos.extend(p.get("logos") or [])
        all_landmarks.extend(p.get("landmarks") or [])
        all_web.extend(p.get("web_entities") or [])
        all_localized.extend(p.get("localized_objects") or [])
        all_colors.extend(p.get("dominant_colors") or [])
        for g in p.get("web_best_guess") or []:
            gs = str(g).strip()
            if gs and gs.lower() not in web_guess_seen:
                web_guess_seen.add(gs.lower())
                web_best_guess.append(gs)

    labels = _merge_scored(all_labels)
    logos = _merge_scored(all_logos)
    landmarks = _merge_scored(all_landmarks)
    web_entities = _merge_scored(all_web)
    localized_objects = _merge_scored(
        [{"description": x.get("name"), "score": x.get("score")} for x in all_localized if x.get("name")]
    )

    best_faces = max(
        parts,
        key=lambda d: (
            int(d.get("face_count") or 0),
            sum(int(f.get("expressive_score") or 0) for f in (d.get("faces") or [])),
        ),
    )
    faces = list(best_faces.get("faces") or [])

    ocr_chunks: List[str] = []
    seen: set[str] = set()
    for p in parts:
        o = (p.get("ocr_text") or "").strip()
        if len(o) < 4 or o in seen:
            continue
        seen.add(o)
        ocr_chunks.append(o)
    ocr_text = "\n---\n".join(ocr_chunks)[:8000]

    logo_names = [x["description"] for x in logos if x.get("description")]
    landmark_names = [x["description"] for x in landmarks if x.get("description")]

    return {
        "faces": faces,
        "face_count": len(faces),
        "has_faces": len(faces) > 0,
        "best_face": max(faces, key=lambda f: f["expressive_score"]) if faces else None,
        "expressive": any(f.get("expressive_score", 0) >= 3 for f in faces),
        "ocr_text": ocr_text,
        "labels": labels,
        "label_names": [lbl["description"] for lbl in labels],
        "logos": logos,
        "landmarks": landmarks,
        "logo_names": logo_names,
        "landmark_names": landmark_names,
        "web_entities": web_entities,
        "web_best_guess": web_best_guess[:12],
        "localized_objects": [
            {"name": x.get("description"), "score": x.get("score")}
            for x in localized_objects
            if x.get("description")
        ],
        "dominant_colors": sorted(
            all_colors,
            key=lambda x: -float(x.get("score") or 0),
        )[:8],
        "vision_multi_frame": len(parts),
    }


def _analyze_sync(image_bytes: bytes) -> Dict[str, Any]:
    """
    Run GCV on a single image (sync, executor thread).
    """
    client = _get_gcv_client()  # sets _vision_module as side effect on first call
    v = _vision_module
    if not client or not v:
        return {}
    label_max, label_min_score = _vision_label_params()
    features = _gcv_feature_list(v, label_max)
    image = v.Image(content=image_bytes)
    request = v.AnnotateImageRequest(image=image, features=features)
    response = client.annotate_image(request=request)
    return _single_response_to_dict(response, label_min_score)


def _analyze_batch_sync(image_bytes_list: List[bytes]) -> Dict[str, Any]:
    """Run GCV batch_annotate_images when multiple JPEGs; merge into one dict."""
    if not image_bytes_list:
        return {}
    if len(image_bytes_list) == 1:
        return _analyze_sync(image_bytes_list[0])

    client = _get_gcv_client()  # sets _vision_module as side effect on first call
    v = _vision_module
    if not client or not v:
        return {}

    label_max, label_min_score = _vision_label_params()
    features = _gcv_feature_list(v, label_max)
    requests: List[Any] = []
    for raw in image_bytes_list:
        img = v.Image(content=raw)
        requests.append(v.AnnotateImageRequest(image=img, features=features))

    try:
        batch = client.batch_annotate_images(requests=requests)
    except Exception as e:
        logger.warning("[vision] batch_annotate_images failed (%s); using first frame only", e)
        return _analyze_sync(image_bytes_list[0])

    dicts: List[Dict[str, Any]] = []
    for resp in batch.responses:
        err = getattr(resp, "error", None)
        if err and getattr(err, "message", None):
            logger.debug("[vision] batch item skipped: %s", err.message)
            continue
        dicts.append(_single_response_to_dict(resp, label_min_score))

    if not dicts:
        return _analyze_sync(image_bytes_list[0])
    if len(dicts) == 1:
        return dicts[0]
    return _merge_vision_dicts(dicts)


def _parse_frame_offset_fractions(n_frames: int) -> List[float]:
    """Fractions along clip duration (0–1) for JPEG extraction; env VISION_FRAME_OFFSETS overrides."""
    n_frames = max(1, min(int(n_frames or 1), 12))
    raw = (os.environ.get("VISION_FRAME_OFFSETS") or "").strip()
    out: List[float] = []
    if raw:
        for x in raw.split(","):
            try:
                out.append(float(x.strip()))
            except ValueError:
                pass
        if out:
            return out[:n_frames]
    # Generate evenly distributed offsets between 8% and 92% so we never
    # sample the leading/trailing black frames intros/outros tend to have.
    if n_frames == 1:
        return [0.5]
    step = (0.92 - 0.08) / (n_frames - 1)
    return [round(0.08 + i * step, 4) for i in range(n_frames)]


def _adaptive_vision_frame_count(
    duration_seconds: float,
    *,
    user_override: Optional[int] = None,
) -> int:
    """Pick how many frames to send to GCV based on clip length.

    Old default was a hard 3 cap, ~5% of a 60s clip. New scaling:

      <  10s →  2 frames
      < 30s  →  4 frames
      < 90s  →  6 frames
      < 240s →  8 frames
      ≥240s  → 10 frames

    Caps:
      - User can hard-pin via ``VISION_MULTI_FRAME`` (legacy env) or the
        ``thumbnailVisionMultiFrame`` user setting.
      - Absolute ceiling (env ``VISION_MULTI_FRAME_MAX``, default 10) keeps
        per-upload spend predictable.
    """
    raw_env = (os.environ.get("VISION_MULTI_FRAME") or "").strip()
    if user_override is not None:
        try:
            n = int(user_override)
        except (TypeError, ValueError):
            n = 0
        if n > 0:
            return max(1, min(n, _vision_frame_ceiling()))
    # Legacy hard pin — only honor when explicitly set non-empty.
    if raw_env:
        try:
            n = int(raw_env)
            if n > 0:
                return max(1, min(n, _vision_frame_ceiling()))
        except ValueError:
            pass
    d = max(0.0, float(duration_seconds or 0.0))
    if d < 10:
        n = 2
    elif d < 30:
        n = 4
    elif d < 90:
        n = 6
    elif d < 240:
        n = 8
    else:
        n = 10
    return max(1, min(n, _vision_frame_ceiling()))


def _vision_frame_ceiling() -> int:
    raw = (os.environ.get("VISION_MULTI_FRAME_MAX") or "10").strip()
    try:
        return max(1, min(int(raw), 12))
    except ValueError:
        return 10


def _vi_shot_offset_fractions(
    ctx: JobContext, duration_seconds: float, target_count: int
) -> Optional[List[float]]:
    """Use Video Intelligence shot boundaries to bias sampling toward
    *content* moments — middle of distinct shots — instead of evenly spaced
    frames that risk landing on transitions.
    """
    vi = getattr(ctx, "video_intelligence_context", None) or getattr(
        ctx, "video_intelligence", None
    )
    if not isinstance(vi, dict):
        return None
    shots = vi.get("shots") or []
    if not isinstance(shots, list) or len(shots) < 2:
        return None
    if not duration_seconds or duration_seconds <= 0:
        return None
    midpoints: List[float] = []
    for s in shots:
        if not isinstance(s, dict):
            continue
        try:
            start = float(s.get("start_s") or s.get("start_seconds") or 0.0)
            end = float(s.get("end_s") or s.get("end_seconds") or 0.0)
        except (TypeError, ValueError):
            continue
        if end > start:
            midpoints.append((start + end) / 2.0)
    if not midpoints:
        return None
    # Pick target_count midpoints: longest shots first (more content there)
    shots_sorted = sorted(
        (
            (s, float(s.get("end_s", 0) or 0) - float(s.get("start_s", 0) or 0))
            for s in shots
            if isinstance(s, dict)
        ),
        key=lambda t: -t[1],
    )
    chosen = []
    for s, _len in shots_sorted[:target_count]:
        try:
            mid = (
                float(s.get("start_s") or 0.0) + float(s.get("end_s") or 0.0)
            ) / 2.0
            chosen.append(round(mid / duration_seconds, 4))
        except (TypeError, ValueError):
            continue
    chosen = sorted(set(0.05 if c < 0.05 else (0.95 if c > 0.95 else c) for c in chosen))
    if len(chosen) < 2:
        return None
    return chosen[:target_count]


async def _extract_frames_at_offsets(
    video_path: Path,
    temp_dir: Path,
    offsets: List[float],
) -> List[Tuple[float, Path]]:
    """Extract one JPEG per fractional offset; returns (fraction_used, path) per success."""
    FFMPEG_PATH = resolve_ffmpeg_executable() or "ffmpeg"
    FFPROBE_PATH = resolve_ffmpeg_executable("ffprobe") or "ffprobe"
    duration = 1.0
    proc = await asyncio.create_subprocess_exec(
        FFPROBE_PATH,
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode == 0 and stdout:
        try:
            data = json.loads(stdout.decode("utf-8", errors="replace"))
            d = data.get("format", {}).get("duration", 1)
            duration = float(d) if d else 1.0
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    duration = max(duration, 0.25)

    out_pairs: List[Tuple[float, Path]] = []
    for i, frac in enumerate(offsets):
        try:
            f = float(frac)
        except (TypeError, ValueError):
            f = 0.2 * (i + 1)
        f = max(0.02, min(f, 0.98))
        offset_s = max(0.15, duration * f)
        out_path = temp_dir / f"vision_mf_{i:02d}.jpg"
        cmd = [
            FFMPEG_PATH,
            "-y",
            "-ss",
            f"{offset_s:.3f}",
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-q:v",
            "2",
            "-vf",
            "scale=1280:-1",
            str(out_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()
        if out_path.exists() and out_path.stat().st_size > 1000:
            out_pairs.append((f, out_path))
    return out_pairs


async def run_vision_stage(ctx: JobContext) -> JobContext:
    """
    Analyze video with Google Cloud Vision (single key frame, or multi-frame batch).

    Multi-frame (default ``VISION_MULTI_FRAME=3``): FFmpeg samples JPEGs at
    ``VISION_FRAME_OFFSETS`` (or built-in spread), then ``batch_annotate_images``
    merges labels / OCR / logos / landmarks for richer captions.

    Env:
      - ``VISION_MULTI_FRAME`` — 1–4 (1 = legacy single-frame path).
      - ``VISION_FRAME_OFFSETS`` — comma fractions, e.g. ``0.15,0.42,0.70``.
    """
    ctx.mark_stage("vision")

    if not VISION_STAGE_ENABLED:
        raise SkipStage("Vision stage disabled via env")

    if not gcp_vision_credentials_configured():
        raise SkipStage(
            "GCP credentials not configured (set GOOGLE_APPLICATION_CREDENTIALS, "
            "GCP_SERVICE_ACCOUNT_JSON, a file under /etc/secrets, or one social-media-up-*.json in repo root)"
        )

    tier_allowed = getattr(ctx.entitlements, "allowed_ai_services", None) if ctx.entitlements else None
    tier_allowed_set = set(tier_allowed) if tier_allowed is not None else None
    if not user_pref_ai_service_enabled(
        ctx.user_settings or {},
        "vision_google",
        default=True,
        allowed_services=tier_allowed_set,
    ):
        raise SkipStage("Vision disabled in upload preferences (aiServiceFrameInspector)")

    try:
        video_path: Optional[Path] = None
        for c in (ctx.processed_video_path, ctx.local_video_path):
            if c and Path(c).exists():
                video_path = Path(c)
                break

        # Duration-aware adaptive sampling. Falls back to legacy 3-frame cap
        # only when VISION_MULTI_FRAME env or user setting forces it.
        duration = float((getattr(ctx, "video_info", None) or {}).get("duration") or 0.0)
        us = ctx.user_settings or {}
        user_override = us.get("thumbnail_vision_multi_frame") or us.get(
            "thumbnailVisionMultiFrame"
        )
        multi = _adaptive_vision_frame_count(duration, user_override=user_override)
        logger.info(
            "[vision] adaptive sampling: duration=%.1fs frames=%d ceiling=%d",
            duration,
            multi,
            _vision_frame_ceiling(),
        )

        loop = asyncio.get_running_loop()
        result: Dict[str, Any] = {}

        if multi > 1 and video_path and ctx.temp_dir:
            # Prefer VI shot midpoints when available (skips transitions);
            # falls back to evenly distributed offsets otherwise.
            offsets = _vi_shot_offset_fractions(ctx, duration, multi) or _parse_frame_offset_fractions(multi)
            frame_pairs = await _extract_frames_at_offsets(video_path, Path(ctx.temp_dir), offsets)
            frames_m = [p for _, p in frame_pairs]
            fracs_used = [f for f, _ in frame_pairs]
            if len(frames_m) >= 2:
                blobs = [p.read_bytes() for p in frames_m]
                result = await loop.run_in_executor(
                    _gcv_executor,
                    partial(_analyze_batch_sync, blobs),
                )
                if isinstance(result, dict) and result:
                    result = dict(result)
                    result["vision_multi_frame_paths"] = [str(p) for p in frames_m]
                    result["vision_sample_fractions"] = fracs_used

        if not result:
            frame_to_analyze = _find_best_frame(ctx)
            if not frame_to_analyze or not frame_to_analyze.exists():
                frame_to_analyze = await _extract_frame_for_vision(ctx)
            if not frame_to_analyze or not frame_to_analyze.exists():
                raise SkipStage("No frame available for vision analysis")
            image_bytes = frame_to_analyze.read_bytes()
            result = await loop.run_in_executor(
                _gcv_executor,
                partial(_analyze_sync, image_bytes),
            )

        ctx.vision_context = result
        try:
            from services.google_visual_recognition import attach_visual_recognition

            attach_visual_recognition(ctx)
        except Exception as _vr_e:
            logger.debug("[vision] visual recognition rollup skipped: %s", _vr_e)

        mf = result.get("vision_multi_frame") if isinstance(result, dict) else None
        logger.info(
            "[vision] multi=%s faces=%s expressive=%s ocr_chars=%s labels=%s logos=%s landmarks=%s web=%s",
            mf or 1,
            result.get("face_count", 0),
            result.get("expressive", False),
            len(result.get("ocr_text", "") or ""),
            (result.get("label_names") or [])[:4],
            (result.get("logo_names") or [])[:5],
            (result.get("landmark_names") or [])[:5],
            len(result.get("web_entities") or []),
        )

        return ctx

    except asyncio.CancelledError:
        raise
    except SkipStage:
        raise
    except _VISION_RUN_NONFATAL as e:
        logger.warning("[vision] Non-fatal error: %s", e)
        append_provider_error(
            ctx,
            provider="google_vision",
            stage="vision_stage",
            operation="batch_or_single_annotate",
            message=str(e),
            exception_type=type(e).__name__,
        )
        ctx.vision_context = {}
        return ctx


def _find_best_frame(ctx: JobContext) -> Optional[Path]:
    """Find the best frame already extracted in temp_dir."""
    if not ctx.temp_dir:
        return None

    temp_dir = Path(ctx.temp_dir)

    # Prefer thumbnail frame (highest quality extraction)
    for pattern in ["thumbnail_final.jpg", "cand_*.jpg", "frame_*.jpg", "cap_frame_*.jpg", "thumb_*.jpg"]:
        matches = sorted(temp_dir.glob(pattern))
        if matches:
            return matches[0]

    return None


async def _extract_frame_for_vision(ctx: JobContext) -> Optional[Path]:
    """Extract one frame from video when none exist (for pipeline order: vision before thumbnail)."""
    video_path = None
    for c in (ctx.processed_video_path, ctx.local_video_path):
        if c and Path(c).exists():
            video_path = Path(c)
            break
    if not video_path or not ctx.temp_dir:
        return None

    out_path = Path(ctx.temp_dir) / "cand_000.jpg"
    FFMPEG_PATH = resolve_ffmpeg_executable() or "ffmpeg"
    FFPROBE_PATH = resolve_ffmpeg_executable("ffprobe") or "ffprobe"
    try:
        # Use ffprobe to get duration, extract at 30%
        import json
        proc = await asyncio.create_subprocess_exec(
            FFPROBE_PATH,
            "-v", "quiet", "-show_entries", "format=duration", "-of", "json",
            str(video_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        duration = 1.0
        if proc.returncode == 0 and stdout:
            try:
                data = json.loads(stdout.decode("utf-8", errors="replace"))
                d = data.get("format", {}).get("duration", 1)
                duration = float(d) if d else 1.0
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        offset = max(0.5, duration * 0.30)

        cmd = [FFMPEG_PATH, "-y", "-ss", f"{offset:.2f}", "-i", str(video_path),
               "-vframes", "1", "-q:v", "2", "-vf", "scale=1280:-1", str(out_path)]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        return out_path if out_path.exists() and out_path.stat().st_size > 1000 else None
    except asyncio.CancelledError:
        raise
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("[vision] Frame extraction failed: %s", e)
        return None


