"""
Canonical hydration snapshot shared by thumbnail styling, Pikzels, M8/captions, and artifacts.

Built once after multimodal stages, before thumbnails. Stored on ``JobContext.hydration_payload``
and ``output_artifacts[\"hydration_payload\"]``.

Env ``UPLOADM8_HYDRATION_PAYLOAD``: default ON when unset; set to ``0``, ``false``,
``no``, ``off``, ``disable``, ``disabled``, or ``2`` to disable.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from core.helpers import sanitize_hashtag_body
from stages.context import build_fusion_summary_text, build_hydration_story_text

logger = logging.getLogger("uploadm8-worker.hydration_payload")

_SNAPSHOT_CAP = 490_000
HYDRATION_PAYLOAD_VERSION = 2

# Required top-level keys and their expected Python types.
_REQUIRED_KEYS: List[Tuple[str, type]] = [
    ("category", str),
    ("anchor_phrase", str),
    ("evidence", dict),
    ("signal_hashtags", list),
    ("fusion_summary", str),
    ("hydration_story", str),
    ("trace_id", str),
]


def validate_hydration_payload(hp: Any) -> Tuple[bool, str]:
    """Return (ok, reason). Used by consumers to detect stale/corrupt blobs."""
    if not isinstance(hp, dict):
        return False, "not a dict"
    v = hp.get("v")
    if v is None or int(v) < HYDRATION_PAYLOAD_VERSION:
        return False, f"stale version: {v!r} (need {HYDRATION_PAYLOAD_VERSION})"
    for key, expected_type in _REQUIRED_KEYS:
        if key not in hp:
            return False, f"missing key: {key!r}"
        val = hp[key]
        if not isinstance(val, expected_type):
            return False, f"wrong type for {key!r}: got {type(val).__name__}"
    ev = hp["evidence"]
    for lane in ("geo", "osd", "music", "speech", "vision", "trill"):
        if lane in ev and not isinstance(ev[lane], dict):
            return False, f"evidence.{lane} must be dict"
    return True, ""


def _minimal_hydration_payload(*, category: str, trace_id: str) -> Dict[str, Any]:
    """Safe fallback: consistent shape, all evidence empty."""
    return {
        "v": HYDRATION_PAYLOAD_VERSION,
        "category": str(category or "general").strip().lower() or "general",
        "anchor_phrase": "",
        "evidence": {
            "geo": {},
            "osd": {},
            "music": {},
            "speech": {},
            "vision": {"labels": [], "ocr": "", "landmarks": [], "logos": []},
            "trill": {},
        },
        "signal_hashtags": [],
        "fusion_summary": "",
        "hydration_story": "",
        "trace_id": str(trace_id or ""),
        "category_source": "minimal_fallback",
    }


def persist_hydration_payload_artifact(ctx: Any) -> None:
    hp = getattr(ctx, "hydration_payload", None)
    if not isinstance(hp, dict):
        return
    arts = getattr(ctx, "output_artifacts", None)
    if not isinstance(arts, dict):
        return
    try:
        arts["hydration_payload"] = json.dumps(hp, default=str)[:_SNAPSHOT_CAP]
    except (TypeError, ValueError):
        return

    # Self-persist: hydration_payload is the spine of every downstream diagnostic
    # (thumbnail brief, M8 prompts, admin trace). Without writing it to DB
    # immediately it depends on save_generated_metadata at the end of caption_stage,
    # which silently drops everything when caption_stage fails or skips.
    try:
        from services.diag_persist import schedule_persist_artifact_now

        schedule_persist_artifact_now(ctx, "hydration_payload")
    except Exception:
        pass


def build_hydration_payload(
    ctx: Any,
    *,
    category: str,
    category_source: str = "stage",
) -> Dict[str, Any]:
    """Assemble JSON-serializable payload from current ctx (no GPT)."""
    from services.hydration_enforcer import build_anchor_phrase, collect_evidence
    from services.signal_hashtags import build_signal_hashtags

    pool = collect_evidence(ctx)
    anchor = build_anchor_phrase(pool, ctx) or ""
    tel = ctx.telemetry or ctx.telemetry_data
    mid_lat = getattr(tel, "mid_lat", None) if tel else None
    mid_lon = getattr(tel, "mid_lon", None) if tel else None

    geo: Dict[str, Any] = {
        "road": pool.road or (getattr(tel, "location_road", None) if tel else None),
        "city": pool.city or (getattr(tel, "location_city", None) if tel else None),
        "state": pool.state_abbr or pool.state or (getattr(tel, "location_state", None) if tel else None),
        "lat": mid_lat,
        "lon": mid_lon,
        "display": getattr(tel, "location_display", None) if tel else None,
        "country": pool.country or (getattr(tel, "location_country", None) if tel else None),
    }

    osd_ctx = ctx.dashcam_osd_context or {}
    fs = osd_ctx.get("first_seen") if isinstance(osd_ctx, dict) else {}
    first_seen = ""
    if isinstance(fs, dict) and (fs.get("date") or fs.get("time")):
        first_seen = f"{str(fs.get('date') or '').strip()} {str(fs.get('time') or '').strip()}".strip()
    max_osd: Optional[float] = None
    if isinstance(osd_ctx, dict) and osd_ctx.get("max_speed_mph") is not None:
        try:
            max_osd = float(osd_ctx.get("max_speed_mph"))
        except (TypeError, ValueError):
            max_osd = None
    osd: Dict[str, Any] = {
        "max_speed_mph": max_osd,
        "driver_name": pool.driver_name or (osd_ctx.get("driver_name") if isinstance(osd_ctx, dict) else None),
        "first_seen": first_seen or None,
        "avg_speed_mph": osd_ctx.get("avg_speed_mph") if isinstance(osd_ctx, dict) else None,
    }

    ac = ctx.audio_context or {}
    music: Dict[str, Any] = {
        "artist": pool.music_artist or ac.get("music_artist"),
        "title": pool.music_title or ac.get("music_title"),
        "genre": pool.music_genre or ac.get("music_genre"),
    }

    speech_phrase = pool.transcript_phrase
    if not speech_phrase:
        t = (ctx.ai_transcript or ac.get("transcript") or "").strip()
        speech_phrase = t[:400] if t else None
    speech = {"phrase": speech_phrase}

    vc = ctx.vision_context or {}
    labels = list(vc.get("label_names") or [])[:40]
    landmarks = [str(x) for x in (vc.get("landmark_names") or [])[:12]]
    vision: Dict[str, Any] = {
        "labels": labels,
        "ocr": ((vc.get("ocr_text") or "")[:4000]).strip(),
        "landmarks": landmarks,
        "logos": [str(x) for x in (vc.get("logo_names") or [])[:12]],
    }

    tr = ctx.trill or ctx.trill_score
    tr_score: Optional[float] = None
    if pool.trill_score and pool.trill_score > 0:
        tr_score = float(pool.trill_score)
    elif tr is not None and getattr(tr, "score", None) is not None:
        try:
            tr_score = float(getattr(tr, "score"))
        except (TypeError, ValueError):
            tr_score = None
    trill: Dict[str, Any] = {
        "bucket": pool.trill_bucket or (getattr(tr, "bucket", None) if tr else None),
        "score": tr_score,
    }

    fusion = (build_fusion_summary_text(ctx) or "")[:4000]
    story = (build_hydration_story_text(ctx) or "")[:1200]

    signals: List[str] = build_signal_hashtags(ctx, max_extra=12)

    def _omit_empty(d: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in d.items():
            if v is None or v == "":
                continue
            if isinstance(v, dict) and not v:
                continue
            if isinstance(v, list) and not v:
                continue
            out[k] = v
        return out

    return {
        "v": HYDRATION_PAYLOAD_VERSION,
        "category": str(category or "general").strip().lower() or "general",
        "category_source": category_source,
        "anchor_phrase": anchor,
        "evidence": {
            "geo": _omit_empty(geo),
            "osd": _omit_empty(osd),
            "music": _omit_empty(music),
            "speech": _omit_empty(speech),
            "vision": vision,
            "trill": _omit_empty(trill),
        },
        "signal_hashtags": signals,
        "fusion_summary": fusion,
        "hydration_story": story,
        "trace_id": str(getattr(ctx, "upload_id", "") or ""),
    }


def hydration_brief_strings(hp: Dict[str, Any]) -> Dict[str, str]:
    """Strings aligned with THUMBNAIL_BRIEF_PROMPT fields."""
    out: Dict[str, str] = {}
    if str(hp.get("hydration_story") or "").strip():
        out["hydration_story"] = str(hp["hydration_story"])[:700]
    if str(hp.get("fusion_summary") or "").strip():
        out["fusion_summary"] = str(hp["fusion_summary"])[:4000]

    ev = hp.get("evidence") if isinstance(hp.get("evidence"), dict) else {}
    geo = ev.get("geo") if isinstance(ev.get("geo"), dict) else {}
    geo_bits: List[str] = []
    rd = str(geo.get("road") or "").strip()
    if rd:
        geo_bits.append(f"road: {rd}")
    city = str(geo.get("city") or "").strip()
    st = str(geo.get("state") or "").strip()
    if city and st:
        geo_bits.append(f"{city}, {st}")
    elif city:
        geo_bits.append(city)
    disp = str(geo.get("display") or "").strip()
    if disp and rd:
        geo_bits.append(f"display: {disp[:120]}")
    elif disp and not rd:
        geo_bits.append(disp[:200])
    lat, lon = geo.get("lat"), geo.get("lon")
    if lat is not None and lon is not None:
        try:
            geo_bits.append(f"GPS: {float(lat):.5f}, {float(lon):.5f}")
        except (TypeError, ValueError):
            pass
    if geo_bits:
        out["geo_context"] = "; ".join(geo_bits)[:500]

    osd = ev.get("osd") if isinstance(ev.get("osd"), dict) else {}
    osd_bits: List[str] = []
    msm = osd.get("max_speed_mph")
    if msm is not None:
        try:
            osd_bits.append(f"peak speed: {float(msm):.1f} mph")
        except (TypeError, ValueError):
            osd_bits.append(f"peak speed: {msm}")
    dn = str(osd.get("driver_name") or "").strip()
    if dn:
        osd_bits.append(f"driver/HUD: {dn}")
    fss = str(osd.get("first_seen") or "").strip()
    if fss:
        osd_bits.append(f"HUD/recording start: {fss}")
    if osd_bits:
        out["osd_context"] = "; ".join(osd_bits)[:500]

    tri = ev.get("trill") if isinstance(ev.get("trill"), dict) else {}
    tb = str(tri.get("bucket") or "").strip()
    tsc = tri.get("score")
    if tb or tsc is not None:
        if tsc is not None:
            try:
                tsf = float(tsc)
                out["trill_context"] = (
                    f"Trill bucket {tb} score {tsf:.0f}" if tb else f"Trill score {tsf:.0f}"
                )[:260]
            except (TypeError, ValueError):
                if tb:
                    out["trill_context"] = tb[:260]
        elif tb:
            out["trill_context"] = f"Trill bucket {tb}"[:260]

    mus = ev.get("music") if isinstance(ev.get("music"), dict) else {}
    ma = str(mus.get("artist") or "").strip()
    mt = str(mus.get("title") or "").strip()
    if ma or mt:
        out["music_context"] = " — ".join(p for p in (ma, mt) if p)[:260]

    sp = ev.get("speech") if isinstance(ev.get("speech"), dict) else {}
    ph = str(sp.get("phrase") or "").strip()
    if ph:
        out["speech_context"] = ph[:420]

    sigs = hp.get("signal_hashtags")
    if isinstance(sigs, list) and sigs:
        clean = [sanitize_hashtag_body(str(x)) for x in sigs if sanitize_hashtag_body(str(x))]
        if clean:
            out["signal_hashtags"] = ", ".join(clean[:14])

    return out


def apply_hydration_payload_to_thumbnail_brief(ctx: Any, brief: Dict[str, Any]) -> Dict[str, Any]:
    """Hydrate thumbnail brief from ``ctx.hydration_payload`` (setdefault — GPT wins if set)."""
    hp = getattr(ctx, "hydration_payload", None)
    if not isinstance(hp, dict):
        return brief
    ok, reason = validate_hydration_payload(hp)
    if not ok:
        logger.debug("[hydration_payload] stale/invalid payload skipped for brief: %s", reason)
        return brief
    if not isinstance(brief, dict):
        brief = {}

    strings = hydration_brief_strings(hp)
    anchor = str(hp.get("anchor_phrase") or "").strip()
    if anchor:
        brief.setdefault("notes", anchor[:200])

    for key in (
        "geo_context",
        "osd_context",
        "trill_context",
        "music_context",
        "speech_context",
        "fusion_summary",
        "hydration_story",
        "signal_hashtags",
    ):
        val = strings.get(key)
        if val:
            brief.setdefault(key, val)

    return brief


def merge_m8_must_use_tokens(base: List[str], ctx: Optional[Any], *, max_tokens: int = 12) -> List[str]:
    """Extend M8 ``must_use`` shortlist with canonical ``ctx.hydration_payload`` evidence (deduped)."""
    if ctx is None:
        return list(base)[:max_tokens]
    hp = getattr(ctx, "hydration_payload", None)
    if not isinstance(hp, dict):
        return list(base)[:max_tokens]

    seen = {str(x).strip().lower() for x in base if str(x).strip()}
    out: List[str] = list(base)

    def _push(token: str) -> None:
        s = str(token or "").strip()
        if not s:
            return
        key = s.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(s)

    ev = hp.get("evidence") if isinstance(hp.get("evidence"), dict) else {}
    geo = ev.get("geo") if isinstance(ev.get("geo"), dict) else {}
    osd = ev.get("osd") if isinstance(ev.get("osd"), dict) else {}

    max_mph = osd.get("max_speed_mph")
    if max_mph is not None:
        try:
            _push(f"{int(round(float(max_mph)))} MPH")
        except (TypeError, ValueError):
            pass

    road = str(geo.get("road") or "").strip()
    if road:
        _push(road)
    city = str(geo.get("city") or "").strip()
    st = str(geo.get("state") or "").strip()
    if city and st:
        _push(f"{city}, {st}")
    elif city:
        _push(city)
    disp = str(geo.get("display") or "").strip()
    if disp and len(disp) <= 100:
        _push(disp)

    mus = ev.get("music") if isinstance(ev.get("music"), dict) else {}
    ma = str(mus.get("artist") or "").strip()
    mt = str(mus.get("title") or "").strip()
    if ma and mt:
        _push(f"{ma} — {mt}")
    elif ma:
        _push(ma)
    elif mt:
        _push(mt)

    tri = ev.get("trill") if isinstance(ev.get("trill"), dict) else {}
    tb = str(tri.get("bucket") or "").strip()
    if tb:
        _push(f"Trill {tb}")

    dn = str(osd.get("driver_name") or "").strip()
    if dn:
        _push(dn)

    vis = ev.get("vision") if isinstance(ev.get("vision"), dict) else {}
    for lm in (vis.get("landmarks") or [])[:2]:
        _push(str(lm))
    for lb in (vis.get("labels") or [])[:3]:
        s = str(lb).strip()
        if len(s) > 2:
            _push(s)

    sp = ev.get("speech") if isinstance(ev.get("speech"), dict) else {}
    phrase = str(sp.get("phrase") or "").strip()
    if phrase:
        _push(phrase[:90] + ("…" if len(phrase) > 90 else ""))

    anch = str(hp.get("anchor_phrase") or "").strip()
    if anch:
        _push(anch[:100] + ("…" if len(anch) > 100 else ""))

    return out[:max_tokens]


def m8_hydration_contract_block(ctx: Any, *, generate_hashtags: bool = True) -> str:
    """
    Hard grounding block for M8: anchor story + hashtag slugs from ``hydration_payload``.
    Empty when no payload or payload is stale/invalid.
    """
    hp = getattr(ctx, "hydration_payload", None)
    if not isinstance(hp, dict):
        return ""
    ok, reason = validate_hydration_payload(hp)
    if not ok:
        logger.debug("[m8_hydration_contract_block] skipping stale payload: %s", reason)
        return ""

    lines: List[str] = []
    anch = str(hp.get("anchor_phrase") or "").strip()
    if anch:
        lines.append(
            "ANCHOR (canonical factual line — every winning YouTube title and every variant caption "
            "must align with this story; do not contradict it; weave at least one concrete noun/number from it): "
            f"{anch[:520]}"
        )

    sigs = hp.get("signal_hashtags")
    if generate_hashtags and isinstance(sigs, list) and sigs:
        slugs = [sanitize_hashtag_body(str(x)) for x in sigs if sanitize_hashtag_body(str(x))]
        if slugs:
            need = min(3, len(slugs))
            lines.append(
                f"HASHTAG GROUNDING: each variant with a non-empty hashtags array MUST include "
                f"at least {need} of these exact slugs (no '#' in JSON; match spelling): "
                + ", ".join(slugs[:12])
            )

    if not lines:
        return ""

    return (
        "\nHYDRATION_PAYLOAD CONTRACT (canonical snapshot from the upload pipeline — obey alongside Scene Graph):\n"
        + "\n".join(f"  • {ln}" for ln in lines)
        + "\n"
    )


def sync_hydration_payload_signal_hashtags(ctx: Any, tags: Optional[List[str]]) -> None:
    """After merge_signal_hashtags_into_ctx, mirror final signal list on payload + artifact."""
    hp = getattr(ctx, "hydration_payload", None)
    if not isinstance(hp, dict):
        persist_hydration_payload_artifact(ctx)
        return
    if tags:
        hp["signal_hashtags"] = list(tags)
    persist_hydration_payload_artifact(ctx)


def hydration_payload_enabled() -> bool:
    """Lazy-load from core.config so tests can monkeypatch easily."""
    from core.config import HYDRATION_PAYLOAD_ENABLED
    return bool(HYDRATION_PAYLOAD_ENABLED)


def attach_hydration_payload_if_enabled(ctx: Any, category: str) -> None:
    """Build + attach ``ctx.hydration_payload`` if feature flag is on and not already set."""
    if not hydration_payload_enabled():
        return
    existing = getattr(ctx, "hydration_payload", None)
    if isinstance(existing, dict) and existing:
        ok, _ = validate_hydration_payload(existing)
        if ok:
            return
    try:
        hp = build_hydration_payload(ctx, category=category, category_source="stage_pre_thumbnail")
        ctx.hydration_payload = hp
        persist_hydration_payload_artifact(ctx)
    except Exception as exc:
        logger.warning("[hydration_payload] attach skipped: %s", exc)


def merge_m8_scene_into_hydration_payload(ctx: Any) -> None:
    """Merge M8 engine's scene graph hydration_story into the central payload."""
    if not hydration_payload_enabled():
        return
    hp = getattr(ctx, "hydration_payload", None)
    if not isinstance(hp, dict):
        return
    m8_scene = getattr(ctx, "m8_scene_graph", None) or {}
    hs = str(m8_scene.get("hydration_story") or "").strip()
    if hs and len(hs) > len(str(hp.get("hydration_story") or "").strip()):
        hp["hydration_story"] = hs[:1200]
        logger.debug("[hydration_payload] m8 hydration_story merged (%d chars)", len(hs))
    persist_hydration_payload_artifact(ctx)


def resolve_publish_metadata(ctx: Any, platform: str) -> Dict[str, Any]:
    """
    Single source of truth for effective title/caption/hashtags/thumbnail_url
    for a given platform. Reads ``ctx.hydration_payload`` + ``get_effective_*``
    so dashboards and email previews stay in sync with what was actually published.

    Returns a dict with keys:
        title, caption, hashtags, thumbnail_url, category, anchor_phrase,
        category_source, payload_version
    """
    hp = getattr(ctx, "hydration_payload", None)
    hp_ok, _ = validate_hydration_payload(hp) if isinstance(hp, dict) else (False, "no payload")

    plat = str(platform or "").strip().lower()

    try:
        from stages.context import get_effective_title, get_effective_caption, get_effective_hashtags
        title = get_effective_title(ctx, platform=plat)
        caption = get_effective_caption(ctx, platform=plat)
        hashtags = get_effective_hashtags(ctx, platform=plat)
    except Exception:
        title = str(getattr(ctx, "ai_title", None) or getattr(ctx, "title", None) or "")
        caption = str(getattr(ctx, "ai_caption", None) or getattr(ctx, "caption", None) or "")
        hashtags = list(getattr(ctx, "ai_hashtags", None) or getattr(ctx, "hashtags", None) or [])

    thumbnail_url: Optional[str] = None
    try:
        plat_thumbnails = getattr(ctx, "platform_thumbnails", None) or {}
        if isinstance(plat_thumbnails, dict):
            thumbnail_url = plat_thumbnails.get(plat) or plat_thumbnails.get("default")
        if not thumbnail_url:
            thumbnail_url = getattr(ctx, "thumbnail_url", None) or getattr(ctx, "thumbnail_r2_key", None)
    except Exception:
        pass

    # Only trust category/anchor from a valid (current-version) payload.
    hp_safe = hp if hp_ok else None

    return {
        "title": title,
        "caption": caption,
        "hashtags": hashtags if isinstance(hashtags, list) else [],
        "thumbnail_url": thumbnail_url,
        "category": str(
            (hp_safe or {}).get("category")
            or getattr(ctx, "thumbnail_category", "")
            or "general"
        ),
        "anchor_phrase": str((hp_safe or {}).get("anchor_phrase") or ""),
        "category_source": str(
            (hp_safe or {}).get("category_source") or ("payload" if hp_ok else "context")
        ),
        "payload_version": int((hp or {}).get("v") or 0),
    }

