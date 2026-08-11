"""FactLedger v1 — single source of truth for publishable evidence classes.

Works *with* persona/speed risks:
  - Soft-weave missing classes into existing voice (never formula-wipe titles).
  - Only ``publishable`` facts are required (high-conf peak, detected music, etc.).
  - Feature flag ``UPLOADM8_FACT_LEDGER`` (default on) — set false to roll back.

Required classes when publishable:
  speed_peak | music_artist | music_title | place_primary | road_primary
  | vehicle_make | vehicle_model | trill_bucket
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("uploadm8.fact_ledger")

FACT_LEDGER_VERSION = 1

# Default on — disable with UPLOADM8_FACT_LEDGER=0/false/off.
def fact_ledger_enabled() -> bool:
    raw = os.environ.get("UPLOADM8_FACT_LEDGER")
    if raw is None:
        try:
            from core.config import FACT_LEDGER_ENABLED

            return bool(FACT_LEDGER_ENABLED)
        except Exception:
            return True
    v = str(raw).strip().lower()
    if not v:
        return True
    return v not in ("0", "false", "no", "off", "disable", "disabled", "2")


def fact_ledger_strict() -> bool:
    raw = os.environ.get("UPLOADM8_FACT_LEDGER_STRICT")
    if raw is None:
        try:
            from core.config import FACT_LEDGER_STRICT

            return bool(FACT_LEDGER_STRICT)
        except Exception:
            return False
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


REQUIRED_CLASSES = (
    "speed_peak",
    "music_artist",
    "music_title",
    "place_primary",
    "road_primary",
    "vehicle_make",
    "vehicle_model",
    "trill_bucket",
)


@dataclass
class FactEntry:
    cls: str
    value: str
    source: str = ""
    confidence: str = "high"  # high | medium | low
    publishable: bool = True
    slug: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FactLedger:
    version: int = FACT_LEDGER_VERSION
    facts: Dict[str, FactEntry] = field(default_factory=dict)
    upload_id: str = ""

    def publishable_classes(self) -> List[str]:
        return [
            c
            for c in REQUIRED_CLASSES
            if c in self.facts and self.facts[c].publishable and self.facts[c].value
        ]

    def missing_in_text(
        self,
        title: str,
        caption: str,
        hashtags: Optional[Sequence[str]] = None,
        *,
        include_hashtags: bool = False,
    ) -> List[str]:
        """Return publishable classes not cited in title/caption (and optionally tags).

        Hashtags are a *separate* discovery lane — by default they do **not**
        satisfy caption weave (artist/song must still land in copy).
        """
        blob = f"{title or ''} {caption or ''}".lower()
        if include_hashtags:
            tags = " ".join(str(t or "").lower().lstrip("#") for t in (hashtags or []))
            hay = f"{blob} {tags}"
        else:
            hay = blob
        missing: List[str] = []
        for cls in self.publishable_classes():
            entry = self.facts[cls]
            if _fact_cited(entry, hay):
                continue
            missing.append(cls)
        return missing

    def to_report(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "upload_id": self.upload_id,
            "publishable": self.publishable_classes(),
            "facts": {k: v.to_dict() for k, v in self.facts.items()},
        }


def _sanitize_slug(raw: str, *, max_len: int = 24) -> str:
    try:
        from core.helpers import sanitize_hashtag_body

        return sanitize_hashtag_body(raw, max_len=max_len) or ""
    except Exception:
        s = re.sub(r"[^\w]", "", str(raw or "").lower())
        return s[:max_len]


def _music_tokens_in(blob: str, value: str) -> bool:
    v = str(value or "").strip().lower()
    if not v:
        return False
    if v in blob:
        return True
    stop = {"wit", "da", "the", "feat", "ft", "featuring", "and", "with", "a", "an"}
    toks = [t for t in re.findall(r"[a-z0-9]+", v) if len(t) >= 2 and t not in stop]
    if len(toks) >= 2 and all(t in blob for t in toks[:2]):
        return True
    if len(toks) == 1 and len(toks[0]) >= 4 and toks[0] in blob:
        return True
    return False


def _fact_cited(entry: FactEntry, hay: str) -> bool:
    val = str(entry.value or "").strip()
    if not val:
        return True
    cls = entry.cls
    low = hay.lower()
    if cls == "speed_peak":
        m = re.search(r"(\d{2,3})", val)
        if m and re.search(rf"\b{re.escape(m.group(1))}\s*mph\b", low):
            return True
        return bool(re.search(r"\b\d{2,3}\s*mph\b", low) and m and m.group(1) in low)
    if cls in ("music_artist", "music_title"):
        if _music_tokens_in(low, val):
            return True
        slug = entry.slug or _sanitize_slug(val)
        return bool(slug and slug in low)
    if cls in ("place_primary", "road_primary", "vehicle_make", "vehicle_model", "trill_bucket"):
        if val.lower() in low:
            return True
        # City/state display: cite if the city token alone already appears.
        if cls == "place_primary" and "," in val:
            city = val.split(",", 1)[0].strip().lower()
            if len(city) >= 3 and city in low:
                return True
        if cls == "road_primary":
            stop = {"road", "hwy", "highway", "street", "ave", "avenue", "blvd", "the", "and"}
            toks = [
                t
                for t in re.findall(r"[a-z0-9]+", val.lower())
                if len(t) >= 3 and t not in stop
            ]
            if toks and all(t in low for t in toks[:2]):
                return True
        # Road: "I 505" vs i505
        compact = re.sub(r"[\s\-]+", "", val.lower())
        if compact and compact in re.sub(r"[\s\-#]+", "", low):
            return True
        slug = entry.slug or _sanitize_slug(val)
        return bool(slug and slug in low)
    return val.lower() in low


def _put(
    facts: Dict[str, FactEntry],
    cls: str,
    value: Any,
    *,
    source: str,
    confidence: str = "high",
    publishable: bool = True,
) -> None:
    s = str(value or "").strip()
    if not s:
        return
    facts[cls] = FactEntry(
        cls=cls,
        value=s,
        source=source,
        confidence=confidence,
        publishable=publishable and confidence == "high",
        slug=_sanitize_slug(s),
    )


def build_fact_ledger(ctx: Any, pool: Any = None) -> FactLedger:
    """Build ledger from EvidencePool + garage vehicle on ctx."""
    if pool is None:
        from services.hydration_enforcer import collect_evidence

        pool = collect_evidence(ctx)

    facts: Dict[str, FactEntry] = {}
    uid = str(getattr(ctx, "upload_id", "") or "")

    # Speed — only high-conf publishable peak (work with consensus risk).
    peak = float(getattr(pool, "max_speed_mph", 0) or 0)
    if peak >= 5:
        _put(
            facts,
            "speed_peak",
            f"{int(round(peak))} MPH",
            source=str(getattr(pool, "speed_source", "") or "consensus"),
            confidence="high",
            publishable=True,
        )

    # Place / road
    place = (
        getattr(pool, "gazetteer_place", None)
        or getattr(pool, "city", None)
        or getattr(pool, "place_sign", None)
    )
    if place:
        state = getattr(pool, "state", None) or getattr(pool, "state_abbr", None)
        disp = str(place).strip()
        if state and str(state).lower() not in disp.lower():
            disp = f"{disp}, {state}"
        _put(facts, "place_primary", disp, source="telemetry_geo")
    road = getattr(pool, "road", None)
    if road:
        _put(facts, "road_primary", str(road).strip(), source="telemetry_road")
    elif getattr(pool, "vision_highways", None):
        hw = pool.vision_highways[0] if pool.vision_highways else None
        if hw:
            _put(facts, "road_primary", str(hw).strip(), source="vision_highway")

    # Music — prefer detected; still accept artist/title if present (ACR flag flake).
    artist = getattr(pool, "music_artist", None)
    title = getattr(pool, "music_title", None)
    if not artist and not title:
        ac = getattr(ctx, "audio_context", None) or {}
        if isinstance(ac, dict):
            artist = (ac.get("music_artist") or "").strip() or None
            title = (ac.get("music_title") or "").strip() or None
    if artist:
        _put(facts, "music_artist", artist, source="acr")
    if title:
        _put(facts, "music_title", title, source="acr")

    # Trill bucket
    bucket = getattr(pool, "trill_bucket", None)
    if bucket:
        _put(facts, "trill_bucket", str(bucket).strip(), source="trill")

    # Garage / Trill vehicle (was never in EvidencePool — root cause for 9020642f)
    make = (
        getattr(ctx, "vehicle_make_name", None)
        or getattr(pool, "vehicle_make", None)
    )
    model = (
        getattr(ctx, "vehicle_model_name", None)
        or getattr(pool, "vehicle_model", None)
    )
    if make:
        _put(facts, "vehicle_make", str(make).strip(), source="garage")
    if model:
        _put(facts, "vehicle_model", str(model).strip(), source="garage")

    return FactLedger(facts=facts, upload_id=uid)


def soft_weave_missing_into_caption(
    caption: str,
    ledger: FactLedger,
    *,
    title: str = "",
    hashtags: Optional[Sequence[str]] = None,
    max_chars: int = 520,
) -> Tuple[str, List[str]]:
    """Append missing publishable classes as a short voice-safe clause.

    Never builds ``N MPH through Place — with Artist`` receipts.
    Returns (new_caption, classes_woven).
    """
    missing = ledger.missing_in_text(
        title, caption, hashtags, include_hashtags=False
    )
    if not missing:
        return (caption or "").strip(), []

    bits: List[str] = []
    woven: List[str] = []

    if "speed_peak" in missing and "speed_peak" in ledger.facts:
        bits.append(ledger.facts["speed_peak"].value)
        woven.append("speed_peak")

    place_bits: List[str] = []
    if "place_primary" in missing and "place_primary" in ledger.facts:
        place_bits.append(f"near {ledger.facts['place_primary'].value}")
        woven.append("place_primary")
    if "road_primary" in missing and "road_primary" in ledger.facts:
        place_bits.append(f"on {ledger.facts['road_primary'].value}")
        woven.append("road_primary")
    if place_bits:
        bits.append(" ".join(place_bits[:2]))

    music_bits: List[str] = []
    if "music_title" in missing and "music_title" in ledger.facts:
        music_bits.append(f"'{ledger.facts['music_title'].value}'")
        woven.append("music_title")
    if "music_artist" in missing and "music_artist" in ledger.facts:
        music_bits.append(f"by {ledger.facts['music_artist'].value}")
        woven.append("music_artist")
    if music_bits:
        bits.append("with " + " ".join(music_bits))

    veh: List[str] = []
    if "vehicle_make" in missing and "vehicle_make" in ledger.facts:
        veh.append(ledger.facts["vehicle_make"].value)
        woven.append("vehicle_make")
    if "vehicle_model" in missing and "vehicle_model" in ledger.facts:
        veh.append(ledger.facts["vehicle_model"].value)
        woven.append("vehicle_model")
    if veh:
        bits.append("in the " + " ".join(veh))

    if "trill_bucket" in missing and "trill_bucket" in ledger.facts:
        bits.append(f"Trill {ledger.facts['trill_bucket'].value}")
        woven.append("trill_bucket")

    if not bits:
        return (caption or "").strip(), []

    clause = " — " + "; ".join(bits)
    base = (caption or "").strip()
    if not base:
        # Prefer scene-less lead that is still not a through-Place receipt.
        peak = ledger.facts.get("speed_peak")
        lead = f"{peak.value} locked in" if peak else "This run"
        out = (lead + clause).strip()
    else:
        out = (base.rstrip(".!?") + clause).strip()
    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip() + "…"
    return out, woven


def ledger_hashtag_bodies(ledger: FactLedger) -> List[str]:
    """Ordered discovery tags from publishable ledger facts (no leading #)."""
    order = (
        "music_artist",
        "music_title",
        "place_primary",
        "road_primary",
        "vehicle_make",
        "vehicle_model",
        "trill_bucket",
        "speed_peak",
    )
    out: List[str] = []
    seen: set = set()
    for cls in order:
        entry = ledger.facts.get(cls)
        if not entry or not entry.publishable:
            continue
        if cls == "speed_peak":
            # Bucket tags instead of raw mph mash
            try:
                n = int(re.search(r"(\d{2,3})", entry.value).group(1))  # type: ignore
            except Exception:
                n = 0
            body = "tripledigits" if n >= 100 else ("highwayheat" if n >= 70 else "")
            if not body:
                continue
        elif cls == "place_primary":
            # Split "City, State" into two tags when possible
            try:
                from core.helpers import expand_geo_runon_hashtag, split_hashtag_source_phrases

                for phrase in split_hashtag_source_phrases(entry.value):
                    for body in expand_geo_runon_hashtag(phrase):
                        b = _sanitize_slug(body)
                        if b and b not in seen:
                            seen.add(b)
                            out.append(b)
                continue
            except Exception:
                body = entry.slug or _sanitize_slug(entry.value)
        elif cls == "road_primary":
            try:
                from core.vision_labels import road_hashtag_tokens

                toks = road_hashtag_tokens(entry.value) or []
                for t in toks:
                    b = _sanitize_slug(t)
                    if b and b not in seen:
                        seen.add(b)
                        out.append(b)
                if toks:
                    continue
            except Exception:
                pass
            body = entry.slug or _sanitize_slug(entry.value)
        else:
            body = entry.slug or _sanitize_slug(entry.value)
        if body and body not in seen:
            seen.add(body)
            out.append(body)
    return out


def pad_hashtags_with_ledger(
    tags: Sequence[str],
    ledger: FactLedger,
    *,
    target: int = 15,
    extras: Optional[Sequence[str]] = None,
) -> Tuple[List[str], int]:
    """Front-load ledger slugs, then keep extras/existing, pad to ``target``.

    Never pads with fyp/viral/meta seeds. Returns (merged, padded_n).
    """
    try:
        from core.vision_labels import is_junk_hashtag_body
    except Exception:

        def is_junk_hashtag_body(_b: str) -> bool:  # type: ignore
            return False

    ban = {"fyp", "viral", "trending", "follow", "like", "subscribe", "foryou", "foryoupage"}
    out: List[str] = []
    seen: set = set()

    def _add(raw: str) -> bool:
        b = _sanitize_slug(str(raw or "").lstrip("#"))
        if not b or b in seen or b in ban or is_junk_hashtag_body(b):
            return False
        seen.add(b)
        out.append(b)
        return True

    for b in ledger_hashtag_bodies(ledger):
        _add(b)
        if len(out) >= target:
            return out[:target], 0

    start = len(out)
    for src in (extras or (), tags):
        for t in src:
            _add(str(t))
            if len(out) >= target:
                break
        if len(out) >= target:
            break

    padded = max(0, len(out) - start) if start == 0 else max(0, len(out) - len(list(tags or [])))
    # Prefer counting how many ledger tags we added beyond original
    original = {_sanitize_slug(str(t).lstrip("#")) for t in (tags or [])}
    added = sum(1 for t in out if t not in original)
    return out[:target], added


def apply_fact_ledger_to_ctx(ctx: Any, pool: Any = None) -> Dict[str, Any]:
    """Soft-weave captions + pad hashtags. Does not wipe titles.

    Safe to call from ``enforce_hydration`` when flag enabled.
    """
    report: Dict[str, Any] = {
        "enabled": fact_ledger_enabled(),
        "woven_classes": [],
        "missing_before": [],
        "missing_after": [],
        "hashtags_padded": 0,
        "hashtag_target": 15,
    }
    if not fact_ledger_enabled():
        return report

    ledger = build_fact_ledger(ctx, pool)
    report["ledger"] = ledger.to_report()

    us = getattr(ctx, "user_settings", None) or {}
    target = 15
    try:
        raw = us.get("maxHashtags")
        if raw is None:
            raw = us.get("max_hashtags")
        if raw is not None:
            target = max(1, min(30, int(raw)))
    except (TypeError, ValueError):
        target = 15
    report["hashtag_target"] = target

    title = str(getattr(ctx, "ai_title", "") or "")
    caption = str(getattr(ctx, "ai_caption", "") or "")
    tags = list(getattr(ctx, "ai_hashtags", None) or [])
    report["missing_before"] = ledger.missing_in_text(
        title, caption, tags, include_hashtags=False
    )

    # Soft-weave into caption only (titles stay LLM-led).
    new_cap, woven = soft_weave_missing_into_caption(
        caption, ledger, title=title, hashtags=tags
    )
    if woven and new_cap != caption:
        ctx.ai_caption = new_cap
        report["woven_classes"] = woven

    m8_caps = getattr(ctx, "m8_platform_captions", None)
    if isinstance(m8_caps, dict):
        for pl, cap in list(m8_caps.items()):
            pl_title = ""
            m8_titles = getattr(ctx, "m8_platform_titles", None) or {}
            if isinstance(m8_titles, dict):
                pl_title = str(m8_titles.get(pl) or "")
            pl_tags = []
            m8_tags = getattr(ctx, "m8_platform_hashtags", None) or {}
            if isinstance(m8_tags, dict):
                pl_tags = list(m8_tags.get(pl) or [])
            nc, w = soft_weave_missing_into_caption(
                str(cap or ""), ledger, title=pl_title or title, hashtags=pl_tags or tags
            )
            if w and nc != str(cap or ""):
                m8_caps[pl] = nc
                for c in w:
                    if c not in report["woven_classes"]:
                        report["woven_classes"].append(c)

    # Pad hashtags — ledger first.
    extras = []
    try:
        from services.hydration_enforcer import build_evidence_hashtags

        if pool is not None:
            extras = build_evidence_hashtags(pool, max_extra=max(target, 14))
    except Exception:
        extras = []

    merged, _added = pad_hashtags_with_ledger(
        list(getattr(ctx, "ai_hashtags", None) or []),
        ledger,
        target=target,
        extras=extras,
    )
    before_n = len(list(getattr(ctx, "ai_hashtags", None) or []))
    ctx.ai_hashtags = merged
    report["hashtags_padded"] = max(0, len(merged) - before_n)

    m8_htags = getattr(ctx, "m8_platform_hashtags", None)
    if isinstance(m8_htags, dict):
        for pl, raw in list(m8_htags.items()):
            mrg, _ = pad_hashtags_with_ledger(
                list(raw or []), ledger, target=target, extras=extras
            )
            m8_htags[pl] = mrg

    title2 = str(getattr(ctx, "ai_title", "") or "")
    caption2 = str(getattr(ctx, "ai_caption", "") or "")
    tags2 = list(getattr(ctx, "ai_hashtags", None) or [])
    missing_pass1 = ledger.missing_in_text(
        title2, caption2, tags2, include_hashtags=False
    )
    # Second soft-weave pass if anything still missing from copy.
    if missing_pass1:
        nc2, w2 = soft_weave_missing_into_caption(
            caption2, ledger, title=title2, hashtags=tags2
        )
        if w2 and nc2 != caption2:
            ctx.ai_caption = nc2
            for c in w2:
                if c not in report["woven_classes"]:
                    report["woven_classes"].append(c)
            report["second_weave"] = True
            caption2 = nc2
            m8_caps2 = getattr(ctx, "m8_platform_captions", None)
            if isinstance(m8_caps2, dict):
                for pl, cap in list(m8_caps2.items()):
                    pl_title = ""
                    m8_titles = getattr(ctx, "m8_platform_titles", None) or {}
                    if isinstance(m8_titles, dict):
                        pl_title = str(m8_titles.get(pl) or "")
                    nc3, w3 = soft_weave_missing_into_caption(
                        str(cap or ""), ledger, title=pl_title or title2, hashtags=tags2
                    )
                    if w3 and nc3 != str(cap or ""):
                        m8_caps2[pl] = nc3

    report["missing_after"] = ledger.missing_in_text(
        title2, str(getattr(ctx, "ai_caption", "") or ""), tags2, include_hashtags=False
    )
    report["missing_after_with_tags"] = ledger.missing_in_text(
        title2,
        str(getattr(ctx, "ai_caption", "") or ""),
        tags2,
        include_hashtags=True,
    )
    if report["missing_after"]:
        report.setdefault("quality_notes", []).append(
            "fact_ledger_classes_still_missing_after_weave"
        )

    arts = getattr(ctx, "output_artifacts", None)
    if not isinstance(arts, dict):
        arts = {}
        try:
            ctx.output_artifacts = arts
        except Exception:
            pass
    if isinstance(arts, dict):
        arts["fact_ledger_v1"] = ledger.to_report()
        arts["fact_ledger_apply"] = {
            "woven_classes": report["woven_classes"],
            "missing_before": report["missing_before"],
            "missing_after": report["missing_after"],
            "missing_after_with_tags": report["missing_after_with_tags"],
            "hashtags_padded": report["hashtags_padded"],
            "hashtag_target": target,
            "second_weave": bool(report.get("second_weave")),
            "quality_notes": list(report.get("quality_notes") or []),
        }

    logger.info(
        "[fact_ledger] upload=%s publishable=%s woven=%s missing_after=%s tags=%s",
        uid if (uid := getattr(ctx, "upload_id", "")) else "?",
        ledger.publishable_classes(),
        report["woven_classes"],
        report["missing_after"],
        len(tags2),
    )
    return report


def publishable_facts_from_scene_graph(scene_graph: Dict[str, Any]) -> Dict[str, FactEntry]:
    """Minimal publishable class map for M8 ranking (no full JobContext)."""
    facts: Dict[str, FactEntry] = {}
    if not isinstance(scene_graph, dict):
        return facts
    cons = scene_graph.get("speed_consensus") if isinstance(scene_graph.get("speed_consensus"), dict) else {}
    try:
        peak = float(cons.get("peak_mph") or 0)
    except (TypeError, ValueError):
        peak = 0.0
    conf = str(cons.get("confidence") or "")
    if peak >= 5 and (not conf or conf == "high"):
        _put(facts, "speed_peak", f"{int(round(peak))} MPH", source="scene_graph", confidence="high")

    geo = scene_graph.get("geo") or {}
    place = geo.get("gazetteer_place") or geo.get("city") or geo.get("place_sign")
    if place:
        state = geo.get("state") or geo.get("gazetteer_state_usps")
        disp = str(place).strip()
        if state and str(state).lower() not in disp.lower():
            disp = f"{disp}, {state}"
        _put(facts, "place_primary", disp, source="scene_graph")
    if geo.get("road"):
        _put(facts, "road_primary", str(geo.get("road")).strip(), source="scene_graph")

    music = scene_graph.get("music") or {}
    if music.get("artist"):
        _put(facts, "music_artist", str(music.get("artist")).strip(), source="scene_graph")
    if music.get("title"):
        _put(facts, "music_title", str(music.get("title")).strip(), source="scene_graph")

    veh = scene_graph.get("vehicle") or {}
    if isinstance(veh, dict):
        if veh.get("make"):
            _put(facts, "vehicle_make", str(veh.get("make")).strip(), source="scene_graph")
        if veh.get("model"):
            _put(facts, "vehicle_model", str(veh.get("model")).strip(), source="scene_graph")

    trill = scene_graph.get("trill") or {}
    if trill.get("bucket"):
        _put(facts, "trill_bucket", str(trill.get("bucket")).strip(), source="scene_graph")
    return facts


def class_coverage_score(
    title: str,
    caption: str,
    facts: Dict[str, FactEntry],
    *,
    hashtags: Optional[Sequence[str]] = None,
) -> float:
    """Class coverage for ranking — work *with* persona voice.

    - Formula stubs never earn the complete-class bonus (hydration rewrites them)
    - 0 hits → −200 (same floor as empty must_use)
    - Partial → soft −12 per missing class (hydration weaves the rest)
    - Complete non-stub → +24
    """
    if not fact_ledger_enabled() or not facts:
        return 0.0
    ledger = FactLedger(facts=facts)
    pubs = ledger.publishable_classes()
    if not pubs:
        return 0.0
    stub = False
    try:
        from services.m8_grounding_pass import is_formula_stub_caption

        # Caption drives stub denial — titles often use soft ``N MPH through…`` leads.
        stub = is_formula_stub_caption(caption or "")
    except Exception:
        stub = bool(re.match(r"(?i)^\s*anchored\s+in\b", (caption or "").strip()))

    missing = ledger.missing_in_text(
        title or "", caption or "", hashtags, include_hashtags=False
    )
    hit = len(pubs) - len(missing)
    if hit <= 0:
        return -200.0
    if stub:
        # Evidence stuffed into a receipt is not a win — soft push only.
        return -35.0 if missing else -20.0
    if missing:
        return -12.0 * float(len(missing))
    return 24.0


def ensure_fact_ledger_at_publish(ctx: Any) -> Dict[str, Any]:
    """Final weave/pad before platform API calls. Fail-soft unless STRICT.

    Returns report; when STRICT and still missing, sets
    ``ctx.fact_ledger_block_publish = True``.
    """
    report: Dict[str, Any] = {
        "enabled": fact_ledger_enabled(),
        "strict": fact_ledger_strict(),
        "missing": [],
        "rewoven": False,
        "blocked": False,
    }
    if not fact_ledger_enabled():
        return report
    try:
        from services.hydration_enforcer import collect_evidence

        pool = collect_evidence(ctx)
        fl = apply_fact_ledger_to_ctx(ctx, pool)
        report["rewoven"] = bool(fl.get("woven_classes") or fl.get("hashtags_padded"))
        missing = list(fl.get("missing_after") or [])
        # Tags may cover discovery; prefer with_tags for publish gap.
        missing_tags = list(fl.get("missing_after_with_tags") or missing)
        report["missing"] = missing_tags
        arts = getattr(ctx, "output_artifacts", None)
        if not isinstance(arts, dict):
            arts = {}
            try:
                ctx.output_artifacts = arts
            except Exception:
                pass
        if missing_tags:
            gap = {
                "missing": missing_tags,
                "strict": report["strict"],
                "upload_id": str(getattr(ctx, "upload_id", "") or ""),
            }
            if isinstance(arts, dict):
                arts["fact_ledger_gap"] = gap
            logger.warning(
                "[fact_ledger] publish gap upload=%s missing=%s strict=%s",
                getattr(ctx, "upload_id", "?"),
                missing_tags,
                report["strict"],
            )
            try:
                import sentry_sdk

                sentry_sdk.add_breadcrumb(
                    category="fact_ledger",
                    message="fact_ledger_gap",
                    level="warning",
                    data=gap,
                )
            except Exception:
                pass
            if report["strict"]:
                report["blocked"] = True
                try:
                    ctx.fact_ledger_block_publish = True
                except Exception:
                    pass
        elif isinstance(arts, dict):
            arts.pop("fact_ledger_gap", None)
    except Exception as exc:
        report["error"] = str(exc)
        logger.warning("[fact_ledger] ensure_at_publish failed: %s", exc)
    return report


__all__ = [
    "FactEntry",
    "FactLedger",
    "REQUIRED_CLASSES",
    "apply_fact_ledger_to_ctx",
    "build_fact_ledger",
    "class_coverage_score",
    "ensure_fact_ledger_at_publish",
    "fact_ledger_enabled",
    "fact_ledger_strict",
    "ledger_hashtag_bodies",
    "pad_hashtags_with_ledger",
    "publishable_facts_from_scene_graph",
    "soft_weave_missing_into_caption",
]
