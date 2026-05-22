"""
Post-hydration metadata quality gate — evidence overlap + filler ban.

Runs **after** :func:`services.hydration_enforcer.enforce_hydration` so titles/captions
already reflect deterministic anchor rewriting. Persists onto
``output_artifacts.metadata_quality_report`` and merges into
structured ``hydration_report["metadata_quality"]`` when the latter exists.

Env:

- ``METADATA_QUALITY_STRICT`` — ``off`` (default report-only, always ``ok``) |
  ``degrade`` / ``true`` / ``1`` / ``on`` | ``halt`` / ``fail`` / ``block``
  (scored like degrade — worker publishes ``halt`` mode blocks publish separately).
- ``METADATA_QUALITY_MIN_ANCHOR_HITS`` — overlap threshold when ``evidence_present``
  (default ``2``).
- ``METADATA_QUALITY_ENFORCE_FILLERS_ON_FIRST_PASS`` — when ``true`` and STRICT is ``off``,
  filler bans still flip ``ok`` to ``False`` without requiring anchor overlap enforcement.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, MutableMapping

logger = logging.getLogger("uploadm8-worker")


def metadata_quality_strict_mode() -> str:
    v = (os.environ.get("METADATA_QUALITY_STRICT") or "").strip().lower()
    if not v or v in ("0", "false", "no", "off"):
        return "off"
    if v in ("halt", "fail", "block", "abort", "hard"):
        return "halt"
    return "degrade"


_FILLER_TERMS = (
    "amazing",
    "go viral",
    "viral",
    "you won't believe",
    "you wont believe",
    "crazy moment",
    "check this out",
    "insane footage",
)


def _text_blob_from_ctx(ctx: Any) -> str:
    chunks: List[str] = []

    def _add(s: Any) -> None:
        t = str(s or "").strip()
        if t:
            chunks.append(t)

    _add(getattr(ctx, "ai_title", ""))
    _add(getattr(ctx, "ai_caption", ""))

    mt = getattr(ctx, "m8_platform_titles", None) or {}
    mc = getattr(ctx, "m8_platform_captions", None) or {}
    if isinstance(mt, dict):
        for v in mt.values():
            _add(v)
    if isinstance(mc, dict):
        for v in mc.values():
            _add(v)

    tags = getattr(ctx, "ai_hashtags", None) or []
    if isinstance(tags, list):
        for t in tags:
            _add(t)
    else:
        _add(tags)

    return " ".join(chunks).lower()


def _structured_hydration_report(ctx: Any) -> Dict[str, Any]:
    arts = getattr(ctx, "output_artifacts", None) or {}
    if not isinstance(arts, MutableMapping):
        return {}
    raw = arts.get("hydration_report")
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            j = json.loads(raw)
            return dict(j) if isinstance(j, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}


def _anchor_clues(anchor: str) -> List[str]:
    if not anchor:
        return []
    s_low = anchor.lower()
    chunks: List[str] = []

    for m in re.finditer(r"\b\d{2,4}\s*mph\b", s_low):
        chunks.append(m.group(0).strip())

    for m in re.finditer(r"-?\d+\.\d{3,}\s*[°nswe]?", anchor, re.I):
        t = m.group(0).strip().lower()
        if len(t.replace(" ", "")) >= 5:
            chunks.append(t)

    for phrase in re.finditer(r"[A-Za-z0-9][A-Za-z0-9 ,'&\.-]{2,48}", anchor):
        token = phrase.group(0).strip().strip("'\"").lower()
        if len(token.replace(" ", "")) >= 3 and token.count(" ") <= 14:
            chunks.append(token)

    seen: set = set()
    out: List[str] = []
    for c in chunks:
        cc = c[:120]
        if cc and cc not in seen:
            seen.add(cc)
            out.append(cc)
        if len(out) >= 42:
            break
    return out


def validate_metadata_quality(ctx: Any) -> Dict[str, Any]:
    mode = metadata_quality_strict_mode()
    fillers_when_off = (
        (os.environ.get("METADATA_QUALITY_ENFORCE_FILLERS_ON_FIRST_PASS") or "")
        .strip()
        .lower()
        in ("1", "true", "yes", "on")
    )

    blob = _text_blob_from_ctx(ctx)
    hr = _structured_hydration_report(ctx)
    evidence_present = bool(hr.get("evidence_present"))
    anchor = str(hr.get("anchor") or "").strip()

    violations: List[str] = []
    filler_hits = [t for t in _FILLER_TERMS if t in blob]

    raw_min = os.environ.get("METADATA_QUALITY_MIN_ANCHOR_HITS", "").strip()
    try:
        min_anchor_hits = int(raw_min) if raw_min else 2
    except ValueError:
        min_anchor_hits = 2
    min_anchor_hits = max(1, min(min_anchor_hits, 12))

    clues = _anchor_clues(anchor)
    hits = sum(1 for c in clues if len(c.replace(" ", "")) >= 3 and c.lower() in blob)

    filler_violations = [f"banned_vague_language:{t}" for t in filler_hits]
    overlap_violation: List[str] = []

    scoring_on = mode in ("degrade", "halt")
    if scoring_on:
        violations.extend(filler_violations)
        if evidence_present and anchor:
            if hits < min_anchor_hits:
                overlap_violation.append(f"thin_evidence_overlap:{hits}/{min_anchor_hits}")
        violations.extend(overlap_violation)
    elif fillers_when_off:
        violations.extend(filler_violations)

    if scoring_on:
        ok_final = not violations
    elif fillers_when_off:
        ok_final = not filler_violations
    else:
        ok_final = True

    report: Dict[str, Any] = {
        "ok": ok_final,
        "mode": mode,
        "filler_hits": filler_hits,
        "hydration_anchor_hits": hits,
        "hydration_anchor_clues": len(clues),
        "evidence_present": evidence_present,
        "violations": violations,
    }

    try:
        setattr(ctx, "metadata_quality_ok", bool(ok_final))
        setattr(ctx, "metadata_quality_violations", list(violations))
    except Exception:
        pass

    if isinstance(getattr(ctx, "output_artifacts", None), dict):
        try:
            ctx.output_artifacts["metadata_quality_report"] = json.dumps(
                report, default=str
            )[:48000]

            hr2 = dict(hr) if hr else {}
            slot = hr2.setdefault("metadata_quality", {})
            if isinstance(slot, dict):
                slot.update({k: v for k, v in report.items() if k != "mode"})
                ctx.output_artifacts["hydration_report"] = hr2
        except Exception as ex:
            logger.debug("[metadata_quality] artifact merge failed: %s", ex)

    return report


__all__ = ["metadata_quality_strict_mode", "validate_metadata_quality"]
