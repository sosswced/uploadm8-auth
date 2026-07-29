"""Live transcode plan + per-platform encode status for queue/upload UI.

Platform encodes stay separate when fps, audio, duration, resolution, or
reframe rules differ (YouTube Shorts copyright/trim, Meta/TikTok caps).
This module explains that plan to users so the stage does not feel stuck.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


_PLATFORM_WHY = {
    "youtube": (
        "YouTube Shorts: separate encode for duration trim, fps, and audio "
        "(copyright-safe packaging)"
    ),
    "tiktok": "TikTok: vertical encode with TikTok bitrate/fps and cover-frame rules",
    "instagram": "Instagram Reels: Meta vertical encode (duration/bitrate caps)",
    "facebook": "Facebook Reels: Meta vertical encode (shared family, own bitrate/duration)",
}


def platform_why(platform: str) -> str:
    p = str(platform or "").strip().lower()
    return _PLATFORM_WHY.get(p, f"{p.title()}: platform-specific format")


def group_split_why(platforms: Sequence[str], *, total_groups: int) -> str:
    plats = [str(p).strip().lower() for p in platforms if str(p).strip()]
    if total_groups <= 1:
        return "One shared encode covers all selected platforms (matching format rules)."
    if len(plats) == 1 and plats[0] == "youtube":
        return platform_why("youtube")
    labels = ", ".join(p.title() for p in plats)
    return (
        f"{labels}: separate encode — fps, audio sample rate, bitrate, or max duration "
        "differ from other platforms"
    )


def build_transcode_status_plan(
    *,
    source: Dict[str, Any],
    groups: Sequence[Dict[str, Any]],
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Initial plan written when the transcode stage starts."""
    total = max(1, len(groups))
    group_rows: List[Dict[str, Any]] = []
    for i, g in enumerate(groups):
        plats = [str(p).strip().lower() for p in (g.get("platforms") or []) if str(p).strip()]
        canonical = str(g.get("canonical") or (plats[0] if plats else f"group{i}")).lower()
        group_rows.append(
            {
                "id": f"g{i}",
                "platforms": plats,
                "canonical": canonical,
                "status": str(g.get("status") or "pending"),
                "target": str(g.get("target") or ""),
                "max_fps": g.get("max_fps"),
                "encode_pct": int(g.get("encode_pct") or 0),
                "why": str(g.get("why") or group_split_why(plats, total_groups=total)),
                "reasons": list(g.get("reasons") or []),
            }
        )
    status = {
        "phase": "planning",
        "source": dict(source or {}),
        "groups": group_rows,
        "groups_total": total,
        "groups_done": 0,
        "note": note
        or (
            "Each platform gets its own encode when fps, audio, duration, or resolution "
            "rules differ. HD/4K clips can take several minutes — progress updates per encode."
        ),
    }
    status["summary"] = summarize_transcode_status(status)
    return status


def summarize_transcode_status(status: Dict[str, Any]) -> str:
    """One-line human summary for queue / upload polling."""
    groups = list(status.get("groups") or [])
    total = int(status.get("groups_total") or len(groups) or 1)
    done = int(status.get("groups_done") or 0)
    phase = str(status.get("phase") or "").strip().lower()
    if phase == "done":
        return f"Platform formats ready ({total} encode group{'s' if total != 1 else ''})"

    encoding = [g for g in groups if str(g.get("status") or "") == "encoding"]
    pending = [g for g in groups if str(g.get("status") or "") == "pending"]
    parts: List[str] = []
    if total > 1:
        parts.append(f"{total} separate encodes")
    else:
        parts.append("1 platform encode")

    if encoding:
        bits = []
        for g in encoding:
            plats = g.get("platforms") or [g.get("canonical")]
            label = "+".join(str(p).title() for p in plats if p)
            pct = int(g.get("encode_pct") or 0)
            tgt = str(g.get("target") or "").strip()
            bit = label
            if tgt:
                bit += f" {tgt}"
            if pct > 0:
                bit += f" {pct}%"
            bits.append(bit)
        parts.append("encoding " + "; ".join(bits))
    elif pending and done == 0:
        labels = []
        for g in groups[:3]:
            plats = g.get("platforms") or [g.get("canonical")]
            labels.append("+".join(str(p).title() for p in plats if p))
        more = f" +{len(groups) - 3}" if len(groups) > 3 else ""
        parts.append("starting " + ", ".join(labels) + more)
    else:
        parts.append(f"{done}/{total} groups done")

    src = status.get("source") or {}
    tier = str(src.get("tier") or "").strip()
    if tier in ("4k", "1080p"):
        parts.append(f"{tier} source")

    return " · ".join(parts)


def patch_group_status(
    status: Dict[str, Any],
    *,
    canonical: str,
    group_status: str,
    encode_pct: Optional[int] = None,
) -> Dict[str, Any]:
    """Return updated status with one group's status/pct refreshed."""
    out = dict(status or {})
    groups = [dict(g) for g in (out.get("groups") or [])]
    canon = str(canonical or "").strip().lower()
    done = 0
    for g in groups:
        plats = [str(p).lower() for p in (g.get("platforms") or [])]
        if g.get("canonical") == canon or canon in plats:
            g["status"] = group_status
            if encode_pct is not None:
                g["encode_pct"] = max(0, min(99, int(encode_pct)))
        if str(g.get("status") or "") in ("done", "copy", "fallback"):
            done += 1
    out["groups"] = groups
    out["groups_done"] = done
    if group_status == "encoding":
        out["phase"] = "encoding"
    elif done >= max(1, int(out.get("groups_total") or len(groups) or 1)):
        out["phase"] = "done"
    else:
        out["phase"] = out.get("phase") or "encoding"
    out["summary"] = summarize_transcode_status(out)
    return out


def _artifact_summary_blob(arts: dict, key: str) -> Optional[str]:
    blob = arts.get(key)
    if isinstance(blob, str) and blob.strip():
        import json

        try:
            blob = json.loads(blob)
        except Exception:
            return blob.strip()
    if isinstance(blob, dict):
        summary = str(blob.get("summary") or "").strip()
        return summary or None
    return None


def stage_detail_from_artifacts(raw: Any) -> Optional[str]:
    """Extract queue-facing detail from transcode_status or stage_status."""
    if raw is None:
        return None
    arts = raw
    if isinstance(raw, str):
        import json

        try:
            arts = json.loads(raw)
        except Exception:
            return None
    if not isinstance(arts, dict):
        return None
    # Prefer live multimodal/status detail when present; fall back to encode plan.
    return _artifact_summary_blob(arts, "stage_status") or _artifact_summary_blob(
        arts, "transcode_status"
    )
