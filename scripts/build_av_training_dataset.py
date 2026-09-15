#!/usr/bin/env python
# /// script
# dependencies = [
#   "asyncpg>=0.29.0,<0.32.0",
#   "pandas>=2.0.0,<3.0.0",
#   "pyarrow>=15.0.0",
#   "datasets>=2.20.0,<4.0.0",
#   "huggingface_hub>=1.10.0,<2.0.0",
#   "python-dotenv>=1.0.0,<2.0.0",
# ]
# ///
"""Build AV training pack dataset from uploads.output_artifacts.av_training_pack_v1."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(_REPO_ROOT / ".env")

from services.av_training_pack_quality import (
    DEFAULT_MIN_EVIDENCE_LANES,
    DEFAULT_MIN_GROUNDING_SCORE,
    filter_pack_training_rows,
)
from services.hf_dataset_export import coerce_dataframe_for_hf, push_dataframe_to_hub

UPLOADS_SQL = """
SELECT
    u.id AS upload_id,
    u.user_id,
    u.created_at,
    u.output_artifacts
FROM uploads u
WHERE u.created_at >= (NOW() - ($1::int || ' days')::interval)
  AND u.output_artifacts ? 'av_training_pack_v1'
ORDER BY u.created_at DESC
LIMIT $2::int
"""


def _as_dict(val: Any) -> Dict[str, Any]:
    if isinstance(val, dict):
        return val
    if isinstance(val, str) and val.strip().startswith("{"):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _row_from_meta(upload_id: str, user_id: str, meta: Dict[str, Any], pack: Dict[str, Any] | None) -> Dict[str, Any]:
    flags = (pack or {}).get("teacher_flags") if isinstance((pack or {}).get("teacher_flags"), dict) else {}
    vision = (pack or {}).get("vision") if isinstance((pack or {}).get("vision"), dict) else {}
    identity = (pack or {}).get("content_identity_v1") if isinstance((pack or {}).get("content_identity_v1"), dict) else {}
    heroes = identity.get("hero_facts") or []
    top_class = meta.get("identity_hero_fact_class") or ""
    if not top_class and heroes and isinstance(heroes[0], dict):
        top_class = str(heroes[0].get("class") or "")
    fusion_text = str((pack or {}).get("fusion_summary") or "")
    transcript_chars = int(meta.get("transcript_chars") or (pack or {}).get("transcript_chars") or 0)
    segs = (pack or {}).get("transcript_segments") or []
    return {
        "upload_id": str(upload_id),
        "user_id": str(user_id),
        "upload_id_hash": meta.get("upload_id_hash") or (pack or {}).get("upload_id_hash"),
        "clip_kind": meta.get("clip_kind") or (pack or {}).get("clip_kind") or "general",
        "transcript_chars": transcript_chars,
        "segment_count": len(segs) if isinstance(segs, list) else 0,
        "vision_label_count": len(vision.get("labels") or []),
        "vision_ocr_chars": len(str(vision.get("ocr") or "")),
        "evidence_lane_count": len(meta.get("lanes_on") or flags.get("lanes_on") or []),
        "identity_confidence": str(meta.get("identity_confidence") or identity.get("confidence") or ""),
        "identity_hero_fact_class": top_class or "unknown",
        "needs_deep_teacher": bool(meta.get("needs_deep_teacher") if meta.get("needs_deep_teacher") is not None else flags.get("needs_deep_teacher")),
        "tl_status": meta.get("tl_status") or flags.get("tl_status") or "unknown",
        "fusion_source": meta.get("fusion_source") or flags.get("fusion_source") or "none",
        "grounding_score": meta.get("grounding_score"),
        "grounding_band": meta.get("grounding_band") or "mid",
        "keyframe_count": int(meta.get("keyframe_count") or 0),
        "fusion_text": fusion_text[:2000],
        "pack_r2_key": meta.get("pack_r2_key") or "",
    }


def _maybe_fetch_pack(meta: Dict[str, Any]) -> Dict[str, Any] | None:
    key = str(meta.get("pack_r2_key") or "").strip()
    if not key:
        return None
    try:
        from core.r2 import get_object_bytes

        body, _ct = get_object_bytes(key)
        return json.loads(body.decode("utf-8"))
    except Exception as e:
        print(f"pack.json fetch skipped ({key}): {e}")
        return None


async def _fetch(dsn: str, lookback_days: int, limit: int) -> List[Dict[str, Any]]:
    from core.db_pool import connect_with_retry

    conn = await connect_with_retry(dsn)
    try:
        rows = await conn.fetch(UPLOADS_SQL, int(lookback_days), int(limit))
    finally:
        await conn.close()
    return [dict(r) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback-days", type=int, default=180)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--output", type=str, default="data/ml/av_training_pack_v1.parquet")
    ap.add_argument("--push-to", type=str, default="")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument(
        "--min-grounding-score",
        type=float,
        default=DEFAULT_MIN_GROUNDING_SCORE,
        help="Refuse packs below this grounding score (teacher disagreement filter)",
    )
    ap.add_argument(
        "--min-evidence-lanes",
        type=int,
        default=DEFAULT_MIN_EVIDENCE_LANES,
        help="Refuse packs with fewer evidence lanes",
    )
    ap.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="Disable teacher disagreement filter (debug only; not for distill train)",
    )
    ap.add_argument(
        "--refused-output",
        type=str,
        default="data/ml/av_training_pack_v1_refused.parquet",
        help="Where to write refused rows for audit (empty string to skip)",
    )
    args = ap.parse_args()

    dsn = (os.environ.get("DATABASE_URL") or "").strip()
    if not dsn:
        raise SystemExit("DATABASE_URL is required")

    rows = asyncio.run(_fetch(dsn, args.lookback_days, args.limit))
    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        arts = r.get("output_artifacts")
        if isinstance(arts, str):
            arts = _as_dict(arts)
        if not isinstance(arts, dict):
            continue
        meta = _as_dict(arts.get("av_training_pack_v1"))
        if not meta:
            continue
        pack = _maybe_fetch_pack(meta)
        out_rows.append(_row_from_meta(str(r["upload_id"]), str(r["user_id"]), meta, pack))

    refused_audit: List[Dict[str, Any]] = []
    if args.no_quality_filter:
        accepted = out_rows
        print("WARNING: --no-quality-filter set; teacher disagreement filter disabled")
    else:
        accepted, refused = filter_pack_training_rows(
            out_rows,
            min_grounding_score=float(args.min_grounding_score),
            min_evidence_lanes=int(args.min_evidence_lanes),
        )
        for item in refused:
            row = dict(item.get("row") or {})
            row["refuse_reason"] = item.get("reason")
            refused_audit.append(row)
        print(
            f"Quality filter: kept {len(accepted)} / {len(out_rows)} "
            f"(refused {len(refused_audit)}; min_grounding={args.min_grounding_score})"
        )

    df = pd.DataFrame(accepted)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows -> {out_path}")

    if refused_audit and str(args.refused_output or "").strip():
        refused_path = Path(args.refused_output)
        refused_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(refused_audit).to_parquet(refused_path, index=False)
        print(f"Wrote {len(refused_audit)} refused rows -> {refused_path}")

    if args.push_to:
        push_dataframe_to_hub(coerce_dataframe_for_hf(df), args.push_to, split=args.split)
        print(f"Pushed to {args.push_to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
