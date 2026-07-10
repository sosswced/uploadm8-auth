"""
Q&A over upload hydration evidence (accuracy ladder Tier 4).

Answers only from persisted artifacts — hydration_report, place_evidence,
shot_list, grounding, M8 claims — with ``evidence_ids`` citations.
No free-form LLM inventing facts outside the evidence index.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


def _as_dict(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            d = json.loads(s)
            return dict(d) if isinstance(d, dict) else {}
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
    return {}


def _as_list(raw: Any) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return list(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            v = json.loads(s)
            return list(v) if isinstance(v, list) else []
        except (json.JSONDecodeError, TypeError, ValueError):
            return []
    return []


def build_evidence_index(output_artifacts: Any) -> List[Dict[str, Any]]:
    """Flatten persisted artifacts into citation-ready evidence rows."""
    arts = _as_dict(output_artifacts)
    rows: List[Dict[str, Any]] = []

    def add(eid: str, text: str, lane: str) -> None:
        t = str(text or "").strip()
        if not t or len(t) < 2:
            return
        rows.append({"id": eid, "text": t[:500], "lane": lane})

    hr = _as_dict(arts.get("hydration_report"))
    ev = _as_dict(hr.get("evidence") or arts.get("hydration_evidence"))
    for k, v in ev.items():
        if isinstance(v, (list, tuple)):
            for i, item in enumerate(v[:12]):
                add(f"hydration.{k}.{i}", str(item), "hydration")
        elif v is not None and str(v).strip():
            add(f"hydration.{k}", str(v), "hydration")

    story = hr.get("hydration_story") or arts.get("hydration_story")
    if story:
        add("hydration.story", str(story), "hydration")

    pe = _as_dict(arts.get("place_evidence_v1") or arts.get("place_evidence"))
    for key in ("landmarks", "beaches", "monuments", "stadiums", "sports_teams", "license_plates", "cities"):
        for i, item in enumerate(pe.get(key) or [] if isinstance(pe.get(key), list) else []):
            if isinstance(item, dict):
                name = item.get("name") or item.get("text") or item.get("value")
            else:
                name = item
            add(f"place.{key}.{i}", str(name), "place")

    for i, shot in enumerate(_as_list(arts.get("shot_list_v1"))[:40]):
        if isinstance(shot, dict):
            bit = shot.get("summary") or shot.get("label") or shot.get("text") or json.dumps(shot, default=str)[:200]
        else:
            bit = str(shot)
        add(f"shot.{i}", bit, "shot_list")

    for i, claim in enumerate(_as_list(arts.get("m8_claims_v1"))[:30]):
        if isinstance(claim, dict):
            text = claim.get("text") or claim.get("claim") or ""
            eids = claim.get("evidence_ids") or []
            add(f"claim.{i}", text, "m8_claim")
            if eids and text:
                rows[-1]["linked_evidence_ids"] = [str(x) for x in eids[:8]]
        else:
            add(f"claim.{i}", str(claim), "m8_claim")

    gs = _as_dict(arts.get("grounding_score_v1"))
    if gs.get("matched_clues"):
        for i, clue in enumerate(gs.get("matched_clues") or [] if isinstance(gs.get("matched_clues"), list) else []):
            add(f"grounding.matched.{i}", str(clue), "grounding")

    # Deduplicate by (lane, lower text)
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = (r["lane"], r["text"].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


_Q_WHERE = re.compile(r"\b(where|location|place|city|beach|landmark|road|stadium)\b", re.I)
_Q_WHAT = re.compile(r"\b(what|who|which|describe|about|happening|scene)\b", re.I)
_Q_SPEED = re.compile(r"\b(speed|mph|fast|how fast)\b", re.I)
_Q_SAID = re.compile(r"\b(say|said|speak|speech|transcript|audio|words)\b", re.I)


def _score_row(question: str, row: Dict[str, Any]) -> float:
    q_tokens = set(re.findall(r"[a-z0-9]{3,}", (question or "").lower()))
    t_tokens = set(re.findall(r"[a-z0-9]{3,}", (row.get("text") or "").lower()))
    if not q_tokens or not t_tokens:
        overlap = 0.0
    else:
        overlap = len(q_tokens & t_tokens) / float(max(len(q_tokens), 1))

    lane = str(row.get("lane") or "")
    boost = 0.0
    if _Q_WHERE.search(question or "") and lane in ("place", "hydration"):
        boost += 0.25
    if _Q_SPEED.search(question or "") and ("mph" in (row.get("text") or "").lower() or "speed" in lane):
        boost += 0.3
    if _Q_SAID.search(question or "") and ("transcript" in lane or "transcript" in (row.get("id") or "")):
        boost += 0.25
    if _Q_WHAT.search(question or "") and lane in ("shot_list", "m8_claim", "hydration"):
        boost += 0.1
    return overlap + boost


def answer_from_evidence(
    question: str,
    output_artifacts: Any,
    *,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Deterministic retrieval answer with citations.

    Returns ``{answer, evidence_ids, citations, grounding_ok, status}``.
    """
    q = (question or "").strip()
    if not q:
        return {
            "answer": "",
            "evidence_ids": [],
            "citations": [],
            "grounding_ok": False,
            "status": "empty_question",
        }

    index = build_evidence_index(output_artifacts)
    if not index:
        return {
            "answer": "No hydration evidence is available for this upload yet.",
            "evidence_ids": [],
            "citations": [],
            "grounding_ok": False,
            "status": "no_evidence",
        }

    ranked = sorted(
        (( _score_row(q, row), row) for row in index),
        key=lambda x: -x[0],
    )
    hits = [(s, r) for s, r in ranked if s > 0.05][: max(1, min(int(top_k), 8))]
    if not hits:
        return {
            "answer": (
                "I could not find evidence that answers that question. "
                "Try asking about place, speed, speech, or what is visible in the clip."
            ),
            "evidence_ids": [],
            "citations": [],
            "grounding_ok": False,
            "status": "no_match",
        }

    citations = [
        {"id": r["id"], "text": r["text"], "lane": r["lane"], "score": round(float(s), 4)}
        for s, r in hits
    ]
    evidence_ids = [c["id"] for c in citations]
    # Compose a short grounded answer from top hits (no LLM).
    bits = [c["text"] for c in citations[:3]]
    answer = "Based on upload evidence: " + " · ".join(bits)
    if len(answer) > 800:
        answer = answer[:797] + "..."

    return {
        "answer": answer,
        "evidence_ids": evidence_ids,
        "citations": citations,
        "grounding_ok": True,
        "status": "ok",
        "evidence_count": len(index),
    }


async def ask_upload_question(
    pool: Any,
    upload_id: str,
    user_id: str,
    question: str,
) -> Dict[str, Any]:
    """Load artifacts for an owned upload and answer with citations."""
    if pool is None:
        return {
            "answer": "",
            "evidence_ids": [],
            "citations": [],
            "grounding_ok": False,
            "status": "no_db",
        }
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, output_artifacts
              FROM uploads
             WHERE id = $1::uuid AND user_id = $2::uuid
            """,
            upload_id,
            user_id,
        )
    if not row:
        return {
            "answer": "",
            "evidence_ids": [],
            "citations": [],
            "grounding_ok": False,
            "status": "not_found",
        }
    result = answer_from_evidence(question, row["output_artifacts"])
    result["upload_id"] = str(row["id"])
    return result


__all__ = [
    "answer_from_evidence",
    "ask_upload_question",
    "build_evidence_index",
]
