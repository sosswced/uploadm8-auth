"""
Hero-fact F1 eval vs teacher labels (accuracy ladder trust layer).

Teacher facts come from content_identity / gold fixtures. Student or candidate
facts are scored with class-aware token Jaccard matching — never used to replace
identity or grounding.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def normalize_hero_text(text: Any) -> str:
    toks = _TOKEN_RE.findall(str(text or "").lower())
    return " ".join(toks)


def _token_set(text: str) -> set:
    return set(_TOKEN_RE.findall(normalize_hero_text(text)))


def fact_match_score(teacher: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    """1.0 exact class+text, else Jaccard on tokens if class matches (or either blank)."""
    tc = str(teacher.get("class") or "").strip().lower()
    cc = str(candidate.get("class") or "").strip().lower()
    tt = normalize_hero_text(teacher.get("text"))
    ct = normalize_hero_text(candidate.get("text"))
    if not tt or not ct:
        return 0.0
    if tc and cc and tc != cc:
        return 0.0
    if tt == ct:
        return 1.0
    a, b = _token_set(tt), _token_set(ct)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def hero_fact_f1(
    teacher_facts: Sequence[Dict[str, Any]],
    candidate_facts: Sequence[Dict[str, Any]],
    *,
    match_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Greedy one-to-one matching by best score above threshold.
    Returns precision, recall, f1.
    """
    teachers = [f for f in teacher_facts if isinstance(f, dict) and str(f.get("text") or "").strip()]
    cands = [f for f in candidate_facts if isinstance(f, dict) and str(f.get("text") or "").strip()]
    if not teachers and not cands:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matched": 0.0}
    if not teachers:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "matched": 0.0}
    if not cands:
        return {"precision": 1.0 if not teachers else 0.0, "recall": 0.0, "f1": 0.0, "matched": 0.0}

    pairs: List[Tuple[float, int, int]] = []
    for i, t in enumerate(teachers):
        for j, c in enumerate(cands):
            s = fact_match_score(t, c)
            if s >= match_threshold:
                pairs.append((s, i, j))
    pairs.sort(reverse=True)
    used_t, used_c = set(), set()
    matched = 0
    for _s, i, j in pairs:
        if i in used_t or j in used_c:
            continue
        used_t.add(i)
        used_c.add(j)
        matched += 1

    precision = matched / len(cands) if cands else 0.0
    recall = matched / len(teachers) if teachers else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "matched": float(matched),
    }


def score_gold_clip(clip: Dict[str, Any], *, match_threshold: float = 0.5) -> Dict[str, float]:
    teacher = clip.get("teacher_hero_facts") or []
    cand = clip.get("student_or_candidate_facts") or []
    return hero_fact_f1(teacher, cand, match_threshold=match_threshold)


__all__ = [
    "normalize_hero_text",
    "fact_match_score",
    "hero_fact_f1",
    "score_gold_clip",
]
