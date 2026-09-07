"""Degenerate-output guards for published titles and captions.

LLMs occasionally emit *degenerate* / stuck-decoding output — e.g. a title that
reads ``"the the the the the"`` — which our cliché / formula / evidence guards do
not catch (it is short, non-generic, and contains no stub template). This module
provides a shared, idempotent sanitizer used both at the hydration rewrite gate
(``services/hydration_enforcer.py``) and as a publish-time safety net
(``stages/context.py`` effective getters) so token-stutter can never reach a
platform.

Design goals:
  * ``collapse_repeated_words`` is *always* safe to run on real copy (it only
    collapses immediate word/short-phrase repetitions, leaving normal prose
    untouched).
  * ``is_degenerate_publish_text`` flags text that is *mostly* repetition so
    callers can prefer an evidence anchor instead.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "collapse_repeated_words",
    "is_degenerate_publish_text",
    "sanitize_publish_text",
]

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _norm_ws(text: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", text.replace("\u00a0", " ")).strip()


def collapse_repeated_words(text: Any, *, max_run: int = 1) -> str:
    """Collapse immediate repeated words / short phrases.

    ``"the the the the the"`` -> ``"the"``; ``"go go go"`` -> ``"go"`` only when
    the run exceeds ``max_run`` copies. Case-insensitive match, first casing
    preserved. Normal prose (``"the road less traveled"``) is untouched.

    Also collapses immediate 2- and 3-word phrase repetitions
    (``"on the road on the road"`` -> ``"on the road"``).
    """
    raw = _norm_ws(str(text or ""))
    if not raw:
        return ""

    # Preserve newlines: process line by line so we do not merge separate lines.
    out_lines = []
    for line in raw.split("\n"):
        line = _norm_ws(line)
        if not line:
            out_lines.append("")
            continue
        tokens = line.split(" ")
        # Immediate single-word repetition.
        collapsed: list[str] = []
        run_key = None
        run_len = 0
        for tok in tokens:
            key = tok.lower()
            if key == run_key:
                run_len += 1
                if run_len > max_run:
                    continue
            else:
                run_key = key
                run_len = 1
            collapsed.append(tok)
        # Immediate short-phrase repetition (2- and 3-word windows).
        collapsed = _collapse_phrase_runs(collapsed, window=3)
        collapsed = _collapse_phrase_runs(collapsed, window=2)
        out_lines.append(" ".join(collapsed))
    return "\n".join(out_lines).strip()


def _collapse_phrase_runs(tokens: list[str], *, window: int) -> list[str]:
    """Collapse immediate, adjacent, identical ``window``-word phrase repeats.

    ``["on","the","road","on","the","road"]`` (window=3) -> ``["on","the","road"]``.
    Only exact back-to-back repetitions of the same window are removed; ordinary
    prose is left untouched.
    """
    n = len(tokens)
    if n < window * 2:
        return list(tokens)
    out: list[str] = []
    i = 0
    while i < n:
        # Try to detect a repeated phrase starting exactly at i.
        if i + 2 * window <= n:
            first = [tokens[i + k].lower() for k in range(window)]
            j = i + window
            while j + window <= n and [tokens[j + k].lower() for k in range(window)] == first:
                j += window
            if j > i + window:
                # Emit one copy of the phrase, skip the rest of the run.
                out.extend(tokens[i : i + window])
                i = j
                continue
        out.append(tokens[i])
        i += 1
    return out


def is_degenerate_publish_text(text: Any, *, min_tokens: int = 3) -> bool:
    """True when text is dominated by repetition / has almost no lexical variety.

    Heuristics (any one triggers):
      * A single word repeated 3+ times in a row (``"the the the"``).
      * >=4 alpha tokens but <=1 unique alpha token (all the same word).
      * >=6 alpha tokens with unique-token ratio < 0.34 (mostly repetition).
    """
    raw = _norm_ws(str(text or "")).lower()
    if not raw:
        return False

    # Immediate triple (or more) repetition of any word.
    if re.search(r"\b([^\W\d_]+)(?:\s+\1){2,}\b", raw, re.UNICODE):
        return True

    words = _WORD_RE.findall(raw)
    if len(words) < min_tokens:
        return False

    unique = set(words)
    if len(words) >= 4 and len(unique) <= 1:
        return True
    if len(words) >= 6 and (len(unique) / len(words)) < 0.34:
        return True
    return False


def sanitize_publish_text(text: Any) -> str:
    """Idempotent publish-time cleanup: collapse immediate word/phrase stutter.

    Safe to call on already-clean copy. Does NOT rewrite meaning — for fully
    degenerate text prefer replacing with an evidence anchor at the caller.
    """
    return collapse_repeated_words(text, max_run=1)
