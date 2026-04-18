"""Strip remaining onclick= patterns (run: python tools/phase3_onclick.py)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "frontend"

SUBS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'\s+onclick="unsubscribeAll\(\)"'), ' data-um8-fn="unsubscribeAll"'),
    (re.compile(r'\s+onclick="savePreferences\(\)"'), ' data-um8-fn="savePreferences"'),
    (re.compile(r'\s+onclick="resendEmail\(\)"'), ' data-um8-fn="resendEmail"'),
    (re.compile(r'\s+onclick="updatePendingEmail\(\)"'), ' data-um8-fn="updatePendingEmail"'),
    (re.compile(r'\s+onclick="handleReset\(\)"'), ' data-um8-fn="handleReset"'),
    (re.compile(r'\s+onclick="toggleMobileMenu\(\)"'), ' data-um8-fn="toggleMobileMenu"'),
    (re.compile(r'\s+onclick="toggleEditMode\(\)"'), ' data-um8-fn="toggleEditMode"'),
    (re.compile(r'\s+onclick="resetLayout\(\)"'), ' data-um8-fn="resetLayout"'),
    (re.compile(r'\s+onclick="refreshKPIsWithSync\(\)"'), ' data-um8-fn="refreshKPIsWithSync"'),
    (re.compile(r'\s+onclick="applyCustomRange\(\)"'), ' data-um8-fn="applyCustomRange"'),
    (re.compile(r'\s+onclick="exportJSON\(\)"'), ' data-um8-fn="exportJSON"'),
    (re.compile(r'\s+onclick="resetAll\(\)"'), ' data-um8-fn="resetAll"'),
    (re.compile(r'\s+onclick="runCalculator\(\)"'), ' data-um8-fn="runCalculator"'),
    (re.compile(r'\s+onclick="runEnterpriseQuote\(\)"'), ' data-um8-fn="runEnterpriseQuote"'),
    (
        re.compile(r'\s+onclick="window\.fetchPricingAndRun\(true\)"'),
        ' data-um8-fn="fetchPricingAndRun" data-um8-args="[true]"',
    ),
    (re.compile(r'\s+onclick="window\.print\(\)"'), ' data-um8-fn="um8WindowPrint"'),
    (
        re.compile(r'onclick="togglePw\(\'newPassword\',\s*this\)"'),
        'data-um8-fn="togglePw" data-um8-arg="newPassword"',
    ),
    (
        re.compile(r'onclick="togglePw\(\'confirmPassword\',\s*this\)"'),
        'data-um8-fn="togglePw" data-um8-arg="confirmPassword"',
    ),
]


def patch_text(t: str) -> str:
    for rx, rep in SUBS:
        t = rx.sub(rep, t)
    t = re.sub(
        r"onclick=\"setColor\('([a-z]+)',\s*'([^']+)'\)\"",
        r'data-um8-fn="setColor" data-um8-args=\'["\1","\2"]\'',
        t,
    )
    t = re.sub(
        r"onclick=\"loadPreset\('([a-z]+)'\)\"",
        r'data-um8-fn="loadPreset" data-um8-arg="\1"',
        t,
    )
    t = re.sub(
        r"onclick=\"switchCalcTab\('([a-z]+)',\s*event\)\"",
        r'data-um8-fn="switchCalcTab" data-um8-arg="\1"',
        t,
    )
    t = re.sub(
        r"onclick=\"toggleSection\('([^']+)'\)\"",
        r'data-um8-fn="toggleSection" data-um8-arg="\1"',
        t,
    )
    return t


def main() -> int:
    paths = list(ROOT.rglob("*.html"))
    for p in paths:
        raw = p.read_text(encoding="utf-8")
        new = patch_text(raw)
        if new != raw:
            p.write_text(new, encoding="utf-8")
            print("updated", p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
