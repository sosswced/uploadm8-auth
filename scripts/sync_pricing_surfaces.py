#!/usr/bin/env python3
"""
Sync PUT/AIC debit numbers into frontend surfaces after weight/pricing changes.

Source of truth: services/pricing_surfaces.py (reads SERVICE_WEIGHTS, PUT_COST_DEFAULTS,
Pikzels/Studio estimators).

Usage:
  python scripts/sync_pricing_surfaces.py           # write generated JS + patch markers
  python scripts/sync_pricing_surfaces.py --check   # exit 1 if drift (CI)
  python scripts/sync_pricing_surfaces.py --json

After changing stages/ai_service_costs.py SERVICE_WEIGHTS, billing PUT defaults,
or thumbnail_studio debit tables, run this script (and deploy frontend).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.pricing_surfaces import (  # noqa: E402
    build_pricing_surfaces_snapshot,
    render_generated_js,
)

GENERATED_JS = ROOT / "frontend" / "js" / "pricing-surfaces.generated.js"

# HTML/JS files with <!-- um8-pricing-sync:KEY --> ... <!-- /um8-pricing-sync:KEY -->
MARKER_FILES = [
    ROOT / "frontend" / "guide.html",
    ROOT / "frontend" / "settings.html",
    ROOT / "frontend" / "how-it-works.html",
]

MARKER_RE = re.compile(
    r"(<!--\s*um8-pricing-sync:([a-z0-9_-]+)\s*-->)(.*?)(<!--\s*/um8-pricing-sync:\2\s*-->)",
    re.DOTALL | re.IGNORECASE,
)


def _snippet_map(snap: dict) -> dict[str, str]:
    copy = snap.get("copy") or {}
    html = snap.get("html_snippets") or {}
    return {
        "guide_put_table_rows": html.get("guide_put_table_rows", ""),
        "guide_aic_table_rows": html.get("guide_aic_table_rows", ""),
        "guide_aic_desc": copy.get("guide_aic_desc", ""),
        "settings_put_blurb": copy.get("settings_put_blurb", ""),
        "settings_aic_blurb": copy.get("settings_aic_blurb", ""),
        "faq_put_aic": copy.get("faq_put_aic", ""),
        "whisper_note": copy.get("whisper_note", ""),
    }


def patch_markers(text: str, snippets: dict[str, str]) -> tuple[str, list[str]]:
    touched: list[str] = []

    def repl(m: re.Match) -> str:
        key = m.group(2)
        if key not in snippets:
            return m.group(0)
        touched.append(key)
        body = snippets[key]
        # Keep one newline around body for readability
        return f"{m.group(1)}\n{body}\n{m.group(4)}"

    return MARKER_RE.sub(repl, text), touched


def write_all(*, check: bool) -> dict:
    snap = build_pricing_surfaces_snapshot()
    js = render_generated_js(snap)
    snippets = _snippet_map(snap)
    report: dict = {
        "calibration": snap.get("calibration"),
        "estimates": snap.get("estimates"),
        "generated_js": str(GENERATED_JS.relative_to(ROOT)).replace("\\", "/"),
        "js_changed": False,
        "files": [],
        "ok": True,
    }

    if check:
        if not GENERATED_JS.is_file() or GENERATED_JS.read_text(encoding="utf-8") != js:
            report["ok"] = False
            report["js_changed"] = True
            report["error"] = "pricing-surfaces.generated.js is stale — run sync_pricing_surfaces.py"
    else:
        GENERATED_JS.parent.mkdir(parents=True, exist_ok=True)
        old = GENERATED_JS.read_text(encoding="utf-8") if GENERATED_JS.is_file() else None
        GENERATED_JS.write_text(js, encoding="utf-8", newline="\n")
        report["js_changed"] = old != js

    for path in MARKER_FILES:
        if not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8")
        if "um8-pricing-sync:" not in raw:
            report["files"].append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "skipped": True})
            continue
        new, keys = patch_markers(raw, snippets)
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        changed = new != raw
        if check:
            if changed:
                report["ok"] = False
                report["files"].append({"path": rel, "stale": True, "keys": keys})
            else:
                report["files"].append({"path": rel, "ok": True, "keys": keys or _list_keys(raw)})
        else:
            if changed:
                path.write_text(new, encoding="utf-8", newline="\n")
            report["files"].append({"path": rel, "changed": changed, "keys": keys})

    return report


def _list_keys(text: str) -> list[str]:
    return sorted(set(MARKER_RE.findall(text) and [m[1] for m in MARKER_RE.finditer(text)]))


def main() -> int:
    ap = argparse.ArgumentParser(description="Sync PUT/AIC pricing surfaces into frontend")
    ap.add_argument("--check", action="store_true", help="Fail if generated artifacts drift")
    ap.add_argument("--json", action="store_true", help="Print JSON report")
    args = ap.parse_args()
    report = write_all(check=args.check)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"calibration={report.get('calibration')} ok={report.get('ok')}")
        print(f"generated={report.get('generated_js')} changed={report.get('js_changed')}")
        for f in report.get("files") or []:
            print(f"  {f}")
        if report.get("error"):
            print(report["error"], file=sys.stderr)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
