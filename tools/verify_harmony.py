from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
APP_PY = ROOT / "app.py"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def get_app_routes() -> set[str]:
    """Collect route paths from the mounted FastAPI app (all ``APIRouter`` includes)."""
    env = os.environ.copy()
    env.setdefault("JWT_SECRET", "harmony-scan-placeholder-32chars-min")
    env.setdefault("TOKEN_ENC_KEYS", "k:" + base64.b64encode(b"x" * 32).decode("ascii"))
    env.setdefault("DATABASE_URL", "postgresql://x:x@127.0.0.1:59999/x")
    env.setdefault("REDIS_URL", "")
    code = (
        "from app import app\n"
        "paths = sorted({getattr(r, 'path', '') for r in app.routes if getattr(r, 'path', None)})\n"
        "print('\\n'.join(paths))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        print("get_app_routes: import app failed:", proc.stderr or proc.stdout, file=sys.stderr)
        # Fallback: only direct @app routes in app.py (legacy behaviour)
        app_txt = read_text(APP_PY)
        pattern = re.compile(r'@app\.(?:get|post|put|patch|delete)\("([^"]+)"')
        return set(pattern.findall(app_txt))
    return {ln for ln in proc.stdout.splitlines() if ln}


def check_api_refs(routes: set[str]) -> dict:
    api_pattern = re.compile(r"/api/[A-Za-z0-9_\-./{}]+")
    missing: dict[str, set[str]] = {}
    checked = 0

    files = list(FRONTEND.rglob("*.html")) + list(FRONTEND.rglob("*.js"))
    for file_path in files:
        txt = read_text(file_path)
        for raw in api_pattern.findall(txt):
            endpoint = raw.split("?")[0].rstrip(".,;:!\"'")
            if "{" in endpoint or "}" in endpoint:
                continue
            if "${" in endpoint or "' +" in endpoint:
                continue
            low = endpoint.lower()
            # User requested excluding Facebook/Instagram checks.
            if "facebook" in low or "instagram" in low:
                continue
            checked += 1
            if endpoint in routes:
                continue
            # Treat collection bases as valid when concrete child routes exist.
            if endpoint.endswith("/"):
                if any(r.startswith(endpoint) for r in routes):
                    continue
            if any(r.startswith(endpoint + "/") for r in routes):
                continue
            if any(endpoint.startswith(r.rstrip("/") + "/") for r in routes):
                continue
            if endpoint not in routes:
                rel = str(file_path.relative_to(ROOT))
                missing.setdefault(endpoint, set()).add(rel)

    return {
        "checked_refs": checked,
        "missing_endpoints": {
            endpoint: sorted(list(files)) for endpoint, files in sorted(missing.items())
        },
    }


def check_links_and_scripts() -> dict:
    href_re = re.compile(r'href="([^"]+)"|href=\'([^\']+)\'')
    src_re = re.compile(r'src="([^"]+)"|src=\'([^\']+)\'')
    viewport_re = re.compile(r'<meta\s+name="viewport"', re.IGNORECASE)
    media_re = re.compile(r"@media\s*\(", re.IGNORECASE)

    missing_links = []
    missing_scripts = []
    missing_viewport = []
    no_media_queries = []
    checked_links = 0
    checked_scripts = 0

    html_files = list(FRONTEND.rglob("*.html"))
    for html in html_files:
        txt = read_text(html)
        if not viewport_re.search(txt):
            missing_viewport.append(str(html.relative_to(ROOT)))
        if "@media" not in txt and html.name not in {"index.html"}:
            # Heuristic only; index uses shared stylesheet.
            no_media_queries.append(str(html.relative_to(ROOT)))

        for m in href_re.findall(txt):
            href = m[0] or m[1]
            if not href:
                continue
            if "${" in href or "' +" in href or '"+' in href:
                continue
            if href.startswith(("http://", "https://", "mailto:", "tel:", "#", "javascript:")):
                continue
            if "/api/" in href:
                continue
            cleaned = href.split("?")[0].split("#")[0]
            if not cleaned:
                continue
            checked_links += 1
            target = (FRONTEND / cleaned.lstrip("/")) if cleaned.startswith("/") else (html.parent / cleaned)
            if not target.exists():
                missing_links.append((str(html.relative_to(ROOT)), cleaned))

        for m in src_re.findall(txt):
            src = m[0] or m[1]
            if not src:
                continue
            if "${" in src or "' +" in src or '"+' in src:
                continue
            if src.startswith(("http://", "https://", "data:", "//")):
                continue
            cleaned = src.split("?")[0].split("#")[0]
            if not cleaned:
                continue
            checked_scripts += 1
            target = (FRONTEND / cleaned.lstrip("/")) if cleaned.startswith("/") else (html.parent / cleaned)
            if not target.exists():
                missing_scripts.append((str(html.relative_to(ROOT)), cleaned))

    return {
        "checked_links": checked_links,
        "checked_scripts": checked_scripts,
        "missing_links": missing_links,
        "missing_scripts": missing_scripts,
        "missing_viewport": missing_viewport,
        "no_media_queries": no_media_queries,
    }


def check_python_syntax() -> dict:
    py_files = [
        p for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts and ".venv" not in p.parts
    ]
    errors = []
    for p in py_files:
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(p)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            errors.append({"file": str(p.relative_to(ROOT)), "error": (proc.stderr or proc.stdout).strip()})
    return {"python_files": len(py_files), "errors": errors}


def check_js_syntax() -> dict:
    js_files = list(FRONTEND.rglob("*.js"))
    errors = []
    for p in js_files:
        proc = subprocess.run(["node", "--check", str(p)], capture_output=True, text=True)
        if proc.returncode != 0:
            errors.append({"file": str(p.relative_to(ROOT)), "error": (proc.stderr or proc.stdout).strip()})
    return {"js_files": len(js_files), "errors": errors}


def main() -> int:
    routes = get_app_routes()
    report = {
        "route_count": len(routes),
        "api_refs": check_api_refs(routes),
        "links": check_links_and_scripts(),
        "python": check_python_syntax(),
        "javascript": check_js_syntax(),
    }
    out = ROOT / "tools" / "harmony_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {out}")
    print(f"Routes: {report['route_count']}")
    print(f"API refs checked: {report['api_refs']['checked_refs']}")
    print(f"Missing API endpoints: {len(report['api_refs']['missing_endpoints'])}")
    print(f"Missing href links: {len(report['links']['missing_links'])}")
    print(f"Missing script/src refs: {len(report['links']['missing_scripts'])}")
    print(f"Python syntax errors: {len(report['python']['errors'])}")
    print(f"JS syntax errors: {len(report['javascript']['errors'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
