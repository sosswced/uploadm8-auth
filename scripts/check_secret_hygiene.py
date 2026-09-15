#!/usr/bin/env python3
"""Fail if tracked files look like committed secrets. Values are never printed."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# High-confidence secret shapes in tracked content (not .env.example placeholders).
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private_key_pem", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("stripe_live_sk", re.compile(r"\bsk_live_[0-9a-zA-Z]{16,}\b")),
    ("stripe_live_rk", re.compile(r"\brk_live_[0-9a-zA-Z]{16,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_pat", re.compile(r"\bghp_[0-9A-Za-z]{20,}\b")),
    ("slack_bot", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b")),
]

_SKIP_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".mp4",
    ".webm",
    ".woff",
    ".woff2",
    ".ttf",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
)

_SKIP_DIRS = (
    "node_modules/",
    ".git/",
    ".venv/",
    "__pycache__/",
    "data/ml/",
)


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "-z"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    paths: list[Path] = []
    for raw in out.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8", errors="replace").replace("\\", "/")
        if any(rel.startswith(d) for d in _SKIP_DIRS):
            continue
        if rel.lower().endswith(_SKIP_SUFFIXES):
            continue
        paths.append(_REPO_ROOT / rel)
    return paths


def _scan_text(rel: str, text: str) -> list[str]:
    hits: list[str] = []
    # Never treat the example env as a leak when it only has placeholders.
    if rel.endswith(".env.example"):
        return hits
    for name, pat in _PATTERNS:
        if pat.search(text):
            hits.append(name)
    # Only flag env-style assignment files (not Python os.environ.get bindings).
    if rel.endswith((".env", ".ini", ".toml", ".yml", ".yaml", ".json", ".md", ".txt", ".ps1", ".sh")):
        if re.search(r"(?m)^TOKEN_ENC_KEYS\s*=\s*.{40,}", text) and "change-me" not in text.lower():
            hits.append("token_enc_keys_assignment")
        if re.search(r"(?m)^JWT_SECRET\s*=\s*.{24,}", text):
            if "change-me" not in text.lower() and "example" not in rel.lower():
                hits.append("jwt_secret_assignment")
    return hits


def main() -> int:
    # Ensure local env-named files are not staged/tracked.
    bad_names = []
    for p in _tracked_files():
        rel = p.relative_to(_REPO_ROOT).as_posix()
        base = p.name.lower()
        if base in ("uploadm8-auth.env", ".env") or (
            base.endswith(".env") and base != ".env.example"
        ):
            bad_names.append(rel)
        if (base.endswith(".local.js") and "secrets" in base) or base == "conversion-pixels.local.js":
            bad_names.append(rel)

    leaks: list[tuple[str, str]] = []
    for p in _tracked_files():
        rel = p.relative_to(_REPO_ROOT).as_posix()
        try:
            raw = p.read_bytes()
        except OSError:
            continue
        if b"\0" in raw[:4096]:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for hit in _scan_text(rel, text):
            leaks.append((rel, hit))

    ok = True
    if bad_names:
        ok = False
        print("FAIL: secret-named files are tracked:")
        for rel in sorted(set(bad_names)):
            print(f"  - {rel}")
    if leaks:
        ok = False
        print("FAIL: high-confidence secret patterns in tracked files (paths only):")
        for rel, hit in sorted(set(leaks)):
            print(f"  - {rel} [{hit}]")
    if ok:
        print("OK: secret hygiene scan passed (no tracked high-risk patterns).")
        return 0
    print("Fix: untrack the file(s), rotate any exposed credentials in host secrets, re-scan.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
