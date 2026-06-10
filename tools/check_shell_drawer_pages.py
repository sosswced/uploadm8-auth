#!/usr/bin/env python3
"""Verify every app-shell page has consistent mobile drawer markup and boot scripts."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"

REQUIRED = (
    ('app-layout', r'class="app-layout"|<div class="app-layout"'),
    ('sidebar', r'id="sidebar"'),
    ('shell-sidebar-boot', r'shell-sidebar-boot\.js'),
    ('sidebarOverlay', r'id="sidebarOverlay"'),
    ('menuToggle', r'id="menuToggle"'),
    ('app.js', r'<script[^>]+src="app\.js"'),
)

FORBIDDEN = (
    ('duplicate shared-sidebar', r'<script[^>]+src="shared-sidebar\.js"'),
    ('broken toggleSidebar fallback', r'toggleSidebar\s*=\s*window\.toggleSidebar\s*\|\|\s*function'),
    ('settings overlay hack', r'um8EnsureSidebarOverlayClosed'),
)


def check_page(path: Path) -> list[str]:
    text = path.read_text(encoding='utf-8', errors='replace')
    errors: list[str] = []
    for label, pattern in REQUIRED:
        if not re.search(pattern, text):
            errors.append(f'missing {label}')
    for label, pattern in FORBIDDEN:
        if re.search(pattern, text):
            errors.append(f'forbidden: {label}')
    return errors


def main() -> int:
    pages = sorted(FRONTEND.glob('*.html'))
    failures: dict[str, list[str]] = {}
    checked = 0
    for page in pages:
        text = page.read_text(encoding='utf-8', errors='replace')
        if 'app-layout' not in text:
            continue
        checked += 1
        errs = check_page(page)
        if errs:
            failures[page.name] = errs

    print(f'Checked {checked} app-shell pages under {FRONTEND}')
    if failures:
        print('FAILURES:')
        for name, errs in sorted(failures.items()):
            print(f'  {name}:')
            for e in errs:
                print(f'    - {e}')
        return 1
    print('All app-shell pages pass shell drawer structure checks.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
