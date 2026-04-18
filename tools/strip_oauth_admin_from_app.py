"""Remove /api/oauth and /api/admin route blocks from app.py (byte-safe)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
p = ROOT / "app.py"
text = p.read_text(encoding="utf-8")

ranges: list[tuple[int, int]] = []

oauth_start = text.find('@app.get("/api/oauth/{platform}/start")')
if oauth_start == -1:
    raise SystemExit("oauth start not found")
oauth_end = text.find("\nasync def _thumbnail_channel_memory_hint", oauth_start)
if oauth_end == -1:
    raise SystemExit("oauth end marker not found")
ranges.append((oauth_start, oauth_end + 1))  # keep leading newline before async def

pat = re.compile(r'^@app\.(get|post|put|delete|patch)\(\"([^\"]+)\"', re.MULTILINE)
matches = list(pat.finditer(text))
for i, m in enumerate(matches):
    path = m.group(2)
    if "/api/admin/" not in path and not path.startswith("/api/admin"):
        continue
    start = m.start()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    ranges.append((start, end))

ranges.sort()
merged: list[list[int]] = []
for s, e in ranges:
    if not merged or s > merged[-1][1]:
        merged.append([s, e])
    else:
        merged[-1][1] = max(merged[-1][1], e)

for s, e in reversed(merged):
    text = text[:s] + text[e:]

p.write_text(text, encoding="utf-8")
print("stripped oauth + admin routes from app.py, merged intervals:", merged)
