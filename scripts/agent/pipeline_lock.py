"""Exclusive lock — do not run checklist + overnight E2E against :8000 in parallel."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "tests" / "e2e" / "artifacts" / ".pipeline.lock"
DEFAULT_STALE_S = 4 * 3600


def acquire(name: str, *, stale_s: int = DEFAULT_STALE_S) -> None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.is_file():
        age = time.time() - LOCK_PATH.stat().st_mtime
        holder = LOCK_PATH.read_text(encoding="utf-8", errors="replace").strip()
        if age < stale_s:
            raise RuntimeError(
                f"Pipeline lock held by: {holder or 'unknown'}. "
                "Do not run checklist and overnight E2E in parallel against the same API."
            )
        LOCK_PATH.unlink(missing_ok=True)
    LOCK_PATH.write_text(
        f"{name} pid={os.getpid()} started={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n",
        encoding="utf-8",
    )


def release() -> None:
    if LOCK_PATH.is_file():
        LOCK_PATH.unlink(missing_ok=True)


def status() -> dict:
    if not LOCK_PATH.is_file():
        return {"locked": False, "path": str(LOCK_PATH)}
    text = LOCK_PATH.read_text(encoding="utf-8", errors="replace").strip()
    return {
        "locked": True,
        "path": str(LOCK_PATH),
        "holder": text,
        "age_s": int(time.time() - LOCK_PATH.stat().st_mtime),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="UploadM8 pipeline exclusive lock")
    parser.add_argument("action", choices=("acquire", "release", "status"))
    parser.add_argument("--name", default="pipeline")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "acquire":
            acquire(args.name)
            out = {"ok": True, "acquired": args.name}
        elif args.action == "release":
            release()
            out = {"ok": True, "released": True}
        else:
            out = {"ok": True, **status()}
    except RuntimeError as e:
        out = {"ok": False, "error": str(e), **status()}
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(out["error"], file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
