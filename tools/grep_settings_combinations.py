#!/usr/bin/env python3
"""Dump or grep upload preference combinatorics signatures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.settings_combinatorics import (
    BOOLEAN_PREF_KEYS,
    PUBLIC_UPLOAD_TIERS,
    boolean_combination_count,
    combo_dict_from_index,
    combo_signature,
    get_tier_effective_baseline,
    validate_settings_combination,
)


def main() -> int:
    p = argparse.ArgumentParser(description="UploadM8 settings combinatorics grep tool")
    p.add_argument("--index", type=int, help="Print one combo by index (0 .. 2^N-1)")
    p.add_argument("--grep", type=str, help="Filter signatures containing substring")
    p.add_argument("--tier", choices=PUBLIC_UPLOAD_TIERS, help="Validate combos for one tier")
    p.add_argument("--validate", action="store_true", help="Run validate_settings_combination")
    p.add_argument("--baseline", action="store_true", help="Print tier baselines (empty prefs)")
    p.add_argument("--max", type=int, default=0, help="Limit output rows (0 = all)")
    args = p.parse_args()

    if args.baseline:
        for tier in PUBLIC_UPLOAD_TIERS:
            b = get_tier_effective_baseline(tier)
            on = [k for k in BOOLEAN_PREF_KEYS if b.get(k)]
            print(f"{tier}: {len(on)} toggles on by default — {', '.join(on[:8])}{'...' if len(on)>8 else ''}")
        return 0

    total = boolean_combination_count()
    print(f"# {total} boolean combinations × {len(PUBLIC_UPLOAD_TIERS)} tiers", file=sys.stderr)

    tiers = (args.tier,) if args.tier else PUBLIC_UPLOAD_TIERS
    limit = args.max or total
    count = 0

    if args.index is not None:
        indices = [args.index]
    else:
        indices = range(total)

    for idx in indices:
        if idx < 0 or idx >= total:
            print(f"index out of range: {idx}", file=sys.stderr)
            return 1
        combo = combo_dict_from_index(idx)
        sig = combo_signature(combo)
        if args.grep and args.grep not in sig:
            continue
        for tier in tiers:
            if args.validate:
                try:
                    validate_settings_combination(combo, tier=tier)
                    status = "OK"
                except Exception as e:
                    status = f"FAIL {e}"
            else:
                status = "—"
            print(f"idx={idx}\ttier={tier}\t{status}\t{sig}")
            count += 1
            if count >= limit:
                return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
