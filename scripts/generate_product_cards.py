#!/usr/bin/env python
"""
CLI wrapper for ``services.product_card_art``. Defaults to repo ``frontend/images``.

Usage (from repo root):
  python scripts/generate_product_cards.py
  python scripts/generate_product_cards.py --out ./frontend/images --cloud-icon ./cloud.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stages.entitlements import TIER_CONFIG, TOPUP_PRODUCTS  # noqa: E402

from services.product_card_art import generate_all  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate UploadM8 product card PNGs.")
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "frontend" / "images",
        help="Output directory for PNG files",
    )
    ap.add_argument(
        "--cloud-icon",
        type=Path,
        default=None,
        help="Optional cloud PNG path; else a vector-style cloud is drawn",
    )
    args = ap.parse_args()
    generate_all(TIER_CONFIG, TOPUP_PRODUCTS, args.out, args.cloud_icon)


if __name__ == "__main__":
    main()
