"""
Push merged catalog metadata + product images to Stripe (Product.modify).

Requires ``STRIPE_SECRET_KEY`` and ``frontend/images/stripe_catalog_manifest.json``
(unless ``STRIPE_CATALOG_MANIFEST`` points elsewhere). Skips entries without a
real ``stripe_product_id`` (e.g. placeholders starting with ``REPLACE_``).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import stripe

from core.config import FRONTEND_STATIC_DIR, STRIPE_SECRET_KEY

logger = logging.getLogger("uploadm8-api")

TAX_CODE = "txcd_10103001"

_MONTHLY_LOOKUP = {
    "creator_lite": "uploadm8_creatorlite_monthly",
    "creator_pro": "uploadm8_creatorpro_monthly",
    "studio": "uploadm8_studio_monthly",
    "agency": "uploadm8_agency_monthly",
}


def default_manifest_path() -> Path:
    raw = os.environ.get("STRIPE_CATALOG_MANIFEST", "").strip()
    if raw:
        return Path(raw)
    return FRONTEND_STATIC_DIR / "images" / "stripe_catalog_manifest.json"


def _queue_depth_str(cfg: dict[str, Any]) -> str:
    qd = int(cfg.get("queue_depth", 0) or 0)
    return "unlimited" if qd >= 9999 else str(qd)


def _statement_descriptor(prefix: str, name_part: str = "") -> str:
    """Stripe statement_descriptor: max 22 chars, alphanumeric only."""
    raw = (prefix + name_part).upper()
    s = "".join(c for c in raw if c.isalnum())
    return s[:22]


def subscription_product_fields(tier_slug: str, cfg: dict[str, Any]) -> Dict[str, Any]:
    name = str(cfg.get("name", tier_slug))
    pm = int(cfg.get("put_monthly", 0) or 0)
    am = int(cfg.get("aic_monthly", 0) or 0)
    ma = int(cfg.get("max_accounts", 0) or 0)
    qh = int(cfg.get("lookahead_hours", 0) or 0)
    qd_s = _queue_depth_str(cfg)
    desc = (
        f"UploadM8 {name} — monthly SaaS subscription. Includes {pm:,} publishing credits (PUT) "
        f"and {am:,} AI credits (AIC) per billing cycle, up to {ma} connected accounts total, "
        f"queue depth {qd_s}, {qh}-hour scheduling lookahead. "
        f"Digital service via uploadm8.com. Renews monthly; cancel at period end."
    )
    if len(desc) > 4500:
        desc = desc[:4497] + "..."
    lk = _MONTHLY_LOOKUP.get(tier_slug, f"uploadm8_{tier_slug}_monthly")
    meta = {
        "type": "subscription",
        "lookup_key": lk,
        "tier": tier_slug,
        "put_monthly": str(pm),
        "aic_monthly": str(am),
        "max_accounts": str(ma),
        "queue_depth": qd_s,
        "lookahead_h": str(qh),
    }
    stmt = _statement_descriptor("UPLOADM8", name.replace(" ", "").replace("-", ""))
    return {
        "name": f"UploadM8 {name} — Monthly Subscription",
        "description": desc,
        "statement_descriptor": stmt,
        "unit_label": "subscription",
        "tax_code": TAX_CODE,
        "metadata": meta,
    }


def topup_product_fields(lookup_key: str, meta: dict[str, Any]) -> Dict[str, Any]:
    wallet = str(meta.get("wallet", ""))
    if wallet == "bundle":
        put = int(meta.get("put") or 0)
        aic = int(meta.get("aic") or 0)
        title = f"UploadM8 Boost — {put} PUT + {aic} AIC"
        desc = (
            f"One-time bundle: {put:,} PUT and {aic:,} AIC credited to your UploadM8 wallet. "
            f"Non-refundable digital purchase via uploadm8.com."
        )
        stmt = _statement_descriptor("UPLOADM8BOOST")
        md = {
            "type": "topup",
            "lookup_key": lookup_key,
            "wallet": "bundle",
            "put": str(put),
            "aic": str(aic),
        }
    else:
        amt = int(meta.get("amount") or 0)
        wu = wallet.upper()
        title = f"UploadM8 {wu} {amt:,} — Token Top-Up"
        desc = (
            f"One-time purchase of {amt:,} {wu} tokens for UploadM8. "
            f"Credited instantly; never expire. Non-refundable digital goods."
        )
        stmt = _statement_descriptor("UPLOADM8", f"{wu}{amt}")
        md = {
            "type": "topup",
            "lookup_key": lookup_key,
            "wallet": wallet,
            "amount": str(amt),
        }
    if len(desc) > 4500:
        desc = desc[:4497] + "..."
    return {
        "name": title[:120],
        "description": desc,
        "statement_descriptor": stmt,
        "unit_label": "token",
        "tax_code": TAX_CODE,
        "metadata": md,
    }


def _upload_product_image(path: Path) -> Optional[str]:
    if not path.is_file():
        logger.warning("stripe_catalog_sync: image missing %s", path)
        return None
    with open(path, "rb") as fp:
        uploaded = stripe.File.create(purpose="product_image", file=fp)
    url = getattr(uploaded, "url", None)
    if not url and isinstance(uploaded, dict):
        url = uploaded.get("url")
    return url


def _load_manifest(path: Path) -> List[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("items")
    if not isinstance(items, list):
        return []
    return [x for x in items if isinstance(x, dict)]


def sync_stripe_catalog(
    tier_config: Dict[str, Dict[str, Any]],
    topup_products: Dict[str, Dict[str, Any]],
    image_dir: Path,
    *,
    manifest_path: Optional[Path] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"updated": [], "skipped": [], "errors": []}
    if not STRIPE_SECRET_KEY:
        out["skipped"].append("no STRIPE_SECRET_KEY")
        return out
    mp = manifest_path or default_manifest_path()
    if not mp.is_file():
        out["skipped"].append(f"manifest not found: {mp}")
        return out
    stripe.api_key = STRIPE_SECRET_KEY
    items = _load_manifest(mp)
    for item in items:
        pid = str(item.get("stripe_product_id") or "").strip()
        if not pid or pid.upper().startswith("REPLACE"):
            out["skipped"].append({"reason": "no product id", "item": item.get("lookup_key")})
            continue
        kind = str(item.get("kind") or "").strip().lower()
        image_file = item.get("image_file")
        image_path = image_dir / image_file if image_file else None
        try:
            if kind == "subscription":
                tier_slug = str(item.get("tier_slug") or "").strip()
                cfg = tier_config.get(tier_slug) or {}
                fields = subscription_product_fields(tier_slug, cfg)
            elif kind == "topup":
                lk = str(item.get("lookup_key") or "").strip()
                meta = topup_products.get(lk) or {}
                fields = topup_product_fields(lk, meta)
            else:
                out["skipped"].append({"reason": "unknown kind", "kind": kind, "id": pid})
                continue

            img_url = None
            if image_path and not dry_run:
                img_url = _upload_product_image(image_path)

            md = dict(fields.get("metadata") or {})
            kwargs: Dict[str, Any] = {
                "name": fields["name"],
                "description": fields["description"],
                "statement_descriptor": fields["statement_descriptor"],
                "unit_label": fields["unit_label"],
                "tax_code": fields["tax_code"],
                "metadata": dict(md),
            }
            if img_url:
                kwargs["images"] = [img_url]

            idem = f"catalog-sync-{pid}-{uuid.uuid4().hex[:24]}"
            if dry_run:
                out["updated"].append({"product": pid, "dry_run": True, "name": kwargs.get("name")})
                continue

            stripe.Product.modify(pid, idempotency_key=idem, **kwargs)
            out["updated"].append({"product": pid, "name": kwargs.get("name")})
        except Exception as e:
            logger.exception("stripe_catalog_sync failed for %s", pid)
            out["errors"].append({"product": pid, "error": str(e)})
    return out
