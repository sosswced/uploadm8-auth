"""
Shared feature schema for promo uplift training and runtime scoring.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from stages.entitlements import normalize_tier

FEATURES_NUM: List[str] = [
    "sent_dow_utc",
    "sent_hour_utc",
    "put_balance",
    "aic_balance",
    "uploads_30d",
    "avg_views_30d",
    "avg_engagement_pct_30d",
    "content_items_30d",
    "pci_avg_views_30d",
]

FEATURES_CAT: List[str] = [
    "channel",
    "delivery_status",
    "subscription_tier",
]

TARGET_CONVERTED = "converted_7d"
TARGET_ENGAGED = "engaged_7d"


def features_row_for_user(
    *,
    subscription_tier: str,
    campaign_features: Dict[str, Any],
    channel: str = "email",
    delivery_status: str = "runtime_score",
    at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Build a scoring row aligned with ``train_promo_uplift_baseline`` columns.
    """
    now = at or datetime.now(timezone.utc)
    tier = normalize_tier(subscription_tier or "free")
    uploads = int(campaign_features.get("uploads_window") or 0)
    ctr = float(campaign_features.get("nudge_ctr_pct") or 0)
    connected = int(campaign_features.get("connected_accounts") or 0)
    rev = float(campaign_features.get("revenue_7d") or 0)
    eng_rate = float(campaign_features.get("engagement_rate_pct_30d") or 0)
    if eng_rate <= 0 and ctr > 0:
        eng_rate = ctr
    avg_views = float(campaign_features.get("avg_views_30d") or 0)
    content_items = int(campaign_features.get("content_items_30d") or 0)
    pci_views = float(campaign_features.get("pci_avg_views_30d") or 0)

    return {
        "sent_dow_utc": int(now.weekday()),
        "sent_hour_utc": int(now.hour),
        "put_balance": int(campaign_features.get("put_balance") or 0),
        "aic_balance": int(campaign_features.get("aic_balance") or 0),
        "uploads_30d": uploads,
        "avg_views_30d": avg_views,
        "avg_engagement_pct_30d": eng_rate,
        "content_items_30d": content_items,
        "pci_avg_views_30d": pci_views,
        "channel": (channel or "email")[:32],
        "delivery_status": (delivery_status or "runtime_score")[:64],
        "subscription_tier": tier,
        # Extra context for toy fallback (ignored by sklearn pipeline).
        "connected_accounts": connected,
        "revenue_7d": rev,
        "nudge_ctr_pct": ctr,
    }


def pick_training_target(df_columns: List[str], converted_col: str = TARGET_CONVERTED) -> str:
    """Prefer strict conversion label; fall back to engagement proxy when needed."""
    if converted_col in df_columns:
        return converted_col
    if TARGET_ENGAGED in df_columns:
        return TARGET_ENGAGED
    return converted_col
