"""
Shared feature schema for promo uplift training and runtime scoring.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from stages.entitlements import normalize_tier
from services.ml_feature_registry import active_cat, active_num, label, label_fallback

# Derived from the feature registry (single source of truth) so training,
# runtime scoring, and the curated views never drift.
FEATURES_NUM: List[str] = active_num("promo")
FEATURES_CAT: List[str] = active_cat("promo")

TARGET_CONVERTED = label("promo")
TARGET_ENGAGED = label_fallback("promo") or "engaged_7d"


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
        "is_snapshot": 1,
        "put_balance": int(campaign_features.get("put_balance") or 0),
        "aic_balance": int(campaign_features.get("aic_balance") or 0),
        "uploads_30d": uploads,
        "avg_views_30d": avg_views,
        "avg_engagement_pct_30d": eng_rate,
        "content_items_30d": content_items,
        "pci_avg_views_30d": pci_views,
        # Marketing history / tenure / recency / trend (best-effort; the view path
        # in score_user_propensity provides exact parity when available).
        "prior_touchpoints": int(campaign_features.get("prior_touchpoints") or 0),
        "opens_all": int(campaign_features.get("opens_all") or 0),
        "clicks_all": int(campaign_features.get("clicks_all") or 0),
        "days_since_last_touchpoint": float(campaign_features.get("days_since_last_touchpoint") or 0.0),
        "account_age_days": float(campaign_features.get("account_age_days") or 0.0),
        "days_since_last_upload": float(campaign_features.get("days_since_last_upload") or 0.0),
        "uploads_trend_30d": float(campaign_features.get("uploads_trend_30d") or 0.0),
        "views_trend_30d": float(campaign_features.get("views_trend_30d") or 0.0),
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
