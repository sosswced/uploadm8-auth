from __future__ import annotations

from typing import Any, Dict

from fastapi import HTTPException

from stages.entitlements import STRIPE_LOOKUP_TO_TIER, entitlements_to_dict, get_entitlements_for_tier


_TIER_RANK = {
    "free": 0,
    "creator_lite": 1,
    "creator_pro": 2,
    "studio": 3,
    "agency": 4,
    "friends_family": 5,
    "lifetime": 6,
    "master_admin": 7,
}


def _tier_is_upgrade(old: str, new: str) -> bool:
    return _TIER_RANK.get(new, 0) >= _TIER_RANK.get(old, 0)


def get_plan(tier: str) -> dict:
    return entitlements_to_dict(get_entitlements_for_tier(tier))


async def ensure_stripe_customer(conn, user: dict, stripe_client) -> str:
    customer_id = user.get("stripe_customer_id")
    if customer_id:
        try:
            stripe_client.Customer.retrieve(customer_id)
            return customer_id
        except stripe_client.error.InvalidRequestError:
            pass
    customer = stripe_client.Customer.create(email=user["email"], name=user.get("name") or user["email"])
    customer_id = customer.id
    await conn.execute("UPDATE users SET stripe_customer_id = $1 WHERE id = $2", customer_id, user["id"])
    return customer_id


def get_active_price_by_lookup(stripe_client, lookup_key: str):
    prices = stripe_client.Price.list(lookup_keys=[lookup_key], active=True)
    if not prices.data:
        raise HTTPException(400, f"Price not found for lookup_key: {lookup_key}. Run stripe_setup.py to create prices.")
    return prices.data[0]


def create_wallet_topup_checkout_session(
    stripe_client,
    customer_id: str,
    lookup_key: str,
    topup_products: Dict[str, Dict[str, Any]],
    success_url: str,
    cancel_url: str,
    user_id: str,
):
    product = topup_products.get(lookup_key)
    if not product:
        raise HTTPException(400, "Invalid product")
    price = get_active_price_by_lookup(stripe_client, lookup_key)
    return stripe_client.checkout.Session.create(
        customer=customer_id,
        line_items=[{"price": price.id, "quantity": 1}],
        mode="payment",
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={
            "user_id": str(user_id),
            "wallet": product["wallet"],
            "amount": str(product["amount"]),
            "kind": "topup",
            "lookup_key": lookup_key,
        },
    )


def create_billing_checkout_session(
    stripe_client,
    customer_id: str,
    kind: str,
    lookup_key: str,
    user_id: str,
    success_url: str,
    cancel_url: str,
    topup_products: Dict[str, Dict[str, Any]],
):
    price = get_active_price_by_lookup(stripe_client, lookup_key)
    if kind == "subscription":
        tier = STRIPE_LOOKUP_TO_TIER.get(lookup_key, "free")
        ent = get_entitlements_for_tier(tier)
        trial_days = ent.trial_days
        session_params = dict(
            customer=customer_id,
            line_items=[{"price": price.id, "quantity": 1}],
            mode="subscription",
            success_url=success_url,
            cancel_url=cancel_url,
            allow_promotion_codes=True,
            metadata={
                "user_id": str(user_id),
                "tier": tier,
                "kind": "subscription",
                "lookup_key": lookup_key,
            },
        )
        if trial_days > 0:
            session_params["subscription_data"] = {
                "trial_period_days": trial_days,
                "metadata": {"user_id": str(user_id), "tier": tier},
            }
        return stripe_client.checkout.Session.create(**session_params)

    product = topup_products.get(lookup_key, {})
    if not product:
        raise HTTPException(400, f"Unknown topup product: {lookup_key}")

    return stripe_client.checkout.Session.create(
        customer=customer_id,
        line_items=[{"price": price.id, "quantity": 1}],
        mode="payment",
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={
            "user_id": str(user_id),
            "wallet": product.get("wallet", "put"),
            "amount": str(product.get("amount", 0)),
            "kind": "topup",
            "lookup_key": lookup_key,
        },
    )
