"""
UploadM8 Billing routes — extracted from app.py.
Stripe checkout, portal, session retrieval, and webhook handling.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import stripe
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request

from api.schemas.billing import BillingSubscriptionActionRequest, UploadCostEstimateRequest
from core.config import (
    BILLING_MODE,
    FRONTEND_URL,
    STRIPE_CANCEL_URL,
    STRIPE_SECRET_KEY,
    STRIPE_SUCCESS_URL,
    STRIPE_WEBHOOK_SECRET,
)
from core.deps import get_current_user
from core.helpers import _now_utc, _tier_is_upgrade
from services.workspace import can_manage_billing, require_can_manage_billing, resolve_billing_user_id
from core.models import CheckoutRequest
from core.notifications import notify_mrr, notify_topup
import core.state
from services.billing_service_weights import fetch_service_weights_map
from core.wallet import credit_wallet, ledger_entry
from migrations.runtime_migrations import ensure_subscription_tier_constraint
from routers.preferences import get_user_prefs_for_upload
from stages.ai_service_costs import compute_presign_put_aic_costs
from stages.entitlements import (
    STRIPE_LOOKUP_TO_TIER,
    get_effective_tier_config,
    get_effective_topup_products,
    get_entitlements_for_tier,
    normalize_tier,
)
from stages.emails import (
    send_subscription_started_email,
    send_trial_started_email,
    send_plan_upgraded_email,
    send_plan_downgraded_email,
    send_topup_receipt_email,
    send_bundle_topup_receipt_email,
    send_renewal_receipt_email,
    send_subscription_cancelled_email,
    send_trial_cancelled_email,
    send_payment_failed_email,
)

logger = logging.getLogger("uploadm8-api")

router = APIRouter(prefix="/api/billing", tags=["billing"])


def _webhook_tier(lookup_key: str, *, fallback: str = "free") -> str:
    """Map Stripe lookup_key → canonical tier slug (launch→creator_lite, etc.)."""
    lk = (lookup_key or "").strip().lower()
    if lk in STRIPE_LOOKUP_TO_TIER:
        return normalize_tier(STRIPE_LOOKUP_TO_TIER[lk])
    return normalize_tier(fallback)


def _stripe_field(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from a Stripe object or plain dict (Basil+ API safe)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            val = getter(key)
            if val is not None:
                return val
        except Exception:
            pass
    return getattr(obj, key, default)


def _subscription_items_data(sub: Any) -> List[Any]:
    items = _stripe_field(sub, "items")
    if items is None:
        return []
    data = _stripe_field(items, "data")
    if isinstance(data, list):
        return data
    return []


def _subscription_price_lookup_key(sub: Any) -> str:
    for item in _subscription_items_data(sub):
        price = _stripe_field(item, "price")
        if price is None:
            continue
        lk = str(_stripe_field(price, "lookup_key") or "").strip().lower()
        if lk:
            return lk
    return ""


def _subscription_period_timestamps(sub: Any) -> Tuple[Optional[int], Optional[int]]:
    """
    Billing period unix timestamps.

    Stripe API 2025-03-31 (Basil) moved ``current_period_*`` from Subscription to
    SubscriptionItem; read both so webhooks work across API versions.
    """
    start = _stripe_field(sub, "current_period_start")
    end = _stripe_field(sub, "current_period_end")
    if start is not None and end is not None:
        return int(start), int(end)
    for item in _subscription_items_data(sub):
        if start is None:
            start = _stripe_field(item, "current_period_start")
        if end is None:
            end = _stripe_field(item, "current_period_end")
        if start is not None and end is not None:
            break
    return (
        int(start) if start is not None else None,
        int(end) if end is not None else None,
    )


def _subscription_period_datetimes(sub: Any) -> Tuple[datetime, datetime]:
    start_ts, end_ts = _subscription_period_timestamps(sub)
    period_start = (
        datetime.fromtimestamp(start_ts, tz=timezone.utc) if start_ts else _now_utc()
    )
    period_end = (
        datetime.fromtimestamp(end_ts, tz=timezone.utc) if end_ts else _now_utc()
    )
    return period_start, period_end


def _invoice_subscription_id(invoice: Any) -> Optional[str]:
    """
    Resolve the subscription id from an Invoice across API versions.

    Pre-Basil:            invoice.subscription
    Basil+ (2025-03-31):  invoice.parent.subscription_details.subscription
    """
    def _as_id(val: Any) -> Optional[str]:
        if not val:
            return None
        if isinstance(val, str):
            return val
        return _stripe_field(val, "id")

    sub_id = _as_id(_stripe_field(invoice, "subscription"))
    if sub_id:
        return sub_id
    parent = _stripe_field(invoice, "parent")
    details = _stripe_field(parent, "subscription_details") if parent is not None else None
    return _as_id(_stripe_field(details, "subscription")) if details is not None else None


async def _event_already_processed(conn, event_id: Optional[str]) -> bool:
    """True if this Stripe event.id was already handled (idempotency fast-path)."""
    if not event_id:
        return False
    try:
        row = await conn.fetchrow(
            "SELECT 1 FROM processed_stripe_events WHERE event_id = $1", event_id
        )
        return row is not None
    except Exception as e:
        logger.warning("stripe webhook: dedup lookup failed for %s: %s", event_id, e)
        return False


async def _mark_event_processed(conn, event_id: Optional[str], event_type: str) -> None:
    """Record a successfully handled event so a later retry short-circuits."""
    if not event_id:
        return
    try:
        await conn.execute(
            "INSERT INTO processed_stripe_events (event_id, event_type) VALUES ($1, $2) "
            "ON CONFLICT (event_id) DO NOTHING",
            event_id, event_type,
        )
    except Exception as e:
        logger.warning("stripe webhook: failed to mark event %s processed: %s", event_id, e)


async def _ensure_tier_constraint(conn) -> None:
    try:
        await ensure_subscription_tier_constraint(conn)
    except Exception as e:
        logger.warning("billing: tier constraint ensure failed: %s", e)


async def _execute_user_tier_update(conn, sql: str, tier: str, *params):
    """Run a tier UPDATE; repair constraint and retry once on check violation."""
    await _ensure_tier_constraint(conn)
    try:
        return await conn.execute(sql, tier, *params)
    except asyncpg.CheckViolationError as e:
        if "users_subscription_tier_check" not in str(e):
            raise
        logger.warning("billing: tier check rejected %r — repairing constraint and retrying", tier)
        await ensure_subscription_tier_constraint(conn)
        return await conn.execute(sql, tier, *params)


# ============================================================
# Routes
# ============================================================

@router.post("/checkout")
async def create_checkout(data: CheckoutRequest, user: dict = Depends(get_current_user)):
    require_can_manage_billing(user)
    if not STRIPE_SECRET_KEY:
        raise HTTPException(503, "Billing not configured")

    bill_id = resolve_billing_user_id(user)
    async with core.state.db_pool.acquire() as conn:
        bill_user = await conn.fetchrow("SELECT * FROM users WHERE id = $1", bill_id)
        if not bill_user:
            raise HTTPException(404, "Billing account not found")
        bill_user = dict(bill_user)
        # ── Double-subscribe guard ────────────────────────────────────
        if data.kind == "subscription":
            existing_sub_id  = bill_user.get("stripe_subscription_id")
            existing_status  = bill_user.get("subscription_status")
            if existing_sub_id and existing_status in ("active", "trialing", "past_due"):
                # User already has an active sub — send straight to billing portal
                # so they can upgrade/downgrade there rather than creating a duplicate
                try:
                    portal = stripe.billing_portal.Session.create(
                        customer=bill_user.get("stripe_customer_id"),
                        return_url=f"{FRONTEND_URL}/settings.html#billing",
                    )
                    return {"checkout_url": portal.url, "session_id": None, "portal_redirect": True}
                except Exception:
                    pass  # Fall through to new checkout if portal fails

        customer_id = bill_user.get("stripe_customer_id")
        if not customer_id:
            customer = stripe.Customer.create(
                email=bill_user["email"],
                name=bill_user.get("name") or bill_user["email"],
            )
            customer_id = customer.id
            await conn.execute("UPDATE users SET stripe_customer_id = $1 WHERE id = $2", customer_id, bill_id)

    prices = stripe.Price.list(lookup_keys=[data.lookup_key], active=True)
    if not prices.data:
        raise HTTPException(400, f"Price not found for lookup_key: {data.lookup_key}. Run stripe_setup.py to create prices.")

    if data.kind == "subscription":
        # Resolve trial days from entitlements (7 days for all paid tiers)
        tier = _webhook_tier(data.lookup_key)
        ent  = get_entitlements_for_tier(tier)
        trial_days = ent.trial_days  # 0 for free/internal, 7 for paid

        session_params = dict(
            customer              = customer_id,
            line_items            = [{"price": prices.data[0].id, "quantity": 1}],
            mode                  = "subscription",
            success_url           = STRIPE_SUCCESS_URL,
            cancel_url            = STRIPE_CANCEL_URL,
            allow_promotion_codes = True,
            metadata              = {"user_id": bill_id, "tier": tier},
        )
        if trial_days > 0:
            session_params["subscription_data"] = {
                "trial_period_days": trial_days,
                "metadata": {"user_id": bill_id, "tier": tier},
            }

        session = stripe.checkout.Session.create(**session_params)

    else:  # topup / one-time payment
        product = get_effective_topup_products().get(data.lookup_key, {})
        if not product:
            raise HTTPException(400, f"Unknown topup product: {data.lookup_key}")

        session = stripe.checkout.Session.create(
            customer    = customer_id,
            line_items  = [{"price": prices.data[0].id, "quantity": 1}],
            mode        = "payment",
            success_url = STRIPE_SUCCESS_URL,
            cancel_url  = STRIPE_CANCEL_URL,
            metadata    = {
                "user_id": bill_id,
                "lookup_key": data.lookup_key,
                "wallet":  product.get("wallet", "put"),
                "amount":  str(product.get("amount", 0)),
            },
        )

    return {"checkout_url": session.url, "session_id": session.id}

@router.post("/portal")
async def create_portal(user: dict = Depends(get_current_user)):
    require_can_manage_billing(user)
    if not user.get("stripe_customer_id"):
        raise HTTPException(400, "No billing account")
    session = stripe.billing_portal.Session.create(
        customer=user["stripe_customer_id"],
        return_url=f"{FRONTEND_URL}/settings.html",
    )
    return {"portal_url": session.url}

@router.get("/session")
async def get_billing_session(
    session_id: str = Query(..., description="Stripe checkout session ID (cs_test_* or cs_live_*)"),
    user: dict = Depends(get_current_user),
):
    """
    Read a Stripe Checkout Session directly from the Stripe API.
    Used by billing/success.html to render the confirmation screen.
    Works for both test (cs_test_*) and live (cs_live_*) sessions.
    The session_id prefix reveals the mode — no separate flag needed.
    """
    if not STRIPE_SECRET_KEY:
        raise HTTPException(503, "Billing not configured")

    try:
        sess = stripe.checkout.Session.retrieve(
            session_id,
            expand=["subscription", "subscription.items.data.price", "line_items"],
        )
    except stripe.error.InvalidRequestError as e:
        raise HTTPException(404, f"Stripe session not found: {e}")
    except stripe.error.AuthenticationError:
        raise HTTPException(503, "Stripe authentication failed — check STRIPE_SECRET_KEY")
    except Exception as e:
        raise HTTPException(502, f"Stripe API error: {e}")

    # Security: session must belong to this workspace billing account
    meta_user_id = (sess.get("metadata") or {}).get("user_id")
    bill_id = resolve_billing_user_id(user)
    if meta_user_id and str(meta_user_id) != str(bill_id):
        raise HTTPException(403, "This session does not belong to your account")

    mode           = sess.get("mode")            # "subscription" | "payment"
    payment_status = sess.get("payment_status")  # "paid" | "unpaid" | "no_payment_required"
    amount_total   = (sess.get("amount_total") or 0) / 100
    currency       = (sess.get("currency") or "usd").upper()

    # ── Subscription fields ──────────────────────────────────────────
    tier                  = None
    lookup_key            = None
    plan_name             = None
    sub_status            = None
    trial_end_ts          = None
    current_period_end_ts = None

    if mode == "subscription":
        sub = sess.get("subscription")
        if isinstance(sub, dict):
            sub_status            = _stripe_field(sub, "status")
            trial_end_ts          = _stripe_field(sub, "trial_end")
            _, current_period_end_ts = _subscription_period_timestamps(sub)
            try:
                lookup_key = _subscription_price_lookup_key(sub)
                price = None
                items = _subscription_items_data(sub)
                if items:
                    price = _stripe_field(items[0], "price")
                lk = (lookup_key or "").strip().lower()
                tier = _webhook_tier(lookup_key) if lk in STRIPE_LOOKUP_TO_TIER else None
                if not tier and price:
                    # Fallback: try to match by product name
                    prod_id = _stripe_field(price, "product")
                    if prod_id:
                        prod = stripe.Product.retrieve(prod_id)
                        plan_name = prod.get("name", "")
            except Exception:
                pass

    # ── Topup fields ─────────────────────────────────────────────────
    topup_wallet = None
    topup_amount = None
    topup_label = None
    if mode == "payment":
        meta = sess.get("metadata") or {}
        topup_wallet = meta.get("wallet")
        topup_amount = meta.get("amount")
        lk = str(meta.get("lookup_key") or "").strip().lower()
        prod = get_effective_topup_products().get(lk) if lk else None
        if prod and str(prod.get("wallet")) == "bundle":
            topup_wallet = "bundle"
            topup_label = f"{int(prod.get('put') or 0)} PUT + {int(prod.get('aic') or 0)} AIC"
            topup_amount = None
        elif prod and str(prod.get("wallet")) in ("put", "aic"):
            topup_wallet = prod.get("wallet")
            topup_amount = str(prod.get("amount", 0))
        elif topup_amount is None and meta.get("amount") is not None:
            topup_amount = meta.get("amount")

    # Resolve display name
    if tier:
        cfg = get_effective_tier_config().get(tier, {})
        plan_name = cfg.get("name", tier.replace("_", " ").title())
    elif not plan_name:
        plan_name = "Your Plan"

    return {
        "session_id":          session_id,
        "mode":                mode,
        "payment_status":      payment_status,
        "amount_total":        amount_total,
        "currency":            currency,
        "tier":                tier,
        "plan_name":           plan_name,
        "lookup_key":          lookup_key,
        "subscription_status": sub_status,
        "trial_end":           trial_end_ts,            # unix timestamp or None
        "current_period_end":  current_period_end_ts,   # unix timestamp or None
        "topup_wallet":        topup_wallet,
        "topup_amount":        int(topup_amount) if topup_amount not in (None, "") else None,
        "topup_label":         topup_label,
        "is_test_mode":        session_id.startswith("cs_test_"),
        "billing_mode":        BILLING_MODE,
    }


def _stripe_obj_to_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (dict, list, str, int, float, bool)):
        return obj
    try:
        return dict(obj)
    except Exception:
        try:
            return json.loads(str(obj))
        except Exception:
            return None


@router.get("/overview")
async def billing_overview(user: dict = Depends(get_current_user)):
    """Stripe subscription snapshot + recent invoices for settings billing UI."""
    empty: Dict[str, Any] = {"subscription": None, "invoices": [], "default_payment_method": None}
    if not STRIPE_SECRET_KEY:
        return empty
    cid = user.get("stripe_customer_id")
    if not cid:
        return empty
    try:
        cust = stripe.Customer.retrieve(cid, expand=["invoice_settings.default_payment_method"])
        dpm = None
        inv_set = cust.get("invoice_settings") or {}
        if inv_set.get("default_payment_method"):
            dpm = _stripe_obj_to_dict(inv_set["default_payment_method"])
        sub_obj = None
        sid = user.get("stripe_subscription_id")
        if sid:
            try:
                sub_obj = stripe.Subscription.retrieve(sid, expand=["default_payment_method"])
            except Exception:
                sub_obj = None
        inv_list: List[Dict[str, Any]] = []
        try:
            invs = stripe.Invoice.list(customer=cid, limit=12)
            for inv in getattr(invs, "data", []) or []:
                inv_list.append(
                    {
                        "id": getattr(inv, "id", None),
                        "status": getattr(inv, "status", None),
                        "created": getattr(inv, "created", None),
                        "amount_due": getattr(inv, "amount_due", None),
                        "amount_paid": getattr(inv, "amount_paid", None),
                        "currency": getattr(inv, "currency", None),
                        "invoice_pdf": getattr(inv, "invoice_pdf", None),
                        "hosted_invoice_url": getattr(inv, "hosted_invoice_url", None),
                    }
                )
        except Exception:
            pass
        return {
            "subscription": _stripe_obj_to_dict(sub_obj),
            "invoices": inv_list,
            "default_payment_method": dpm,
        }
    except Exception as e:
        logger.warning("billing overview stripe error: %s", e)
        return empty


@router.post("/upload-estimate")
async def billing_upload_estimate(body: UploadCostEstimateRequest, user: dict = Depends(get_current_user)):
    """PUT/AIC estimate for billing settings calculator (same model as upload presign)."""
    tier = user.get("subscription_tier") or "free"
    ent = get_entitlements_for_tier(tier)
    async with core.state.db_pool.acquire() as conn:
        user_prefs = await get_user_prefs_for_upload(conn, user["id"])
        db_weights = await fetch_service_weights_map(conn)
    put, aic, billing_breakdown = compute_presign_put_aic_costs(
        ent,
        num_publish_targets=body.num_publish_targets,
        file_size=body.file_size,
        duration_hint=body.duration_seconds,
        has_telemetry=body.has_telemetry,
        use_ai_checkbox=body.use_ai,
        user_prefs=user_prefs,
        num_thumbnails_override=body.num_thumbnails,
        service_weights_map=db_weights,
    )
    return {"put_cost": put, "aic_cost": aic, "billing_breakdown": billing_breakdown}


@router.get("/subscription/actions")
async def billing_subscription_actions(user: dict = Depends(get_current_user)):
    st = str(user.get("subscription_status") or "").lower()
    sid = user.get("stripe_subscription_id")
    can = bool(sid and st in ("active", "trialing", "past_due") and can_manage_billing(user))
    return {"can_manage_subscription": can}


@router.post("/subscription/action")
async def billing_subscription_action(
    body: BillingSubscriptionActionRequest,
    user: dict = Depends(get_current_user),
):
    """Stripe Customer Portal flows + subscription maintenance (best-effort)."""
    require_can_manage_billing(user)
    if not STRIPE_SECRET_KEY:
        raise HTTPException(503, "Billing not configured")
    cid = user.get("stripe_customer_id")
    sub_id = user.get("stripe_subscription_id")
    if not cid:
        raise HTTPException(400, "No Stripe customer on file")
    action = body.action
    try:
        if action == "share_payment_update_link":
            sess = stripe.billing_portal.Session.create(
                customer=cid,
                return_url=f"{FRONTEND_URL}/settings.html#billing",
                flow_data={"type": "payment_method_update"},
            )
            return {"share_url": sess.url, "ok": True}
        if not sub_id:
            raise HTTPException(400, "No active subscription to modify")
        if action == "cancel_subscription":
            stripe.Subscription.modify(sub_id, cancel_at_period_end=True)
            return {"ok": True}
        if action == "pause_payment_collection":
            stripe.Subscription.modify(
                sub_id,
                pause_collection={"behavior": "mark_uncollectible"},
            )
            return {"ok": True}
        if action == "create_one_time_invoice":
            if not body.amount_cents:
                raise HTTPException(400, "amount_cents required (min 100)")
            desc = (body.description or "One-time invoice").strip() or "One-time invoice"
            stripe.InvoiceItem.create(
                customer=cid,
                amount=int(body.amount_cents),
                currency=(body.currency or "usd").lower(),
                description=desc,
            )
            inv = stripe.Invoice.create(
                customer=cid,
                collection_method="charge_automatically",
                auto_advance=True,
            )
            fin = stripe.Invoice.finalize_invoice(inv.id)
            return {"ok": True, "invoice_id": getattr(fin, "id", None) or inv.id}
    except stripe.error.StripeError as e:
        raise HTTPException(400, str(e.user_message or e)) from e
    raise HTTPException(400, "Unsupported action")


@router.post("/webhook")
async def stripe_webhook(request: Request, background_tasks: BackgroundTasks):
    """Stripe billing webhook — idempotent, signature-verified."""
    payload = await request.body()
    sig = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        raise HTTPException(400, f"Invalid signature: {e}")

    etype = event.type
    event_id = _stripe_field(event, "id")

    # Preflight: repair the tier constraint and short-circuit duplicate deliveries.
    # Stripe re-sends events on timeout; the per-operation idempotency below makes
    # reprocessing safe, but this avoids redundant Stripe API calls and emails.
    try:
        async with core.state.db_pool.acquire() as _pre_conn:
            await _ensure_tier_constraint(_pre_conn)
            if await _event_already_processed(_pre_conn, event_id):
                logger.info("stripe webhook: duplicate event %s (%s) — skipping", event_id, etype)
                return {"status": "duplicate", "event_id": event_id}
    except Exception as e:
        logger.warning("stripe webhook: preflight check failed: %s", e)

    # ── checkout.session.completed ──────────────────────────────────────
    if etype == "checkout.session.completed":
        session = event.data.object
        user_id = session.metadata.get("user_id")
        if not user_id:
            return {"status": "no_user_id"}

        async with core.state.db_pool.acquire() as conn:
            user_row = await conn.fetchrow("SELECT email, name FROM users WHERE id = $1", user_id)
            email = user_row["email"] if user_row else ""
            uname = (user_row["name"] if user_row else None) or "there"

            if session.mode == "subscription":
                sub = stripe.Subscription.retrieve(
                    session.subscription,
                    expand=["items.data.price"],
                )
                lookup_key = _subscription_price_lookup_key(sub)
                tier = _webhook_tier(lookup_key)
                ent  = get_entitlements_for_tier(tier)
                status = _stripe_field(sub, "status")

                period_start, period_end = _subscription_period_datetimes(sub)
                trial_ts = _stripe_field(sub, "trial_end")
                trial_end = (
                    datetime.fromtimestamp(trial_ts, tz=timezone.utc) if trial_ts else None
                )

                await _execute_user_tier_update(
                    conn,
                    """
                    UPDATE users SET
                        subscription_tier      = $1,
                        stripe_subscription_id = $2,
                        subscription_status    = $3,
                        current_period_end     = $4,
                        trial_end              = $5,
                        updated_at             = NOW()
                    WHERE id = $6
                    """,
                    tier,
                    session.subscription,
                    status,
                    period_end,
                    trial_end,
                    user_id,
                )

                # Seed wallet for first period / trial — deduped by invoice id
                refill_ref = _stripe_field(sub, "latest_invoice") or session.id
                await _do_monthly_refill(conn, user_id, tier, ent, refill_ref, period_start, period_end)

                amount = (session.amount_total or 0) / 100
                await conn.execute(
                    "INSERT INTO revenue_tracking (user_id, amount, source, stripe_event_id, plan) "
                    "VALUES ($1,$2,'subscription',$3,$4) ON CONFLICT (stripe_event_id) DO NOTHING",
                    user_id, amount, session.id, tier
                )
                background_tasks.add_task(notify_mrr, amount, email, tier, status)

                # ── Welcome email: trial vs paid ──────────────────────────────
                if trial_end:
                    trial_days = _stripe_field(sub, "trial_period_days") or 14
                    background_tasks.add_task(
                        send_trial_started_email,
                        email, uname, tier,
                        trial_end.strftime("%B %d, %Y"),
                        trial_days,
                    )
                else:
                    next_date = period_end.strftime("%B %d, %Y")
                    background_tasks.add_task(
                        send_subscription_started_email,
                        email, uname, tier, amount, next_date,
                    )

            elif session.mode == "payment":
                # Idempotency: credit_wallet stamps token_ledger.stripe_event_id with
                # session.id, so a re-delivered checkout.session.completed must not
                # re-credit the wallet (the credit path is not otherwise idempotent).
                already_credited = await conn.fetchval(
                    "SELECT 1 FROM token_ledger "
                    "WHERE stripe_event_id = $1 AND reason = 'topup_purchase' LIMIT 1",
                    session.id,
                )
                if already_credited:
                    logger.info("topup already credited for session %s — skipping", session.id)
                    return {"status": "topup_already_processed"}

                meta = session.metadata or {}
                lookup_key = str(meta.get("lookup_key") or "").strip().lower()
                prod = get_effective_topup_products().get(lookup_key) if lookup_key else None

                async def _wallet_balances() -> tuple[int, int]:
                    row = await conn.fetchrow(
                        "SELECT put_balance, aic_balance FROM wallets WHERE user_id = $1", user_id
                    )
                    if not row:
                        return (0, 0)
                    return (int(row["put_balance"] or 0), int(row["aic_balance"] or 0))

                if prod is None:
                    # Legacy sessions: wallet + amount only (no lookup_key)
                    wallet_type = meta.get("wallet", "put")
                    amount_tokens = int(meta.get("amount", 0))
                    if amount_tokens <= 0:
                        return {"status": "no_topup_amount"}
                    prior = await conn.fetchval(
                        "SELECT 1 FROM token_ledger WHERE user_id = $1 AND reason = 'topup_purchase' LIMIT 1",
                        user_id,
                    )
                    bonus = int(amount_tokens * 0.25) if not prior else 0
                    total = amount_tokens + bonus
                    await credit_wallet(conn, user_id, wallet_type, total, "topup_purchase", session.id)
                    amount = (session.amount_total or 0) / 100
                    await conn.execute(
                        "INSERT INTO revenue_tracking (user_id, amount, source, stripe_event_id, plan) "
                        "VALUES ($1,$2,'topup',$3,$4) ON CONFLICT (stripe_event_id) DO NOTHING",
                        user_id, amount, session.id, f"{wallet_type}_{amount_tokens}",
                    )
                    background_tasks.add_task(notify_topup, amount, email, wallet_type, total)
                    background_tasks.add_task(
                        send_topup_receipt_email,
                        email, uname, wallet_type, total, amount, 0, session.id, bonus_tokens=bonus,
                    )
                elif prod.get("wallet") == "bundle":
                    put_base = int(prod.get("put", 0))
                    aic_base = int(prod.get("aic", 0))
                    if put_base <= 0 and aic_base <= 0:
                        return {"status": "invalid_bundle"}
                    prior = await conn.fetchval(
                        "SELECT 1 FROM token_ledger WHERE user_id = $1 AND reason = 'topup_purchase' LIMIT 1",
                        user_id,
                    )
                    bonus_put = int(put_base * 0.25) if (put_base > 0 and not prior) else 0
                    bonus_aic = int(aic_base * 0.25) if (aic_base > 0 and not prior) else 0
                    put_total = put_base + bonus_put
                    aic_total = aic_base + bonus_aic
                    if put_total > 0:
                        await credit_wallet(conn, user_id, "put", put_total, "topup_purchase", session.id)
                    if aic_total > 0:
                        await credit_wallet(conn, user_id, "aic", aic_total, "topup_purchase", session.id)
                    put_bal, aic_bal = await _wallet_balances()
                    amount = (session.amount_total or 0) / 100
                    await conn.execute(
                        "INSERT INTO revenue_tracking (user_id, amount, source, stripe_event_id, plan) "
                        "VALUES ($1,$2,'topup',$3,$4) ON CONFLICT (stripe_event_id) DO NOTHING",
                        user_id, amount, session.id, f"bundle_{lookup_key}",
                    )
                    tok_label = f"{put_total} PUT + {aic_total} AIC"
                    background_tasks.add_task(notify_topup, amount, email, "bundle", tok_label)
                    background_tasks.add_task(
                        send_bundle_topup_receipt_email,
                        email,
                        uname,
                        put_total,
                        aic_total,
                        amount,
                        session.id,
                        bonus_put=bonus_put,
                        bonus_aic=bonus_aic,
                        put_balance=put_bal,
                        aic_balance=aic_bal,
                    )
                else:
                    wallet_type = str(prod.get("wallet", "put"))
                    amount_tokens = int(prod.get("amount", 0))
                    if amount_tokens <= 0:
                        return {"status": "no_topup_amount"}
                    prior = await conn.fetchval(
                        "SELECT 1 FROM token_ledger WHERE user_id = $1 AND reason = 'topup_purchase' LIMIT 1",
                        user_id,
                    )
                    bonus = int(amount_tokens * 0.25) if not prior else 0
                    total = amount_tokens + bonus
                    await credit_wallet(conn, user_id, wallet_type, total, "topup_purchase", session.id)
                    amount = (session.amount_total or 0) / 100
                    await conn.execute(
                        "INSERT INTO revenue_tracking (user_id, amount, source, stripe_event_id, plan) "
                        "VALUES ($1,$2,'topup',$3,$4) ON CONFLICT (stripe_event_id) DO NOTHING",
                        user_id, amount, session.id, f"{wallet_type}_{amount_tokens}",
                    )
                    background_tasks.add_task(notify_topup, amount, email, wallet_type, total)
                    background_tasks.add_task(
                        send_topup_receipt_email,
                        email, uname, wallet_type, total, amount, 0, session.id, bonus_tokens=bonus,
                    )

    # ── invoice.paid — monthly wallet refill on every renewal ──────────
    elif etype == "invoice.paid":
        invoice = event.data.object
        sub_id  = _invoice_subscription_id(invoice)
        if not sub_id:
            return {"status": "no_subscription"}

        async with core.state.db_pool.acquire() as conn:
            user_row = await conn.fetchrow(
                "SELECT id, email, name, subscription_tier FROM users WHERE stripe_subscription_id = $1", sub_id
            )
            if not user_row:
                logger.warning(f"invoice.paid: no user for subscription {sub_id}")
                return {"status": "user_not_found"}

            user_id = str(user_row["id"])
            email   = user_row["email"]
            uname   = user_row["name"] or "there"
            sub = stripe.Subscription.retrieve(sub_id, expand=["items.data.price"])
            lookup_key = _subscription_price_lookup_key(sub)
            tier = _webhook_tier(lookup_key, fallback=user_row["subscription_tier"] or "free")
            ent  = get_entitlements_for_tier(tier)

            inv_period_start = _stripe_field(invoice, "period_start")
            inv_period_end = _stripe_field(invoice, "period_end")
            period_start = (
                datetime.fromtimestamp(inv_period_start, tz=timezone.utc)
                if inv_period_start is not None
                else _now_utc()
            )
            period_end = (
                datetime.fromtimestamp(inv_period_end, tz=timezone.utc)
                if inv_period_end is not None
                else _now_utc()
            )
            invoice_id = _stripe_field(invoice, "id")

            await _execute_user_tier_update(
                conn,
                """
                UPDATE users SET
                    subscription_tier   = $1,
                    subscription_status = 'active',
                    current_period_end  = $2,
                    updated_at          = NOW()
                WHERE id = $3
                """,
                tier,
                period_end,
                user_id,
            )

            # Monthly wallet refill — deduped by invoice_id
            await _do_monthly_refill(conn, user_id, tier, ent, invoice_id, period_start, period_end)

            amount = (invoice.amount_paid or 0) / 100
            await conn.execute(
                "INSERT INTO revenue_tracking (user_id, amount, source, stripe_event_id, plan) "
                "VALUES ($1,$2,'renewal',$3,$4) ON CONFLICT (stripe_event_id) DO NOTHING",
                user_id, amount, invoice_id, tier
            )
            background_tasks.add_task(notify_mrr, amount, email, tier, "renewal")
            background_tasks.add_task(
                send_renewal_receipt_email,
                email, uname, tier, amount,
                invoice_id,
                f"{period_start.strftime('%b %d')} – {period_end.strftime('%b %d, %Y')}",
                period_end.strftime("%B %d, %Y"),
            )

    # ── subscription.updated — status changes, upgrades, downgrades ────
    elif etype == "customer.subscription.updated":
        sub = event.data.object
        async with core.state.db_pool.acquire() as conn:
            lookup_key = _subscription_price_lookup_key(sub)
            new_tier = _webhook_tier(lookup_key)
            _, period_end = _subscription_period_datetimes(sub)

            # Fetch user before updating so we have old_tier for comparison
            user_row = await conn.fetchrow(
                "SELECT id, email, name, subscription_tier FROM users WHERE stripe_subscription_id = $1", sub.id
            )
            old_tier = user_row["subscription_tier"] if user_row else new_tier

            await _execute_user_tier_update(
                conn,
                """
                UPDATE users SET
                    subscription_tier   = $1,
                    subscription_status = $2,
                    current_period_end  = $3,
                    updated_at          = NOW()
                WHERE stripe_subscription_id = $4
                """,
                new_tier,
                _stripe_field(sub, "status"),
                period_end,
                _stripe_field(sub, "id"),
            )

            # Send tier change email only when tier actually changed
            if user_row and old_tier != new_tier:
                _email = user_row["email"]
                _name  = user_row["name"] or "there"
                _amount = 0.0  # Stripe doesn't provide amount here directly
                if _tier_is_upgrade(old_tier, new_tier):
                    background_tasks.add_task(
                        send_plan_upgraded_email,
                        _email, _name, old_tier, new_tier, _amount,
                        period_end.strftime("%B %d, %Y"),
                    )
                else:
                    background_tasks.add_task(
                        send_plan_downgraded_email,
                        _email, _name, old_tier, new_tier, _amount,
                        period_end.strftime("%B %d, %Y"),
                    )

    # ── invoice.payment_failed — notify user to update payment method ────
    elif etype == "invoice.payment_failed":
        inv = event.data.object
        sub_id = _invoice_subscription_id(inv)
        if sub_id:
            async with core.state.db_pool.acquire() as conn:
                user_row = await conn.fetchrow(
                    "SELECT email, name, subscription_tier FROM users WHERE stripe_subscription_id = $1",
                    sub_id,
                )
            if user_row:
                retry_ts = inv.get("next_payment_attempt")
                retry_date = (
                    datetime.fromtimestamp(retry_ts, tz=timezone.utc).strftime("%B %d, %Y")
                    if retry_ts else ""
                )
                failure_reason = ""
                try:
                    err = inv.get("last_finalization_error") or {}
                    failure_reason = err.get("message", "") if isinstance(err, dict) else str(err)
                except Exception:
                    pass
                background_tasks.add_task(
                    send_payment_failed_email,
                    user_row["email"],
                    user_row["name"] or "there",
                    user_row["subscription_tier"] or "free",
                    (inv.get("amount_due") or 0) / 100,
                    retry_date,
                    inv.get("id", ""),
                    failure_reason,
                )

    # ── subscription.deleted — downgrade to free OR execute deferred account deletion ─
    elif etype == "customer.subscription.deleted":
        sub = event.data.object
        async with core.state.db_pool.acquire() as conn:
            user_row = await conn.fetchrow(
                "SELECT * FROM users WHERE stripe_subscription_id = $1", sub.id
            )
            if not user_row:
                return {"status": "user_not_found"}

            user_dict = dict(user_row)

            if user_row.get("deletion_requested_at"):
                # User requested account deletion; period ended — execute full deletion now
                from routers.me import _execute_account_deletion

                deletion_log = await conn.fetchrow(
                    "SELECT id FROM account_deletion_log WHERE user_id = $1 AND completed_at IS NULL ORDER BY requested_at DESC LIMIT 1",
                    str(user_row["id"]),
                )
                result = await _execute_account_deletion(
                    conn,
                    user_dict,
                    initiated_by="account_deletion",
                    background_tasks=background_tasks,
                )
                if deletion_log:
                    await conn.execute(
                        """
                        UPDATE account_deletion_log
                        SET completed_at = NOW(), r2_keys_deleted = $2, tokens_revoked = $3,
                            stripe_cancelled = TRUE, rows_deleted = $4
                        WHERE id = $1
                        """,
                        deletion_log["id"],
                        result["r2_deleted"],
                        result["tokens_revoked"],
                        json.dumps(result["rows_deleted"]),
                    )
                logger.info(
                    f"[DELETION COMPLETE via subscription.deleted] user={user_row['id']} "
                    f"r2={result['r2_deleted']} tokens={result['tokens_revoked']}"
                )
            else:
                # Normal cancellation: downgrade to free, send emails
                await conn.execute("""
                    UPDATE users SET
                        subscription_tier   = 'free',
                        subscription_status = 'cancelled',
                        updated_at          = NOW()
                    WHERE stripe_subscription_id = $1
                """, sub.id)

                _email    = user_row["email"]
                _name     = user_row["name"] or "there"
                _old_tier = user_row["subscription_tier"] or "free"
                _trial_end = user_row.get("trial_end")
                _end_ts = _subscription_period_timestamps(sub)[1]
                _access_until = (
                    datetime.fromtimestamp(_end_ts, tz=timezone.utc).strftime("%B %d, %Y")
                    if _end_ts
                    else "now"
                )

                if _trial_end and _trial_end > _now_utc():
                    background_tasks.add_task(
                        send_trial_cancelled_email,
                        _email, _name, _old_tier, _access_until,
                    )
                else:
                    background_tasks.add_task(
                        send_subscription_cancelled_email,
                        _email, _name, _old_tier, _access_until,
                    )

    # Record the event last: if handling above raised, the event is NOT marked,
    # so a Stripe retry reprocesses it (per-operation idempotency dedups the
    # writes that already succeeded).
    async with core.state.db_pool.acquire() as _done_conn:
        await _mark_event_processed(_done_conn, event_id, etype)

    return {"status": "ok"}


# ============================================================
# Helpers
# ============================================================

async def _do_monthly_refill(conn, user_id, tier, ent, invoice_id, period_start, period_end):
    """Credit monthly PUT+AIC. Deduped by invoice_id — safe to call on webhook retry."""
    # Check dedup table
    try:
        existing = await conn.fetchrow(
            "SELECT invoice_id FROM stripe_invoice_log WHERE invoice_id = $1", invoice_id
        )
        if existing:
            logger.info(f"Monthly refill already processed for {invoice_id}, skipping.")
            return False
    except Exception:
        pass  # Table may not exist yet — proceed, credit_wallet is idempotent enough

    put_amount = ent.put_monthly
    aic_amount = ent.aic_monthly

    if put_amount > 0:
        await conn.execute(
            "UPDATE wallets SET put_balance = put_balance + $1, updated_at = NOW() WHERE user_id = $2",
            put_amount, user_id
        )
        await ledger_entry(conn, user_id, "put", put_amount, "monthly_refill",
                           stripe_event_id=invoice_id)

    if aic_amount > 0:
        await conn.execute(
            "UPDATE wallets SET aic_balance = aic_balance + $1, updated_at = NOW() WHERE user_id = $2",
            aic_amount, user_id
        )
        await ledger_entry(conn, user_id, "aic", aic_amount, "monthly_refill",
                           stripe_event_id=invoice_id)

    try:
        await conn.execute("""
            INSERT INTO stripe_invoice_log
                (invoice_id, user_id, tier_slug, put_credited, aic_credited, period_start, period_end)
            VALUES ($1,$2,$3,$4,$5,$6,$7)
            ON CONFLICT (invoice_id) DO NOTHING
        """, invoice_id, user_id, tier, put_amount, aic_amount, period_start, period_end)
    except Exception:
        pass  # Non-critical — dedup log insert failure doesn't break billing

    logger.info(f"Monthly refill: user={user_id} tier={tier} +{put_amount} PUT +{aic_amount} AIC invoice={invoice_id}")
    return True
