"""
UploadM8 Billing routes — extracted from app.py.
Stripe checkout, portal, session retrieval, and webhook handling.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

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
from core.models import CheckoutRequest
from core.notifications import notify_mrr, notify_topup
import core.state
from core.wallet import credit_wallet, ledger_entry
from routers.preferences import get_user_prefs_for_upload
from stages.ai_service_costs import compute_presign_put_aic_costs
from stages.entitlements import (
    STRIPE_LOOKUP_TO_TIER,
    TIER_CONFIG,
    TOPUP_PRODUCTS,
    get_entitlements_for_tier,
)
from stages.emails import (
    send_subscription_started_email,
    send_trial_started_email,
    send_plan_upgraded_email,
    send_plan_downgraded_email,
    send_topup_receipt_email,
    send_renewal_receipt_email,
    send_subscription_cancelled_email,
    send_trial_cancelled_email,
    send_payment_failed_email,
)

logger = logging.getLogger("uploadm8-api")

router = APIRouter(prefix="/api/billing", tags=["billing"])


# ============================================================
# Routes
# ============================================================

@router.post("/checkout")
async def create_checkout(data: CheckoutRequest, user: dict = Depends(get_current_user)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(503, "Billing not configured")

    async with core.state.db_pool.acquire() as conn:
        # ── Double-subscribe guard ────────────────────────────────────
        if data.kind == "subscription":
            existing_sub_id  = user.get("stripe_subscription_id")
            existing_status  = user.get("subscription_status")
            if existing_sub_id and existing_status in ("active", "trialing"):
                # User already has an active sub — send straight to billing portal
                # so they can upgrade/downgrade there rather than creating a duplicate
                try:
                    portal = stripe.billing_portal.Session.create(
                        customer=user["stripe_customer_id"],
                        return_url=f"{FRONTEND_URL}/settings.html#billing",
                    )
                    return {"checkout_url": portal.url, "session_id": None, "portal_redirect": True}
                except Exception:
                    pass  # Fall through to new checkout if portal fails

        customer_id = user.get("stripe_customer_id")
        if not customer_id:
            customer = stripe.Customer.create(email=user["email"], name=user.get("name") or user["email"])
            customer_id = customer.id
            await conn.execute("UPDATE users SET stripe_customer_id = $1 WHERE id = $2", customer_id, user["id"])

    prices = stripe.Price.list(lookup_keys=[data.lookup_key], active=True)
    if not prices.data:
        raise HTTPException(400, f"Price not found for lookup_key: {data.lookup_key}. Run stripe_setup.py to create prices.")

    if data.kind == "subscription":
        # Resolve trial days from entitlements (7 days for all paid tiers)
        tier = STRIPE_LOOKUP_TO_TIER.get(data.lookup_key, "free")
        ent  = get_entitlements_for_tier(tier)
        trial_days = ent.trial_days  # 0 for free/internal, 7 for paid

        session_params = dict(
            customer              = customer_id,
            line_items            = [{"price": prices.data[0].id, "quantity": 1}],
            mode                  = "subscription",
            success_url           = STRIPE_SUCCESS_URL,
            cancel_url            = STRIPE_CANCEL_URL,
            allow_promotion_codes = True,
            metadata              = {"user_id": str(user["id"]), "tier": tier},
        )
        if trial_days > 0:
            session_params["subscription_data"] = {
                "trial_period_days": trial_days,
                "metadata": {"user_id": str(user["id"]), "tier": tier},
            }

        session = stripe.checkout.Session.create(**session_params)

    else:  # topup / one-time payment
        product = TOPUP_PRODUCTS.get(data.lookup_key, {})
        if not product:
            raise HTTPException(400, f"Unknown topup product: {data.lookup_key}")

        session = stripe.checkout.Session.create(
            customer    = customer_id,
            line_items  = [{"price": prices.data[0].id, "quantity": 1}],
            mode        = "payment",
            success_url = STRIPE_SUCCESS_URL,
            cancel_url  = STRIPE_CANCEL_URL,
            metadata    = {
                "user_id": str(user["id"]),
                "wallet":  product.get("wallet", "put"),
                "amount":  str(product.get("amount", 0)),
            },
        )

    return {"checkout_url": session.url, "session_id": session.id}

@router.post("/portal")
async def create_portal(user: dict = Depends(get_current_user)):
    if not user.get("stripe_customer_id"): raise HTTPException(400, "No billing account")
    session = stripe.billing_portal.Session.create(customer=user["stripe_customer_id"], return_url=f"{FRONTEND_URL}/settings.html")
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

    # Security: session must belong to this user (metadata.user_id match)
    meta_user_id = (sess.get("metadata") or {}).get("user_id")
    if meta_user_id and str(meta_user_id) != str(user.get("id")):
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
            sub_status            = sub.get("status")        # active | trialing | past_due
            trial_end_ts          = sub.get("trial_end")     # unix ts or None
            current_period_end_ts = sub.get("current_period_end")
            try:
                price      = sub["items"]["data"][0]["price"]
                lookup_key = price.get("lookup_key", "")
                tier       = STRIPE_LOOKUP_TO_TIER.get(lookup_key)
                if not tier:
                    # Fallback: try to match by product name
                    prod_id = price.get("product")
                    if prod_id:
                        prod = stripe.Product.retrieve(prod_id)
                        plan_name = prod.get("name", "")
            except Exception:
                pass

    # ── Topup fields ─────────────────────────────────────────────────
    topup_wallet = None
    topup_amount = None
    if mode == "payment":
        meta         = sess.get("metadata") or {}
        topup_wallet = meta.get("wallet")        # "put" | "aic"
        topup_amount = meta.get("amount")        # token count as string

    # Resolve display name
    if tier:
        cfg       = TIER_CONFIG.get(tier, {})
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
        "topup_amount":        int(topup_amount) if topup_amount else None,
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
    use_hud = bool(body.use_hud) and ent.can_burn_hud
    put, aic = compute_presign_put_aic_costs(
        ent,
        num_publish_targets=body.num_publish_targets,
        file_size=body.file_size,
        duration_hint=body.duration_seconds,
        has_telemetry=body.has_telemetry,
        use_ai_checkbox=body.use_ai,
        hud_enabled_effective=use_hud,
        user_prefs=user_prefs,
        num_thumbnails_override=body.num_thumbnails,
    )
    return {"put_cost": put, "aic_cost": aic}


@router.get("/subscription/actions")
async def billing_subscription_actions(user: dict = Depends(get_current_user)):
    st = str(user.get("subscription_status") or "").lower()
    sid = user.get("stripe_subscription_id")
    can = bool(sid and st in ("active", "trialing", "past_due"))
    return {"can_manage_subscription": can}


@router.post("/subscription/action")
async def billing_subscription_action(
    body: BillingSubscriptionActionRequest,
    user: dict = Depends(get_current_user),
):
    """Stripe Customer Portal flows + subscription maintenance (best-effort)."""
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
                sub = stripe.Subscription.retrieve(session.subscription)
                lookup_key = sub["items"]["data"][0]["price"].get("lookup_key", "")
                tier = STRIPE_LOOKUP_TO_TIER.get(lookup_key, "free")
                ent  = get_entitlements_for_tier(tier)
                status = sub.status

                period_start = datetime.fromtimestamp(sub.current_period_start, tz=timezone.utc) if sub.get("current_period_start") else _now_utc()
                period_end   = datetime.fromtimestamp(sub.current_period_end, tz=timezone.utc)
                trial_end    = datetime.fromtimestamp(sub.trial_end, tz=timezone.utc) if sub.get("trial_end") else None

                await conn.execute("""
                    UPDATE users SET
                        subscription_tier      = $1,
                        stripe_subscription_id = $2,
                        subscription_status    = $3,
                        current_period_end     = $4,
                        trial_end              = $5,
                        updated_at             = NOW()
                    WHERE id = $6
                """, tier, session.subscription, status, period_end, trial_end, user_id)

                # Seed wallet for first period / trial — deduped by invoice id
                refill_ref = sub.get("latest_invoice") or session.id
                await _do_monthly_refill(conn, user_id, tier, ent, refill_ref, period_start, period_end)

                amount = (session.amount_total or 0) / 100
                await conn.execute(
                    "INSERT INTO revenue_tracking (user_id, amount, source, stripe_event_id, plan) "
                    "VALUES ($1,$2,'subscription',$3,$4) ON CONFLICT DO NOTHING",
                    user_id, amount, session.id, tier
                )
                background_tasks.add_task(notify_mrr, amount, email, tier, status)

                # ── Welcome email: trial vs paid ──────────────────────────────
                if trial_end:
                    trial_days = sub.get("trial_period_days") or 14
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
                wallet_type   = session.metadata.get("wallet", "put")
                amount_tokens = int(session.metadata.get("amount", 0))
                if amount_tokens > 0:
                    # First top-up bonus: +25% to incentivize trying paid credits
                    prior = await conn.fetchval(
                        "SELECT 1 FROM token_ledger WHERE user_id = $1 AND reason = 'topup_purchase' LIMIT 1",
                        user_id
                    )
                    bonus = int(amount_tokens * 0.25) if not prior else 0
                    total = amount_tokens + bonus
                    await credit_wallet(conn, user_id, wallet_type, total, "topup_purchase", session.id)
                    amount = (session.amount_total or 0) / 100
                    await conn.execute(
                        "INSERT INTO revenue_tracking (user_id, amount, source, stripe_event_id, plan) "
                        "VALUES ($1,$2,'topup',$3,$4) ON CONFLICT DO NOTHING",
                        user_id, amount, session.id, f"{wallet_type}_{amount_tokens}"
                    )
                    background_tasks.add_task(notify_topup, amount, email, wallet_type, total)
                    background_tasks.add_task(
                        send_topup_receipt_email,
                        email, uname, wallet_type, total, amount, 0, session.id, bonus_tokens=bonus,
                    )

    # ── invoice.paid — monthly wallet refill on every renewal ──────────
    elif etype == "invoice.paid":
        invoice = event.data.object
        sub_id  = invoice.get("subscription")
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
            sub = stripe.Subscription.retrieve(sub_id)
            lookup_key = sub["items"]["data"][0]["price"].get("lookup_key", "")
            tier = STRIPE_LOOKUP_TO_TIER.get(lookup_key, user_row["subscription_tier"] or "free")
            ent  = get_entitlements_for_tier(tier)

            period_start = datetime.fromtimestamp(invoice.period_start, tz=timezone.utc)
            period_end   = datetime.fromtimestamp(invoice.period_end, tz=timezone.utc)
            invoice_id   = invoice.id

            await conn.execute("""
                UPDATE users SET
                    subscription_tier   = $1,
                    subscription_status = 'active',
                    current_period_end  = $2,
                    updated_at          = NOW()
                WHERE id = $3
            """, tier, period_end, user_id)

            # Monthly wallet refill — deduped by invoice_id
            await _do_monthly_refill(conn, user_id, tier, ent, invoice_id, period_start, period_end)

            amount = (invoice.amount_paid or 0) / 100
            await conn.execute(
                "INSERT INTO revenue_tracking (user_id, amount, source, stripe_event_id, plan) "
                "VALUES ($1,$2,'renewal',$3,$4) ON CONFLICT DO NOTHING",
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
            lookup_key = sub["items"]["data"][0]["price"].get("lookup_key", "")
            new_tier = STRIPE_LOOKUP_TO_TIER.get(lookup_key, "free")
            period_end = datetime.fromtimestamp(sub.current_period_end, tz=timezone.utc)

            # Fetch user before updating so we have old_tier for comparison
            user_row = await conn.fetchrow(
                "SELECT id, email, name, subscription_tier FROM users WHERE stripe_subscription_id = $1", sub.id
            )
            old_tier = user_row["subscription_tier"] if user_row else new_tier

            await conn.execute("""
                UPDATE users SET
                    subscription_tier   = $1,
                    subscription_status = $2,
                    current_period_end  = $3,
                    updated_at          = NOW()
                WHERE stripe_subscription_id = $4
            """, new_tier, sub.status, period_end, sub.id)

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
        sub_id = inv.get("subscription")
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
                result = await _execute_account_deletion(conn, user_dict, initiated_by="account_deletion")
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
                _access_until = datetime.fromtimestamp(
                    sub.current_period_end, tz=timezone.utc
                ).strftime("%B %d, %Y") if sub.get("current_period_end") else "now"

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
