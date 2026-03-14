"""
UploadM8 Email Package  —  stages/emails/
==========================================
All 20 email functions in one import:

    from stages.emails import (
        # Auth
        send_password_reset_email,
        send_password_changed_email,
        send_account_deleted_email,
        # Billing — Subscriptions
        send_subscription_started_email,
        send_trial_started_email,
        send_trial_cancelled_email,
        send_subscription_cancelled_email,
        send_renewal_receipt_email,
        # Billing — Changes
        send_plan_upgraded_email,
        send_plan_downgraded_email,
        send_topup_receipt_email,
        # Uploads
        send_upload_completed_email,
        send_upload_failed_email,
        # Admin actions
        send_admin_wallet_topup_email,
        send_admin_tier_switch_email,
        # Heartfelt welcomes
        send_friends_family_welcome_email,
        send_agency_welcome_email,
        send_master_admin_welcome_email,
        # Announcements
        send_announcement_email,
    )

─────────────────────────────────────────────────────────────────────────────
APP.PY INTEGRATION GUIDE  —  exactly where to add each call
─────────────────────────────────────────────────────────────────────────────

PHASE 1 — AUTH
--------------
1. /api/auth/forgot-password  (line ~1908)
   REPLACE:
       html=f"<p>{reset_link}</p>"
   WITH:
       await send_password_reset_email(email, reset_link)

2. POST /api/auth/reset-password  (line ~1912, on success)
   ADD after password update:
       await send_password_changed_email(email, name)

3. POST /api/auth/change-password  (line ~2025, on success)
   ADD after password update:
       await send_password_changed_email(email, name)

4. DELETE /api/me  (line ~2341, BEFORE the DELETE query)
   ADD:
       await send_account_deleted_email(email, name)

PHASE 2 — BILLING: SUBSCRIPTIONS
----------------------------------
Inside the Stripe webhook handler for checkout.session.completed:

5. Subscription mode, no trial:
       await send_subscription_started_email(
           email, name, tier, amount/100, next_billing_date
       )

6. Subscription mode WITH trial:
       await send_trial_started_email(
           email, name, tier, trial_end_date, trial_days
       )

Inside customer.subscription.deleted:

7. Cancelled during trial (subscription_status was "trialing"):
       await send_trial_cancelled_email(email, name, tier, access_until)

8. Cancelled paid subscription:
       await send_subscription_cancelled_email(email, name, tier, access_until)

Inside invoice.paid (recurring only — skip first invoice):

9.    await send_renewal_receipt_email(
           email, name, tier, amount/100,
           invoice_id, billing_period, next_billing_date, payment_method
       )

PHASE 3 — BILLING: CHANGES
---------------------------
Inside customer.subscription.updated — compare old vs new tier:

10. Upgrade:
        await send_plan_upgraded_email(
            email, name, old_tier, new_tier, new_amount/100, next_billing_date
        )

11. Downgrade:
        await send_plan_downgraded_email(
            email, name, old_tier, new_tier, new_amount/100, effective_date
        )

Inside checkout.session.completed (mode=payment — token top-up):

12.     await send_topup_receipt_email(
            email, name, wallet_type, tokens_added, amount/100,
            new_balance, stripe_payment_id
        )

PHASE 4 — UPLOADS
------------------
In your upload worker finish handler:

13. On success:
        prefs = await conn.fetchrow(
            "SELECT email_notifications FROM user_preferences WHERE user_id=$1", user_id
        )
        if prefs and prefs["email_notifications"]:
            await send_upload_completed_email(
                email, name, filename, platforms, put_spent, aic_spent, upload_id
            )

14. On failure:
        if prefs and prefs["email_notifications"]:
            await send_upload_failed_email(
                email, name, filename, platforms, error_reason, upload_id, stage
            )

Admin action hooks:

15. After credit_wallet() is called by an admin:
        await send_admin_wallet_topup_email(
            email, name, wallet_type, tokens_added, new_balance, reason, admin_note
        )

16. After subscription_tier is manually updated by admin:
        is_up = _tier_rank(new_tier) >= _tier_rank(old_tier)
        await send_admin_tier_switch_email(
            email, name, old_tier, new_tier, admin_note, is_upgrade=is_up
        )

PHASE 5 — HEARTFELT WELCOMES
------------------------------
17. friends_family tier granted (admin grant or invite):
        await send_friends_family_welcome_email(email, name)

18. agency tier activated:
        await send_agency_welcome_email(email, name, amount/100, next_billing_date)

19. master_admin granted:
        await send_master_admin_welcome_email(email, name)

ANNOUNCEMENTS
-------------
20. In _execute_announcement_deliveries() (line ~7566):
    REPLACE:
        await send_email(dest, title, f"<h1>{title}</h1><p>{body}</p>")
    WITH:
        from stages.emails.announcements import send_announcement_email
        await send_announcement_email(dest, title, body)

─────────────────────────────────────────────────────────────────────────────
"""

# ── Phase 1: Auth ─────────────────────────────────────────────────────────────
from .auth import (
    send_password_reset_email,
    send_password_changed_email,
    send_account_deleted_email,
)

# ── Phase 2: Billing — Subscriptions ─────────────────────────────────────────
from .billing_subscriptions import (
    send_subscription_started_email,
    send_trial_started_email,
    send_trial_cancelled_email,
    send_subscription_cancelled_email,
    send_renewal_receipt_email,
)

# ── Phase 3: Billing — Changes & Receipts ────────────────────────────────────
from .billing_changes import (
    send_plan_upgraded_email,
    send_plan_downgraded_email,
    send_topup_receipt_email,
)

# ── Phase 4: Uploads ──────────────────────────────────────────────────────────
from .uploads import (
    send_upload_completed_email,
    send_upload_failed_email,
)

# ── Phase 4b: Admin Actions ───────────────────────────────────────────────────
from .admin_actions import (
    send_admin_wallet_topup_email,
    send_admin_tier_switch_email,
)

# ── Phase 5: Heartfelt Welcomes ───────────────────────────────────────────────
from .welcome_special import (
    send_friends_family_welcome_email,
    send_agency_welcome_email,
    send_master_admin_welcome_email,
)

# ── Phase 5b: Announcements ───────────────────────────────────────────────────
from .announcements import send_announcement_email


__all__ = [
    # Auth
    "send_password_reset_email",
    "send_password_changed_email",
    "send_account_deleted_email",
    # Billing — Subscriptions
    "send_subscription_started_email",
    "send_trial_started_email",
    "send_trial_cancelled_email",
    "send_subscription_cancelled_email",
    "send_renewal_receipt_email",
    # Billing — Changes
    "send_plan_upgraded_email",
    "send_plan_downgraded_email",
    "send_topup_receipt_email",
    # Uploads
    "send_upload_completed_email",
    "send_upload_failed_email",
    # Admin actions
    "send_admin_wallet_topup_email",
    "send_admin_tier_switch_email",
    # Heartfelt welcomes
    "send_friends_family_welcome_email",
    "send_agency_welcome_email",
    "send_master_admin_welcome_email",
    # Announcements
    "send_announcement_email",
]
