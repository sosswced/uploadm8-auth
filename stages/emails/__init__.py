"""
UploadM8 Email Package  —  stages/emails/
==========================================
All email functions exported from one import:

    from stages.emails import (
        # Auth
        send_welcome_email,
        send_password_reset_email,
        send_password_changed_email,
        send_account_deleted_email,
        send_email_change_email,
        send_admin_reset_password_email,
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
        # Lifecycle
        send_payment_failed_email,
        send_trial_ending_reminder_email,
        send_low_token_warning_email,
    )

─────────────────────────────────────────────────────────────────────────────
APP.PY INTEGRATION GUIDE  —  exactly where to add each new call
─────────────────────────────────────────────────────────────────────────────

NEW CALLS TO WIRE IN app.py:
─────────────────────────────

1. /api/auth/register  (line ~1902) — Replace raw welcome email:
   BEFORE:
       background_tasks.add_task(send_email, data.email, "Welcome to UploadM8!", f"<h1>Welcome, {data.name}!</h1>...")
   AFTER:
       background_tasks.add_task(send_welcome_email, data.email, data.name)

2. /api/admin/users/{id}/reset-password — After the DB write (line ~7355):
       target_row = await conn.fetchrow("SELECT email, name FROM users WHERE id=$1", user_id)
       if target_row:
           background_tasks.add_task(
               send_admin_reset_password_email,
               target_row["email"], target_row["name"] or "there", payload.temp_password
           )

3. /api/admin/users/{id}/email — After inserting verification_token (line ~7292):
       verify_link = f"{FRONTEND_URL}/verify-email?token={verification_token}"
       background_tasks.add_task(
           send_email_change_email,
           new_email, old["email"], target_user["name"] or "there", verify_link
       )

4. Stripe webhook — invoice.payment_failed (add new elif block):
       elif event_type == "invoice.payment_failed":
           inv = event["data"]["object"]
           sub_id = inv.get("subscription")
           if sub_id:
               async with db_pool.acquire() as conn:
                   user_row = await conn.fetchrow(
                       "SELECT email, name, subscription_tier FROM users WHERE stripe_subscription_id=$1", sub_id
                   )
               if user_row:
                   retry_ts = inv.get("next_payment_attempt")
                   retry_date = datetime.fromtimestamp(retry_ts, tz=timezone.utc).strftime("%B %d, %Y") if retry_ts else ""
                   background_tasks.add_task(
                       send_payment_failed_email,
                       user_row["email"], user_row["name"] or "there",
                       user_row["subscription_tier"],
                       inv.get("amount_due", 0) / 100,
                       retry_date, inv.get("id", ""),
                   )

EXISTING WIRING (already in app.py — confirmed active):
─────────────────────────────────────────────────────────
- send_password_reset_email      → /api/auth/forgot-password         ✓
- send_password_changed_email    → /api/auth/reset-password           ✓
- send_password_changed_email    → /api/auth/change-password          ✓
- send_account_deleted_email     → DELETE /api/me                     ✓
- send_subscription_started_email→ checkout.session.completed         ✓
- send_trial_started_email       → checkout.session.completed (trial) ✓
- send_trial_cancelled_email     → customer.subscription.deleted      ✓
- send_subscription_cancelled_email → customer.subscription.deleted   ✓
- send_renewal_receipt_email     → invoice.paid                       ✓
- send_plan_upgraded_email       → customer.subscription.updated      ✓
- send_plan_downgraded_email     → customer.subscription.updated      ✓
- send_topup_receipt_email       → checkout.session.completed (pay)   ✓
- send_announcement_email        → admin announcement endpoint        ✓
- send_friends_family_welcome_email → tier grant                      ✓
- send_agency_welcome_email      → tier grant                         ✓
- send_master_admin_welcome_email→ tier grant                         ✓
- send_admin_wallet_topup_email  → admin wallet credit                ✓
- send_admin_tier_switch_email   → admin tier change                  ✓

─────────────────────────────────────────────────────────────────────────────
FRONTEND PAGES NEEDED (now created):
─────────────────────────────────────
- /reset-password.html   → password reset link destination     ✓ NEW
- /verify-email.html     → email change verification           ✓ NEW
- /unsubscribe.html      → email preferences management        ✓ NEW
─────────────────────────────────────────────────────────────────────────────
"""

# ── Phase 1: Auth ─────────────────────────────────────────────────────────────
from .auth import (
    send_welcome_email,
    send_password_reset_email,
    send_password_changed_email,
    send_account_deleted_email,
    send_email_change_email,
    send_admin_reset_password_email,
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

# ── Phase 6: Lifecycle ────────────────────────────────────────────────────────
from .lifecycle import (
    send_payment_failed_email,
    send_trial_ending_reminder_email,
    send_low_token_warning_email,
)


__all__ = [
    # Auth
    "send_welcome_email",
    "send_password_reset_email",
    "send_password_changed_email",
    "send_account_deleted_email",
    "send_email_change_email",
    "send_admin_reset_password_email",
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
    # Lifecycle
    "send_payment_failed_email",
    "send_trial_ending_reminder_email",
    "send_low_token_warning_email",
]
