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
APP.PY / WORKER INTEGRATION — wiring status (verified)
─────────────────────────────────────────────────────────────────────────────

REMAINING: send_trial_ending_reminder_email
────────────────────────────────────────────
  Requires a scheduled job (e.g. APScheduler or Render cron) that runs daily and:
    - Queries users WHERE subscription_status='trialing' AND trial_end BETWEEN NOW() AND NOW()+4 days
    - Fires send_trial_ending_reminder_email, sets trial_reminder_sent=NOW()
  Schema: ALTER TABLE users ADD COLUMN IF NOT EXISTS trial_reminder_sent TIMESTAMPTZ;

EXISTING WIRING (confirmed active):
─────────────────────────────────────────────────────────
- send_welcome_email             → /api/auth/register                 
- send_password_reset_email      → /api/auth/forgot-password           
- send_password_changed_email    → /api/auth/reset-password            
- send_password_changed_email    → /api/auth/change-password            
- send_account_deleted_email     → DELETE /api/me                      
- send_email_change_email        → PUT /api/admin/users/{id}/email       
- send_admin_reset_password_email→ POST /api/admin/users/{id}/reset-password 
- send_subscription_started_email→ checkout.session.completed          
- send_trial_started_email       → checkout.session.completed (trial)  
- send_trial_cancelled_email     → customer.subscription.deleted       
- send_subscription_cancelled_email → customer.subscription.deleted    
- send_renewal_receipt_email     → invoice.paid                        
- send_plan_upgraded_email       → customer.subscription.updated       
- send_plan_downgraded_email     → customer.subscription.updated      
- send_topup_receipt_email       → checkout.session.completed (pay)   
- send_payment_failed_email      → invoice.payment_failed              
- send_announcement_email        → _execute_announcement_deliveries    
- send_friends_family_welcome_email → admin tier grant                 
- send_agency_welcome_email     → admin tier grant                     
- send_master_admin_welcome_email→ admin tier grant                     
- send_admin_wallet_topup_email  → POST /api/admin/users/{id}/wallet   
- send_admin_tier_switch_email   → admin tier change                   
- send_upload_completed_email    → worker (job succeeded/partial)      
- send_upload_failed_email       → worker (job failed)                  
- send_low_token_warning_email   → worker _capture_tokens (balance ≤5)  

NOT WIRED (requires scheduled job):
- send_trial_ending_reminder_email → needs daily cron + trial_reminder_sent column

─────────────────────────────────────────────────────────────────────────────
FRONTEND PAGES NEEDED (now created):
─────────────────────────────────────
- /reset-password.html   → password reset link destination      NEW
- /verify-email.html     → email change verification            NEW
- /unsubscribe.html      → email preferences management         NEW
─────────────────────────────────────────────────────────────────────────────
"""

# ── Phase 1: Auth ─────────────────────────────────────────────────────────────
from .auth import (
    send_signup_confirmation_email,
    send_welcome_email,
    send_fully_signed_up_guide_email,
    send_password_reset_email,
    send_password_changed_email,
    send_account_deleted_email,
    send_email_change_email,
    send_admin_email_change_notice_to_old_email,
    send_user_email_change_notice_to_old_email,
    send_admin_reset_password_email,
    send_login_anomaly_email,
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
    send_refund_receipt_email,
)

# ── Phase 4: Uploads ──────────────────────────────────────────────────────────
from .uploads import (
    send_upload_completed_email,
    send_upload_failed_email,
    send_scheduled_publish_alert_email,
)

# ── Phase 4b: Admin Actions ───────────────────────────────────────────────────
from .admin_actions import (
    send_admin_wallet_topup_email,
    send_admin_tier_switch_email,
    send_admin_account_status_email,
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

# ── Phase 7: Digests ──────────────────────────────────────────────────────────
from .digests import (
    send_monthly_user_kpi_digest_email,
    send_admin_weekly_kpi_digest_email,
    send_report_ready_email,
)


__all__ = [
    # Auth
    "send_signup_confirmation_email",
    "send_welcome_email",
    "send_fully_signed_up_guide_email",
    "send_password_reset_email",
    "send_password_changed_email",
    "send_account_deleted_email",
    "send_email_change_email",
    "send_admin_email_change_notice_to_old_email",
    "send_user_email_change_notice_to_old_email",
    "send_admin_reset_password_email",
    "send_login_anomaly_email",
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
    "send_refund_receipt_email",
    # Uploads
    "send_upload_completed_email",
    "send_upload_failed_email",
    "send_scheduled_publish_alert_email",
    # Admin actions
    "send_admin_wallet_topup_email",
    "send_admin_tier_switch_email",
    "send_admin_account_status_email",
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
    # Digests
    "send_monthly_user_kpi_digest_email",
    "send_admin_weekly_kpi_digest_email",
    "send_report_ready_email",
]
