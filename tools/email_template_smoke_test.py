import asyncio
import os
import sys
from typing import Any, Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stages.emails import admin_actions
from stages.emails import announcements
from stages.emails import auth
from stages.emails import billing_changes
from stages.emails import billing_subscriptions
from stages.emails import digests
from stages.emails import lifecycle
from stages.emails import uploads
from stages.emails import welcome_special


Captured = Tuple[str, str, str]
captured: List[Captured] = []


async def _fake_send_email(to: str, subject: str, html: str, from_addr: str = None, reply_to: str = None) -> None:
    assert to and "@" in to, f"invalid recipient: {to!r}"
    assert subject, "empty subject"
    assert subject.strip() == subject, f"subject has leading/trailing spaces: {subject!r}"
    assert "<html" in html.lower(), "html envelope missing"
    assert "uploadm8" in html.lower(), "brand text missing"
    captured.append((to, subject, html))


def _patch_module(module: Any) -> None:
    if hasattr(module, "mailgun_ready"):
        setattr(module, "mailgun_ready", lambda: True)
    if hasattr(module, "send_email"):
        setattr(module, "send_email", _fake_send_email)


def _get_new_email(before: int, label: str) -> Captured:
    after = len(captured)
    assert after == before + 1, f"{label}: expected 1 email, got {after - before}"
    return captured[-1]


async def _invoke(label: str, coro) -> Captured:
    before = len(captured)
    await coro
    return _get_new_email(before, label)


async def main() -> None:
    modules = [
        auth,
        billing_subscriptions,
        billing_changes,
        uploads,
        admin_actions,
        welcome_special,
        announcements,
        lifecycle,
        digests,
    ]
    for m in modules:
        _patch_module(m)

    results: Dict[str, Captured] = {}

    # Auth
    results["signup_confirmation"] = await _invoke(
        "signup_confirmation",
        auth.send_signup_confirmation_email("user@example.com", "Taylor", "https://app.uploadm8.com/confirm-email.html?token=abc123"),
    )
    results["welcome"] = await _invoke(
        "welcome",
        auth.send_welcome_email("user@example.com", "Taylor"),
    )
    results["fully_signed_up_guide"] = await _invoke(
        "fully_signed_up_guide",
        auth.send_fully_signed_up_guide_email("user@example.com", "Taylor", "creator_pro", 1200, 350, 12, 3),
    )
    results["password_reset"] = await _invoke(
        "password_reset",
        auth.send_password_reset_email("user@example.com", "https://app.uploadm8.com/reset-password.html?token=xyz789"),
    )
    results["password_changed"] = await _invoke(
        "password_changed",
        auth.send_password_changed_email("user@example.com", "Taylor"),
    )
    results["account_deleted"] = await _invoke(
        "account_deleted",
        auth.send_account_deleted_email("user@example.com", "Taylor"),
    )
    results["email_change"] = await _invoke(
        "email_change",
        auth.send_email_change_email(
            "new@example.com",
            "old@example.com",
            "Taylor",
            "https://app.uploadm8.com/verify-email.html?token=verify123",
        ),
    )
    results["admin_email_change_old_notice"] = await _invoke(
        "admin_email_change_old_notice",
        auth.send_admin_email_change_notice_to_old_email("old@example.com", "new@example.com", "Taylor"),
    )
    results["user_email_change_old_notice"] = await _invoke(
        "user_email_change_old_notice",
        auth.send_user_email_change_notice_to_old_email("old@example.com", "new@example.com", "Taylor"),
    )
    results["admin_reset_password"] = await _invoke(
        "admin_reset_password",
        auth.send_admin_reset_password_email("user@example.com", "Taylor", "TempP@ss123"),
    )
    results["login_anomaly"] = await _invoke(
        "login_anomaly",
        auth.send_login_anomaly_email("user@example.com", "Taylor", "203.0.113.10", "US", "Mozilla/5.0", "198.51.100.20"),
    )

    # Billing
    results["subscription_started"] = await _invoke(
        "subscription_started",
        billing_subscriptions.send_subscription_started_email("user@example.com", "Taylor", "creator_pro", 19.99, "April 10, 2026"),
    )
    results["trial_started"] = await _invoke(
        "trial_started",
        billing_subscriptions.send_trial_started_email("user@example.com", "Taylor", "creator_pro", "April 10, 2026", 14),
    )
    results["trial_cancelled"] = await _invoke(
        "trial_cancelled",
        billing_subscriptions.send_trial_cancelled_email("user@example.com", "Taylor", "creator_pro", "April 10, 2026"),
    )
    results["subscription_cancelled"] = await _invoke(
        "subscription_cancelled",
        billing_subscriptions.send_subscription_cancelled_email("user@example.com", "Taylor", "creator_pro", "April 10, 2026"),
    )
    results["renewal_receipt"] = await _invoke(
        "renewal_receipt",
        billing_subscriptions.send_renewal_receipt_email(
            "user@example.com",
            "Taylor",
            "creator_pro",
            19.99,
            "in_123456",
            "Mar 01 - Mar 31, 2026",
            "April 01, 2026",
            "Visa **** 4242",
        ),
    )
    results["plan_upgraded"] = await _invoke(
        "plan_upgraded",
        billing_changes.send_plan_upgraded_email("user@example.com", "Taylor", "creator_lite", "creator_pro", 19.99, "April 01, 2026"),
    )
    results["plan_downgraded"] = await _invoke(
        "plan_downgraded",
        billing_changes.send_plan_downgraded_email("user@example.com", "Taylor", "studio", "creator_pro", 19.99, "April 01, 2026"),
    )
    results["topup_receipt"] = await _invoke(
        "topup_receipt",
        billing_changes.send_topup_receipt_email("user@example.com", "Taylor", "put", 250, 9.99, 1500, "pi_123", bonus_tokens=62),
    )
    results["refund_receipt"] = await _invoke(
        "refund_receipt",
        billing_changes.send_refund_receipt_email("user@example.com", "Taylor", 9.99, "usd", "ch_123", "requested_by_customer", "refund"),
    )

    # Upload emails
    platform_results = [
        {"platform": "tiktok", "success": True, "platform_video_id": "abc", "account_username": "@demo"},
        {"platform": "youtube", "success": False, "error_message": "API quota exceeded"},
    ]
    results["upload_completed"] = await _invoke(
        "upload_completed",
        uploads.send_upload_completed_email(
            "user@example.com",
            "Taylor",
            "clip.mp4",
            ["tiktok", "youtube"],
            10,
            3,
            "up_123",
            45,
            "My Clip",
            "Caption here",
            ["tag1", "tag2"],
            platform_results,
        ),
    )
    results["upload_failed"] = await _invoke(
        "upload_failed",
        uploads.send_upload_failed_email(
            "user@example.com",
            "Taylor",
            "clip.mp4",
            ["tiktok", "youtube"],
            "publish failed",
            "up_123",
            "publish",
            "My Clip",
            "Caption here",
            ["tag1", "tag2"],
            platform_results,
        ),
    )
    results["scheduled_publish_alert"] = await _invoke(
        "scheduled_publish_alert",
        uploads.send_scheduled_publish_alert_email(
            "user@example.com",
            "Taylor",
            "clip.mp4",
            "March 30, 2026 18:00 UTC",
            "failed",
            "queue timeout",
            "up_123",
        ),
    )

    # Admin + special welcomes
    results["admin_wallet_topup"] = await _invoke(
        "admin_wallet_topup",
        admin_actions.send_admin_wallet_topup_email("user@example.com", "Taylor", "put", 100, 550, "support grant", "Thanks for your patience"),
    )
    results["admin_tier_switch"] = await _invoke(
        "admin_tier_switch",
        admin_actions.send_admin_tier_switch_email("user@example.com", "Taylor", "creator_lite", "studio", "manual promotion", True),
    )
    results["admin_account_status"] = await _invoke(
        "admin_account_status",
        admin_actions.send_admin_account_status_email("user@example.com", "Taylor", "banned", "policy violation"),
    )
    results["friends_family"] = await _invoke(
        "friends_family",
        welcome_special.send_friends_family_welcome_email("user@example.com", "Taylor"),
    )
    results["agency"] = await _invoke(
        "agency",
        welcome_special.send_agency_welcome_email("user@example.com", "Taylor", 99.99, "April 01, 2026"),
    )
    results["master_admin"] = await _invoke(
        "master_admin",
        welcome_special.send_master_admin_welcome_email("user@example.com", "Taylor"),
    )

    # Announcement, lifecycle, digests
    results["announcement"] = await _invoke(
        "announcement",
        announcements.send_announcement_email("user@example.com", "New Feature Drop", "We shipped a big update.", "Read more", "https://app.uploadm8.com/guide.html", "Creator Pro"),
    )
    results["payment_failed"] = await _invoke(
        "payment_failed",
        lifecycle.send_payment_failed_email("user@example.com", "Taylor", "creator_pro", 19.99, "April 05, 2026", "in_123", "Card declined"),
    )
    results["trial_ending"] = await _invoke(
        "trial_ending",
        lifecycle.send_trial_ending_reminder_email("user@example.com", "Taylor", "creator_pro", "April 10, 2026", 3, 19.99),
    )
    results["low_token"] = await _invoke(
        "low_token",
        lifecycle.send_low_token_warning_email("user@example.com", "Taylor", "put", 4, 5),
    )
    results["monthly_digest"] = await _invoke(
        "monthly_digest",
        digests.send_monthly_user_kpi_digest_email(
            "user@example.com",
            "Taylor",
            "creator_pro",
            "March 2026",
            45,
            96,
            120000,
            8400,
            700,
            200,
            500,
            120,
            comments=120,
            shares=30,
            platform_breakdown=[("tiktok", "20 uploads targeted"), ("youtube", "15 uploads targeted")],
        ),
    )
    results["admin_weekly_digest"] = await _invoke(
        "admin_weekly_digest",
        digests.send_admin_weekly_kpi_digest_email(
            "admin@example.com",
            "Admin",
            "Mar 20 - Mar 27, 2026",
            1200,
            90,
            320,
            1800,
            4200.0,
            1800.0,
            57.1,
            upload_success_pct=88,
            trialing_paid_users=12,
            platform_summary_lines=["tiktok: 900 targeted / 820 completed — 1,200,000 views, 40,000 likes"],
        ),
    )
    results["report_ready"] = await _invoke(
        "report_ready",
        digests.send_report_ready_email("user@example.com", "Taylor", "Analytics export (30d)", "https://app.uploadm8.com/api/exports/download?token=abc", "March 29, 2026 10:00 UTC"),
    )

    # Critical link checks
    assert "confirm-email.html?token=abc123" in results["signup_confirmation"][2], "signup confirmation link mismatch"
    assert "reset-password.html?token=xyz789" in results["password_reset"][2], "password reset link mismatch"
    assert "verify-email.html?token=verify123" in results["email_change"][2], "email change verify link mismatch"

    print(f"PASS: {len(results)} email scenarios rendered and validated.")


if __name__ == "__main__":
    asyncio.run(main())
