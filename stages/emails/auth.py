"""
UploadM8 — Phase 1: Auth Emails
================================
  send_welcome_email               → /api/auth/register  (replaces raw HTML)
  send_password_reset_email        → /api/auth/forgot-password
  send_password_changed_email      → /api/auth/reset-password + /api/auth/change-password
  send_account_deleted_email       → DELETE /api/me
  send_email_change_email          → /api/admin/users/{id}/email  (verification link)
  send_admin_reset_password_email  → /api/admin/users/{id}/reset-password (temp password)
"""

import logging
from .base import (
    send_email, mailgun_ready,
    email_shell, intro_row, body_row, cta_button, tinted_box,
    check_list, alert_banner, numbered_steps, spacer,
    GRAD_ORANGE, GRAD_RED, GRAD_DARK, GRAD_GREEN, GRAD_BLUE,
    URL_DASHBOARD, URL_SETTINGS, SUPPORT_EMAIL, FRONTEND_URL,
)

logger = logging.getLogger("uploadm8-worker")


# ─────────────────────────────────────────────────────────────────────────────
# 1. New user welcome  (replaces the bare <h1>Welcome</h1> in /api/auth/register)
# ─────────────────────────────────────────────────────────────────────────────
async def send_welcome_email(email: str, name: str) -> None:
    """
    Warm onboarding email fired immediately after a new user registers.
    Replaces the raw HTML send_email call at line ~1902 in app.py:

    BEFORE:
        background_tasks.add_task(send_email, data.email, "Welcome to UploadM8!", f"<h1>Welcome, {data.name}!</h1>...")
    AFTER:
        from stages.emails.auth import send_welcome_email
        background_tasks.add_task(send_welcome_email, data.email, data.name)
    """
    if not mailgun_ready():
        return

    html = email_shell(
        gradient=GRAD_ORANGE,
        tagline="Upload once. Publish everywhere.",
        body_rows=(
            intro_row(
                f"Welcome to UploadM8, {name}! 🎉",
                "You've just unlocked the fastest way to publish short-form video across "
                "TikTok, YouTube Shorts, Instagram Reels, and Facebook Reels — simultaneously. "
                "You also have <strong style='color:#f97316;'>30 free upload tokens</strong> "
                "waiting in your wallet to get you started.",
            )
            + numbered_steps(
                ("Connect Your Platforms",
                 "Link TikTok, YouTube, Instagram, and Facebook from your Settings page."),
                ("Upload Your First Video",
                 "Drop in your video and we handle transcoding, captions, and publishing."),
                ("Watch the Numbers Move",
                 "Track views, engagement, and performance from your Analytics dashboard."),
            )
            + cta_button("Get Started Now", URL_DASHBOARD, pt="12px", pb="20px")
            + tinted_box(
                f'<p style="margin:0;color:#9ca3af;font-size:13px;line-height:1.6;">'
                f'Questions? We\'re here. Reply to this email or reach out at '
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#f97316;text-decoration:none;">'
                f'{SUPPORT_EMAIL}</a>.</p>',
                hex_color="#374151",
                pb="36px",
            )
        ),
        footer_note="You received this because you created an UploadM8 account.",
    )

    await send_email(email, f"Welcome to UploadM8, {name}! 🎉 Your 30 free tokens are ready", html)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Password reset link
# ─────────────────────────────────────────────────────────────────────────────
async def send_password_reset_email(email: str, reset_link: str) -> None:
    """
    Fired from /api/auth/forgot-password.
    reset_link points to {FRONTEND_URL}/reset-password?token=xxx
    (the reset-password.html page now exists in the frontend).
    Expiry (60 min) is shown clearly so users don't sit on a stale link.
    """
    if not mailgun_ready():
        return

    html = email_shell(
        gradient=GRAD_ORANGE,
        tagline="Keeping your account secure",
        body_rows=(
            intro_row(
                "Reset your password 🔑",
                "We received a request to reset the password on your UploadM8 account. "
                "Click the button below — this link is single-use and expires in "
                "<strong style='color:#f97316;'>60 minutes</strong>.",
            )
            + cta_button("Reset My Password", reset_link, pt="24px", pb="20px")
            + tinted_box(
                '<p style="margin:0;color:#9ca3af;font-size:13px;line-height:1.6;">'
                '⚠️&nbsp; If you did <strong style="color:#ffffff;">not</strong> '
                'request this reset, you can safely ignore this email — your password will '
                'not change. If you\'re concerned, contact us at '
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#f97316;text-decoration:none;">'
                f'{SUPPORT_EMAIL}</a>.</p>',
                hex_color="#374151",
                pb="36px",
            )
        ),
        footer_note="You received this because a password reset was requested for your account.",
    )

    await send_email(email, "Reset your UploadM8 password 🔑", html)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Password successfully changed — security confirmation
# ─────────────────────────────────────────────────────────────────────────────
async def send_password_changed_email(email: str, name: str) -> None:
    """
    Security receipt sent after a successful password change or reset.
    Hook into:
      POST /api/auth/reset-password   on success
      POST /api/auth/change-password  on success
    """
    if not mailgun_ready():
        return

    html = email_shell(
        gradient=GRAD_RED,
        tagline="Account security alert",
        body_rows=(
            intro_row(
                f"Password changed, {name} 🔒",
                "Your UploadM8 password was just changed successfully. "
                "All existing sessions have been signed out for your security.",
            )
            + check_list(
                "Password updated successfully",
                "All previous sessions signed out automatically",
                "Your videos, data, and connections are safe",
                hex_color="#22c55e",
            )
            + alert_banner(
                "⚠️&nbsp; <strong>Wasn't you?</strong> Contact us immediately at "
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#ffffff;text-decoration:underline;">'
                f"{SUPPORT_EMAIL}</a> — your account may be at risk.",
                hex_color="#ef4444",
            )
            + spacer("20px")
            + cta_button("Sign In to Your Account", f"{FRONTEND_URL}/login.html")
        ),
        footer_note="You received this security alert because your password was changed.",
    )

    await send_email(email, "Your UploadM8 password was changed ⚠️", html)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Account deletion confirmation
# ─────────────────────────────────────────────────────────────────────────────
async def send_account_deleted_email(email: str, name: str) -> None:
    """
    Goodbye email sent immediately after DELETE /api/me completes.
    Send BEFORE the user row is wiped so the address still resolves.
    """
    if not mailgun_ready():
        return

    html = email_shell(
        gradient=GRAD_DARK,
        tagline="We're sorry to see you go",
        body_rows=(
            intro_row(
                f"Goodbye, {name}.",
                "Your UploadM8 account and all associated data have been permanently deleted. "
                "Thank you for being part of the community — we genuinely hope we got to help "
                "your content reach more people.",
            )
            + check_list(
                "All personal data permanently deleted",
                "Platform OAuth tokens revoked from our system",
                "Videos and storage cleared",
                "Stripe subscription cancelled (if active)",
                hex_color="#6b7280",
            )
            + tinted_box(
                '<p style="margin:0 0 8px;color:#ffffff;font-size:15px;font-weight:600;">'
                'Changed your mind?</p>'
                '<p style="margin:0;color:#9ca3af;font-size:14px;line-height:1.65;">'
                'You\'re always welcome back. Creating a new account takes less than a minute at '
                f'<a href="{FRONTEND_URL}" style="color:#f97316;text-decoration:none;">app.uploadm8.com</a>.'
                '</p>',
                hex_color="#374151",
            )
            + tinted_box(
                f'<p style="margin:0;color:#9ca3af;font-size:13px;line-height:1.6;">'
                f'If this deletion was made in error, email us within 7 days at '
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#f97316;text-decoration:none;">'
                f'{SUPPORT_EMAIL}</a> and we\'ll do everything we can to help.</p>',
                hex_color="#f97316",
                pb="36px",
            )
        ),
        footer_note="This is a confirmation that your account deletion request was completed.",
    )

    await send_email(email, "Your UploadM8 account has been deleted", html)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Email change verification link
# ─────────────────────────────────────────────────────────────────────────────
async def send_email_change_email(
    new_email: str,
    old_email: str,
    name: str,
    verification_link: str,
) -> None:
    """
    Sent to the NEW email address when a user (or admin) requests an email change.
    The verification_link points to {FRONTEND_URL}/verify-email?token=xxx

    Hook into /api/admin/users/{id}/email after inserting into email_changes:

        verification_link = f"{FRONTEND_URL}/verify-email?token={verification_token}"
        background_tasks.add_task(
            send_email_change_email,
            new_email, old["email"], target_user["name"], verification_link
        )
    """
    if not mailgun_ready():
        return

    html = email_shell(
        gradient=GRAD_BLUE,
        tagline="Account email update",
        body_rows=(
            intro_row(
                f"Verify your new email, {name} ✉️",
                f"A request was made to change your UploadM8 account email from "
                f"<strong style='color:#9ca3af;'>{old_email}</strong> to "
                f"<strong style='color:#60a5fa;'>{new_email}</strong>. "
                "Click the button below to confirm this change.",
            )
            + cta_button("Verify New Email", verification_link, pt="24px", pb="20px")
            + tinted_box(
                '<p style="margin:0;color:#9ca3af;font-size:13px;line-height:1.6;">'
                '⚠️&nbsp; If you did <strong style="color:#ffffff;">not</strong> '
                'request an email change, contact us immediately at '
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#f97316;text-decoration:none;">'
                f'{SUPPORT_EMAIL}</a>. Your account may be at risk.</p>',
                hex_color="#374151",
                pb="36px",
            )
        ),
        footer_note="You received this because an email change was requested for your UploadM8 account.",
    )

    await send_email(new_email, "Verify your new UploadM8 email address ✉️", html)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Admin-forced password reset (temp password delivered to user)
# ─────────────────────────────────────────────────────────────────────────────
async def send_admin_reset_password_email(
    email: str,
    name: str,
    temp_password: str,
) -> None:
    """
    Sent to the user when an admin resets their password via
    POST /api/admin/users/{user_id}/reset-password.

    The user receives their temporary password and is instructed to change it
    immediately on login (must_reset_password=true is already set in the DB).

    Hook into admin_reset_password() after the DB write:

        background_tasks.add_task(
            send_admin_reset_password_email,
            target_user["email"], target_user["name"] or "there", payload.temp_password
        )
    """
    if not mailgun_ready():
        return

    html = email_shell(
        gradient=GRAD_RED,
        tagline="Account security — action required",
        body_rows=(
            intro_row(
                f"Your password has been reset, {name} 🔐",
                "An UploadM8 administrator has reset your account password. "
                "Use the temporary password below to sign in, then you'll be "
                "prompted to set a new one immediately.",
            )
            + tinted_box(
                f'<p style="margin:0 0 8px;color:#6b7280;font-size:11px;text-transform:uppercase;'
                f'letter-spacing:1px;">Temporary Password</p>'
                f'<p style="margin:0;color:#f97316;font-size:22px;font-weight:700;'
                f'font-family:monospace;letter-spacing:2px;">{temp_password}</p>'
                f'<p style="margin:10px 0 0;color:#6b7280;font-size:12px;">'
                f'This temporary password expires in 7 days.</p>',
                hex_color="#f97316",
            )
            + cta_button("Sign In Now", f"{FRONTEND_URL}/login.html", pt="8px", pb="20px")
            + alert_banner(
                "⚠️&nbsp; <strong>Change your password immediately</strong> after signing in. "
                "You will be prompted to do so automatically. "
                f'If you didn\'t expect this, contact us at '
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#ffffff;text-decoration:underline;">'
                f'{SUPPORT_EMAIL}</a>.',
                hex_color="#ef4444",
                pb="36px",
            )
        ),
        footer_note="You received this because an admin reset your UploadM8 account password.",
    )

    await send_email(email, "Your UploadM8 password has been reset — action required 🔐", html)
