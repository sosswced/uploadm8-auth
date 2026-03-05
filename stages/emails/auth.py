"""
UploadM8 — Phase 1: Auth Emails
================================
  send_password_reset_email    → /api/auth/forgot-password
  send_password_changed_email  → /api/auth/reset-password + /api/auth/change-password
  send_account_deleted_email   → DELETE /api/me
"""

import logging
from .base import (
    send_email, mailgun_ready,
    email_shell, intro_row, body_row, cta_button, tinted_box,
    check_list, alert_banner, spacer,
    GRAD_ORANGE, GRAD_RED, GRAD_DARK,
    URL_DASHBOARD, SUPPORT_EMAIL, FRONTEND_URL,
)

logger = logging.getLogger("uploadm8-worker")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Password reset link
# ─────────────────────────────────────────────────────────────────────────────
async def send_password_reset_email(email: str, reset_link: str) -> None:
    """
    Replaces the bare <p>reset_link</p> currently in /api/auth/forgot-password.
    Expiry (60 min) is shown clearly so users don't sit on a stale link.
    """
    if not mailgun_ready():
        return

    html = email_shell(
        gradient=GRAD_ORANGE,
        tagline="Keeping your account secure",
        body_rows=(
            intro_row(
                "Reset your password &#128272;",
                "We received a request to reset the password on your UploadM8 account. "
                "Click the button below — this link is single-use and expires in "
                "<strong style='color:#f97316;'>60 minutes</strong>.",
            )
            + cta_button("Reset My Password", reset_link, pt="24px", pb="20px")
            + tinted_box(
                '<p style="margin:0;color:#9ca3af;font-size:13px;line-height:1.6;">'
                '&#9888;&#65039;&nbsp; If you did <strong style="color:#ffffff;">not</strong> '
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

    await send_email(email, "Reset your UploadM8 password &#128272;", html)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Password successfully changed — security confirmation
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
                f"Password changed, {name} &#128274;",
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
                "&#9888;&#65039;&nbsp; <strong>Wasn't you?</strong> Contact us immediately at "
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#ffffff;text-decoration:underline;">'
                f"{SUPPORT_EMAIL}</a> — your account may be at risk.",
                hex_color="#ef4444",
            )
            + spacer("20px")
            + cta_button("Sign In to Your Account", f"{FRONTEND_URL}/login.html")
        ),
        footer_note="You received this security alert because your password was changed.",
    )

    await send_email(email, "Your UploadM8 password was changed &#9888;&#65039;", html)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Account deletion confirmation
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
