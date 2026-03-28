"""
UploadM8 — Phase 1: Auth Emails  (v2 — Enhanced Design)
=========================================================
  send_welcome_email               → /api/auth/register
  send_password_reset_email        → /api/auth/forgot-password
  send_password_changed_email      → /api/auth/reset-password + /api/auth/change-password
  send_account_deleted_email       → DELETE /api/me
  send_email_change_email          → /api/admin/users/{id}/email
  send_admin_reset_password_email  → /api/admin/users/{id}/reset-password

v2 upgrades:
  - All emails have preheader_text (inbox preview)
  - Welcome: metric_hero for the 30 free tokens, section_tag
  - Password reset: section_tag "Action Required", urgency styling
  - Password changed: section_tag "Security Alert"
  - Account deleted: section_tag "Account Closed"
  - Email change: section_tag "Verify Email"
  - Admin reset: metric_hero for temp password display
"""

import logging
from .base import (
    send_email, mailgun_ready, MAIL_FROM_SUPPORT,
    email_shell, intro_row, body_row, cta_button, tinted_box,
    check_list, alert_banner, numbered_steps, spacer, stat_grid, secondary_links,
    section_tag, metric_hero, divider_accent,
    GRAD_ORANGE, GRAD_RED, GRAD_DARK, GRAD_GREEN, GRAD_BLUE,
    URL_DASHBOARD, URL_SETTINGS, URL_BILLING, URL_PRICING, DISCORD_INVITE_URL, SUPPORT_EMAIL, FRONTEND_URL,
)

logger = logging.getLogger("uploadm8-worker")


# ─────────────────────────────────────────────────────────────────────────────
# 0. Signup confirmation (must verify email before full access)
# ─────────────────────────────────────────────────────────────────────────────
async def send_signup_confirmation_email(email: str, name: str, confirmation_link: str) -> None:
    """
    Sent immediately after signup. User must click the link to verify their email.
    confirmation_link points to {FRONTEND_URL}/confirm-email.html?token=xxx
    """
    if not mailgun_ready():
        return

    html = email_shell(
        gradient=GRAD_ORANGE,
        tagline="One more step to get started",
        preheader_text=f"Click the link to confirm your UploadM8 account and start publishing.",
        body_rows=(
            section_tag("Confirm Your Email", "#f97316")
            + intro_row(
                f"Almost there, {name}! &#128640;",
                "Thanks for signing up for UploadM8. Click the button below to confirm your "
                "email address and activate your account. This link expires in "
                "<strong style='color:#f97316;'>24 hours</strong>.",
            )
            + cta_button("Confirm My Email", confirmation_link, pt="24px", pb="20px")
            + alert_banner(
                "&#9888;&#65039;&nbsp; If you did <strong>not</strong> create an UploadM8 account, "
                "you can safely ignore this email.",
                hex_color="#374151",
                pb="36px",
            )
        ),
        footer_note="You received this because you signed up for an UploadM8 account.",
    )

    await send_email(email, "Confirm your UploadM8 email address", html, from_addr=MAIL_FROM_SUPPORT, reply_to=SUPPORT_EMAIL)


# ─────────────────────────────────────────────────────────────────────────────
# 1. New user welcome (sent AFTER email confirmation)
# ─────────────────────────────────────────────────────────────────────────────
async def send_welcome_email(email: str, name: str) -> None:
    """
    Warm onboarding email fired immediately after a new user registers.

    BEFORE (app.py ~line 1902):
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
        preheader_text=f"Welcome, {name}! Your 30 free upload tokens are waiting — let's get your content live.",
        body_rows=(
            section_tag("Welcome to the Platform", "#f97316")
            + intro_row(
                f"Welcome to UploadM8, {name}! &#127881;",
                "You've just unlocked the fastest way to publish short-form video across "
                "TikTok, YouTube Shorts, Instagram Reels, and Facebook Reels — simultaneously. "
                "We dropped some tokens in your wallet to get you started.",
            )
            + metric_hero(
                "30",
                "Free Upload Tokens",
                "ready to use right now — no credit card needed",
                "#f97316",
            )
            + divider_accent()
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
                f'<p style="margin:0;color:#9ca3af;font-size:13px;line-height:1.65;">'
                f'Questions? We\'re here. Reply to this email or reach out at '
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#f97316;text-decoration:none;">'
                f'{SUPPORT_EMAIL}</a>.</p>',
                hex_color="#374151",
                pb="36px",
            )
        ),
        footer_note="You received this because you created an UploadM8 account.",
    )

    await send_email(email, f"Welcome to UploadM8, {name} — your 30 free tokens are ready", html, from_addr=MAIL_FROM_SUPPORT)


async def send_fully_signed_up_guide_email(
    email: str,
    name: str,
    tier: str,
    put_monthly: int,
    aic_monthly: int,
    max_accounts: int,
    max_accounts_per_platform: int,
) -> None:
    """Brochure-style onboarding email sent after email verification completes."""
    if not mailgun_ready():
        return

    plan_label = tier.replace("_", " ").title()
    html = email_shell(
        gradient=GRAD_BLUE,
        tagline="You are fully signed up and ready to publish",
        preheader_text=f"Your account is fully active. See your {plan_label} features and first steps to get your first successful upload live.",
        body_rows=(
            section_tag("Account Fully Activated", "#2563eb")
            + intro_row(
                f"Welcome aboard, {name}",
                "Your account is now fully verified. This quick guide shows what you have access to and the fastest path to your first successful upload.",
            )
            + metric_hero(
                plan_label,
                "Current Plan",
                f"{put_monthly:,} PUT monthly • {aic_monthly:,} AIC monthly",
                "#2563eb",
            )
            + stat_grid(
                ("Plan", plan_label),
                ("Monthly PUT", f"{put_monthly:,}"),
                ("Monthly AIC", f"{aic_monthly:,}"),
                ("Account Slots", f"{max_accounts:,}"),
            )
            + check_list(
                f"Up to {max_accounts_per_platform:,} accounts per platform",
                "AI caption tools are available from Upload",
                "Scheduling and queue management are available in Dashboard",
                "Billing supports one-time top-ups and plan upgrades",
                hex_color="#2563eb",
            )
            + numbered_steps(
                ("Connect platforms", "Open Settings and connect TikTok, YouTube, Instagram, and Facebook accounts."),
                ("Upload your first video", "Go to Upload, add title/caption/hashtags, and select your platforms."),
                ("Review results", "Use Queue and Dashboard to confirm publish status and retry any failed platforms."),
            )
            + tinted_box(
                f'<p style="margin:0;color:#d1d5db;font-size:14px;line-height:1.7;">'
                f'<strong style="color:#ffffff;">Upgrade option:</strong> Need more monthly quota? '
                f'Open Billing to upgrade your plan or buy a one-time top-up. '
                f'<a href="{URL_BILLING}" style="color:#60a5fa;text-decoration:none;">Go to Billing &#8594;</a><br><br>'
                f'<strong style="color:#ffffff;">Community:</strong> Join our Discord for release updates, support, and roadmap feedback. '
                f'<a href="{DISCORD_INVITE_URL}" style="color:#5865F2;text-decoration:none;">Join Discord &#8599;</a><br><br>'
                f'<strong style="color:#ffffff;">Guide:</strong> '
                f'<a href="{FRONTEND_URL}/guide.html" style="color:#60a5fa;text-decoration:none;">Open getting started guide</a>.'
                f'</p>',
                hex_color="#2563eb",
            )
            + cta_button("Start First Upload", URL_DASHBOARD, pt="16px", pb="20px")
            + secondary_links(
                ("Billing", URL_BILLING),
                ("Plans", URL_PRICING),
                ("Settings", URL_SETTINGS),
            )
        ),
        footer_note="You received this because your email was verified and your account is fully active.",
    )
    await send_email(
        email,
        f"UploadM8 getting started guide — {plan_label} plan",
        html,
        from_addr=MAIL_FROM_SUPPORT,
        reply_to=SUPPORT_EMAIL,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Password reset link
# ─────────────────────────────────────────────────────────────────────────────
async def send_password_reset_email(email: str, reset_link: str) -> None:
    """
    Fired from /api/auth/forgot-password.
    reset_link points to {FRONTEND_URL}/reset-password.html?token=xxx
    """
    if not mailgun_ready():
        return

    html = email_shell(
        gradient=GRAD_ORANGE,
        tagline="Keeping your account secure",
        preheader_text="Your password reset link is inside — expires in 60 minutes.",
        body_rows=(
            section_tag("Action Required", "#f97316")
            + intro_row(
                "Reset your password &#128273;",
                "We received a request to reset the password on your UploadM8 account. "
                "Click the button below — this link is single-use and expires in "
                "<strong style='color:#f97316;'>60 minutes</strong>.",
            )
            + cta_button("Reset My Password", reset_link, pt="24px", pb="20px")
            + alert_banner(
                "&#9888;&#65039;&nbsp; If you did <strong>not</strong> request this reset, "
                "you can safely ignore this email — your password will not change. "
                f'If you\'re concerned, contact us at '
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#ffffff;text-decoration:underline;">'
                f'{SUPPORT_EMAIL}</a>.',
                hex_color="#374151",
                pb="36px",
            )
        ),
        footer_note="You received this because a password reset was requested for your account.",
    )

    await send_email(email, "Reset your UploadM8 password", html, from_addr=MAIL_FROM_SUPPORT, reply_to=SUPPORT_EMAIL)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Password successfully changed — security confirmation
# ─────────────────────────────────────────────────────────────────────────────
async def send_password_changed_email(email: str, name: str) -> None:
    """
    Security receipt sent after a successful password change or reset.
    Hook into POST /api/auth/reset-password and POST /api/auth/change-password on success.
    """
    if not mailgun_ready():
        return

    html = email_shell(
        gradient=GRAD_RED,
        tagline="Account security alert",
        preheader_text=f"Your UploadM8 password was just changed successfully. All sessions signed out.",
        body_rows=(
            section_tag("Security Alert", "#ef4444")
            + intro_row(
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

    await send_email(email, "Your UploadM8 password was changed", html, from_addr=MAIL_FROM_SUPPORT, reply_to=SUPPORT_EMAIL)


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
        preheader_text=f"Your UploadM8 account has been permanently deleted. Thank you for being part of the community.",
        body_rows=(
            section_tag("Account Closed", "#6b7280")
            + intro_row(
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
            + divider_accent(
                "linear-gradient(90deg,rgba(107,114,128,0) 0%,#6b7280 50%,rgba(107,114,128,0) 100%)"
            )
            + tinted_box(
                '<p style="margin:0 0 8px;color:#ffffff;font-size:15px;font-weight:700;">'
                'Changed your mind?</p>'
                '<p style="margin:0;color:#9ca3af;font-size:14px;line-height:1.65;">'
                'You\'re always welcome back. Creating a new account takes less than a minute at '
                f'<a href="{FRONTEND_URL}" style="color:#f97316;text-decoration:none;">app.uploadm8.com</a>.'
                '</p>',
                hex_color="#374151",
            )
            + tinted_box(
                f'<p style="margin:0;color:#9ca3af;font-size:13px;line-height:1.65;">'
                f'If this deletion was made in error, email us within 7 days at '
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#f97316;text-decoration:none;">'
                f'{SUPPORT_EMAIL}</a> and we\'ll do everything we can to help.</p>',
                hex_color="#f97316",
                pb="36px",
            )
        ),
        footer_note="This is a confirmation that your account deletion request was completed.",
    )

    await send_email(email, "Your UploadM8 account has been deleted", html, from_addr=MAIL_FROM_SUPPORT, reply_to=SUPPORT_EMAIL)


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
    verification_link points to {FRONTEND_URL}/verify-email.html?token=xxx

    Hook into /api/admin/users/{id}/email after inserting into email_changes:
        verification_link = f"{FRONTEND_URL}/verify-email.html?token={verification_token}"
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
        preheader_text=f"Click to verify your new UploadM8 email address: {new_email}",
        body_rows=(
            section_tag("Verify Email", "#2563eb")
            + intro_row(
                f"Verify your new email, {name} &#9993;&#65039;",
                f"Your UploadM8 sign-in email was updated from "
                f"<strong style='color:#9ca3af;'>{old_email}</strong> to "
                f"<strong style='color:#60a5fa;'>{new_email}</strong>. "
                "Click the button below to confirm you can receive mail at this address. "
                "Until you verify, some account notices may still go to your previous email.",
            )
            + cta_button("Verify New Email", verification_link, pt="24px", pb="20px")
            + alert_banner(
                "&#9888;&#65039;&nbsp; If you did <strong>not</strong> request an email change, "
                "contact us immediately at "
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#ffffff;text-decoration:underline;">'
                f'{SUPPORT_EMAIL}</a>. Your account may be at risk.',
                hex_color="#374151",
                pb="36px",
            )
        ),
        footer_note="You received this because an email change was requested for your UploadM8 account.",
    )

    await send_email(new_email, "Verify your new UploadM8 email address", html, from_addr=MAIL_FROM_SUPPORT, reply_to=SUPPORT_EMAIL)


# ─────────────────────────────────────────────────────────────────────────────
# 5b. Security notice to the OLD inbox (admin-initiated change)
# ─────────────────────────────────────────────────────────────────────────────
async def send_admin_email_change_notice_to_old_email(
    email: str,
    new_email: str,
    name: str,
) -> None:
    """
    Sent to the OLD email address when an UploadM8 administrator changes the
    user's sign-in email via:
      PUT /api/admin/users/{id}/email

    The NEW email receives the verification link email (send_email_change_email).
    """
    if not mailgun_ready():
        return

    html = email_shell(
        gradient=GRAD_RED,
        tagline="Security notice — email updated",
        preheader_text=f"An admin updated your UploadM8 email to: {new_email}",
        body_rows=(
            section_tag("Security Notice", "#ef4444")
            + intro_row(
                f"Your UploadM8 email was updated, {name} &#128274;",
                f"An <strong>UploadM8 administrator</strong> updated your sign-in email to "
                f"<strong style='color:#60a5fa;'>{new_email}</strong>. "
                "A verification link was sent to that new address. "
                f"If you did <strong>not</strong> request this change, sign in and review your account settings immediately: "
                f'<a href="{URL_SETTINGS}" style="color:#f97316;text-decoration:none;">Settings</a>.',
            )
            + tinted_box(
                f'<p style="margin:0 0 6px;color:#6b7280;font-size:10px;text-transform:uppercase;'
                f'letter-spacing:1.5px;font-weight:600;">New Email Address</p>'
                f'<p style="margin:0;color:#60a5fa;font-size:18px;font-weight:800;">{new_email}</p>',
                hex_color="#ef4444",
                pb="28px",
            )
            + cta_button("Sign In to Review", f"{FRONTEND_URL}/login.html", pt="24px", pb="20px")
            + alert_banner(
                "&#9888;&#65039;&nbsp; If you did <strong>not</strong> request this change, contact us immediately at "
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#ffffff;text-decoration:underline;">'
                f"{SUPPORT_EMAIL}</a>.",
                hex_color="#ef4444",
                pb="36px",
            )
        ),
        footer_note="You received this because an administrator changed the email on your UploadM8 account.",
    )

    await send_email(
        email,
        "Security notice: your UploadM8 email was updated",
        html,
        from_addr=MAIL_FROM_SUPPORT,
        reply_to=SUPPORT_EMAIL,
    )


async def send_user_email_change_notice_to_old_email(
    email: str,
    new_email: str,
    name: str,
) -> None:
    """
    Sent to the old email address when the user changes their own sign-in email.
    """
    if not mailgun_ready():
        return

    html = email_shell(
        gradient=GRAD_RED,
        tagline="Security notice — email updated",
        preheader_text=f"Your UploadM8 sign-in email was changed to: {new_email}",
        body_rows=(
            section_tag("Security Notice", "#ef4444")
            + intro_row(
                f"Your UploadM8 email was updated, {name} &#128274;",
                f"You changed your sign-in email to "
                f"<strong style='color:#60a5fa;'>{new_email}</strong>. "
                "A verification link was sent to that address. "
                f"If this wasn't you, sign in and secure your account immediately: "
                f'<a href="{URL_SETTINGS}" style="color:#f97316;text-decoration:none;">Settings</a>.',
            )
            + cta_button("Review Account Security", f"{FRONTEND_URL}/settings.html#security", pt="24px", pb="20px")
            + alert_banner(
                "&#9888;&#65039;&nbsp; If you did <strong>not</strong> request this change, contact us immediately at "
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#ffffff;text-decoration:underline;">'
                f"{SUPPORT_EMAIL}</a>.",
                hex_color="#ef4444",
                pb="36px",
            )
        ),
        footer_note="You received this because your UploadM8 account email was changed.",
    )

    await send_email(
        email,
        "Security notice: your UploadM8 email was changed",
        html,
        from_addr=MAIL_FROM_SUPPORT,
        reply_to=SUPPORT_EMAIL,
    )


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
        preheader_text=f"An admin reset your UploadM8 password. Your temporary password is inside — change it immediately.",
        body_rows=(
            section_tag("Action Required", "#ef4444")
            + intro_row(
                f"Your password has been reset, {name} &#128272;",
                "An <strong>UploadM8 administrator</strong> set a new temporary password on your account "
                "for security or support reasons. Use the temporary password below to sign in at "
                f'<a href="{FRONTEND_URL}/login.html" style="color:#f97316;text-decoration:none;">login</a>. '
                "After you sign in, you will be asked to choose a new password before using the app.",
            )
            + tinted_box(
                f'<p style="margin:0 0 6px;color:#6b7280;font-size:10px;text-transform:uppercase;'
                f'letter-spacing:1.5px;font-weight:600;">Temporary Password</p>'
                f'<p style="margin:0;color:#f97316;font-size:24px;font-weight:800;'
                f'font-family:\'Courier New\',Courier,monospace;letter-spacing:3px;">{temp_password}</p>'
                f'<p style="margin:10px 0 0;color:#6b7280;font-size:12px;">'
                f'This temporary password expires in 7 days.</p>',
                hex_color="#f97316",
            )
            + cta_button("Sign In Now", f"{FRONTEND_URL}/login.html", pt="8px", pb="20px")
            + alert_banner(
                "&#9888;&#65039;&nbsp; <strong>Change your password immediately</strong> after signing in. "
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

    await send_email(email, "Your UploadM8 password has been reset — action required", html, from_addr=MAIL_FROM_SUPPORT, reply_to=SUPPORT_EMAIL)


async def send_login_anomaly_email(
    email: str,
    name: str,
    ip_address: str,
    country_code: str = "",
    user_agent: str = "",
    previous_ip: str = "",
) -> None:
    """Security alert for sign-ins from a new IP/country/device fingerprint."""
    if not mailgun_ready():
        return

    ua_short = (user_agent or "Unknown device")[:180]
    prev_html = (
        f'<p style="margin:10px 0 0;color:#9ca3af;font-size:13px;line-height:1.6;">Previous sign-in IP: {previous_ip}</p>'
        if previous_ip else ""
    )
    html = email_shell(
        gradient=GRAD_RED,
        tagline="Security sign-in alert",
        preheader_text="We detected a sign-in to your account from a new device or location.",
        body_rows=(
            section_tag("Security Alert", "#ef4444")
            + intro_row(
                f"New sign-in detected, {name}",
                "We noticed a sign-in from a new location or device fingerprint. "
                "If this was you, no action is required.",
            )
            + tinted_box(
                f'<p style="margin:0;color:#d1d5db;font-size:14px;line-height:1.7;">'
                f'<strong style="color:#ffffff;">IP:</strong> {ip_address or "Unknown"}<br>'
                f'<strong style="color:#ffffff;">Country:</strong> {country_code or "Unknown"}<br>'
                f'<strong style="color:#ffffff;">Device:</strong> {ua_short}'
                f'{prev_html}</p>',
                hex_color="#ef4444",
            )
            + check_list(
                "If this was you, you can ignore this message",
                "If this was not you, change your password immediately",
                "Review connected sessions and platform tokens in Settings",
                hex_color="#22c55e",
            )
            + cta_button("Sign In and Review Security", f"{FRONTEND_URL}/login.html", pt="16px", pb="20px")
        ),
        footer_note="You received this security alert because a new sign-in was detected.",
    )
    await send_email(
        email,
        "UploadM8 security alert — new sign-in detected",
        html,
        from_addr=MAIL_FROM_SUPPORT,
        reply_to=SUPPORT_EMAIL,
    )
