"""Workspace invite emails."""

import logging
from typing import Any, Optional

from .base import (
    GRAD_BLUE,
    URL_LOGIN,
    email_shell,
    intro_row,
    cta_button,
    send_email,
    mailgun_ready,
    MAIL_FROM_HELLO,
)

logger = logging.getLogger("uploadm8-worker")


async def send_workspace_invite_email(
    email: str,
    inviter_name: str,
    workspace_name: str,
    accept_link: str,
    owner_user_id: Optional[str] = None,
    db_pool: Any = None,
) -> None:
    if not mailgun_ready():
        return

    brand = None
    product = "UploadM8"
    if db_pool and owner_user_id:
        try:
            from services.white_label import load_effective_brand_context

            async with db_pool.acquire() as conn:
                brand = await load_effective_brand_context(conn, str(owner_user_id))
            if brand:
                product = brand.product_name or brand.company_name or product
        except Exception:
            logger.debug("workspace invite brand context skipped", exc_info=True)

    signup_hint = accept_link.replace("accept-invite.html", "signup.html")
    html = email_shell(
        gradient=GRAD_BLUE,
        tagline="You've been invited to collaborate",
        preheader_text=f"{inviter_name} invited you to {workspace_name} on {product}.",
        brand=brand,
        body_rows=(
            intro_row(
                f"Join {workspace_name}",
                f"<strong>{inviter_name}</strong> invited you to collaborate on uploads, queue, and shared credits.",
            )
            + cta_button("Accept invite", accept_link)
            + intro_row(
                "New to the platform?",
                f'<a href="{signup_hint}" style="color:#f97316;">Create an account</a> with this email, then open the invite link again.',
            )
            + intro_row(
                "Already have an account?",
                f"Sign in with the invited email, then open the link again. "
                f'<a href="{URL_LOGIN}" style="color:#f97316;">Sign in</a>',
            )
        ),
        footer_note="If you did not expect this invite, you can ignore this email.",
    )
    await send_email(
        email,
        f"You're invited to {workspace_name} on {product}",
        html,
        from_addr=MAIL_FROM_HELLO,
    )
