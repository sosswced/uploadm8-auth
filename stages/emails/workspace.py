"""Workspace invite emails."""

import logging

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
) -> None:
    if not mailgun_ready():
        return
    html = email_shell(
        gradient=GRAD_BLUE,
        tagline="You've been invited to collaborate",
        preheader_text=f"{inviter_name} invited you to {workspace_name} on UploadM8.",
        body_rows=(
            intro_row(
                f"Join {workspace_name}",
                f"<strong>{inviter_name}</strong> invited you to collaborate on uploads, queue, and shared credits.",
            )
            + cta_button("Accept invite", accept_link)
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
        f"You're invited to {workspace_name} on UploadM8",
        html,
        from_addr=MAIL_FROM_HELLO,
    )
