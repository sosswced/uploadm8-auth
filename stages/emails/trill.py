"""Trill leaderboard engagement emails."""

import html as html_module
import logging

from .base import (
    FRONTEND_URL,
    GRAD_PURPLE,
    URL_DASHBOARD,
    cta_button,
    email_shell,
    intro_row,
    mailgun_ready,
    send_email,
    section_tag,
)

logger = logging.getLogger("uploadm8-worker")

URL_LEADERBOARD = f"{FRONTEND_URL.rstrip('/')}/trill-leaderboard.html"


async def send_trill_rival_overtake_email(
    email: str,
    name: str,
    rival_handle: str,
    rival_rank: int,
    your_rank: int,
    sort_label: str = "best Trill",
) -> None:
    if not mailgun_ready() or not email:
        return
    safe_name = html_module.escape(name or "there")
    safe_rival = html_module.escape(rival_handle or "A rival")
    body = (
        section_tag("Trill leaderboard", "#a855f7")
        + intro_row(
            f"{safe_name}, you were passed on the board",
            f"<strong>{safe_rival}</strong> moved to <strong>#{rival_rank}</strong> while you are "
            f"<strong>#{your_rank}</strong> on the community leaderboard (sorted by {html_module.escape(sort_label)}).",
        )
        + cta_button("View leaderboard", URL_LEADERBOARD)
    )
    html = email_shell(
        gradient=GRAD_PURPLE,
        tagline="Trill community leaderboard",
        preheader_text=f"{rival_handle} passed you on the Trill leaderboard.",
        body_rows=body,
    )
    await send_email(
        email,
        f"⚔ {rival_handle} passed you on the Trill leaderboard",
        html,
    )
