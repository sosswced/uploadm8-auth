"""SQL fragment + bind values for optional Trill vehicle filter (analytics + map-feed)."""

from __future__ import annotations

from typing import Any


def trill_vehicle_where_fragment(start_param: int, make: str | None, model: str | None) -> tuple[str, list[Any]]:
    """
    When ``make`` is non-empty, restrict to rows with ``trill_metadata`` vehicle
    matching that make; if ``model`` is also non-empty, require exact model match.
    Placeholders use asyncpg-style $n starting at ``start_param`` (typically 3 after uid, since).
    """
    m = (make or "").strip()
    if not m:
        return "", []
    mo = (model or "").strip()
    if mo:
        frag = (
            " AND u.trill_metadata ? 'vehicle' "
            f"AND COALESCE(NULLIF(btrim(u.trill_metadata#>>'{{vehicle,make_name}}'), ''), '(unknown)') = ${start_param} "
            f"AND COALESCE(NULLIF(btrim(u.trill_metadata#>>'{{vehicle,model_name}}'), ''), '') = ${start_param + 1}"
        )
        return frag, [m, mo]
    frag = (
        " AND u.trill_metadata ? 'vehicle' "
        f"AND COALESCE(NULLIF(btrim(u.trill_metadata#>>'{{vehicle,make_name}}'), ''), '(unknown)') = ${start_param}"
    )
    return frag, [m]
