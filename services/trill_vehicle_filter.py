"""SQL fragment + bind values for optional Trill vehicle filter (analytics + map-feed)."""

from __future__ import annotations

from typing import Any


def _jsonb_vehicle_fragment(start_param: int, make: str, model: str | None) -> tuple[str, list[Any]]:
    mo = (model or "").strip()
    if mo:
        frag = (
            " AND u.trill_metadata ? 'vehicle' "
            f"AND COALESCE(NULLIF(btrim(u.trill_metadata#>>'{{vehicle,make_name}}'), ''), '(unknown)') = ${start_param} "
            f"AND COALESCE(NULLIF(btrim(u.trill_metadata#>>'{{vehicle,model_name}}'), ''), '') = ${start_param + 1}"
        )
        return frag, [make, mo]
    frag = (
        " AND u.trill_metadata ? 'vehicle' "
        f"AND COALESCE(NULLIF(btrim(u.trill_metadata#>>'{{vehicle,make_name}}'), ''), '(unknown)') = ${start_param}"
    )
    return frag, [make]


def trill_vehicle_where_fragment(start_param: int, make: str | None, model: str | None) -> tuple[str, list[Any]]:
    """
    Synchronous JSONB-only filter (legacy/tests). Prefer ``build_trill_vehicle_filter``.
    """
    m = (make or "").strip()
    if not m:
        return "", []
    return _jsonb_vehicle_fragment(start_param, m, model)


async def resolve_trill_vehicle_ids(
    conn: Any,
    make: str | None,
    model: str | None,
    *,
    make_id: int | None = None,
    model_id: int | None = None,
) -> tuple[int | None, int | None]:
    """Resolve catalog ids from explicit ids or make/model names."""
    if model_id is not None:
        model_id = int(model_id)
        if make_id is None:
            row = await conn.fetchrow(
                "SELECT make_id FROM vehicle_models WHERE id = $1 LIMIT 1",
                model_id,
            )
            make_id = int(row["make_id"]) if row else None
        else:
            make_id = int(make_id)
        return make_id, model_id
    if make_id is not None:
        return int(make_id), None

    m = (make or "").strip()
    if not m:
        return None, None
    mo = (model or "").strip()
    if mo:
        row = await conn.fetchrow(
            """
            SELECT vm.id AS model_id, vm.make_id
            FROM vehicle_models vm
            JOIN vehicle_makes mk ON mk.id = vm.make_id
            WHERE upper(mk.name) = upper($1) AND upper(vm.name) = upper($2)
            LIMIT 1
            """,
            m,
            mo,
        )
        if row:
            return int(row["make_id"]), int(row["model_id"])
    row = await conn.fetchrow(
        "SELECT id FROM vehicle_makes WHERE upper(name) = upper($1) LIMIT 1",
        m,
    )
    return (int(row["id"]) if row else None), None


async def build_trill_vehicle_filter(
    conn: Any,
    start_param: int,
    make: str | None,
    model: str | None,
    *,
    make_id: int | None = None,
    model_id: int | None = None,
) -> tuple[str, list[Any]]:
    """
    Prefer indexed ``uploads.vehicle_make_id`` / ``vehicle_model_id``; fall back to
    JSONB when the make name is not in the catalog.
    """
    m = (make or "").strip()
    if not m and make_id is None and model_id is None:
        return "", []

    resolved_make_id, resolved_model_id = await resolve_trill_vehicle_ids(
        conn, make, model, make_id=make_id, model_id=model_id
    )

    if resolved_make_id is not None:
        mo = (model or "").strip()
        if resolved_model_id is not None:
            if m:
                frag = (
                    f" AND ((u.vehicle_make_id = ${start_param} AND u.vehicle_model_id = ${start_param + 1})"
                    f" OR (u.vehicle_make_id IS NULL AND u.trill_metadata ? 'vehicle'"
                    f" AND COALESCE(NULLIF(btrim(u.trill_metadata#>>'{{vehicle,make_name}}'), ''), '(unknown)') = ${start_param + 2}"
                    f" AND COALESCE(NULLIF(btrim(u.trill_metadata#>>'{{vehicle,model_name}}'), ''), '') = ${start_param + 3}))"
                )
                return frag, [resolved_make_id, resolved_model_id, m, mo]
            return (
                f" AND u.vehicle_make_id = ${start_param} AND u.vehicle_model_id = ${start_param + 1}",
                [resolved_make_id, resolved_model_id],
            )
        if m:
            frag = (
                f" AND (u.vehicle_make_id = ${start_param}"
                f" OR (u.vehicle_make_id IS NULL AND u.trill_metadata ? 'vehicle'"
                f" AND COALESCE(NULLIF(btrim(u.trill_metadata#>>'{{vehicle,make_name}}'), ''), '(unknown)') = ${start_param + 1}))"
            )
            return frag, [resolved_make_id, m]
        return f" AND u.vehicle_make_id = ${start_param}", [resolved_make_id]

    if not m:
        return "", []
    return _jsonb_vehicle_fragment(start_param, m, model)
