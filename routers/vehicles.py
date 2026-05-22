"""Vehicle make/model catalog (NHTSA vPIC-backed, DB-cached)."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

import core.state
from core.deps import get_current_user
from pydantic import BaseModel, Field

from services import vehicle_catalog as vc

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/vehicles", tags=["vehicles"])


class VehicleProfileBody(BaseModel):
    make_id: Optional[int] = Field(None, description="vehicle_makes.id")
    model_id: Optional[int] = Field(None, description="vehicle_models.id")


@router.get("/makes")
async def list_makes(
    q: str = Query("", max_length=80),
    limit: int = Query(80, ge=1, le=200),
    consumer_only: bool = Query(True, description="Passenger car / light-truck brands only"),
    user: dict = Depends(get_current_user),
):
    _ = user
    async with core.state.db_pool.acquire() as conn:
        makes = await vc.fetch_makes_list(conn, q=q, limit=limit, consumer_only=consumer_only)
    return {"makes": makes}


@router.get("/models")
async def list_models(
    make_id: int = Query(..., ge=1),
    q: str = Query("", max_length=80),
    limit: int = Query(120, ge=1, le=400),
    user: dict = Depends(get_current_user),
):
    _ = user
    async with core.state.db_pool.acquire() as conn:
        mk = await conn.fetchrow(
            "SELECT id, nhtsa_make_id, name FROM vehicle_makes WHERE id = $1",
            make_id,
        )
        if not mk:
            raise HTTPException(404, "Make not found")
        await vc.ensure_models_for_make(conn, make_id)
        if not mk["nhtsa_make_id"]:
            raise HTTPException(
                400,
                detail=f"Make “{mk['name']}” is not linked to NHTSA — pick a brand from search or refresh the catalog.",
            )
        qn = (q or "").strip().lower()
        if qn:
            rows = await conn.fetch(
                """
                SELECT id, name FROM vehicle_models
                WHERE make_id = $1 AND LOWER(name) LIKE '%' || $2 || '%'
                ORDER BY name ASC
                LIMIT $3
                """,
                make_id,
                qn,
                limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id, name FROM vehicle_models
                WHERE make_id = $1
                ORDER BY name ASC
                LIMIT $2
                """,
                make_id,
                limit,
            )
    return {"models": [{"id": r["id"], "name": r["name"]} for r in rows]}


@router.put("/profile")
async def set_default_vehicle(body: VehicleProfileBody, user: dict = Depends(get_current_user)):
    """Set user's default vehicle (garage) for new uploads."""
    uid = user["id"]
    make_id = body.make_id
    model_id = body.model_id
    async with core.state.db_pool.acquire() as conn:
        if make_id is None and model_id is None:
            await conn.execute(
                """
                UPDATE user_preferences
                SET default_vehicle_make_id = NULL, default_vehicle_model_id = NULL, updated_at = NOW()
                WHERE user_id = $1
                """,
                uid,
            )
            return {"ok": True, "default_vehicle_make_id": None, "default_vehicle_model_id": None}

        if not make_id:
            raise HTTPException(400, "make_id required when setting a vehicle")
        mk = await conn.fetchrow("SELECT id FROM vehicle_makes WHERE id = $1", make_id)
        if not mk:
            raise HTTPException(404, "Make not found")
        if model_id:
            mo = await conn.fetchrow(
                "SELECT id FROM vehicle_models WHERE id = $1 AND make_id = $2",
                model_id,
                make_id,
            )
            if not mo:
                raise HTTPException(400, "Model does not belong to this make")
        await conn.execute(
            """
            INSERT INTO user_preferences (user_id) VALUES ($1)
            ON CONFLICT (user_id) DO NOTHING
            """,
            uid,
        )
        await conn.execute(
            """
            UPDATE user_preferences
            SET default_vehicle_make_id = $2,
                default_vehicle_model_id = $3,
                updated_at = NOW()
            WHERE user_id = $1
            """,
            uid,
            make_id,
            model_id,
        )
    return {"ok": True, "default_vehicle_make_id": make_id, "default_vehicle_model_id": model_id}
