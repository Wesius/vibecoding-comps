from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from server.config import ServerConfig
from server.tournament import run_tournament, update_leaderboard

# Simple lock to prevent concurrent tournaments
_tournament_lock = asyncio.Lock()


def create_run_router(config: ServerConfig, auth_dependency) -> APIRouter:
    r = APIRouter()

    @r.post("/run")
    async def trigger_run(
        player_name: str = Depends(auth_dependency),
    ):
        if _tournament_lock.locked():
            raise HTTPException(
                status_code=409,
                detail="A tournament is already running.",
            )

        async with _tournament_lock:
            try:
                # Run tournament in a thread to not block the event loop
                result = await asyncio.to_thread(run_tournament, config)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Tournament failed: {str(e)}",
                )

            # Update leaderboard
            await asyncio.to_thread(update_leaderboard, config, result)

        return result

    return r
