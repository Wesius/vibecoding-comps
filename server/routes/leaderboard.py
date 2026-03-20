from __future__ import annotations

import json

from fastapi import APIRouter

from server.config import ServerConfig


def create_leaderboard_router(config: ServerConfig) -> APIRouter:
    r = APIRouter()

    @r.get("/leaderboard")
    async def get_leaderboard():
        if not config.leaderboard_path.exists():
            return {"updated_at": None, "standings": []}

        data = json.loads(config.leaderboard_path.read_text())
        return data

    return r
