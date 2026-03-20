from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from server.config import ServerConfig
from server.storage import validate_and_store_agent

router = APIRouter()


def create_submit_router(config: ServerConfig, auth_dependency) -> APIRouter:
    r = APIRouter()

    @r.post("/submit")
    async def submit_agent(
        player_name: str = Depends(auth_dependency),
        agent_zip: UploadFile = File(...),
    ):
        zip_bytes = await agent_zip.read()

        try:
            path = validate_and_store_agent(
                name=player_name,
                zip_bytes=zip_bytes,
                agents_dir=config.agents_dir,
                max_bytes=config.max_agent_zip_bytes,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return {
            "status": "ok",
            "message": f"Agent for '{player_name}' uploaded successfully.",
        }

    return r
