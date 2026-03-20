from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Header, HTTPException

from server.config import ServerConfig


def create_auth_dependency(config: ServerConfig):
    """Create a FastAPI dependency that validates player credentials."""

    async def verify_player(
        x_player_name: Annotated[str, Header()],
        x_player_token: Annotated[str, Header()],
    ) -> str:
        """Validates player name + token. Returns player name."""
        expected_token = config.players.get(x_player_name)
        if expected_token is None:
            raise HTTPException(status_code=401, detail="Unknown player name.")

        if not secrets.compare_digest(x_player_token, expected_token):
            raise HTTPException(status_code=401, detail="Invalid token.")

        return x_player_name

    return verify_player
