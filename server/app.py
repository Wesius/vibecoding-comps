import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import json

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from server.config import load_config
from server.auth import create_auth_dependency
from server.routes.submit import create_submit_router
from server.routes.run import create_run_router
from server.routes.leaderboard import create_leaderboard_router
from server.tournament import run_tournament, update_leaderboard

logger = logging.getLogger("comp-server")

config = load_config()
auth = create_auth_dependency(config)

TOURNAMENT_INTERVAL = 60  # seconds


async def _auto_tournament_loop():
    """Run a tournament every TOURNAMENT_INTERVAL seconds."""
    while True:
        await asyncio.sleep(TOURNAMENT_INTERVAL)
        # Check if any agents are submitted
        agent_dirs = [
            d for d in config.agents_dir.iterdir()
            if d.is_dir() and (d / "agent.py").exists()
        ] if config.agents_dir.exists() else []

        if not agent_dirs:
            continue

        try:
            result = await asyncio.to_thread(run_tournament, config)
            await asyncio.to_thread(update_leaderboard, config, result)
            names = [r["name"] for r in result["results"]]
            logger.info(f"Auto-tournament complete: {names}")
        except Exception as e:
            logger.error(f"Auto-tournament failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_auto_tournament_loop())
    yield
    task.cancel()


app = FastAPI(title="Order Execution Competition", version="0.1.0", lifespan=lifespan)

app.include_router(create_submit_router(config, auth))
app.include_router(create_run_router(config, auth))
app.include_router(create_leaderboard_router(config))

_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def index():
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/replay")
async def replay_page():
    return FileResponse(_STATIC_DIR / "replay.html")


@app.get("/replay-data")
async def replay_data():
    path = config.data_dir / "replay.json"
    if not path.exists():
        return {"ticks": [], "agents": []}
    return JSONResponse(content=json.loads(path.read_text()))
