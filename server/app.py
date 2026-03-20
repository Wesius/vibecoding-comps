from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from server.config import load_config
from server.auth import create_auth_dependency
from server.routes.submit import create_submit_router
from server.routes.run import create_run_router
from server.routes.leaderboard import create_leaderboard_router

config = load_config()
auth = create_auth_dependency(config)

app = FastAPI(title="Order Execution Competition", version="0.1.0")

app.include_router(create_submit_router(config, auth))
app.include_router(create_run_router(config, auth))
app.include_router(create_leaderboard_router(config))

_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def index():
    return FileResponse(_STATIC_DIR / "index.html")
