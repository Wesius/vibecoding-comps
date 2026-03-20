from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ServerConfig:
    players: dict[str, str] = field(default_factory=dict)  # name -> token
    max_agent_zip_bytes: int = 5 * 1024 * 1024  # 5MB
    ticks_per_sim: int = 500
    seeds_per_tournament: int = 50
    per_tick_timeout_ms: int = 100
    tournament_timeout_seconds: int = 600
    container_memory_limit: str = "512m"
    container_cpu_limit: float = 1.0
    data_dir: Path = Path("server/data")

    @property
    def agents_dir(self) -> Path:
        return self.data_dir / "agents"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def leaderboard_path(self) -> Path:
        return self.data_dir / "leaderboard.json"


def load_config() -> ServerConfig:
    config_path = Path(os.environ.get("PLAYERS_CONFIG", "players.yaml"))

    config = ServerConfig()

    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        config.players = data.get("players", {})

        settings = data.get("settings", {})
        for key, value in settings.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Ensure directories exist
    config.agents_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    return config
