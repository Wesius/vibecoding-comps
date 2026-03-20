from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from engine.simulation import Simulation
from engine.scoring import implementation_shortfall
from engine.types import SimulationConfig
from agents.base import BaseAgent
from server.config import ServerConfig


def _load_agent_class(agent_dir: Path, player_name: str) -> type | None:
    """Dynamically load an Agent class from a player's directory."""
    agent_py = agent_dir / "agent.py"
    if not agent_py.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        f"submitted_agent_{player_name}", agent_py
    )
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None

    return getattr(module, "Agent", None)


def run_tournament(config: ServerConfig) -> dict:
    """Run a tournament with all submitted agents.

    Returns a result dict suitable for JSON serialization.
    """
    agents_dir = config.agents_dir
    player_dirs = sorted(
        [d for d in agents_dir.iterdir() if d.is_dir() and (d / "agent.py").exists()]
    )

    if not player_dirs:
        raise ValueError("No agents have been submitted yet.")

    # Load all agent classes
    agent_factories: list[tuple[type, dict]] = []
    errors: dict[str, str] = {}

    for player_dir in player_dirs:
        player_name = player_dir.name
        agent_cls = _load_agent_class(player_dir, player_name)
        if agent_cls is None:
            errors[player_name] = "Failed to load agent."
            continue
        agent_factories.append((agent_cls, {"agent_id": player_name}))

    if not agent_factories:
        raise ValueError("No valid agents could be loaded.")

    # Run tournament
    sim_config = SimulationConfig(
        n_ticks=config.ticks_per_sim,
    )

    master_ss = np.random.SeedSequence(42)
    sim_seeds = master_ss.spawn(config.seeds_per_tournament)

    n_ticks = sim_config.n_ticks
    agent_scores: dict[str, list[float]] = {
        kwargs["agent_id"]: [] for _, kwargs in agent_factories
    }
    # Accumulate per-tick data across seeds for averaging
    agent_fill_pct_sum: dict[str, np.ndarray] = {
        kwargs["agent_id"]: np.zeros(n_ticks) for _, kwargs in agent_factories
    }
    agent_price_sum: dict[str, np.ndarray] = {
        kwargs["agent_id"]: np.zeros(n_ticks) for _, kwargs in agent_factories
    }
    agent_seed_count: dict[str, int] = {
        kwargs["agent_id"]: 0 for _, kwargs in agent_factories
    }

    for seed_seq in sim_seeds:
        # Fresh instances per seed
        agents: list[BaseAgent] = []
        for agent_cls, kwargs in agent_factories:
            try:
                agent = agent_cls(
                    agent_id=kwargs["agent_id"],
                    target_qty=sim_config.target_qty,
                )
                agents.append(agent)
            except Exception:
                agent_scores[kwargs["agent_id"]].append(float("inf"))

        if not agents:
            continue

        sim = Simulation(agents=agents, config=sim_config, seed=seed_seq)
        results = sim.run()

        for result in results:
            agent_scores[result.agent_id].append(result.implementation_shortfall)
            if result.cumulative_fill_pct is not None:
                agent_fill_pct_sum[result.agent_id] += np.array(result.cumulative_fill_pct)
                agent_price_sum[result.agent_id] += np.array(result.running_avg_price)
                agent_seed_count[result.agent_id] += 1

    # Compute rankings
    rankings: list[dict] = []
    for agent_id, scores in agent_scores.items():
        finite = [s for s in scores if s != float("inf")]
        mean_is = float(np.mean(finite)) if finite else float("inf")
        count = agent_seed_count[agent_id]
        # Downsample to 50 points for the frontend
        if count > 0:
            avg_pct = agent_fill_pct_sum[agent_id] / count
            avg_price = agent_price_sum[agent_id] / count
            step = max(1, n_ticks // 50)
            fill_curve = [round(float(avg_pct[i]), 4) for i in range(0, n_ticks, step)]
            price_curve = [round(float(avg_price[i]), 4) for i in range(0, n_ticks, step)]
        else:
            fill_curve = []
            price_curve = []
        rankings.append({
            "name": agent_id,
            "mean_is": round(mean_is, 2),
            "seeds_completed": len(finite),
            "fill_curve": fill_curve,
            "price_curve": price_curve,
        })

    rankings.sort(key=lambda x: x["mean_is"])
    for i, entry in enumerate(rankings, 1):
        entry["rank"] = i

    # Add errors
    for name, error in errors.items():
        rankings.append({
            "rank": len(rankings) + 1,
            "name": name,
            "mean_is": float("inf"),
            "seeds_completed": 0,
            "error": error,
        })

    tournament_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

    return {
        "tournament_id": tournament_id,
        "results": rankings,
    }


def update_leaderboard(config: ServerConfig, tournament_result: dict) -> None:
    """Update the leaderboard JSON file with new tournament results."""
    leaderboard_path = config.leaderboard_path

    # Leaderboard just shows the most recent tournament
    standings: list[dict] = []
    for entry in tournament_result["results"]:
        s = {
            "name": entry["name"],
            "mean_is": entry["mean_is"],
            "seeds_completed": entry.get("seeds_completed", 0),
            "rank": entry["rank"],
        }
        if "fill_curve" in entry:
            s["fill_curve"] = entry["fill_curve"]
        if "price_curve" in entry:
            s["price_curve"] = entry["price_curve"]
        standings.append(s)

    data = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "tournament_id": tournament_result["tournament_id"],
        "standings": standings,
    }

    # Atomic write
    tmp_path = leaderboard_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2))
    os.replace(tmp_path, leaderboard_path)
