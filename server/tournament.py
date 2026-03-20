from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
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


def _run_seed_worker(agent_dirs_info, sim_config, seed_seq, collect_replay):
    """Run one simulation seed. Called in a worker process."""
    agents = []
    for dir_str, player_name in agent_dirs_info:
        agent_cls = _load_agent_class(Path(dir_str), player_name)
        if agent_cls is None:
            continue
        try:
            agent = agent_cls(agent_id=player_name, target_qty=sim_config.target_qty)
            agents.append(agent)
        except Exception:
            continue

    if not agents:
        return None

    sim = Simulation(
        agents=agents, config=sim_config, seed=seed_seq,
        collect_replay=collect_replay,
    )
    return sim.run()


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

    # Validate agents can load (also populates errors for broken agents)
    valid_dirs: list[Path] = []
    errors: dict[str, str] = {}

    for player_dir in player_dirs:
        player_name = player_dir.name
        agent_cls = _load_agent_class(player_dir, player_name)
        if agent_cls is None:
            errors[player_name] = "Failed to load agent."
        else:
            valid_dirs.append(player_dir)

    if not valid_dirs:
        raise ValueError("No valid agents could be loaded.")

    agent_ids = [d.name for d in valid_dirs]
    agent_dirs_info = [(str(d), d.name) for d in valid_dirs]

    # Run tournament
    sim_config = SimulationConfig(
        n_ticks=config.ticks_per_sim,
    )

    master_ss = np.random.SeedSequence(42)
    sim_seeds = master_ss.spawn(config.seeds_per_tournament)

    n_ticks = sim_config.n_ticks
    agent_scores: dict[str, list[float]] = {aid: [] for aid in agent_ids}
    # Accumulate per-tick data across seeds for averaging
    agent_fill_pct_sum: dict[str, np.ndarray] = {
        aid: np.zeros(n_ticks) for aid in agent_ids
    }
    agent_price_sum: dict[str, np.ndarray] = {
        aid: np.zeros(n_ticks) for aid in agent_ids
    }
    agent_seed_count: dict[str, int] = {aid: 0 for aid in agent_ids}
    mid_price_sum = np.zeros(n_ticks)
    spread_sum = np.zeros(n_ticks)
    market_seed_count = 0
    replay_data: list[dict] = []

    # Run seeds in parallel across CPU cores
    max_workers = min(os.cpu_count() or 4, config.seeds_per_tournament)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_seed_worker, agent_dirs_info, sim_config, ss, i == 0
            )
            for i, ss in enumerate(sim_seeds)
        ]

        for i, future in enumerate(futures):
            try:
                output = future.result()
            except Exception:
                for aid in agent_ids:
                    agent_scores[aid].append(float("inf"))
                continue

            if output is None:
                for aid in agent_ids:
                    agent_scores[aid].append(float("inf"))
                continue

            results, tick_mids, tick_spreads, replay_ticks = output

            mid_price_sum += np.array(tick_mids)
            spread_sum += np.array(tick_spreads)

            if i == 0:
                replay_data = replay_ticks

            market_seed_count += 1

            seen = set()
            for result in results:
                agent_scores[result.agent_id].append(result.implementation_shortfall)
                if result.cumulative_fill_pct is not None:
                    agent_fill_pct_sum[result.agent_id] += np.array(result.cumulative_fill_pct)
                    agent_price_sum[result.agent_id] += np.array(result.running_avg_price)
                    agent_seed_count[result.agent_id] += 1
                seen.add(result.agent_id)

            # Agents that failed to instantiate in this worker get inf
            for aid in agent_ids:
                if aid not in seen:
                    agent_scores[aid].append(float("inf"))

    # Compute rankings
    rankings: list[dict] = []
    for agent_id, scores in agent_scores.items():
        finite = [s for s in scores if s != float("inf")]
        mean_is = float(np.mean(finite)) if finite else float("inf")
        count = agent_seed_count[agent_id]
        if count > 0:
            avg_pct = agent_fill_pct_sum[agent_id] / count
            avg_price = agent_price_sum[agent_id] / count
            fill_curve = [round(float(v), 4) for v in avg_pct]
            price_curve = [round(float(v), 4) for v in avg_price]
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

    if market_seed_count > 0:
        avg_mid = mid_price_sum / market_seed_count
        avg_spread = spread_sum / market_seed_count
        mid_curve = [round(float(v), 4) for v in avg_mid]
        spread_curve = [round(float(v), 4) for v in avg_spread]
    else:
        mid_curve = []
        spread_curve = []

    tournament_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

    # Save replay data to disk
    replay_path = config.data_dir / "replay.json"
    tmp_replay = replay_path.with_suffix(".tmp")
    tmp_replay.write_text(json.dumps({"ticks": replay_data, "agents": agent_ids}))
    os.replace(tmp_replay, replay_path)

    return {
        "tournament_id": tournament_id,
        "results": rankings,
        "mid_curve": mid_curve,
        "spread_curve": spread_curve,
    }


def update_leaderboard(config: ServerConfig, tournament_result: dict) -> None:
    """Update the leaderboard JSON file with new tournament results."""
    leaderboard_path = config.leaderboard_path

    # Load existing to preserve history
    if leaderboard_path.exists():
        existing = json.loads(leaderboard_path.read_text())
        history = existing.get("history", [])
    else:
        history = []

    # Leaderboard shows the most recent tournament
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

    # Append to history (compact: just name->score per tournament)
    history.append({
        "t": tournament_result["tournament_id"],
        "s": {entry["name"]: entry["mean_is"] for entry in tournament_result["results"]},
    })
    # Cap at 50 entries
    history = history[-50:]

    data = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "tournament_id": tournament_result["tournament_id"],
        "standings": standings,
        "mid_curve": tournament_result.get("mid_curve", []),
        "spread_curve": tournament_result.get("spread_curve", []),
        "history": history,
    }

    # Atomic write
    tmp_path = leaderboard_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2))
    os.replace(tmp_path, leaderboard_path)
