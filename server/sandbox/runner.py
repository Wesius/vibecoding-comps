#!/usr/bin/env python3
"""Sandbox runner — executes inside a Docker container.

Discovers all agents in /app/agents/*/agent.py, runs the tournament,
and writes results to /app/output/results.json.
"""

import importlib.util
import json
import signal
import sys
import traceback
from pathlib import Path

import numpy as np

# Engine is mounted at /app/engine
sys.path.insert(0, "/app")

from engine.simulation import Simulation
from engine.types import SimulationConfig
from agents.base import BaseAgent


class TickTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TickTimeout()


def _load_agent_class(agent_dir: Path, name: str):
    """Load Agent class from a player's directory."""
    agent_py = agent_dir / "agent.py"
    spec = importlib.util.spec_from_file_location(f"agent_{name}", agent_py)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "Agent", None)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--tick-timeout-ms", type=int, default=100)
    args = parser.parse_args()

    agents_root = Path("/app/agents")
    output_dir = Path("/app/output")
    output_dir.mkdir(exist_ok=True)

    # Discover agents
    player_dirs = sorted([
        d for d in agents_root.iterdir()
        if d.is_dir() and (d / "agent.py").exists()
    ])

    if not player_dirs:
        json.dump({"error": "No agents found"}, open(output_dir / "results.json", "w"))
        return

    # Load agents
    agent_classes: dict[str, type] = {}
    errors: dict[str, str] = {}

    for d in player_dirs:
        name = d.name
        try:
            cls = _load_agent_class(d, name)
            if cls is None:
                errors[name] = "No Agent class found"
            else:
                agent_classes[name] = cls
        except Exception as e:
            errors[name] = f"Import error: {e}"

    # Run tournament
    config = SimulationConfig(n_ticks=args.ticks)
    master_ss = np.random.SeedSequence(42)
    sim_seeds = master_ss.spawn(args.seeds)

    agent_scores: dict[str, list[float]] = {name: [] for name in agent_classes}

    for seed_idx, seed_seq in enumerate(sim_seeds):
        # Fresh instances
        agents: list[BaseAgent] = []
        for name, cls in agent_classes.items():
            try:
                agent = cls(agent_id=name, target_qty=config.target_qty)
                agents.append(agent)
            except Exception:
                agent_scores[name].append(float("inf"))

        if not agents:
            continue

        sim = Simulation(agents=agents, config=config, seed=seed_seq)
        results = sim.run()

        for result in results:
            agent_scores[result.agent_id].append(result.implementation_shortfall)

        print(f"Seed {seed_idx + 1}/{args.seeds} complete", flush=True)

    # Compute rankings
    rankings = []
    for name, scores in agent_scores.items():
        finite = [s for s in scores if s != float("inf")]
        mean_is = float(np.mean(finite)) if finite else float("inf")
        rankings.append({
            "name": name,
            "mean_is": round(mean_is, 2),
            "seeds_completed": len(finite),
        })

    for name, error in errors.items():
        rankings.append({
            "name": name,
            "mean_is": float("inf"),
            "seeds_completed": 0,
            "error": error,
        })

    rankings.sort(key=lambda x: x["mean_is"])
    for i, entry in enumerate(rankings, 1):
        entry["rank"] = i

    output = {
        "results": rankings,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Tournament complete!", flush=True)


if __name__ == "__main__":
    main()
