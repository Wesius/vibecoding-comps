#!/usr/bin/env python3
"""CLI for the order execution competition.

Commands:
    test        Run your agent locally against example bots
    submit      Submit your agent to the competition server
    run         Trigger a tournament on the server
    leaderboard Show current standings
"""

import argparse
import importlib.util
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np


def _load_env() -> dict[str, str]:
    """Load .env file if it exists."""
    env: dict[str, str] = {}
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _get_credentials(args: argparse.Namespace) -> tuple[str, str, str]:
    """Get server URL, name, and token from args or .env."""
    env = _load_env()
    server = args.server or env.get("COMP_SERVER", "http://localhost:8000")
    name = args.name or env.get("COMP_NAME")
    token = args.token or env.get("COMP_TOKEN")

    if not name or not token:
        print("Error: name and token required.")
        print("Set via --name/--token flags or in .env file:")
        print('  COMP_SERVER=http://localhost:8000')
        print('  COMP_NAME=yourname')
        print('  COMP_TOKEN=yourtoken')
        sys.exit(1)

    return server, name, token


def _load_player_agent(agent_dir: str = "agent"):
    """Dynamically load the player's Agent class."""
    agent_path = Path(agent_dir) / "agent.py"
    if not agent_path.exists():
        print(f"Error: {agent_path} not found.")
        print("Create your agent in the agent/ directory.")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("player_agent", agent_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "Agent"):
        print(f"Error: {agent_path} must define a class named 'Agent'.")
        sys.exit(1)

    return module.Agent


def cmd_test(args: argparse.Namespace) -> None:
    """Run your agent locally against example bots."""
    from engine.tournament import Tournament
    from engine.types import SimulationConfig
    from agents.naive import NaiveAgent
    from agents.twap import TWAPAgent
    from agents.vwap import VWAPAgent

    PlayerAgent = _load_player_agent()

    config = SimulationConfig()
    n_seeds = args.seeds

    agent_factories: list[tuple[type, dict]] = [
        (PlayerAgent, {"agent_id": "You"}),
        (NaiveAgent, {"agent_id": "Naive"}),
        (TWAPAgent, {"agent_id": "TWAP"}),
        (VWAPAgent, {"agent_id": "VWAP"}),
    ]

    print(f"Running local tournament ({n_seeds} seeds, {config.n_ticks} ticks)...\n")

    tournament = Tournament(
        agent_factories=agent_factories,
        n_seeds=n_seeds,
        master_seed=args.seed,
        config=config,
    )
    result = tournament.run()

    # Print results
    print(f"{'Rank':<6}{'Agent':<15}{'Mean IS (bps)':<16}{'Std IS':<12}{'Median IS':<12}{'Best IS':<12}")
    print("-" * 73)

    for rank, (agent_id, mean_is) in enumerate(result.rankings, 1):
        scores = result.agent_scores[agent_id]
        finite = [s for s in scores if s != float("inf")]
        std_is = float(np.std(finite)) if finite else float("inf")
        med_is = float(np.median(finite)) if finite else float("inf")
        best_is = min(finite) if finite else float("inf")

        marker = " <-- YOU" if agent_id == "You" else ""
        print(
            f"{rank:<6}{agent_id:<15}{mean_is:<16.2f}{std_is:<12.2f}"
            f"{med_is:<12.2f}{best_is:<12.2f}{marker}"
        )

    print()


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit your agent to the competition server."""
    import requests

    server, name, token = _get_credentials(args)
    agent_dir = Path("agent")

    if not (agent_dir / "agent.py").exists():
        print("Error: agent/agent.py not found.")
        sys.exit(1)

    # Zip the agent directory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(agent_dir.rglob("*.py")):
            zf.write(path, path.relative_to(agent_dir))
    buf.seek(0)

    print(f"Submitting to {server}...")
    resp = requests.post(
        f"{server}/submit",
        headers={"X-Player-Name": name, "X-Player-Token": token},
        files={"agent_zip": ("agent.zip", buf, "application/zip")},
    )

    if resp.status_code != 200:
        print(f"Error ({resp.status_code}): {resp.json().get('detail', resp.text)}")
        sys.exit(1)

    print("Submitted! Running tournament...\n")

    resp = requests.post(
        f"{server}/run",
        headers={"X-Player-Name": name, "X-Player-Token": token},
        timeout=600,
    )

    if resp.status_code != 200:
        print(f"Tournament error ({resp.status_code}): {resp.json().get('detail', resp.text)}")
        sys.exit(1)

    data = resp.json()
    results = data.get("results", [])

    print(f"{'Rank':<6}{'Agent':<15}{'Mean IS (bps)':<16}{'Seeds OK':<10}")
    print("-" * 47)

    for entry in results:
        marker = " <--" if entry["name"] == name else ""
        print(
            f"{entry['rank']:<6}{entry['name']:<15}"
            f"{entry['mean_is']:<16.2f}{entry['seeds_completed']:<10}{marker}"
        )
    print()


def cmd_leaderboard(args: argparse.Namespace) -> None:
    """Show current standings."""
    import requests

    env = _load_env()
    server = args.server or env.get("COMP_SERVER", "http://localhost:8000")

    resp = requests.get(f"{server}/leaderboard", timeout=10)

    if resp.status_code != 200:
        print(f"Error ({resp.status_code}): {resp.text}")
        sys.exit(1)

    data = resp.json()
    standings = data.get("standings", [])

    if not standings:
        print("No tournaments have been run yet.")
        return

    print(f"Updated: {data.get('updated_at', 'unknown')}\n")
    print(f"{'Rank':<6}{'Agent':<15}{'Mean IS (bps)':<16}{'Tournaments':<12}")
    print("-" * 49)

    for entry in standings:
        print(
            f"{entry['rank']:<6}{entry['name']:<15}"
            f"{entry['mean_is']:<16.2f}{entry['tournaments_played']:<12}"
        )

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Order Execution Competition CLI",
    )
    parser.add_argument("--server", help="Server URL")
    parser.add_argument("--name", help="Player name")
    parser.add_argument("--token", help="Player token")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # test
    test_parser = subparsers.add_parser("test", help="Test locally against example bots")
    test_parser.add_argument("--seeds", type=int, default=5, help="Number of seeds (default: 5)")
    test_parser.add_argument("--seed", type=int, default=42, help="Master seed (default: 42)")

    # submit
    subparsers.add_parser("submit", help="Submit your agent to the server")

    # leaderboard
    subparsers.add_parser("leaderboard", help="Show current standings")

    args = parser.parse_args()

    commands = {
        "test": cmd_test,
        "submit": cmd_submit,
        "leaderboard": cmd_leaderboard,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
