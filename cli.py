#!/usr/bin/env python3
"""CLI for the order execution competition.

Commands:
    submit      Submit your agent to the competition server
    leaderboard Show current standings
"""

import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path

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

    # submit
    subparsers.add_parser("submit", help="Submit your agent to the server")

    # leaderboard
    subparsers.add_parser("leaderboard", help="Show current standings")

    args = parser.parse_args()

    commands = {
        "submit": cmd_submit,
        "leaderboard": cmd_leaderboard,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
