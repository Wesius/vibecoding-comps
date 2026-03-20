#!/usr/bin/env python3
"""Quick demo: run a tournament with all example agents."""

import numpy as np

from engine.tournament import Tournament
from engine.types import SimulationConfig
from agents import NaiveAgent, TWAPAgent, VWAPAgent


def main() -> None:
    config = SimulationConfig()

    agent_factories = [
        (NaiveAgent, {"agent_id": "Naive"}),
        (TWAPAgent, {"agent_id": "TWAP"}),
        (VWAPAgent, {"agent_id": "VWAP"}),
    ]

    print("Running tournament (50 seeds, 500 ticks)...\n")

    tournament = Tournament(
        agent_factories=agent_factories,
        n_seeds=50,
        master_seed=42,
        config=config,
    )
    result = tournament.run()

    print(f"{'Rank':<6}{'Agent':<15}{'Mean IS (bps)':<16}{'Std IS':<12}{'Median IS':<12}{'Best IS':<12}")
    print("-" * 73)

    for rank, (agent_id, mean_is) in enumerate(result.rankings, 1):
        scores = result.agent_scores[agent_id]
        finite = [s for s in scores if s != float("inf")]
        std_is = float(np.std(finite)) if finite else float("inf")
        med_is = float(np.median(finite)) if finite else float("inf")
        best_is = min(finite) if finite else float("inf")
        print(
            f"{rank:<6}{agent_id:<15}{mean_is:<16.2f}{std_is:<12.2f}"
            f"{med_is:<12.2f}{best_is:<12.2f}"
        )


if __name__ == "__main__":
    main()
