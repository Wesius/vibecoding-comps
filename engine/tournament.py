from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from engine.simulation import Simulation
from engine.types import SimulationConfig

if TYPE_CHECKING:
    from agents.base import BaseAgent


@dataclass
class TournamentResult:
    """Aggregated results across all seeds."""

    # agent_id -> list of IS (bps) per seed
    agent_scores: dict[str, list[float]] = field(default_factory=dict)
    # (agent_id, mean_IS) sorted ascending (best first)
    rankings: list[tuple[str, float]] = field(default_factory=list)
    # agent_id -> agent class name
    agent_classes: dict[str, str] = field(default_factory=dict)


class Tournament:
    """Runs multiple seeds and aggregates results."""

    def __init__(
        self,
        agent_factories: Sequence[
            tuple[type[BaseAgent], Mapping[str, object]]
        ],
        n_seeds: int = 50,
        master_seed: int = 42,
        config: SimulationConfig | None = None,
    ) -> None:
        """
        Args:
            agent_factories: List of (AgentClass, kwargs) tuples.
                Each seed gets fresh instances.
            n_seeds: Number of random seeds to run.
            master_seed: Master seed for generating per-sim seeds.
            config: Simulation configuration.
        """
        self._agent_factories = list(agent_factories)
        self._n_seeds = n_seeds
        self._master_seed = master_seed
        self._config = config or SimulationConfig()

    def run(self) -> TournamentResult:
        """Run all seeds and return aggregated results."""
        master_ss = np.random.SeedSequence(self._master_seed)
        sim_seeds = master_ss.spawn(self._n_seeds)

        result = TournamentResult()

        # Initialize score tracking
        for agent_cls, kwargs in self._agent_factories:
            agent_id = str(kwargs.get("agent_id", agent_cls.__name__))
            result.agent_scores[agent_id] = []
            result.agent_classes[agent_id] = agent_cls.__name__

        for seed_seq in sim_seeds:
            # Fresh agent instances per seed
            agents: list[BaseAgent] = []
            for agent_cls, kwargs in self._agent_factories:
                agent_id = str(kwargs.get("agent_id", agent_cls.__name__))
                agent = agent_cls(
                    agent_id=agent_id,
                    target_qty=self._config.target_qty,
                    **{k: v for k, v in kwargs.items() if k != "agent_id"},
                )
                agents.append(agent)

            sim = Simulation(
                agents=agents,
                config=self._config,
                seed=seed_seq,
            )
            sim_results, _, _, _ = sim.run()

            for agent_result in sim_results:
                result.agent_scores[agent_result.agent_id].append(
                    agent_result.implementation_shortfall
                )

        # Compute rankings (mean IS, ascending = best first)
        rankings: list[tuple[str, float]] = []
        for agent_id, scores in result.agent_scores.items():
            if any(not np.isfinite(s) for s in scores):
                mean_is = float("inf")
            else:
                mean_is = float(np.mean(scores)) if scores else float("inf")
            rankings.append((agent_id, mean_is))

        result.rankings = sorted(rankings, key=lambda x: x[1])
        return result
