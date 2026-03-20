from __future__ import annotations

import math
from collections import deque

import numpy as np


class MidPriceProcess:
    """Generates the true mid price path for a simulation.

    Model: Geometric Brownian Motion with permanent price impact.

    P_new = P_old * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    Then: P_new += permanent_impact_bps/10000 * recent_agent_flow * P_new

    The initial price is always 100.0 for simplicity.
    """

    def __init__(
        self,
        initial_price: float = 100.0,
        volatility: float = 0.001,
        drift: float = 0.0,
        permanent_impact_bps: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._price = initial_price
        self._volatility = volatility
        self._drift = drift
        self._permanent_impact_bps = permanent_impact_bps
        self._rng = rng or np.random.default_rng()
        self._returns: deque[float] = deque(maxlen=50)

    def step(self, agent_flow: int = 0) -> float:
        """Advance one tick. Returns new mid price."""
        old_price = self._price

        # GBM step
        z = self._rng.standard_normal()
        log_return = (
            self._drift - 0.5 * self._volatility**2
        ) + self._volatility * z
        self._price = old_price * math.exp(log_return)

        # Permanent impact from the most recent tick's executed agent flow
        if agent_flow != 0:
            impact = (
                (self._permanent_impact_bps / 10_000)
                * (agent_flow / 1000)
                * self._price
            )
            self._price += impact

        # Track returns for volatility estimation
        if old_price > 0:
            self._returns.append(math.log(self._price / old_price))

        return self._price

    @property
    def price(self) -> float:
        return self._price

    @property
    def recent_volatility(self) -> float:
        """Rolling window realized vol for MM spread adjustment."""
        if len(self._returns) < 2:
            return self._volatility
        arr = np.array(self._returns)
        return float(np.std(arr))
