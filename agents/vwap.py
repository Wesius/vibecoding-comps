from __future__ import annotations

import numpy as np

from engine.types import Order, OrderType, Side, TickState
from agents.base import BaseAgent


class VWAPAgent(BaseAgent):
    """Volume-Weighted Average Price: sizes based on a U-shaped volume profile.

    Executes more heavily at the open and close, matching typical
    intraday volume patterns.
    """

    def __init__(self, agent_id: str, target_qty: int, **kwargs: object) -> None:
        super().__init__(agent_id, target_qty)
        # Pre-compute a U-shaped volume profile
        ticks = np.arange(500)
        raw = 1.0 + 0.5 * np.cos(2 * np.pi * (ticks / 500 - 0.5))
        self._volume_profile = raw / raw.sum()

    def on_tick(self, state: TickState) -> list[Order]:
        if state.remaining_qty <= 0:
            return []
        if state.tick >= len(self._volume_profile):
            return [Order(
                side=Side.BUY,
                size=state.remaining_qty,
                order_type=OrderType.MARKET,
            )]

        target_this_tick = max(1, int(self.target_qty * self._volume_profile[state.tick]))
        size = min(target_this_tick, state.remaining_qty)

        # On last tick, send whatever remains
        if state.tick == state.total_ticks - 1:
            size = state.remaining_qty

        return [Order(
            side=Side.BUY,
            size=size,
            order_type=OrderType.MARKET,
        )]
