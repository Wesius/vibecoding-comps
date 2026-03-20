from __future__ import annotations

from engine.types import Order, OrderType, Side, TickState
from agents.base import BaseAgent


class TWAPAgent(BaseAgent):
    """Time-Weighted Average Price: equal-sized market orders each tick."""

    def on_tick(self, state: TickState) -> list[Order]:
        if state.remaining_qty <= 0:
            return []

        ticks_remaining = state.total_ticks - state.tick
        if ticks_remaining <= 0:
            return []

        # Even split across remaining ticks
        size = max(1, state.remaining_qty // ticks_remaining)
        # On last tick, send whatever remains
        if ticks_remaining == 1:
            size = state.remaining_qty

        return [Order(
            side=Side.BUY,
            size=size,
            order_type=OrderType.MARKET,
        )]
