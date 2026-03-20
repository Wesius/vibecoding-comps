from __future__ import annotations

from engine.types import Order, OrderType, Side, TickState
from agents.base import BaseAgent


class NaiveAgent(BaseAgent):
    """Market orders all remaining quantity every tick.

    The simplest possible strategy: always be as aggressive as possible.
    Gets destroyed by slippage (walking thin books) and market impact.
    """

    def on_tick(self, state: TickState) -> list[Order]:
        if state.remaining_qty <= 0:
            return []
        return [Order(
            side=Side.BUY,
            size=state.remaining_qty,
            order_type=OrderType.MARKET,
        )]
