from __future__ import annotations

from dataclasses import replace

import numpy as np

from engine.orderbook import OrderBook
from engine.types import Fill, Order, OrderType, Side, TradeTapeEntry


class MatchingEngine:
    """Processes orders within a single tick.

    Agent orders are shuffled (seeded for reproducibility) then
    executed sequentially against the book. This means later agents
    in the shuffle see depleted liquidity from earlier agents.
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng or np.random.default_rng()

    @staticmethod
    def _compute_expiry_tick(
        tick: int,
        ttl_ticks: int | None,
    ) -> int | None:
        if ttl_ticks is None:
            return None
        return tick + max(ttl_ticks - 1, 0)

    def _execute_orders(
        self,
        book: OrderBook,
        orders: list[Order],
        tick: int,
        resting_order_ttl_ticks: int | None = None,
    ) -> tuple[dict[str, list[Fill]], dict[str, list[Fill]], list[TradeTapeEntry]]:
        """Execute orders and return aggressor fills, resting fills, and tape."""
        aggressor_fills: dict[str, list[Fill]] = {}
        tape: list[TradeTapeEntry] = []
        resting_fills: dict[str, list[Fill]] = {}

        for order in orders:
            owner_id = order.agent_id or "anonymous"
            fills: list[tuple[float, int, str, int]]
            if order.order_type == OrderType.MARKET:
                fills = book.match_market_order(order.side, order.size)
            else:
                assert order.price is not None
                fills, remainder = book.match_limit_order(
                    order.side, order.price, order.size
                )
                # Rest any unfilled portion
                if remainder > 0:
                    book.add_limit_order(
                        order.side,
                        order.price,
                        remainder,
                        owner_id,
                        submitted_tick=tick,
                        expires_tick=self._compute_expiry_tick(
                            tick,
                            resting_order_ttl_ticks,
                        ),
                        persistent=True,
                    )

            resting_side = Side.SELL if order.side == Side.BUY else Side.BUY
            for price, size, resting_id, _resting_order_id in fills:
                tape.append(TradeTapeEntry(
                    price=price,
                    size=size,
                    aggressor_side=order.side,
                    tick=tick,
                ))
                aggressor_fills.setdefault(owner_id, []).append(Fill(
                    price=price,
                    size=size,
                    tick=tick,
                    side=order.side,
                ))
                resting_fills.setdefault(resting_id, []).append(Fill(
                    price=price,
                    size=size,
                    tick=tick,
                    side=resting_side,
                ))

        return aggressor_fills, resting_fills, tape

    def execute_background_orders(
        self,
        book: OrderBook,
        orders: list[Order],
        tick: int,
        resting_order_ttl_ticks: int | None = None,
    ) -> tuple[list[TradeTapeEntry], dict[str, list[Fill]]]:
        """Execute noise trader orders against the book.

        Returns public tape entries and fills earned by resting orders.
        """
        _aggressor_fills, resting_fills, tape = self._execute_orders(
            book,
            orders,
            tick,
            resting_order_ttl_ticks,
        )
        return tape, resting_fills

    def execute_orders(
        self,
        book: OrderBook,
        orders: list[Order],
        tick: int,
        resting_order_ttl_ticks: int | None = None,
    ) -> tuple[dict[str, list[Fill]], dict[str, list[Fill]], list[TradeTapeEntry]]:
        """Execute generic participant orders against the live book."""
        return self._execute_orders(
            book,
            orders,
            tick,
            resting_order_ttl_ticks,
        )

    def execute_agent_orders(
        self,
        book: OrderBook,
        agent_id: str,
        orders: list[Order],
        tick: int,
        agent_order_ttl_ticks: int | None,
    ) -> tuple[list[Fill], list[TradeTapeEntry], dict[str, list[Fill]]]:
        """Execute one agent's orders against the live book."""
        stamped_orders = [
            replace(order, agent_id=agent_id)
            for order in orders
        ]
        aggressor_fills, resting_fills, tape = self._execute_orders(
            book,
            stamped_orders,
            tick,
            agent_order_ttl_ticks,
        )
        return aggressor_fills.get(agent_id, []), tape, resting_fills
