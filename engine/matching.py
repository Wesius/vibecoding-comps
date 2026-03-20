from __future__ import annotations

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

    def execute_background_orders(
        self,
        book: OrderBook,
        orders: list[Order],
        tick: int,
    ) -> tuple[list[TradeTapeEntry], dict[str, list[Fill]]]:
        """Execute noise trader orders against the book.

        Returns public tape entries and fills earned by resting orders.
        """
        tape: list[TradeTapeEntry] = []
        resting_fills: dict[str, list[Fill]] = {}

        for order in orders:
            fills: list[tuple[float, int, str, int]]
            if order.order_type == OrderType.MARKET:
                fills = book.match_market_order(order.side, order.size)
                for price, size, _resting_id, _resting_order_id in fills:
                    tape.append(TradeTapeEntry(
                        price=price,
                        size=size,
                        aggressor_side=order.side,
                        tick=tick,
                    ))
            else:
                assert order.price is not None
                fills, remainder = book.match_limit_order(
                    order.side, order.price, order.size
                )
                for price, size, _resting_id, _resting_order_id in fills:
                    tape.append(TradeTapeEntry(
                        price=price,
                        size=size,
                        aggressor_side=order.side,
                        tick=tick,
                    ))
                # Rest any unfilled portion
                if remainder > 0:
                    book.add_limit_order(
                        order.side,
                        order.price,
                        remainder,
                        order.agent_id or "noise",
                        submitted_tick=tick,
                    )

            resting_side = Side.SELL if order.side == Side.BUY else Side.BUY
            for price, size, resting_id, _resting_order_id in fills:
                resting_fills.setdefault(resting_id, []).append(Fill(
                    price=price,
                    size=size,
                    tick=tick,
                    side=resting_side,
                ))

        return tape, resting_fills

    def execute_agent_orders(
        self,
        book: OrderBook,
        agent_orders: dict[str, list[Order]],
        tick: int,
        agent_order_ttl_ticks: int,
    ) -> tuple[dict[str, list[Fill]], list[TradeTapeEntry]]:
        """Execute all agent orders for one tick.

        Agents are shuffled randomly, then each agent's orders
        are executed sequentially against the book.

        Returns (fills_per_agent, tape_entries).
        """
        fills_per_agent: dict[str, list[Fill]] = {
            aid: [] for aid in agent_orders
        }
        tape: list[TradeTapeEntry] = []

        # Shuffle agent execution order
        agent_ids = list(agent_orders.keys())
        self._rng.shuffle(agent_ids)

        for agent_id in agent_ids:
            orders = agent_orders[agent_id]
            for order in orders:
                order_fills: list[tuple[float, int, str, int]]
                expiry_tick = tick + max(agent_order_ttl_ticks - 1, 0)

                if order.order_type == OrderType.MARKET:
                    order_fills = book.match_market_order(order.side, order.size)
                else:
                    assert order.price is not None
                    order_fills, remainder = book.match_limit_order(
                        order.side, order.price, order.size
                    )
                    # Rest any unfilled limit order (will be cancelled at tick end)
                    if remainder > 0:
                        book.add_limit_order(
                            order.side,
                            order.price,
                            remainder,
                            agent_id,
                            submitted_tick=tick,
                            expires_tick=expiry_tick,
                            persistent=True,
                        )

                for price, size, _resting_id, _resting_order_id in order_fills:
                    fills_per_agent[agent_id].append(Fill(
                        price=price,
                        size=size,
                        tick=tick,
                        side=order.side,
                    ))
                    tape.append(TradeTapeEntry(
                        price=price,
                        size=size,
                        aggressor_side=order.side,
                        tick=tick,
                    ))

        return fills_per_agent, tape
