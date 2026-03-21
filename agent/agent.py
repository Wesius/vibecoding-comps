"""Your agent! Modify this file to build your execution strategy.

Run `python cli.py submit` to submit to the competition server.
"""

from agents.base import BaseAgent
from engine.types import Order, OrderType, Side, TickState


class Agent(BaseAgent):
    """Your execution agent.

    You need to buy target_qty units at the lowest average price.
    Implement your strategy in on_tick().
    """

    def __init__(self, agent_id: str, target_qty: int, **kwargs: object) -> None:
        super().__init__(agent_id, target_qty)
        # Load any config, model weights, or precomputed data here

    def on_tick(self, state: TickState) -> list[Order]:
        """Called every tick. Return a list of orders.

        state contains:
            - tick / total_ticks: current progress
            - order_book: full book with bids/asks and depth
            - remaining_qty: how much more you need to buy (net)
            - net_position: shares you currently hold (can sell up to this)
            - fills: your fills so far (both buys and sells)
            - avg_fill_price: net cost / target_qty
            - trade_tape: recent trades (price, size, aggressor side)
            - arrival_price: mid price at tick 0 (your benchmark)

        Return:
            List of Order objects. Can be BUY or SELL side.
            Market buy:  Order(side=Side.BUY, size=N, order_type=OrderType.MARKET)
            Limit buy:   Order(side=Side.BUY, size=N, order_type=OrderType.LIMIT, price=P)
            Market sell:  Order(side=Side.SELL, size=N, order_type=OrderType.MARKET)
        """
        if state.remaining_qty <= 0:
            return []

        # TODO: Replace this simple TWAP with your strategy!
        ticks_remaining = state.total_ticks - state.tick
        if ticks_remaining <= 0:
            return []

        size = max(1, state.remaining_qty // ticks_remaining)
        if ticks_remaining == 1:
            size = state.remaining_qty

        return [Order(
            side=Side.BUY,
            size=size,
            order_type=OrderType.MARKET,
        )]
