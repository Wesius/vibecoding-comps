from __future__ import annotations

from dataclasses import dataclass

from engine.types import BookLevel, OrderBookSnapshot, RestingOrderInfo, Side


@dataclass(slots=True)
class _RestingBookOrder:
    order_id: int
    agent_id: str
    remaining_size: int
    submitted_tick: int
    expires_tick: int | None
    persistent: bool


class OrderBook:
    """Mutable order book used internally by the matching engine.

    Orders are tracked individually at each price level in FIFO order,
    allowing queue priority and passive fill attribution.
    """

    def __init__(self) -> None:
        self._bids: dict[float, list[_RestingBookOrder]] = {}
        self._asks: dict[float, list[_RestingBookOrder]] = {}
        self._next_order_id = 1

    def add_limit_order(
        self,
        side: Side,
        price: float,
        size: int,
        agent_id: str,
        *,
        submitted_tick: int = 0,
        expires_tick: int | None = None,
        persistent: bool = False,
    ) -> int:
        """Place a resting limit order on the book and return its order id."""
        book = self._bids if side == Side.BUY else self._asks
        if price not in book:
            book[price] = []

        order_id = self._next_order_id
        self._next_order_id += 1
        book[price].append(
            _RestingBookOrder(
                order_id=order_id,
                agent_id=agent_id,
                remaining_size=size,
                submitted_tick=submitted_tick,
                expires_tick=expires_tick,
                persistent=persistent,
            )
        )
        return order_id

    def match_market_order(
        self, side: Side, size: int
    ) -> list[tuple[float, int, str, int]]:
        """Walk the book for a market order.

        Returns (price, filled_size, resting_agent_id, resting_order_id).
        """
        if side == Side.BUY:
            return self._walk_book(self._asks, size, ascending=True)
        return self._walk_book(self._bids, size, ascending=False)

    def match_limit_order(
        self, side: Side, price: float, size: int
    ) -> tuple[list[tuple[float, int, str, int]], int]:
        """Try to fill a limit order at the given price or better."""
        if side == Side.BUY:
            fills = self._walk_book(
                self._asks, size, ascending=True, limit_price=price
            )
        else:
            fills = self._walk_book(
                self._bids, size, ascending=False, limit_price=price
            )

        filled = sum(fill[1] for fill in fills)
        return fills, size - filled

    def _walk_book(
        self,
        book: dict[float, list[_RestingBookOrder]],
        size: int,
        ascending: bool,
        limit_price: float | None = None,
    ) -> list[tuple[float, int, str, int]]:
        fills: list[tuple[float, int, str, int]] = []
        remaining = size
        prices = sorted(book.keys(), reverse=not ascending)
        empty_prices: list[float] = []

        for price in prices:
            if remaining <= 0:
                break
            if limit_price is not None:
                if ascending and price > limit_price:
                    break
                if not ascending and price < limit_price:
                    break

            orders = book[price]
            empty_orders: list[int] = []

            for index, order in enumerate(orders):
                if remaining <= 0:
                    break
                fill_size = min(order.remaining_size, remaining)
                fills.append(
                    (price, fill_size, order.agent_id, order.order_id)
                )
                remaining -= fill_size
                order.remaining_size -= fill_size
                if order.remaining_size <= 0:
                    empty_orders.append(index)

            for index in reversed(empty_orders):
                orders.pop(index)

            if not orders:
                empty_prices.append(price)

        for price in empty_prices:
            del book[price]

        return fills

    def get_resting_orders(self, agent_id: str) -> tuple[RestingOrderInfo, ...]:
        """Return live resting orders for an agent with queue-ahead sizes."""
        orders: list[RestingOrderInfo] = []

        for side, book in ((Side.BUY, self._bids), (Side.SELL, self._asks)):
            price_levels = sorted(book.keys(), reverse=side == Side.BUY)
            for price in price_levels:
                queue_ahead = 0
                for order in book[price]:
                    if order.agent_id == agent_id:
                        orders.append(
                            RestingOrderInfo(
                                order_id=order.order_id,
                                side=side,
                                price=price,
                                remaining_size=order.remaining_size,
                                queue_ahead=queue_ahead,
                                submitted_tick=order.submitted_tick,
                                expires_tick=order.expires_tick,
                            )
                        )
                    queue_ahead += order.remaining_size

        return tuple(orders)

    def total_resting_qty(self, agent_id: str, side: Side | None = None) -> int:
        """Return total live resting quantity for an agent."""
        total = 0
        books = []
        if side in (None, Side.BUY):
            books.append(self._bids)
        if side in (None, Side.SELL):
            books.append(self._asks)

        for book in books:
            for orders in book.values():
                total += sum(
                    order.remaining_size
                    for order in orders
                    if order.agent_id == agent_id
                )
        return total

    def cancel_order(self, agent_id: str, order_id: int) -> int:
        """Cancel one live resting order owned by the given agent.

        Returns the removed quantity, or 0 if not found.
        """
        for book in (self._bids, self._asks):
            empty_prices: list[float] = []
            for price, orders in book.items():
                for index, order in enumerate(orders):
                    if order.order_id == order_id and order.agent_id == agent_id:
                        removed_qty = order.remaining_size
                        orders.pop(index)
                        if not orders:
                            empty_prices.append(price)
                        for empty_price in empty_prices:
                            del book[empty_price]
                        return removed_qty
            for empty_price in empty_prices:
                del book[empty_price]
        return 0

    def expire_orders(self, current_tick: int) -> None:
        """Remove expired resting orders."""
        self._prune_orders(
            lambda order: order.expires_tick is not None
            and order.expires_tick < current_tick
        )

    def clear_transient(self) -> None:
        """Remove non-persistent orders at the end of a tick."""
        self._prune_orders(lambda order: not order.persistent)

    def _prune_orders(self, should_remove) -> None:
        for book in (self._bids, self._asks):
            empty_prices: list[float] = []
            for price, orders in book.items():
                book[price] = [order for order in orders if not should_remove(order)]
                if not book[price]:
                    empty_prices.append(price)
            for price in empty_prices:
                del book[price]

    def snapshot(self) -> OrderBookSnapshot:
        """Create an immutable aggregated snapshot for agents."""
        bids = tuple(
            BookLevel(
                price=price,
                size=sum(order.remaining_size for order in orders),
            )
            for price, orders in sorted(self._bids.items(), reverse=True)
            if orders
        )
        asks = tuple(
            BookLevel(
                price=price,
                size=sum(order.remaining_size for order in orders),
            )
            for price, orders in sorted(self._asks.items())
            if orders
        )

        if bids and asks:
            mid = (bids[0].price + asks[0].price) / 2.0
        elif bids:
            mid = bids[0].price
        elif asks:
            mid = asks[0].price
        else:
            mid = 0.0

        return OrderBookSnapshot(bids=bids, asks=asks, mid_price=mid)

    def clear(self) -> None:
        self._bids.clear()
        self._asks.clear()

    @property
    def best_bid(self) -> float | None:
        if not self._bids:
            return None
        return max(self._bids.keys())

    @property
    def best_ask(self) -> float | None:
        if not self._asks:
            return None
        return min(self._asks.keys())

    @property
    def mid_price(self) -> float:
        best_bid = self.best_bid
        best_ask = self.best_ask
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        if best_bid is not None:
            return best_bid
        if best_ask is not None:
            return best_ask
        return 0.0
