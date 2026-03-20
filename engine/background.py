from __future__ import annotations

import math

import numpy as np

from engine.types import Order, OrderType, Side


class MarketMaker:
    """Simulated market maker providing background liquidity.

    Places N levels on each side of the mid price. Spread widens
    with recent volatility and recent agent order flow.
    """

    def __init__(
        self,
        n_levels: int = 10,
        base_spread_bps: float = 5.0,
        level_spacing_bps: float = 2.0,
        base_size: int = 200,
        depth_growth: float = 1.3,
        flow_sensitivity: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._n_levels = n_levels
        self._base_spread_bps = base_spread_bps
        self._level_spacing_bps = level_spacing_bps
        self._base_size = base_size
        self._depth_growth = depth_growth
        self._flow_sensitivity = flow_sensitivity
        self._rng = rng or np.random.default_rng()
        self._current_spread: float = 0.0

    @property
    def current_spread(self) -> float:
        return self._current_spread

    def generate_quotes(
        self,
        mid_price: float,
        net_agent_flow: int,
        recent_volatility: float,
    ) -> list[Order]:
        """Generate limit orders for both sides of the book."""
        half_spread = self._compute_half_spread(
            mid_price, net_agent_flow, recent_volatility
        )
        self._current_spread = half_spread * 2

        orders: list[Order] = []
        for i in range(self._n_levels):
            level_offset = i * (self._level_spacing_bps / 10_000) * mid_price
            level_size = self._compute_level_size(i)

            # Bid side
            bid_price = round(mid_price - half_spread - level_offset, 4)
            if bid_price > 0:
                orders.append(
                    Order(
                        side=Side.BUY,
                        size=level_size,
                        order_type=OrderType.LIMIT,
                        price=bid_price,
                        agent_id="mm",
                    )
                )

            # Ask side
            ask_price = round(mid_price + half_spread + level_offset, 4)
            orders.append(
                Order(
                    side=Side.SELL,
                    size=level_size,
                    order_type=OrderType.LIMIT,
                    price=ask_price,
                    agent_id="mm",
                )
            )

        return orders

    def _compute_half_spread(
        self,
        mid_price: float,
        net_agent_flow: int,
        recent_volatility: float,
    ) -> float:
        """Compute half-spread in price units."""
        base = (self._base_spread_bps / 10_000) * mid_price

        # Widen for volatility (vol ratio vs baseline)
        vol_scaling = max(1.0, recent_volatility / 0.001)
        vol_component = base * (vol_scaling - 1.0) * 0.5

        # Widen for agent flow pressure
        flow_component = (
            self._flow_sensitivity
            * abs(net_agent_flow)
            / 10_000
            * (self._base_spread_bps / 10_000)
            * mid_price
        )

        # Small random jitter
        jitter = self._rng.uniform(-0.05, 0.05) * base

        return max(base + vol_component + flow_component + jitter, base * 0.5)

    def _compute_level_size(self, level_idx: int) -> int:
        """Compute size at a given depth level (thinner near top, thicker deeper)."""
        raw = self._base_size * (self._depth_growth**level_idx)
        # Add some noise
        noise = self._rng.uniform(0.8, 1.2)
        return max(1, int(raw * noise))


class NoiseTrader:
    """Generates random buy/sell flow each tick."""

    def __init__(
        self,
        avg_orders_per_tick: float = 5.0,
        market_order_prob: float = 0.7,
        mean_size: int = 50,
        size_std: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._avg_orders = avg_orders_per_tick
        self._market_prob = market_order_prob
        self._mean_size = mean_size
        self._size_std = size_std
        self._rng = rng or np.random.default_rng()

    def generate_orders(
        self,
        mid_price: float,
        spread: float,
        intensity_scale: float = 1.0,
    ) -> list[Order]:
        """Generate random orders for this tick."""
        n_orders = self._rng.poisson(
            max(0.0, self._avg_orders * intensity_scale)
        )
        orders: list[Order] = []

        for _ in range(n_orders):
            side = Side.BUY if self._rng.random() < 0.5 else Side.SELL
            size = max(1, int(self._rng.lognormal(
                math.log(self._mean_size), self._size_std
            )))

            if self._rng.random() < self._market_prob:
                orders.append(
                    Order(
                        side=side,
                        size=size,
                        order_type=OrderType.MARKET,
                        agent_id="noise",
                    )
                )
            else:
                # Limit order placed near mid
                half_spread = spread / 2 if spread > 0 else 0.01
                if side == Side.BUY:
                    price = mid_price - self._rng.uniform(0, half_spread * 2)
                else:
                    price = mid_price + self._rng.uniform(0, half_spread * 2)
                price = round(max(0.01, price), 4)
                orders.append(
                    Order(
                        side=side,
                        size=size,
                        order_type=OrderType.LIMIT,
                        price=price,
                        agent_id="noise",
                    )
                )

        return orders
