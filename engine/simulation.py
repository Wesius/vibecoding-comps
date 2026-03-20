from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Sequence

import numpy as np

from engine.background import MarketMaker, NoiseTrader
from engine.matching import MatchingEngine
from engine.orderbook import OrderBook
from engine.price_model import MidPriceProcess
from engine.scoring import implementation_shortfall
from engine.types import (
    AgentResult,
    CancelOrder,
    Fill,
    Order,
    RestingOrderInfo,
    Side,
    SimulationConfig,
    TickState,
    TradeTapeEntry,
)

if TYPE_CHECKING:
    from agents.base import BaseAgent


def _validate_agent_actions(
    actions: Sequence[Order | CancelOrder],
    remaining_qty: int,
    resting_orders: tuple[RestingOrderInfo, ...],
) -> tuple[list[int], list[Order]]:
    """Validate agent actions.

    - CancelOrder may only target the agent's current resting orders
    - All new orders must be BUY side
    - Total new size clipped to remaining_qty
    - Invalid actions are dropped
    """
    resting_order_ids = {order.order_id for order in resting_orders}
    cancel_ids: list[int] = []
    validated: list[Order] = []
    budget = remaining_qty

    for action in actions:
        if isinstance(action, CancelOrder):
            if (
                action.order_id in resting_order_ids
                and action.order_id not in cancel_ids
            ):
                cancel_ids.append(action.order_id)
            continue

        order = action
        if order.side != Side.BUY:
            continue
        if order.size <= 0:
            continue
        size = min(order.size, budget)
        if size <= 0:
            break
        validated.append(dataclasses.replace(order, size=size))
        budget -= size

    return cancel_ids, validated


def _compute_avg_price(fills: list[Fill]) -> float:
    if not fills:
        return 0.0
    total_cost = sum(f.price * f.size for f in fills)
    total_size = sum(f.size for f in fills)
    return total_cost / total_size if total_size > 0 else 0.0


class Simulation:
    """Runs one full game (N ticks) for a set of agents under a single seed."""

    def __init__(
        self,
        agents: Sequence[BaseAgent],
        config: SimulationConfig,
        seed: int | np.random.SeedSequence,
    ) -> None:
        self._agents = list(agents)
        self._config = config
        self._seed = seed

    def run(self) -> list[AgentResult]:
        """Execute the full simulation. Returns results for each agent."""
        cfg = self._config

        # Create independent RNG streams for reproducibility
        if isinstance(self._seed, np.random.SeedSequence):
            ss = self._seed
        else:
            ss = np.random.SeedSequence(self._seed)
        child_seeds = ss.spawn(4)
        price_rng = np.random.default_rng(child_seeds[0])
        mm_rng = np.random.default_rng(child_seeds[1])
        noise_rng = np.random.default_rng(child_seeds[2])
        shuffle_rng = np.random.default_rng(child_seeds[3])

        # Initialize subsystems
        price_model = MidPriceProcess(
            initial_price=cfg.initial_price,
            volatility=cfg.volatility,
            drift=cfg.drift,
            permanent_impact_bps=cfg.permanent_impact_bps,
            rng=price_rng,
        )
        market_maker = MarketMaker(
            n_levels=cfg.mm_n_levels,
            base_spread_bps=cfg.mm_base_spread_bps,
            level_spacing_bps=cfg.mm_level_spacing_bps,
            base_size=cfg.mm_base_size,
            depth_growth=cfg.mm_depth_growth,
            flow_sensitivity=cfg.mm_flow_sensitivity,
            rng=mm_rng,
        )
        noise_trader = NoiseTrader(
            avg_orders_per_tick=cfg.noise_avg_orders,
            market_order_prob=cfg.noise_market_prob,
            mean_size=cfg.noise_mean_size,
            size_std=cfg.noise_size_std,
            rng=noise_rng,
        )
        matching = MatchingEngine(rng=shuffle_rng)
        book = OrderBook()

        arrival_price = price_model.price

        # Per-agent state
        agent_fills: dict[str, list[Fill]] = {
            a.agent_id: [] for a in self._agents
        }
        # Per-tick tracking for charts
        tick_cumulative_pct: dict[str, list[float]] = {
            a.agent_id: [] for a in self._agents
        }
        tick_avg_price: dict[str, list[float]] = {
            a.agent_id: [] for a in self._agents
        }
        tick_mid_prices: list[float] = []
        tick_spreads: list[float] = []
        tape: list[TradeTapeEntry] = []
        prev_tick_agent_flow = 0

        for tick in range(cfg.n_ticks):
            book.expire_orders(tick)

            # [A] PRICE EVOLUTION
            mid = price_model.step(agent_flow=prev_tick_agent_flow)

            # [B] BACKGROUND ORDER GENERATION
            mm_orders = market_maker.generate_quotes(
                mid_price=mid,
                net_agent_flow=prev_tick_agent_flow,
                recent_volatility=price_model.recent_volatility,
            )
            noise_orders = noise_trader.generate_orders(
                mid_price=mid,
                spread=market_maker.current_spread,
                intensity_scale=0.5,
            )

            # [C] REFRESH TRANSIENT LIQUIDITY AND EXECUTE NOISE
            for order in mm_orders:
                assert order.price is not None
                book.add_limit_order(
                    order.side,
                    order.price,
                    order.size,
                    order.agent_id or "mm",
                    submitted_tick=tick,
                )

            noise_tape, background_fills = matching.execute_background_orders(
                book, noise_orders, tick
            )
            tape.extend(noise_tape)
            for agent_id, fills in background_fills.items():
                if agent_id in agent_fills:
                    agent_fills[agent_id].extend(fills)

            # [D] SNAPSHOT FOR AGENTS
            snapshot = book.snapshot()
            tick_mid_prices.append(snapshot.mid_price)
            if snapshot.asks and snapshot.bids:
                tick_spreads.append(snapshot.asks[0].price - snapshot.bids[0].price)
            else:
                tick_spreads.append(0.0)

            # Trim tape to rolling window for agents
            tape_window_entries = [
                e for e in tape if e.tick >= tick - cfg.tape_window
            ]

            # [E] COLLECT AGENT ORDERS
            all_agent_orders: dict[str, list[Order]] = {}
            pending_cancels: dict[str, list[int]] = {}
            for agent in self._agents:
                total_filled = sum(
                    f.size for f in agent_fills[agent.agent_id]
                )
                true_remaining = cfg.target_qty - total_filled
                resting_orders = book.get_resting_orders(agent.agent_id)
                resting_qty = sum(
                    order.remaining_size for order in resting_orders
                )
                state = TickState(
                    tick=tick,
                    total_ticks=cfg.n_ticks,
                    order_book=snapshot,
                    remaining_qty=max(0, true_remaining),
                    fills=tuple(agent_fills[agent.agent_id]),
                    avg_fill_price=_compute_avg_price(
                        agent_fills[agent.agent_id]
                    ),
                    trade_tape=tuple(tape_window_entries),
                    arrival_price=arrival_price,
                    resting_orders=resting_orders,
                )

                try:
                    actions = agent.on_tick(state)
                except Exception:
                    actions = []

                cancel_ids = [
                    action.order_id
                    for action in actions
                    if isinstance(action, CancelOrder)
                ]
                cancelled_resting_qty = sum(
                    order.remaining_size
                    for order in resting_orders
                    if order.order_id in cancel_ids
                )
                order_budget = max(
                    0,
                    true_remaining - (resting_qty - cancelled_resting_qty),
                )

                validated_cancels, validated = _validate_agent_actions(
                    actions,
                    order_budget,
                    resting_orders,
                )
                pending_cancels[agent.agent_id] = validated_cancels
                all_agent_orders[agent.agent_id] = validated

            for agent_id, order_ids in pending_cancels.items():
                for order_id in order_ids:
                    book.cancel_order(agent_id, order_id)

            # [F] EXECUTE AGENT ORDERS (shuffled)
            tick_fills, tick_tape = matching.execute_agent_orders(
                book,
                all_agent_orders,
                tick,
                cfg.agent_order_ttl_ticks,
            )

            for agent_id, fills in tick_fills.items():
                agent_fills[agent_id].extend(fills)
            tape.extend(tick_tape)

            post_noise_orders = noise_trader.generate_orders(
                mid_price=mid,
                spread=market_maker.current_spread,
                intensity_scale=0.5,
            )
            post_noise_tape, post_background_fills = (
                matching.execute_background_orders(
                book, post_noise_orders, tick
                )
            )
            tape.extend(post_noise_tape)
            for agent_id, fills in post_background_fills.items():
                if agent_id in agent_fills:
                    agent_fills[agent_id].extend(fills)

            prev_tick_agent_flow = sum(
                fill.size
                for fills in tick_fills.values()
                for fill in fills
            )

            # Track per-tick stats for charts
            for agent in self._agents:
                aid = agent.agent_id
                fills = agent_fills[aid]
                total = sum(f.size for f in fills)
                tick_cumulative_pct[aid].append(total / cfg.target_qty)
                tick_avg_price[aid].append(
                    _compute_avg_price(fills) if fills else arrival_price
                )

            # [G] CLEANUP
            book.clear_transient()

        # Compute final results
        results: list[AgentResult] = []
        for agent in self._agents:
            fills = agent_fills[agent.agent_id]
            total_filled = sum(f.size for f in fills)
            avg_price = _compute_avg_price(fills)
            is_bps = implementation_shortfall(
                fills, arrival_price, cfg.target_qty
            )
            results.append(AgentResult(
                agent_id=agent.agent_id,
                agent_class=type(agent).__name__,
                fills=fills,
                total_filled=total_filled,
                avg_price=avg_price,
                arrival_price=arrival_price,
                implementation_shortfall=is_bps,
                remaining_qty=cfg.target_qty - total_filled,
                cumulative_fill_pct=tick_cumulative_pct[agent.agent_id],
                running_avg_price=tick_avg_price[agent.agent_id],
            ))

        return results, tick_mid_prices, tick_spreads
