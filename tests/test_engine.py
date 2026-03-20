"""Basic engine tests."""

import numpy as np
import pytest

from agents.base import BaseAgent
from engine.background import MarketMaker
from engine.matching import MatchingEngine
from agents.twap import TWAPAgent
from engine.orderbook import OrderBook
from engine.price_model import MidPriceProcess
from engine.scoring import implementation_shortfall
from engine.simulation import Simulation
from engine.types import CancelOrder, Fill, Order, OrderType, Side, SimulationConfig


class PassiveLimitAgent(BaseAgent):
    def on_tick(self, state):
        if state.remaining_qty <= 0:
            return []
        return [
            Order(
                side=Side.BUY,
                size=state.remaining_qty,
                order_type=OrderType.LIMIT,
                price=100.0,
            )
        ]


class FixedSliceAgent(BaseAgent):
    def on_tick(self, state):
        if state.remaining_qty <= 0:
            return []
        return [
            Order(
                side=Side.BUY,
                size=min(2, state.remaining_qty),
                order_type=OrderType.MARKET,
            )
        ]


class CancelThenMarketAgent(BaseAgent):
    def on_tick(self, state):
        if state.tick == 0 and state.remaining_qty > 0:
            return [
                Order(
                    side=Side.BUY,
                    size=state.remaining_qty,
                    order_type=OrderType.LIMIT,
                    price=100.0,
                )
            ]
        if state.resting_orders and state.remaining_qty > 0:
            return [
                CancelOrder(order_id=state.resting_orders[0].order_id),
                Order(
                    side=Side.BUY,
                    size=state.remaining_qty,
                    order_type=OrderType.MARKET,
                ),
            ]
        return []


class LongLivedPassiveAgent(BaseAgent):
    def __init__(self, agent_id: str, target_qty: int, **kwargs):
        super().__init__(agent_id, target_qty, **kwargs)
        self.saw_resting_on_tick_five = False

    def on_tick(self, state):
        if state.tick == 0:
            return [
                Order(
                    side=Side.BUY,
                    size=1,
                    order_type=OrderType.LIMIT,
                    price=100.0,
                )
            ]
        if state.tick == 5:
            self.saw_resting_on_tick_five = bool(state.resting_orders)
            if state.resting_orders:
                return [
                    CancelOrder(order_id=state.resting_orders[0].order_id),
                    Order(
                        side=Side.BUY,
                        size=1,
                        order_type=OrderType.MARKET,
                    ),
                ]
        return []


class TickOneMarketAgent(BaseAgent):
    def on_tick(self, state):
        if state.tick == 1 and state.remaining_qty > 0:
            return [
                Order(
                    side=Side.BUY,
                    size=state.remaining_qty,
                    order_type=OrderType.MARKET,
                )
            ]
        return []


class OneShotMarketAgent(BaseAgent):
    def __init__(self, agent_id: str, target_qty: int, size: int = 1, **kwargs):
        super().__init__(agent_id, target_qty, **kwargs)
        self._size = size

    def on_tick(self, state):
        if state.tick == 0 and state.remaining_qty > 0:
            return [
                Order(
                    side=Side.BUY,
                    size=min(self._size, state.remaining_qty),
                    order_type=OrderType.MARKET,
                )
            ]
        return []


class ObserveBestAskAgent(BaseAgent):
    def __init__(self, agent_id: str, target_qty: int, **kwargs):
        super().__init__(agent_id, target_qty, **kwargs)
        self.seen_best_ask = None

    def on_tick(self, state):
        self.seen_best_ask = (
            state.order_book.asks[0].price if state.order_book.asks else None
        )
        return []


class NoOpAgent(BaseAgent):
    def on_tick(self, state):
        return []


class FakePriceModel:
    seen_flows: list[int] = []

    def __init__(self, initial_price=100.0, **kwargs):
        self._price = initial_price

    def step(self, agent_flow: int = 0) -> float:
        type(self).seen_flows.append(agent_flow)
        return self._price

    @property
    def price(self) -> float:
        return self._price

    @property
    def recent_volatility(self) -> float:
        return 0.0002


class StaticMarketMaker:
    seen_flows: list[int] = []

    def __init__(self, **kwargs):
        self._current_spread = 0.2

    @property
    def current_spread(self) -> float:
        return self._current_spread

    def generate_quotes(self, mid_price: float, net_agent_flow: int, recent_volatility: float):
        type(self).seen_flows.append(net_agent_flow)
        return [
            Order(side=Side.BUY, size=100, order_type=OrderType.LIMIT, price=99.9, agent_id="mm"),
            Order(side=Side.SELL, size=100, order_type=OrderType.LIMIT, price=100.1, agent_id="mm"),
        ]


class EmptyMarketMaker:
    def __init__(self, **kwargs):
        self._current_spread = 0.2

    @property
    def current_spread(self) -> float:
        return self._current_spread

    def generate_quotes(self, mid_price: float, net_agent_flow: int, recent_volatility: float):
        return []


class TwoLevelBookMarketMaker:
    def __init__(self, **kwargs):
        self._current_spread = 0.2

    @property
    def current_spread(self) -> float:
        return self._current_spread

    def observe_passive_fills(self, fills):
        pass

    def generate_quotes(self, mid_price: float, net_agent_flow: int, recent_volatility: float):
        return [
            Order(side=Side.BUY, size=100, order_type=OrderType.LIMIT, price=99.9, agent_id="mm"),
            Order(side=Side.SELL, size=1, order_type=OrderType.LIMIT, price=100.1, agent_id="mm"),
            Order(side=Side.SELL, size=100, order_type=OrderType.LIMIT, price=101.0, agent_id="mm"),
        ]


class TwoPhaseNoiseTrader:
    def __init__(self, **kwargs):
        self._calls = 0

    def generate_orders(self, mid_price: float, spread: float, intensity_scale: float = 1.0):
        self._calls += 1
        if self._calls == 2:
            return [Order(side=Side.SELL, size=5, order_type=OrderType.MARKET, agent_id="noise")]
        return []


class SilentNoiseTrader:
    def __init__(self, **kwargs):
        pass

    def generate_orders(self, mid_price: float, spread: float, intensity_scale: float = 1.0):
        return []


class DelayedNoiseLimitTrader:
    def __init__(self, **kwargs):
        self._calls = 0

    def generate_orders(self, mid_price: float, spread: float, intensity_scale: float = 1.0):
        self._calls += 1
        if self._calls == 1:
            return [
                Order(
                    side=Side.SELL,
                    size=3,
                    order_type=OrderType.LIMIT,
                    price=100.3,
                    agent_id="noise",
                )
            ]
        return []


class _NoShuffleRng:
    def shuffle(self, values):
        return None


class DeterministicMatchingEngine(MatchingEngine):
    def __init__(self, rng=None):
        super().__init__(rng=np.random.default_rng(0))
        self._rng = _NoShuffleRng()


class TestOrder:
    def test_market_order(self):
        o = Order(side=Side.BUY, size=100)
        assert o.order_type == OrderType.MARKET
        assert o.price is None

    def test_limit_order(self):
        o = Order(side=Side.BUY, size=100, order_type=OrderType.LIMIT, price=99.5)
        assert o.price == 99.5

    def test_limit_order_requires_price(self):
        with pytest.raises(ValueError):
            Order(side=Side.BUY, size=100, order_type=OrderType.LIMIT)

    def test_size_must_be_positive(self):
        with pytest.raises(ValueError):
            Order(side=Side.BUY, size=0)


class TestOrderBook:
    def test_add_and_snapshot(self):
        book = OrderBook()
        book.add_limit_order(Side.BUY, 99.0, 100, "mm")
        book.add_limit_order(Side.SELL, 101.0, 100, "mm")

        snap = book.snapshot()
        assert len(snap.bids) == 1
        assert len(snap.asks) == 1
        assert snap.bids[0].price == 99.0
        assert snap.asks[0].price == 101.0
        assert snap.mid_price == 100.0

    def test_match_market_order(self):
        book = OrderBook()
        book.add_limit_order(Side.SELL, 101.0, 50, "mm")
        book.add_limit_order(Side.SELL, 102.0, 50, "mm")

        fills = book.match_market_order(Side.BUY, 75)
        assert len(fills) == 2
        assert fills[0][:3] == (101.0, 50, "mm")  # consume all at 101
        assert fills[1][:3] == (102.0, 25, "mm")  # partial at 102

    def test_match_limit_order_price_limit(self):
        book = OrderBook()
        book.add_limit_order(Side.SELL, 101.0, 50, "mm")
        book.add_limit_order(Side.SELL, 103.0, 50, "mm")

        fills, remainder = book.match_limit_order(Side.BUY, 102.0, 100)
        assert len(fills) == 1  # only 101 level matches
        assert fills[0][:3] == (101.0, 50, "mm")
        assert remainder == 50

    def test_clear(self):
        book = OrderBook()
        book.add_limit_order(Side.BUY, 99.0, 100, "mm")
        book.clear()
        snap = book.snapshot()
        assert len(snap.bids) == 0

    def test_partial_fill_leaves_remainder(self):
        book = OrderBook()
        book.add_limit_order(Side.SELL, 101.0, 100, "mm")

        fills = book.match_market_order(Side.BUY, 30)
        assert [fill[:3] for fill in fills] == [(101.0, 30, "mm")]

        # 70 should remain
        snap = book.snapshot()
        assert snap.asks[0].size == 70

    def test_queue_priority_same_price_level(self):
        book = OrderBook()
        book.add_limit_order(Side.BUY, 100.0, 10, "first", submitted_tick=0)
        book.add_limit_order(Side.BUY, 100.0, 10, "second", submitted_tick=1)

        fills = book.match_market_order(Side.SELL, 15)

        assert [fill[:3] for fill in fills] == [
            (100.0, 10, "first"),
            (100.0, 5, "second"),
        ]

    def test_resting_orders_hide_exact_queue_ahead(self):
        book = OrderBook()
        book.add_limit_order(Side.BUY, 100.0, 8, "mm", submitted_tick=0)
        book.add_limit_order(
            Side.BUY,
            100.0,
            5,
            "agent",
            submitted_tick=1,
            expires_tick=5,
            persistent=True,
        )

        resting = book.get_resting_orders("agent")

        assert len(resting) == 1
        assert resting[0].queue_ahead is None
        assert resting[0].remaining_size == 5

    def test_expire_orders_removes_old_passive_orders(self):
        book = OrderBook()
        book.add_limit_order(
            Side.BUY,
            100.0,
            5,
            "agent",
            submitted_tick=1,
            expires_tick=2,
            persistent=True,
        )

        book.expire_orders(3)

        assert book.get_resting_orders("agent") == ()

    def test_cancel_order_removes_live_resting_order(self):
        book = OrderBook()
        order_id = book.add_limit_order(
            Side.BUY,
            100.0,
            5,
            "agent",
            submitted_tick=1,
            expires_tick=5,
            persistent=True,
        )

        removed_qty = book.cancel_order("agent", order_id)

        assert removed_qty == 5
        assert book.get_resting_orders("agent") == ()

    def test_cancel_all_orders_removes_all_live_resting_orders(self):
        book = OrderBook()
        book.add_limit_order(Side.BUY, 100.0, 5, "mm", submitted_tick=0)
        book.add_limit_order(Side.SELL, 101.0, 7, "mm", submitted_tick=0)
        book.add_limit_order(Side.BUY, 99.5, 3, "agent", submitted_tick=0)

        removed_qty = book.cancel_all_orders("mm")

        assert removed_qty == 12
        snap = book.snapshot()
        assert tuple((level.price, level.size) for level in snap.bids) == ((99.5, 3),)
        assert snap.asks == ()


class TestPriceModel:
    def test_initial_price(self):
        pm = MidPriceProcess(initial_price=100.0)
        assert pm.price == 100.0

    def test_step_changes_price(self):
        pm = MidPriceProcess(initial_price=100.0, rng=np.random.default_rng(42))
        pm.step()
        assert pm.price != 100.0

    def test_reproducible(self):
        pm1 = MidPriceProcess(initial_price=100.0, rng=np.random.default_rng(42))
        pm2 = MidPriceProcess(initial_price=100.0, rng=np.random.default_rng(42))
        for _ in range(100):
            pm1.step()
            pm2.step()
        assert pm1.price == pm2.price

    def test_permanent_impact(self):
        pm = MidPriceProcess(
            initial_price=100.0,
            volatility=0.0,  # no randomness
            permanent_impact_bps=1.0,
            rng=np.random.default_rng(42),
        )
        pm.step(agent_flow=10000)
        # Price should be higher due to buying pressure
        assert pm.price > 100.0


class TestBackgroundParticipants:
    def test_market_maker_skews_quotes_after_inventory_builds(self):
        flat_mm = MarketMaker(
            n_levels=1,
            base_spread_bps=10.0,
            base_size=100,
            rng=np.random.default_rng(7),
        )
        long_mm = MarketMaker(
            n_levels=1,
            base_spread_bps=10.0,
            base_size=100,
            rng=np.random.default_rng(7),
        )

        flat_quotes = flat_mm.generate_quotes(
            mid_price=100.0,
            net_agent_flow=0,
            recent_volatility=0.001,
        )
        long_mm.observe_passive_fills([
            Fill(price=99.9, size=300, tick=0, side=Side.BUY)
        ])
        skewed_quotes = long_mm.generate_quotes(
            mid_price=100.0,
            net_agent_flow=0,
            recent_volatility=0.001,
        )

        flat_bid = max(order.price for order in flat_quotes if order.side == Side.BUY)
        flat_ask = min(order.price for order in flat_quotes if order.side == Side.SELL)
        skewed_bid = max(order.price for order in skewed_quotes if order.side == Side.BUY)
        skewed_ask = min(order.price for order in skewed_quotes if order.side == Side.SELL)

        assert skewed_bid < flat_bid
        assert skewed_ask < flat_ask


class TestScoring:
    def test_basic_shortfall(self):
        fills = [Fill(price=101.0, size=100, tick=0, side=Side.BUY)]
        is_bps = implementation_shortfall(fills, arrival_price=100.0, target_qty=100)
        assert abs(is_bps - 100.0) < 0.01  # 1% above = 100bps

    def test_no_fills(self):
        assert implementation_shortfall([], 100.0, 100) == float("inf")

    def test_unfilled_penalty(self):
        fills = [Fill(price=101.0, size=50, tick=0, side=Side.BUY)]
        is_full = implementation_shortfall(fills, 100.0, 50)
        is_partial = implementation_shortfall(fills, 100.0, 100)
        assert is_partial > is_full  # penalty for unfilled

    def test_negative_shortfall(self):
        # Price moved in our favor
        fills = [Fill(price=99.0, size=100, tick=0, side=Side.BUY)]
        is_bps = implementation_shortfall(fills, 100.0, 100)
        assert is_bps < 0

    def test_invalid_arrival_price_raises(self):
        fills = [Fill(price=101.0, size=100, tick=0, side=Side.BUY)]
        with pytest.raises(ValueError, match="arrival_price must be positive"):
            implementation_shortfall(fills, 0.0, 100)


class TestSimulation:
    def test_duplicate_agent_ids_rejected(self):
        agents = [TWAPAgent("dup", 10), TWAPAgent("dup", 10)]
        with pytest.raises(ValueError, match="duplicate agent_id"):
            Simulation(agents, SimulationConfig(n_ticks=1, target_qty=10), seed=1)

    def test_reproducible(self):
        agents1 = [TWAPAgent("twap", 10000)]
        agents2 = [TWAPAgent("twap", 10000)]
        config = SimulationConfig(n_ticks=50)

        r1 = Simulation(agents1, config, seed=123).run()[0]
        r2 = Simulation(agents2, config, seed=123).run()[0]

        assert r1[0].implementation_shortfall == r2[0].implementation_shortfall

    def test_agents_get_fills(self):
        agents = [TWAPAgent("twap", 10000)]
        config = SimulationConfig(n_ticks=100)
        results = Simulation(agents, config, seed=42).run()[0]
        assert results[0].total_filled > 0

    def test_passive_limits_can_fill_from_post_agent_noise(self, monkeypatch):
        config = SimulationConfig(n_ticks=1, target_qty=5, noise_avg_orders=0)

        monkeypatch.setattr("engine.simulation.MidPriceProcess", FakePriceModel)
        monkeypatch.setattr("engine.simulation.MarketMaker", StaticMarketMaker)
        monkeypatch.setattr("engine.simulation.NoiseTrader", TwoPhaseNoiseTrader)

        result = Simulation(
            agents=[PassiveLimitAgent("passive", 5)],
            config=config,
            seed=123,
        ).run()[0][0]

        assert result.total_filled == 5
        assert result.remaining_qty == 0

    def test_replay_activity_counts_passive_fills(self, monkeypatch):
        config = SimulationConfig(n_ticks=1, target_qty=5, noise_avg_orders=0)

        monkeypatch.setattr("engine.simulation.MidPriceProcess", FakePriceModel)
        monkeypatch.setattr("engine.simulation.MarketMaker", StaticMarketMaker)
        monkeypatch.setattr("engine.simulation.NoiseTrader", TwoPhaseNoiseTrader)

        _results, _tick_mids, _tick_spreads, replay_ticks = Simulation(
            agents=[PassiveLimitAgent("passive", 5)],
            config=config,
            seed=123,
        ).run()

        replay_agent = replay_ticks[0]["agents"]["passive"]
        assert replay_agent["filled_this_tick"] == 5
        assert replay_agent["cumulative_filled"] == 5

    def test_flow_signals(self, monkeypatch):
        FakePriceModel.seen_flows = []
        StaticMarketMaker.seen_flows = []
        config = SimulationConfig(n_ticks=3, target_qty=6, noise_avg_orders=0)

        monkeypatch.setattr("engine.simulation.MidPriceProcess", FakePriceModel)
        monkeypatch.setattr("engine.simulation.MarketMaker", StaticMarketMaker)
        monkeypatch.setattr("engine.simulation.NoiseTrader", SilentNoiseTrader)

        Simulation(
            agents=[FixedSliceAgent("slicer", 6)],
            config=config,
            seed=123,
        ).run()[0]

        # Price model sees per-tick flow
        assert FakePriceModel.seen_flows == [0, 2, 2]
        # Market maker sees cumulative flow
        assert StaticMarketMaker.seen_flows == [0, 2, 4]

    def test_cancel_order_frees_budget_for_repricing(self, monkeypatch):
        config = SimulationConfig(n_ticks=2, target_qty=5, noise_avg_orders=0)

        monkeypatch.setattr("engine.simulation.MidPriceProcess", FakePriceModel)
        monkeypatch.setattr("engine.simulation.MarketMaker", StaticMarketMaker)
        monkeypatch.setattr("engine.simulation.NoiseTrader", SilentNoiseTrader)

        result = Simulation(
            agents=[CancelThenMarketAgent("repricer", 5)],
            config=config,
            seed=123,
        ).run()[0][0]

        assert result.total_filled == 5
        assert result.remaining_qty == 0

    def test_invalid_action_payload_is_ignored(self, monkeypatch):
        class BadPayloadAgent(BaseAgent):
            def on_tick(self, state):
                return ["bad", None, 123]

        config = SimulationConfig(n_ticks=1, target_qty=5, noise_avg_orders=0)

        monkeypatch.setattr("engine.simulation.MidPriceProcess", FakePriceModel)
        monkeypatch.setattr("engine.simulation.MarketMaker", StaticMarketMaker)
        monkeypatch.setattr("engine.simulation.NoiseTrader", SilentNoiseTrader)

        result = Simulation(
            agents=[BadPayloadAgent("bad", 5)],
            config=config,
            seed=123,
        ).run()[0][0]

        assert result.total_filled == 0
        assert result.remaining_qty == 5

    def test_agent_limit_orders_persist_until_cancelled(self, monkeypatch):
        config = SimulationConfig(n_ticks=6, target_qty=1, noise_avg_orders=0)
        agent = LongLivedPassiveAgent("passive", 1)

        monkeypatch.setattr("engine.simulation.MidPriceProcess", FakePriceModel)
        monkeypatch.setattr("engine.simulation.MarketMaker", StaticMarketMaker)
        monkeypatch.setattr("engine.simulation.NoiseTrader", SilentNoiseTrader)

        result = Simulation(
            agents=[agent],
            config=config,
            seed=123,
        ).run()[0][0]

        assert agent.saw_resting_on_tick_five
        assert result.total_filled == 1
        assert result.remaining_qty == 0

    def test_noise_limit_orders_persist_across_ticks(self, monkeypatch):
        config = SimulationConfig(n_ticks=2, target_qty=3, noise_avg_orders=0)

        monkeypatch.setattr("engine.simulation.MidPriceProcess", FakePriceModel)
        monkeypatch.setattr("engine.simulation.MarketMaker", EmptyMarketMaker)
        monkeypatch.setattr("engine.simulation.NoiseTrader", DelayedNoiseLimitTrader)

        result = Simulation(
            agents=[TickOneMarketAgent("taker", 3)],
            config=config,
            seed=123,
        ).run()[0][0]

        assert result.total_filled == 3
        assert result.avg_price == pytest.approx(100.3)

    def test_market_maker_quotes_replace_instead_of_accumulating(self, monkeypatch):
        config = SimulationConfig(n_ticks=3, target_qty=1, noise_avg_orders=0)

        monkeypatch.setattr("engine.simulation.MidPriceProcess", FakePriceModel)
        monkeypatch.setattr("engine.simulation.MarketMaker", StaticMarketMaker)
        monkeypatch.setattr("engine.simulation.NoiseTrader", SilentNoiseTrader)

        _results, _tick_mids, _tick_spreads, replay_ticks = Simulation(
            agents=[NoOpAgent("idle", 1)],
            config=config,
            seed=123,
        ).run()

        final_tick = replay_ticks[-1]
        assert final_tick["bids"][0] == (99.9, 100)
        assert final_tick["asks"][0] == (100.1, 100)

    def test_later_agents_see_live_book_after_earlier_agent_executes(self, monkeypatch):
        observer = ObserveBestAskAgent("observer", 1)
        config = SimulationConfig(n_ticks=1, target_qty=1, noise_avg_orders=0)

        monkeypatch.setattr("engine.simulation.MidPriceProcess", FakePriceModel)
        monkeypatch.setattr("engine.simulation.MarketMaker", TwoLevelBookMarketMaker)
        monkeypatch.setattr("engine.simulation.NoiseTrader", SilentNoiseTrader)
        monkeypatch.setattr(
            "engine.simulation.MatchingEngine",
            DeterministicMatchingEngine,
        )

        Simulation(
            agents=[OneShotMarketAgent("first", 1), observer],
            config=config,
            seed=123,
        ).run()

        assert observer.seen_best_ask == pytest.approx(101.0)

