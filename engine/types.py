from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class Side(Enum):
    BUY = auto()
    SELL = auto()


class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()


@dataclass(frozen=True, slots=True)
class Order:
    """An order submitted by an agent or background trader."""

    side: Side
    size: int
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    agent_id: Optional[str] = None  # stamped by the engine, not by agents

    def __post_init__(self) -> None:
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("Limit orders require a price")
        if self.size <= 0:
            raise ValueError("Order size must be positive")


@dataclass(frozen=True, slots=True)
class CancelOrder:
    """Cancel a previously submitted resting order by id."""

    order_id: int

    def __post_init__(self) -> None:
        if self.order_id <= 0:
            raise ValueError("order_id must be positive")


@dataclass(frozen=True, slots=True)
class Fill:
    """A single fill (partial or full) received by an agent."""

    price: float
    size: int
    tick: int
    side: Side


@dataclass(frozen=True, slots=True)
class TradeTapeEntry:
    """A trade visible on the public tape."""

    price: float
    size: int
    aggressor_side: Side
    tick: int


@dataclass(frozen=True, slots=True)
class BookLevel:
    """One price level in the order book snapshot."""

    price: float
    size: int


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    """Immutable snapshot of the order book visible to agents."""

    bids: tuple[BookLevel, ...]  # sorted descending by price (best bid first)
    asks: tuple[BookLevel, ...]  # sorted ascending by price (best ask first)
    mid_price: float


@dataclass(frozen=True, slots=True)
class RestingOrderInfo:
    """Visible status for one of the agent's currently resting orders."""

    order_id: int
    side: Side
    price: float
    remaining_size: int
    queue_ahead: int | None
    submitted_tick: int
    expires_tick: int | None


@dataclass(frozen=True, slots=True)
class TickState:
    """Everything an agent sees on a given tick."""

    tick: int
    total_ticks: int
    order_book: OrderBookSnapshot
    remaining_qty: int
    net_position: int
    fills: tuple[Fill, ...]
    avg_fill_price: float
    trade_tape: tuple[TradeTapeEntry, ...]
    arrival_price: float
    resting_orders: tuple[RestingOrderInfo, ...] = ()


@dataclass
class AgentResult:
    """Final result for one agent after a simulation run."""

    agent_id: str
    agent_class: str
    fills: list[Fill]
    total_filled: int
    avg_price: float
    arrival_price: float
    implementation_shortfall: float  # basis points
    remaining_qty: int
    # Per-tick tracking (length = n_ticks)
    cumulative_fill_pct: list[float] | None = None  # 0.0 to 1.0
    running_avg_price: list[float] | None = None


@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""

    n_ticks: int = 500
    target_qty: int = 10_000
    initial_price: float = 100.0
    # Price model
    volatility: float = 0.0002
    drift: float = 0.0
    permanent_impact_bps: float = 2.0
    # Market maker
    mm_n_levels: int = 10
    mm_base_spread_bps: float = 10.0
    mm_level_spacing_bps: float = 10.0
    mm_base_size: int = 15
    mm_depth_growth: float = 1.15
    mm_flow_sensitivity: float = 0.3
    # Noise trader
    noise_avg_orders: float = 5.0
    noise_market_prob: float = 0.7
    noise_mean_size: int = 50
    noise_size_std: float = 0.5
    # Noise sell budget as fraction of total agent demand (target_qty * n_agents).
    # Drawn uniformly per seed from [low, high]. None disables the cap.
    noise_sell_cap_low: float | None = None
    noise_sell_cap_high: float | None = None
    # Optional passive order lifetime cap. None means rest until cancelled.
    agent_order_ttl_ticks: int | None = None
    # Noise trader passive orders rest for a bounded time to preserve
    # queue continuity without letting stale orders accumulate forever.
    noise_order_ttl_ticks: int = 20
    # Tape
    tape_window: int = 50

    def __post_init__(self) -> None:
        if self.n_ticks <= 0:
            raise ValueError("n_ticks must be positive")
        if self.target_qty <= 0:
            raise ValueError("target_qty must be positive")
        if self.initial_price <= 0:
            raise ValueError("initial_price must be positive")
        if (
            self.agent_order_ttl_ticks is not None
            and self.agent_order_ttl_ticks <= 0
        ):
            raise ValueError("agent_order_ttl_ticks must be positive or None")
        if self.noise_order_ttl_ticks <= 0:
            raise ValueError("noise_order_ttl_ticks must be positive")
