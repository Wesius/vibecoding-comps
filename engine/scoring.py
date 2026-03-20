from __future__ import annotations

from engine.types import Fill


def implementation_shortfall(
    fills: list[Fill],
    arrival_price: float,
    target_qty: int,
) -> float:
    """Compute implementation shortfall in basis points.

    IS_bps = (avg_execution_price - arrival_price) / arrival_price * 10,000

    Unfilled quantity is handled by the simulation's terminal sweep, so
    this function only scores actual fills. Returns float('inf') if no
    fills at all.
    """
    if arrival_price <= 0:
        raise ValueError("arrival_price must be positive")
    if target_qty <= 0:
        raise ValueError("target_qty must be positive")
    if not fills:
        return float("inf")

    total_filled = sum(f.size for f in fills)
    total_cost = sum(f.price * f.size for f in fills)

    avg_price = total_cost / total_filled
    is_bps = (avg_price - arrival_price) / arrival_price * 10_000
    return is_bps
