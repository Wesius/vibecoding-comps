from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from engine.types import CancelOrder, Order, TickState


class BaseAgent(ABC):
    """Base class for all execution agents.

    Subclasses must implement on_tick(). May optionally override __init__
    for loading config or model weights.
    """

    def __init__(self, agent_id: str, target_qty: int, **kwargs: object) -> None:
        self.agent_id = agent_id
        self.target_qty = target_qty

    @abstractmethod
    def on_tick(self, state: TickState) -> Sequence[Order | CancelOrder]:
        """Called once per tick. Return a list of actions.

        Rules:
        - BUY orders: fill your buy mandate (target_qty shares)
        - SELL orders: sell shares you already hold (no shorting).
          You can sell up to state.net_position shares.
        - CancelOrder may target only your own live resting orders
        - Buy size clipped to remaining buy budget; sell size clipped
          to available sell budget (engine enforces no-short)
        - Empty list is valid (skip this tick)
        """
        ...
