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
        - All orders must be BUY side (you are filling a buy order)
        - CancelOrder may target only your own live resting orders
        - Total size across orders must not exceed state.remaining_qty
          (engine will clip if you exceed)
        - Empty list is valid (skip this tick)
        """
        ...
