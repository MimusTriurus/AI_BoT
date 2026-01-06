from dataclasses import dataclass
from typing import Tuple, Any, List

from AI_BoT.common.constants import UnitType
from AI_BoT.common.reachable_positions_calculator import ReachabilityCache


@dataclass
class Unit:
    id: Any
    # текущая позиция
    pos: Tuple[int, int]
    # weapon damage
    wd: Tuple[int, int]
    # weapon range
    wr: Tuple[int, int]
    mp: int
    hp: int
    # ценность юнита (его стоимость)
    value: int

    # вместимость пассажиров
    capacity: int

    inside_transport: bool

    unit_type: UnitType

    reachability_cache: ReachabilityCache = None

    def is_Transport(self):
        return self.capacity > 0

    def can_fire(self):
        return not self.inside_transport and self.wr[0] > 0

    def can_move(self):
        return self.mp > 0