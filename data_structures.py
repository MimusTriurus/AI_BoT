# Константы для ключей
from dataclasses import dataclass
from enum import Enum
from math import log2
from typing import List, Dict, Tuple, Optional, Set
from clustering import cluster_by_proximity, kmeans_hex, soft_clustering
from w9_pathfinding.pf import IDAStar, AStar
from w9_pathfinding.envs import HexGrid

POS_KEY = 'pos'
ID_KEY = 'id'
TYPE_KEY = 'type'
MOVE_RANGE_KEY = 'move_range'
ATTACK_RANGE_KEY = 'attack_range'
DAMAGE_KEY = 'damage'
VALUE_KEY = 'value'
HP_KEY = 'hp'
CAPACITY_KEY = 'capacity'

class UnitType(Enum):
    TANK                = 0
    LAND_TRANSPORT      = 1
    ABSTRACT_TARGET     = 2
    LAV                 = 4

@dataclass
class Squad:
    units: List[Dict]
    total_damage: int
    unit_indices: List[int]


@dataclass
class PickupPlan:
    """План погрузки одного юнита"""
    unit_idx: int
    unit_pos: Tuple[int, int]  # Начальная позиция юнита
    meeting_point: Tuple[int, int]  # Позиция транспорта для погрузки
    unit_meeting_point: Tuple[int, int] = None  # Позиция юнита при погрузке (соседняя с транспортом)
    transport_cost: int = 0
    unit_cost: int = 0


@dataclass
class TransportMission:
    transport_idx: int
    target: Dict
    squad: Squad
    pickup_sequence: List[PickupPlan]
    unload_point: Tuple[int, int]
    full_route: List[Tuple[int, int]]
    efficiency: float

# m - move,
# w - weapon
# d - damage,
# r - weapon range,
# h - hp,
# v - value,
# c - capacity (transport)
unit_data = {
    #                                 m    d   r   h   v   c
    UnitType.TANK:                  ( 2 ,  2,  1,  5,  2,  0 ),
    UnitType.LAND_TRANSPORT:        ( 7,   0,  0,  3,  1,  2 ),
    UnitType.ABSTRACT_TARGET:       ( 0 ,  0,  0,  4,  9,  0 ),
    UnitType.LAV:                   ( 1 ,  1,  1,  4,  1,  0 ),
}


def make_unit(u_id, pos, unit_type: UnitType):
    unit = {
        ID_KEY: u_id,
        POS_KEY: pos,
        MOVE_RANGE_KEY: unit_data[unit_type][0],
        DAMAGE_KEY: unit_data[unit_type][1],
        ATTACK_RANGE_KEY: unit_data[unit_type][2],
        HP_KEY: unit_data[unit_type][3],
        VALUE_KEY: unit_data[unit_type][4],
        CAPACITY_KEY: unit_data[unit_type][5],
    }
    return unit

class UnitsStorage:
    def __init__(self):
        self.units = dict()

    def add_unit(self, u_id: str, pos: Tuple, unit_type: UnitType) -> dict:
        unit = make_unit(u_id, pos, unit_type)
        self.units[pos] = unit
        return unit

    def get_units_pos(self):
        return list(self.units.keys())

    def get_units(self, units_pos: List[Tuple[int, int]] = None) -> List:
        if units_pos:
            results = []
            for u_p in units_pos:
                if u_p in self.units:
                    results.append(self.units[u_p])
            return results
        else:
            return list(self.units.values())

    def get_unit(self, pos: Tuple):
        return self.units[pos]

    def get_unit_by_id(self, u_id: str) -> Optional[dict]:
        for p, u in self.units.items():
            if u[ID_KEY] == u_id:
                return u
        return None

    def get_clusters(self, grid):
        units_poses = self.get_units_pos()
        pf = AStar(grid)
        centers, clusters = kmeans_hex(pf, units_poses, k=3)
        # кластеризация по степени близости
        # prox_clusters = cluster_by_proximity(pf, units_poses, grid, max_range=2)
        # кластеризация с перекрытием
        clusters = soft_clustering(units_poses, centers, pf, move_range=5)
        return clusters


class TransportPlan:
    def __init__(
            self,
            transport: dict,
            target: dict,
            passengers: List[dict],
            path: List[Tuple[int, int]],
            grid: HexGrid,
            pf: AStar
    ):
        self.transport: dict = transport
        self.target: dict = target
        self.passengers: List[dict] = passengers
        self.path: Dict[str, Tuple[int, int]] = path
        self.grid: HexGrid = grid
        self.pf: AStar = pf

        self.meeting_points: Dict[str, Tuple[int, int]] = self.calculate_meeting_points()
        self.delivery_path: List[Tuple[int, int]] = self.calculate_delivery_path()

        self.utility: float = self._calculate_utility()

    def calculate_meeting_points(self):
        result = dict()
        for p in self.path:
            for passenger in self.passengers:
                p_id = passenger[ID_KEY]
                if self.grid.adjacent(passenger[POS_KEY], p) and p_id not in result:
                    result[p_id] = p
        return result

    # путь до цели после загрузки
    def calculate_delivery_path(self) -> List[Tuple[int, int]]:
        last_loading_idx = max(self.path.index(lp) for lp in self.meeting_points.values() if lp in self.path)
        return self.path[last_loading_idx:]

    def _calculate_utility(self) -> float:
        """
        Рассчитывает скор (Utility) плана.
        Учитывает: Урон, Ценность цели, Оверхед, Бонус за убийство,
        и главное - Marginal Utility (нужен ли вообще транспорт).
        """

        # 1. Стоимость пути транспорта (Очки Передвижения)
        # Используем метод сетки для точного расчета (учет дорог, болот и т.д.)
        transport_path_cost = self.grid.calculate_cost(self.delivery_path)

        target_hp = self.target[HP_KEY]
        target_val = self.target[VALUE_KEY]
        target_pos = self.target[POS_KEY]

        total_potential_damage = 0
        damage_added_by_transport = 0  # Урон, который невозможен без транспорта
        damage_available_on_foot = 0  # Урон, который можно нанести и так

        cost_denominator = max(0.5, transport_path_cost)

        for unit in self.passengers:
            dmg = unit[DAMAGE_KEY]
            total_potential_damage += dmg

            # --- ПРОВЕРКА: МОЖЕТ ЛИ ЮНИТ ДОЙТИ САМ? ---
            # Используем AStar, чтобы проверить реальную проходимость.
            # Находим путь от юнита до цели.
            walk_path = self.pf.find_path(unit[POS_KEY], target_pos)

            can_reach_on_foot = False

            if walk_path:
                # Стоимость прохода пешком
                walk_cost = self.grid.calculate_cost(walk_path)

                # Эвристика для дистанционной атаки:
                # Юниту не обязательно вставать НА цель, достаточно подойти на radius_attack.
                # Предполагаем, что последние N шагов (равные радиусу) можно не делать.
                # Это грубая оценка, но эффективная.
                attack_range_discount = unit.get(ATTACK_RANGE_KEY, 1)
                adjusted_walk_cost = max(0, walk_cost - attack_range_discount)

                if adjusted_walk_cost <= unit[MOVE_RANGE_KEY]:
                    can_reach_on_foot = True

            if can_reach_on_foot:
                damage_available_on_foot += dmg
            else:
                damage_added_by_transport += dmg

        # А нужно ли нам грузить юнит если он может атаковать цель и сам?
        if damage_available_on_foot > 0.0:
            return 0

        if transport_path_cost < 0.1 and damage_available_on_foot >= target_hp:
            return 0
        # --- ЛОГИКА ОЦЕНКИ ---

        # 1. Взвешиваем урон (Marginal Utility)
        # Урон, требующий транспорта -> Вес 1.0 (Высокий приоритет)
        # Урон, доступный пешком -> Вес 0.1 (Низкий приоритет, используем транспорт только если некуда девать)
        weighted_damage = (damage_added_by_transport * 1.0) + (damage_available_on_foot * 0.1)

        # Коэффициент качества плана (0.0 - 1.0). Насколько этот план оправдывает использование машины?
        if total_potential_damage > 0:
            transport_relevance = weighted_damage / total_potential_damage
        else:
            transport_relevance = 0

        # 2. Эффективность по здоровью (Capping)
        # Мы не можем получить пользы больше, чем у цели есть здоровья.
        real_damage_dealt = min(total_potential_damage, target_hp)

        # Применяем релевантность транспорта к нанесенному урону
        utility_score_base = real_damage_dealt * transport_relevance

        # 3. Бонус за убийство (Kill Bonus)
        kill_multiplier = 1.0
        if total_potential_damage >= target_hp:
            # Если убиваем...
            if damage_available_on_foot >= target_hp:
                # ...но могли убить и пешком. Небольшой бонус за скорость/гарантию.
                kill_multiplier = 1.1
            else:
                # ...и без транспорта убийство было бы невозможно. Огромный бонус.
                kill_multiplier = 2.0

        # 4. Штраф за Оверхед (Waste Penalty)
        # Если у цели 1 HP, а мы везем 50 урона -> штрафуем.
        waste_damage = max(0, total_potential_damage - target_hp)
        waste_penalty = log2(1 + waste_damage)# или waste_damage * 0.5 если хотим пожестче

        # --- ИТОГОВАЯ ФОРМУЛА ---
        # (Скорректированный урон * Ценность Цели * Множитель Убийства) - Штраф
        numerator = (utility_score_base * target_val * kill_multiplier) - waste_penalty

        # Делим на стоимость пути (Урон в пересчете на 1 Очко Движения транспорта)
        final_utility = max(0.0, numerator / cost_denominator)
        # На подумать...
        '''
        # === Учитываем коэффициент загрузки ===
        capacity = self.transport.get(CAPACITY_KEY, len(self.passengers))
        load_factor = len(self.passengers) / max(1, capacity)
        final_utility *= (0.5 + 0.5 * load_factor)

        # === Учитываем остаток очков движения ===
        move_range = self.transport[MOVE_RANGE_KEY]
        delivery_cost = self.grid.calculate_cost(self.delivery_path)
        remaining = max(0.1, move_range - delivery_cost)
        remaining_factor = remaining / move_range
        final_utility *= remaining_factor
        '''
        return final_utility

    def __repr__(self):
        return str(self)

    def __str__(self):
        passengers_id = []
        for u in self.passengers:
            u_id = u[ID_KEY]
            passengers_id.append(u_id)
        passengers_id = sorted(passengers_id)
        passengers_str = ', '.join(passengers_id)
        return f"Transport: {self.transport[ID_KEY]} took units {passengers_str} and transit to target: {self.target[ID_KEY]} ({self.path[-1]}). Utility: {self.utility}"

    def to_path(self):
        return self.transport[ID_KEY], self.path
