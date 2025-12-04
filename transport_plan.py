from math import log2
from typing import List, Dict, Tuple, Optional, Set
from w9_pathfinding.pf import IDAStar, AStar
from w9_pathfinding.envs import HexGrid
from common.constants import *

class TransportPlan:
    def __init__(
            self,
            transport: dict,
            target: dict,
            passengers: List[dict],
            path: List[Tuple[int, int]],
            unload_map: Dict[str, Tuple[int, int]],
            grid: HexGrid,
            pf: AStar
    ):
        self.transport: dict = transport
        self.target: dict = target
        self.passengers: List[dict] = passengers
        self.path: List[Tuple[int, int]] = path
        # вариант разгрузки юнитов
        self.unload_map = unload_map
        self.grid: HexGrid = grid
        self.pf: AStar = pf

        # пока костыль - транспортируем сами себя
        self.transit_ourself: bool = False
        if len(self.passengers) == 1 and passengers[0][POS_KEY] == self.transport[POS_KEY]:
            self.transit_ourself = True

        self.meeting_points: Dict[str, Tuple[int, int]] = self.calculate_meeting_points()
        self.delivery_path: List[Tuple[int, int]] = self.calculate_delivery_path()

        self.occupied_hexes_set = set(unload_map.values())
        if self.path:
            self.occupied_hexes_set.add(self.path[-1])

        self.actual_damage_contribution = self._calculate_actual_damage()

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
        try:
            last_loading_idx = max(self.path.index(lp) for lp in self.meeting_points.values() if lp in self.path)
            return self.path[last_loading_idx:]
        except Exception as e:
            return []

    def _calculate_actual_damage(self) -> float:
        """
        Рассчитывает урон, который действительно будет нанесен, исходя из
        оптимальной расстановки выгрузки (self.unload_map).
        """
        total_damage = 0.0
        target_pos = self.target[POS_KEY]

        for unit in self.passengers:
            unit_id = unit[ID_KEY]
            unload_pos = self.unload_map.get(unit_id)

            if unload_pos:
                path = self.pf.find_path(unload_pos, target_pos)
                dist = self.grid.calculate_cost(path)

                # Если юнит достает до цели с назначенной ему точки выгрузки
                if dist <= unit[MAX_ATTACK_RANGE_KEY]:
                    total_damage += unit[DAMAGE_KEY]

        return total_damage

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

        actual_damage = self.actual_damage_contribution

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
                attack_range_discount = unit.get(MAX_ATTACK_RANGE_KEY, 1)
                adjusted_walk_cost = max(0, walk_cost - attack_range_discount)

                if adjusted_walk_cost <= unit[MOVE_RANGE_KEY]:
                    can_reach_on_foot = True

            cost_path_2_target = self.grid.calculate_cost(self.pf.find_path(self.unload_map[unit[ID_KEY]], self.target[POS_KEY]))
            is_unit_attacking = self.unload_map.get(unit[ID_KEY]) is not None and cost_path_2_target <= unit[MAX_ATTACK_RANGE_KEY]

            if is_unit_attacking:
                if can_reach_on_foot:
                    damage_available_on_foot += dmg
                else:
                    damage_added_by_transport += dmg

        if actual_damage == 0.0:
            return 0.0

        # А нужно ли нам грузить юнит если он может атаковать цель и сам?
        if damage_available_on_foot > 0.0 and not self.transit_ourself:
            return 0

        if transport_path_cost < 0.1 and damage_available_on_foot >= target_hp:
            return 0
        # --- ЛОГИКА ОЦЕНКИ ---

        # 1. Взвешиваем урон (Marginal Utility)
        # Урон, требующий транспорта -> Вес 1.0 (Высокий приоритет)
        # Урон, доступный пешком -> Вес 0.1 (Низкий приоритет, используем транспорт только если некуда девать)
        weighted_damage = (damage_added_by_transport * 1) + (damage_available_on_foot * 0.1)

        # Коэффициент качества плана (0.0 - 1.0)
        # Используем actual_damage как нормализатор для весов.
        transport_relevance = weighted_damage / actual_damage

        # 2. Эффективность по здоровью (Capping)
        real_damage_dealt = min(actual_damage, target_hp)  # <--- ИСПОЛЬЗУЕМ actual_damage

        # Применяем релевантность транспорта к нанесенному урону
        utility_score_base = real_damage_dealt * transport_relevance

        # 3. Бонус за убийство (Kill Bonus)
        kill_multiplier = 1.0
        if actual_damage >= target_hp:  # <--- ИСПОЛЬЗУЕМ actual_damage
            if damage_available_on_foot >= target_hp:
                kill_multiplier = 1.1
            else:
                kill_multiplier = 4.0

        # 4. Штраф за Оверхед (Waste Penalty)
        # Штрафуем только за УРОН, который реально мог быть нанесен, но превышает HP цели.
        waste_damage = max(0, actual_damage - target_hp)  # <--- ИСПОЛЬЗУЕМ actual_damage
        waste_penalty = log2(1 + waste_damage)

        # --- ИТОГОВАЯ ФОРМУЛА ---
        numerator = (utility_score_base * target_val * kill_multiplier) - waste_penalty

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

        mp_max = self.transport[MOVE_RANGE_KEY]
        mp_use = int(self.grid.calculate_cost(self.path))

        unloading_str = ''
        for u_id, pos in self.unload_map.items():
            unloading_str += f'{u_id} {pos} '

        return (
            f"Transport: {self.transport[ID_KEY]} took units {passengers_str} "
            f"and transit to unload position {self.path[-1]}\n"
            f"Units attack target {self.target[ID_KEY]} {self.target[POS_KEY]} from positions: {unloading_str}\n"
            f"MP: [{mp_use}/{mp_max}]\n"
            f"Utility: [{self.utility}]"
        )

    def to_path(self):
        return self.transport[ID_KEY], self.path

    @staticmethod
    def solve_best_unload_configuration(
            drop_zone: Tuple[int, int],
            target_pos: Tuple[int, int],
            passengers: List[dict],
            enemy_positions: List[Tuple[int, int]],
            other_obstacles: List[Tuple[int, int]],
            grid: HexGrid,
            pf: AStar
    ) -> Tuple[Dict[str, Tuple[int, int]], float]:
        """
        Находит такую расстановку юнитов по свободным клеткам,
        которая дает МАКСИМАЛЬНЫЙ суммарный урон.
        """
        # 1. Находим доступные слоты (гексы)
        neighbors = grid.get_neighbors(drop_zone, include_self=False)
        available_hexes = []

        blocked_set = set(enemy_positions) | set(other_obstacles)

        for h, _ in neighbors:
            if not grid.has_obstacle(h) and h not in blocked_set:
                available_hexes.append(h)

        # Если мест меньше, чем пассажиров, придется кого-то оставить в машине (или выбрать подмножество)
        # Но для простоты предположим, что мы пытаемся высадить максимум

        best_mapping = {}
        max_total_damage = -1.0

        # 2. Перебираем варианты: Какой юнит на какой гекс?
        # itertools.permutations генерирует варианты назначения.
        # Если гексов больше чем юнитов, берем permutations(hexes, len(units))

        num_units = len(passengers)
        num_slots = len(available_hexes)

        # Генерируем все возможные назначения (Slots -> Units)
        # Если слотов 4, а юнитов 2: выбираем 2 слота из 4 и расставляем юнитов

        # Чтобы не усложнять, сделаем перебор назначений:
        # Вариант: [(Unit1, HexA), (Unit2, HexB)], [(Unit1, HexB), (Unit2, HexA)] ...

        if num_slots == 0:
            return {}, 0.0

        # Оптимизация: перебираем слоты для юнитов
        import itertools

        # Генерируем все возможные распределения (длиной min(units, slots))
        limit = min(num_units, num_slots)

        # Берем комбинации слотов, которые будем использовать
        for hex_subset in itertools.permutations(available_hexes, limit):
            # hex_subset[0] достается passengers[0]
            # hex_subset[1] достается passengers[1]

            current_mapping = {}
            current_damage = 0.0

            for i in range(limit):
                unit = passengers[i]
                hex_pos = hex_subset[i]

                # Проверка дистанции атаки
                path = pf.find_path(hex_pos, target_pos)
                dist = grid.calculate_cost(path)
                if dist <= unit[MAX_ATTACK_RANGE_KEY]:
                    dmg = unit[DAMAGE_KEY]
                else:
                    dmg = 0.0  # Юнит высадился, но не достает

                current_mapping[unit[ID_KEY]] = hex_pos
                current_damage += dmg

            if current_damage > max_total_damage:
                max_total_damage = current_damage
                best_mapping = current_mapping.copy()

        return best_mapping, max_total_damage