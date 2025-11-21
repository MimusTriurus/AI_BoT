import heapq
from enum import Enum

import pygame
from typing import List, Dict, Tuple, Optional, Set
from visualizer import HexVisualizer
from w9_pathfinding.envs import HexGrid, HexLayout
from w9_pathfinding.pf import IDAStar, AStar
from w9_pathfinding.mapf import CBS, SpaceTimeAStar, ReservationTable, MultiAgentAStar
from w9_pathfinding.visualization import animate_grid

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from itertools import combinations, permutations
import heapq

# Константы для ключей
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


class TransportPlanner:
    def __init__(self, grid, mapf_solver, pathfinder, reservation_table):
        self.grid = grid
        self.mapf = mapf_solver
        self.pf = pathfinder
        self.rt = reservation_table

    def plan_transport_operations(
            self,
            units: List[Dict],
            transports: List[Dict],
            targets: List[Dict]
    ) -> List[TransportMission]:
        # 1. Анализ и приоритизация целей
        prioritized_targets = self._prioritize_targets(targets, units)

        # 2. Планирование миссий для каждого транспорта
        transport_missions = []
        available_units = set(range(len(units)))

        for transport_idx, transport in enumerate(transports):
            mission = self._plan_best_mission(
                transport, transport_idx, units, available_units, prioritized_targets
            )

            if mission:
                transport_missions.append(mission)
                # Удаляем использованные юниты
                available_units -= set(mission.squad.unit_indices)

        # 3. Координация путей (MAPF)
        # под вопросом пока
        # coordinated_missions = self._coordinate_missions(transport_missions, transports)
        coordinated_missions = transport_missions
        return coordinated_missions

    def _prioritize_targets(
            self,
            targets: List[Dict],
            units: List[Dict]
    ) -> List[Tuple[Dict, float]]:
        """
        Приоритизация целей по эффективности

        Returns:
            Список (цель, приоритет), отсортированный по убыванию приоритета
        """
        prioritized = []

        for target in targets:
            hp = target[HP_KEY]
            value = target[VALUE_KEY]

            # Приоритет = ценность / требуемое здоровье
            # Чем меньше HP нужно для уничтожения, тем выше приоритет
            priority = value / max(hp, 0.1)

            prioritized.append((target, priority))

        # Сортировка по убыванию приоритета
        prioritized.sort(key=lambda x: x[1], reverse=True)

        return prioritized

    def _plan_best_mission(
            self,
            transport: Dict,
            transport_idx: int,
            units: List[Dict],
            available_units: Set[int],
            prioritized_targets: List[Tuple[Dict, float]]
    ) -> Optional[TransportMission]:
        best_mission = None
        best_efficiency = -float('inf')

        for target, _ in prioritized_targets:
            # Формирование оптимального отряда с учетом транспортировки
            squad = self._form_optimal_squad(
                units,
                available_units,
                target,
                transport[CAPACITY_KEY],
                transport,
                transport[POS_KEY]
            )

            if not squad:
                continue

            # Поиск точки выгрузки (теперь для конкретных юнитов отряда)
            unload_point = self._find_unload_position(target, squad, units)

            if not unload_point:
                continue

            # Планирование последовательности погрузки
            pickup_sequence = self._plan_pickup_sequence(
                transport, squad, units
            )

            if not pickup_sequence:
                continue

            # Построение полного маршрута
            full_route = self._build_transport_route(
                transport, pickup_sequence, unload_point
            )

            if not full_route:
                continue
            if self.grid.calculate_cost(full_route) > transport[MOVE_RANGE_KEY]:
                continue
            # Оценка эффективности
            efficiency = self._calculate_efficiency(
                target, squad, full_route, pickup_sequence
            )

            if efficiency > best_efficiency:
                best_mission = TransportMission(
                    transport_idx=transport_idx,
                    target=target,
                    squad=squad,
                    pickup_sequence=pickup_sequence,
                    unload_point=unload_point,
                    full_route=full_route,
                    efficiency=efficiency
                )
                best_efficiency = efficiency

        return best_mission
    def _find_attack_positions_for_unit(self, target_pos: Tuple, weapon_range: int, move_range: int = 0) -> List[Tuple]:
        visited = {target_pos}
        frontier = [target_pos]
        attack_range = weapon_range + move_range
        for _ in range(attack_range):
            next_frontier = []
            for node in frontier:
                for n, _ in self.grid.get_neighbors(node):
                    if n not in visited and not self.grid.has_obstacle(n):
                        visited.add(n)
                        next_frontier.append(n)
            frontier = next_frontier

        visited.remove(target_pos)
        return list(visited)

    def _estimate_meeting_cost(
            self,
            unit_pos: Tuple[int, int],
            unit_move: int,
            transport_pos: Tuple[int, int],
            transport_move: int
    ) -> Optional[int]:
        distance = self._hex_distance(unit_pos, transport_pos)

        # Вариант 1: Транспорт идет к юниту
        if distance <= transport_move:
            return distance
        '''
        # Вариант 2: Юнит идет к транспорту
        if distance <= unit_move:
            return distance

        # Вариант 3: Встреча на середине
        # Проверяем, могут ли они встретиться, двигаясь навстречу
        combined_move = unit_move + transport_move
        if distance <= combined_move:
            # Оптимальная точка встречи примерно посередине
            # Стоимость = расстояние до середины для каждого
            return distance // 2 + distance % 2
        '''
        # Не могут встретиться за один ход
        return None

    def _can_deliver_unit_to_target(
            self,
            unit: Dict,
            transport_start: Tuple[int, int],
            transport_move: int,
            target_pos: Tuple[int, int]
    ) -> Optional[Dict]:
        unit_pos = unit[POS_KEY]
        unit_move = unit[MOVE_RANGE_KEY]
        attack_range = unit[ATTACK_RANGE_KEY]

        # Находим все возможные точки выгрузки (откуда юнит может атаковать цель)
        possible_unload_positions = self._find_attack_positions_for_unit(
            target_pos,
            attack_range
        )

        if not possible_unload_positions:
            return None

        # Для каждой точки выгрузки оцениваем возможность доставки
        best_delivery = None
        min_cost = float('inf')

        for unload_pos in possible_unload_positions:
            # 1. Расстояние от юнита до транспорта (для погрузки)
            pickup_cost = self._estimate_meeting_cost(
                unit_pos, unit_move, transport_start, transport_move
            )

            if pickup_cost is None:
                continue

            # 2. Расстояние от точки погрузки до точки выгрузки
            # Упрощенная оценка: транспорт должен дойти от места погрузки до выгрузки
            transport_distance = self._hex_distance(unit_pos, unload_pos)

            # Проверяем, хватит ли хода транспорта
            # (это грубая оценка, точный путь построим позже)
            if transport_distance > transport_move * 2:  # запас на обход препятствий
                continue

            # 3. Юнит после выгрузки должен быть в радиусе атаки от цели
            distance_to_target = self._hex_distance(unload_pos, target_pos)
            if distance_to_target > attack_range:
                continue

            # Общая стоимость операции
            total_cost = pickup_cost + transport_distance

            if total_cost < min_cost:
                min_cost = total_cost
                best_delivery = {
                    'can_deliver': True,
                    'unload_positions': [unload_pos],
                    'total_cost': total_cost,
                    'pickup_cost': pickup_cost,
                    'transport_distance': transport_distance
                }

        return best_delivery

    def _form_optimal_squad(
            self,
            units: List[Dict],
            available_units: Set[int],
            target: Dict,
            capacity: int,
            transport: Dict,
            transport_pos: Tuple[int, int]
    ) -> Optional[Squad]:
        target_hp = target[HP_KEY]
        target_pos = target[POS_KEY]
        transport_move = transport[MOVE_RANGE_KEY]

        candidate_scores = []

        for idx in available_units:
            unit = units[idx]

            # Проверяем, можно ли доставить юнита до цели
            delivery_info = self._can_deliver_unit_to_target(
                unit, transport_pos, transport_move, target_pos
            )

            if not delivery_info:
                continue

            # Оценка полезности юнита
            damage = unit[DAMAGE_KEY]
            attack_range = unit[ATTACK_RANGE_KEY]
            total_cost = delivery_info['total_cost']

            # Эффективность: урон / стоимость доставки
            efficiency = damage / max(total_cost, 1)

            # Бонус за большую дальность атаки (безопаснее)
            range_bonus = attack_range * 0.1

            score = efficiency + range_bonus

            candidate_scores.append({
                'idx': idx,
                'damage': damage,
                'score': score,
                'delivery_info': delivery_info
            })

        if not candidate_scores:
            return None

        # Сортируем юнитов по эффективности
        candidate_scores.sort(key=lambda x: x['score'], reverse=True)

        # Жадный подбор: берем самых эффективных юнитов до достижения нужного урона
        selected_units = []
        selected_indices = []
        total_damage = 0

        for candidate in candidate_scores:
            if len(selected_units) >= capacity:
                break

            selected_indices.append(candidate['idx'])
            selected_units.append(units[candidate['idx']])
            total_damage += candidate['damage']

            # Проверяем, достаточно ли урона (с запасом 10%)
            if total_damage >= target_hp * 1.1:
                return Squad(
                    units=selected_units,
                    total_damage=total_damage,
                    unit_indices=selected_indices
                )

        # Если набрали максимум юнитов, но урона недостаточно - не подходит
        # надо реализовать более гибкую эвристику. ввести порог или что-то типо того
        # если мы можем использовать транспорт для атаки цели, то его НУЖНО использовать
        if total_damage < target_hp:
            return None

        return Squad(
            units=selected_units,
            total_damage=total_damage,
            unit_indices=selected_indices
        )

    def _can_unit_attack_target(
            self,
            unit: Dict,
            target_pos: Tuple[int, int]
    ) -> bool:
        unit_pos = unit[POS_KEY]
        move_range = unit[MOVE_RANGE_KEY]
        attack_range = unit[ATTACK_RANGE_KEY]

        # Максимальная дистанция, на которую юнит может переместиться и атаковать
        max_reach = move_range + attack_range

        distance = self._hex_distance(unit_pos, target_pos)

        return distance <= max_reach

    def _hex_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:

        path = self.pf.find_path(pos1, pos2)
        if path:
            return self.grid.calculate_cost(path)
        return 9999

# точка выгрузки должна определяться по юниту с наименьшим кол-ом ОП и ДА
    def _find_unload_position(
            self,
            target: Dict,
            squad: Squad,
            units: List[Dict]
    ) -> Optional[Tuple[int, int]]:
        target_pos = target[POS_KEY]

        # Находим все возможные позиции атаки для каждого юнита
        unit_attack_positions = []
        for unit in squad.units:
            positions = self._find_attack_positions_for_unit(
                target_pos,
                unit[ATTACK_RANGE_KEY],
                unit[MOVE_RANGE_KEY],
            )
            unit_attack_positions.append(set(positions))

        # Находим пересечение - позиции, откуда ВСЕ юниты могут атаковать
        if not unit_attack_positions:
            return None

        common_positions = unit_attack_positions[0]
        for positions_set in unit_attack_positions[1:]:
            common_positions = common_positions.intersection(positions_set)

        if not common_positions:
            # Если нет общих позиций, берем позицию для юнита с наименьшей дальностью
            min_weapon_range = min(unit[ATTACK_RANGE_KEY] for unit in squad.units)
            min_move_range = min(unit[MOVE_RANGE_KEY] for unit in squad.units)
            min_attack_range = min(unit[ATTACK_RANGE_KEY] + unit[MOVE_RANGE_KEY] for unit in squad.units)
            min_range_positions = []
            for unit in squad.units:
                attack_range = unit[ATTACK_RANGE_KEY] + unit[MOVE_RANGE_KEY]
                if attack_range == min_attack_range:
                    min_range_positions = self._find_attack_positions_for_unit(
                        target_pos,
                        min_weapon_range,
                        min_move_range
                    )
                    break
            common_positions = min_range_positions

        if not common_positions:
            return None

        # Оцениваем каждую позицию
        best_pos = None
        best_score = -float('inf')

        for pos in common_positions:
            distance = self._hex_distance(pos, target_pos)
            score = self._evaluate_unload_position(pos, target_pos, distance)

            if score > best_score:
                best_score = score
                best_pos = pos

        return best_pos

    def _evaluate_unload_position(
            self,
            pos: Tuple[int, int],
            target_pos: Tuple[int, int],
            distance: int
    ) -> float:
        """
        Оценка качества позиции выгрузки
        """
        # Чем ближе к цели, тем лучше (но не вплотную)
        proximity_score = 10.0 / (distance + 1)

        # Безопасность (предпочитаем не самые близкие позиции)
        safety_score = min(distance, 3) * 2

        return proximity_score + safety_score

    def _plan_pickup_sequence(
            self,
            transport: Dict,
            squad: Squad,
            units: List[Dict]
    ) -> Optional[List[PickupPlan]]:
        transport_pos = transport[POS_KEY]
        transport_move = transport[MOVE_RANGE_KEY]

        pickup_plans = []
        reserved_positions = {transport_pos}

        for unit_idx in squad.unit_indices:
            unit = units[unit_idx]
            unit_pos = unit[POS_KEY]
            unit_move = unit[MOVE_RANGE_KEY]

            reserved_positions.add(unit_pos)

            t_point, u_point, t_cost, u_cost = self._find_meeting_point(
                transport_pos,
                transport_move,
                unit_pos,
                unit_move,
                reserved_positions
            )

            if not t_point or not u_point:
                return None

            reserved_positions.add(t_point)
            reserved_positions.add(u_point)

            pickup_plans.append(PickupPlan(
                unit_idx=unit_idx,
                unit_pos=unit_pos,
                meeting_point=t_point,  # Где будет транспорт
                unit_meeting_point=u_point,  # Где будет юнит (соседний гекс)
                transport_cost=t_cost,
                unit_cost=u_cost
            ))

        optimized_sequence = self._optimize_pickup_order(transport_pos, pickup_plans)

        return optimized_sequence

    def _find_adjacent_meeting_points(
            self,
            transport_pos: Tuple[int, int],
            transport_move: int,
            unit_pos: Tuple[int, int],
            unit_move: int,
            reserved_positions: Set[Tuple[int, int]]
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int], int, int]]:
        """
        Находит пары соседних гексов для встречи транспорта и юнита

        Returns:
            List[(transport_hex, unit_hex, transport_cost, unit_cost)]
        """
        candidates = []

        # Строим достижимые области для обоих
        transport_reachable = self._get_reachable_hexes(transport_pos, transport_move)
        unit_reachable = self._get_reachable_hexes(unit_pos, unit_move)

        # Для каждого достижимого гекса транспорта
        for t_hex, t_cost in transport_reachable.items():
            if t_hex in reserved_positions:
                continue

            # Проверяем его соседей
            neighbors = self.grid.get_neighbors(t_hex, include_self=False)

            for neighbor_hex, _ in neighbors:
                if self.grid.has_obstacle(neighbor_hex):
                    continue
                if neighbor_hex in reserved_positions:
                    continue

                # Проверяем, может ли юнит дойти до соседнего гекса
                if neighbor_hex in unit_reachable:
                    u_cost = unit_reachable[neighbor_hex]
                    candidates.append((t_hex, neighbor_hex, t_cost, u_cost))

        return candidates
# Переделать!!!
    def _get_reachable_hexes(
            self,
            start: Tuple[int, int],
            max_move: int
    ) -> Dict[Tuple[int, int], int]:
        """
        Находит все достижимые гексы в пределах max_move

        Returns:
            Dict[hex_position -> cost_to_reach]
        """
        reachable = {start: 0}
        queue = [(0, start)]
        visited = {start}

        while queue:
            cost, current = heapq.heappop(queue)

            if cost >= max_move:
                continue

            neighbors = self.grid.get_neighbors(current, include_self=False)

            for neighbor_pos, move_cost in neighbors:
                if self.grid.has_obstacle(neighbor_pos):
                    continue

                new_cost = cost + move_cost

                if new_cost > max_move:
                    continue

                if neighbor_pos not in visited or new_cost < reachable.get(neighbor_pos, float('inf')):
                    visited.add(neighbor_pos)
                    reachable[neighbor_pos] = new_cost
                    heapq.heappush(queue, (new_cost, neighbor_pos))

        return reachable


    def _find_meeting_point(
            self,
            transport_pos: Tuple[int, int],
            transport_move: int,
            unit_pos: Tuple[int, int],
            unit_move: int,
            reserved_positions: Set[Tuple[int, int]] = None
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]], int, int]:
        """
        Поиск оптимальной точки встречи транспорта и юнита для погрузки

        Погрузка происходит когда транспорт и юнит находятся на соседних гексах

        Args:
            reserved_positions: уже зарезервированные позиции другими юнитами

        Returns:
            (transport_meeting_point, unit_meeting_point, transport_cost, unit_cost)
            где transport_meeting_point и unit_meeting_point - соседние гексы
        """
        if reserved_positions is None:
            reserved_positions = set()

        best_option = None
        best_total_cost = float('inf')

        # Получаем соседей для текущей позиции юнита
        unit_neighbors = self.grid.get_neighbors(unit_pos, include_self=False)
        unit_neighbor_positions = [n[0] for n in unit_neighbors if not self.grid.has_obstacle(n[0])]

        # Вариант 0: Транспорт и юнит уже рядом
        if self.grid.adjacent(transport_pos, unit_pos):
            return transport_pos, unit_pos, 0, 0

        # Вариант 1: Транспорт подъезжает к юниту (юнит стоит на месте)
        for transport_target in unit_neighbor_positions:
            if transport_target in reserved_positions:
                continue

            path_t = self.pf.find_path(transport_pos, transport_target)
            if path_t:
                cost_t = len(path_t) - 1
                if cost_t <= transport_move:
                    total = cost_t
                    if total < best_total_cost:
                        best_option = (transport_target, unit_pos, cost_t, 0)
                        best_total_cost = total
        '''
        # Вариант 2: Юнит подходит к транспорту (транспорт стоит на месте)
        transport_neighbors = self.grid.get_neighbors(transport_pos, include_self=False)
        transport_neighbor_positions = [n[0] for n in transport_neighbors if not self.grid.has_obstacle(n[0])]

        for unit_target in transport_neighbor_positions:
            if unit_target in reserved_positions:
                continue

            path_u = self.pf.find_path(unit_pos, unit_target)
            if path_u:
                cost_u = len(path_u) - 1
                if cost_u <= unit_move:
                    total = cost_u
                    if total < best_total_cost:
                        best_option = (transport_pos, unit_target, 0, cost_u)
                        best_total_cost = total

        # Вариант 3: Оба движутся навстречу друг другу
        # Ищем пары соседних гексов, до которых могут дойти оба
        meeting_pairs = self._find_adjacent_meeting_points(
            transport_pos, transport_move,
            unit_pos, unit_move,
            reserved_positions
        )
        for transport_point, unit_point, cost_t, cost_u in meeting_pairs:
            total = cost_t + cost_u
            if total < best_total_cost:
                best_option = (transport_point, unit_point, cost_t, cost_u)
                best_total_cost = total
        '''
        if best_option:
            return best_option

        return None, None, 0, 0

    def _optimize_pickup_order(
            self,
            start_pos: Tuple[int, int],
            pickup_plans: List[PickupPlan]
    ) -> List[PickupPlan]:
        """
        Оптимизация порядка погрузки (жадный алгоритм ближайшего соседа)
        """
        if len(pickup_plans) <= 1:
            return pickup_plans

        optimized = []
        remaining = pickup_plans.copy()
        current_pos = start_pos

        while remaining:
            # Находим ближайшую точку погрузки
            nearest = min(
                remaining,
                key=lambda p: self._hex_distance(current_pos, p.meeting_point)
            )

            optimized.append(nearest)
            remaining.remove(nearest)
            current_pos = nearest.meeting_point

        return optimized

    def _build_transport_route(
            self,
            transport: Dict,
            pickup_sequence: List[PickupPlan],
            unload_point: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        route = [transport[POS_KEY]]
        current_pos = transport[POS_KEY]

        # Добавляем точки погрузки
        for pickup in pickup_sequence:
            # Транспорт движется к своей точке встречи
            if current_pos != pickup.meeting_point:
                path = self.pf.find_path(current_pos, pickup.meeting_point)
                if not path:
                    return None
                route.extend(path[1:])
                current_pos = pickup.meeting_point

        # Добавляем маршрут к точке выгрузки
        if current_pos != unload_point:
            path = self.pf.find_path(current_pos, unload_point)
            if not path:
                return None
            route.extend(path[1:])
        if (3,0) in route:
            print('!')
        return route

    def _calculate_efficiency(
            self,
            target: Dict,
            squad: Squad,
            route: List[Tuple[int, int]],
            pickup_sequence: List[PickupPlan]
    ) -> float:
        """
        Расчёт эффективности миссии
        """
        target_value = target[VALUE_KEY]

        # Стоимость транспортировки
        transport_cost = len(route) - 1

        # Стоимость движения юнитов
        unit_cost = sum(p.unit_cost for p in pickup_sequence)

        total_cost = transport_cost + unit_cost

        # Эффективность = ценность / стоимость
        efficiency = target_value / max(total_cost, 1)

        # Бонус за использование вместимости транспорта
        capacity_bonus = len(squad.units) * 0.1

        # Бонус за гарантированное уничтожение цели
        overkill_penalty = max(0, squad.total_damage - target[HP_KEY] * 1.5) * 0.05

        return efficiency + capacity_bonus - overkill_penalty

    def _coordinate_missions(
            self,
            missions: List[TransportMission],
            transports: List[Dict]
    ) -> List[TransportMission]:
        """
        Координация путей транспортов с использованием MAPF
        """
        if len(missions) <= 1:
            return missions

        # Собираем начальные и конечные позиции
        starts = []
        goals = []

        for mission in missions:
            starts.append(transports[mission.transport_idx][POS_KEY])
            goals.append(mission.unload_point)

        # Запускаем MAPF
        try:
            # Определяем максимальную длину пути
            max_length = max(len(m.full_route) for m in missions) + 10

            coordinated_paths = self.mapf.mapf(
                starts,
                goals,
                reservation_table=self.rt,
                max_length=max_length
            )

            # Обновляем маршруты в миссиях
            for i, mission in enumerate(missions):
                if coordinated_paths and i < len(coordinated_paths):
                    mission.full_route = coordinated_paths[i]

        except Exception as e:
            # Если MAPF не справился, используем приоритизацию
            print(f"MAPF failed: {e}, using prioritized planning")
            missions.sort(key=lambda m: m.efficiency, reverse=True)

        return missions

def insert_after(lst, target, new_value):
    try:
        index = lst.index(target)
        lst.insert(index + 1, new_value)
    except ValueError:
        print("!")

# Пример использования
def example_usage():
    weights = [
        # 0   1   2   3   4   5   6   7   8   9   10
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ],  # 0
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ],  # 1
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ],  # 2
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ],  # 3
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ],  # 4
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ],  # 5
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ],  # 6
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ],  # 7
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ],  # 8
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ],  # 9
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ],  # 10
    ]

    grid = HexGrid(weights=weights, edge_collision=True, layout=HexLayout.odd_q)
    rt = ReservationTable(grid)
    pf = AStar(grid)
    mapf = CBS(grid)
    planner = TransportPlanner(grid, mapf, pf, rt)

    # m - move,
    # w - weapon
    # d - damage,
    # r - weapon range,
    # h - hp,
    # v - value,
    # c - capacity (transport)
    unit_data = {
        #                                m   d  r  h  v  c
        UnitType.TANK:                  ( 2 ,  2,  1,  5,  2,  0 ),
        UnitType.LAND_TRANSPORT:        ( 15,  0,  0,  3,  1,  2 ),
        UnitType.ABSTRACT_TARGET:       ( 0 ,  0,  0,  3,  9,  0 ),
        UnitType.LAV:                   ( 1 ,  1,  1,  4,  1,  0 ),
    }

    def make_unit(u_id, pos, unit_type: UnitType):
        unit = {
            ID_KEY: u_id,
            POS_KEY: pos,
            MOVE_RANGE_KEY: unit_data[unit_type]    [0],
            DAMAGE_KEY: unit_data[unit_type]        [1],
            ATTACK_RANGE_KEY: unit_data[unit_type]  [2],
            HP_KEY: unit_data[unit_type]            [3],
            VALUE_KEY: unit_data[unit_type]         [4],
            CAPACITY_KEY: unit_data[unit_type]      [5],
        }
        return unit

    transports = [
        # transport #1 for group 1
        make_unit(999, (0, 0), UnitType.LAND_TRANSPORT),
        # transport #2 for group 2
        make_unit(888, (0, 6), UnitType.LAND_TRANSPORT),
    ]

    units = [
        # group 1
        make_unit(0, (1, 0), UnitType.TANK),
        make_unit(1, (3, 1), UnitType.TANK),
        # group 2
        make_unit(2, (0, 5), UnitType.TANK),
        make_unit(3, (3, 6), UnitType.TANK),

        make_unit(4, (0, 1), UnitType.LAV),
    ]

    targets = [
        # target #1 for group 1
        make_unit(100, (6, 1), UnitType.ABSTRACT_TARGET),
        # target #2 for group 2
        make_unit(101, (6, 5), UnitType.ABSTRACT_TARGET),
    ]

    missions = planner.plan_transport_operations(units, transports, targets)
    # визуализация. синхронизация путей. не на продакшен
    units_paths = dict()
    for mission in missions:
        insertions = dict()
        units_2_load = dict()
        for step in mission.full_route:
            wait_steps = 0
            for ps in mission.pickup_sequence:
                if ps.meeting_point == step:
                    if step not in units_2_load:
                        units_2_load[step] = list()
                    units_2_load[step].append(ps.unit_idx)
                    wait_steps += 1
            for s in range(wait_steps):
                insertions[step] = wait_steps

        transport_route = list(mission.full_route)
        for pos, waiting in insertions.items():
            for i in range(waiting):
                insert_after(transport_route, pos, pos)
            i = 0
            head_len = len(transport_route[0: transport_route.index(pos)])
            while waiting > 0:
                unit_idx = units_2_load[pos][i]
                unit_id = units[unit_idx][ID_KEY]

                if unit_id not in units_paths:
                    units_paths[unit_id] = list()

                while head_len > 0:
                    units_paths[unit_id].append(units[unit_idx][POS_KEY])
                    head_len -= 1

                for j in range(waiting):
                    units_paths[unit_id].append(units[unit_idx][POS_KEY])
                i += 1
                waiting -= 1
                units_paths[unit_id].append(pos)

        transport_id = transports[mission.transport_idx][ID_KEY]
        units_paths[transport_id] = transport_route
        # движение юнитов в транспортов
        for u_id, path in units_paths.items():
            if u_id == transport_id:
                continue

            result = next((item for item in mission.squad.units if item[ID_KEY] == u_id), None)
            if result:
                path_tail = transport_route[len(path):]
                path = path + path_tail
                units_paths[u_id] = path

        print(f"Transport {mission.transport_idx}:")
        print(f"  Target: {mission.target[POS_KEY]}")
        print(f"  Squad: {len(mission.squad.units)} units")
        print(f"  Route length: {len(mission.full_route)}")
        print(f"  Efficiency: {mission.efficiency:.2f}")
        print(f"  Pickup sequence: {len(mission.pickup_sequence)} pickups")

    visualizer = HexVisualizer(grid)

    solution = {
        'assignments': [],
        'paths': units_paths
    }
    for transport in transports:
        units.append(transport)

    while True:
        restart = visualizer.animate_solution(solution, units, targets)
        if restart:
            for target in targets:
                target['current_hp'] = target['hp']
            continue
        else:
            break

    pygame.quit()

if __name__ == '__main__':
    example_usage()