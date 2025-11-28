from itertools import combinations, chain
from typing import Dict, Tuple, List, Set, Optional, Any
from itertools import permutations

import pygame

from AI_BoT.common.helpers import find_unload_positions, insert_after
from AI_BoT.data_structures import *
from AI_BoT.clustering import *
from AI_BoT.transport_plan_optimization import TransportPlanOptimizer
from AI_BoT.visualizer import HexVisualizer
from w9_pathfinding.envs import HexGrid, HexLayout
from w9_pathfinding.pf import IDAStar, AStar
from w9_pathfinding.mapf import CBS, SpaceTimeAStar, ReservationTable, MultiAgentAStar

from best_transport_plans import *


def generate_transport_loads(cluster_units: Dict[int, List[Tuple[int, int]]], transport_capacity: int):
    """
    cluster_units: dict {cluster_id: [(x1,y1), (x2,y2), ...]}
    T: вместимость транспорта (максимальное число юнитов)

    Возвращает:
        dict {cluster_id: [ [u1,u2], [u1], ... ] } - все возможные пакеты
    """
    all_loads = {}

    for cid, units in cluster_units.items():
        # генерируем все комбинации размера 1..T
        loads = []
        for r in range(1, min(transport_capacity, len(units)) + 1):
            loads.extend(combinations(units, r))

        # преобразуем из tuple в list
        all_loads[cid] = sorted([list(l) for l in loads], key=lambda x: -len(x))

    return all_loads


def filter_transport_options(transport_pos, unit_packages, targets, transport_mp, pf):
    valid_targets = {}

    # --- 1. Фильтруем недоступные цели ---
    reachable_targets = []
    for t in targets:
        path = pf.find_path(transport_pos, t)
        if path:  # путь найден
            reachable_targets.append(t)
        # иначе цель недостижима, пропускаем

    # --- 2. Фильтруем пакеты по каждой цели ---
    for t in reachable_targets:
        valid_packages = []

        for package in unit_packages:
            # вычисляем минимальный маршрут сбора всех юнитов и доставки
            current = transport_pos
            remaining = set(package)
            cost = 0

            # жадный алгоритм: всегда идём к ближайшему юниту
            while remaining:
                next_u = min(remaining, key=lambda u: len(pf.find_path(current, u)))
                path_len = len(pf.find_path(current, next_u))
                if path_len == 999999:
                    cost = float('inf')  # недостижимый юнит
                    break
                cost += path_len
                current = next_u
                remaining.remove(next_u)

            if cost == float('inf'):
                continue  # пакет недостижим

            # доставка к цели
            delivery_cost = len(pf.find_path(current, t))
            if delivery_cost == 999999:
                continue  # цель недостижима
            cost += delivery_cost

            # проверка по MP транспорта
            if cost <= transport_mp:
                valid_packages.append(package)

        if valid_packages:
            valid_targets[t] = valid_packages

    return valid_targets


def evaluate_transport_loads(loads, transport_start, delivery_pos, pf):
    """
    loads: list пакетов юнитов, где юнит = (x,y)
    transport_start: стартовая позиция транспорта (x,y)
    delivery_pos: позиция доставки (x,y)
    pf: AStar pathfinder

    Возвращает:
        dict {tuple(unit_positions): cost}
    """
    load_costs = {}

    for package in loads:
        if not package:
            continue

        min_cost = float('inf')

        # перебираем все порядки забора юнитов
        for order in permutations(package):
            cost = 0
            current = transport_start

            # путь до каждого юнита
            for u in order:
                cost += len(pf.find_path(current, u))
                current = u

            # доставка к цели
            cost += len(pf.find_path(current, delivery_pos))

            if cost < min_cost:
                min_cost = cost

        load_costs[tuple(package)] = min_cost

    return load_costs


def solve_transport_mission(
        transport_pos: Tuple[int, int],
        transport_mp: int,
        passengers: List[Tuple[int, int]],
        drop_zone: Tuple[int, int],
        enemy_positions: List[Tuple[int, int]],  # <--- НОВЫЙ АРГУМЕНТ: Все враги на карте
        grid: HexGrid,  # HexGrid
        pf: AStar
) -> tuple[Optional[tuple], float, list]:
    # Список для глобальной очистки (чтобы вернуть сетку в исходное состояние)
    mission_obstacles = []

    try:
        # --- ЭТАП SETUP: Превращаем юнитов в препятствия ---

        # 1. Добавляем ПАССАЖИРОВ (своих)
        for p in passengers:
            if not grid.has_obstacle(p):
                grid.add_obstacle(p)
                mission_obstacles.append(p)

        # 2. Добавляем ВСЕХ ВРАГОВ
        # Враги - это статичные препятствия для движения транспорта.
        for enemy in enemy_positions:
            # Защита: Drop Zone не должна быть занята врагом (иначе туда не приехать).
            # Но если drop_zone это соседняя клетка, то все ок.
            if enemy == drop_zone:
                # Если цель высадки занята врагом, транспорт туда физически не встанет.
                # A* вернет None, если destination занят.
                # Но мы все равно помечаем, так как стоять на враге нельзя.
                pass

            if not grid.has_obstacle(enemy):
                grid.add_obstacle(enemy)
                mission_obstacles.append(enemy)

        # --- Переменные поиска ---
        best_sequence = None
        min_mp_cost = float('inf')
        best_full_path = []

        # 3. ПЕРЕБОР (TSP)
        for sequence in permutations(passengers):

            current_pos = transport_pos
            current_accumulated_cost = 0
            full_path_hexes = []
            is_sequence_valid = True

            # Локальные изменения внутри одной ветки (только свои пассажиры)
            sequence_removed_obstacles = []

            try:
                # --- Этап А: Сбор пассажиров ---
                for target_unit in sequence:
                    # Временно "подбираем" пассажира -> клетка освобождается
                    grid.remove_obstacle(target_unit)
                    sequence_removed_obstacles.append(target_unit)

                    if current_accumulated_cost >= transport_mp:
                        is_sequence_valid = False
                        break

                    neighbors = grid.get_neighbors(target_unit, include_self=False)
                    best_leg_path = None
                    best_leg_cost = float('inf')
                    found_entry_point = False

                    for neighbor_hex, _ in neighbors:
                        # has_obstacle теперь вернет True для:
                        # - Гор/Стен
                        # - Других пассажиров
                        # - ВСЕХ ВРАГОВ (мы их добавили выше)
                        if grid.has_obstacle(neighbor_hex): continue

                        path = pf.find_path(current_pos, neighbor_hex)

                        if path:
                            cost = grid.calculate_cost(path) if isinstance(path, list) else getattr(path, 'cost',
                                                                                                    len(path))

                            if current_accumulated_cost + cost > transport_mp: continue

                            if cost < best_leg_cost:
                                best_leg_cost = cost
                                best_leg_path = path
                                found_entry_point = True

                    if not found_entry_point:
                        is_sequence_valid = False;
                        break

                    current_accumulated_cost += best_leg_cost
                    path_nodes = best_leg_path if isinstance(best_leg_path, list) else getattr(best_leg_path, 'nodes',
                                                                                               [])

                    if full_path_hexes:
                        full_path_hexes.extend(path_nodes[1:])
                    else:
                        full_path_hexes.extend(path_nodes)

                    if path_nodes:
                        current_pos = path_nodes[-1]

                # --- Этап Б: Доставка к цели ---
                if is_sequence_valid:
                    # Путь к drop_zone.
                    # Благодаря setup-фазе, A* будет огибать всех врагов.
                    path_drop = pf.find_path(current_pos, drop_zone)

                    if path_drop:
                        cost_drop = grid.calculate_cost(path_drop) if isinstance(path_drop, list) else getattr(
                            path_drop, 'cost', 0)
                        total_cost = current_accumulated_cost + cost_drop

                        if total_cost <= transport_mp:
                            if total_cost < min_mp_cost:
                                min_mp_cost = total_cost
                                best_sequence = sequence
                                final_nodes = path_drop if isinstance(path_drop, list) else getattr(path_drop, 'nodes',
                                                                                                    [])

                                if final_nodes:
                                    best_full_path = list(full_path_hexes) + final_nodes[1:]
                                else:
                                    best_full_path = list(full_path_hexes)

            finally:
                # Восстанавливаем пассажиров текущей ветки
                for removed_unit in sequence_removed_obstacles:
                    if current_pos != removed_unit:
                        grid.add_obstacle(removed_unit)

        if best_sequence is None:
            return None, 0.0, []

        return best_sequence, min_mp_cost, best_full_path

    finally:
        # 4. ГЛОБАЛЬНАЯ ОЧИСТКА
        # Убираем пассажиров И всех врагов из списка препятствий сетки
        for obs in mission_obstacles:
            if grid.has_obstacle(obs):
                grid.remove_obstacle(obs)

if __name__ == '__main__':
    map_size = 22
    map_data = [[1] * map_size] * map_size

    grid = HexGrid(weights=map_data, edge_collision=True, layout=HexLayout.odd_q)
    pf = AStar(grid)
    # === MY UNITS ===
    my_units_storage = UnitsStorage()
    my_units_storage.add_unit('T_1', (0, 0), UnitType.TANK)
    my_units_storage.add_unit('T_2', (1, 0), UnitType.TANK)

    my_units_storage.add_unit('T_3', (1, 2), UnitType.TANK)
    my_units_storage.add_unit('T_4', (1, 6), UnitType.TANK)

    my_units_storage.add_unit('T_5', (6, 6), UnitType.TANK)
    my_units_storage.add_unit('T_6', (5, 6), UnitType.TANK)

    units_clusters = my_units_storage.get_clusters(grid)

    # === ENEMY UNITS ===
    en_units_storage = UnitsStorage()
    en_units_storage.add_unit('#1', (5, 1), UnitType.ABSTRACT_TARGET)
    en_units_storage.add_unit('#2', (5, 4), UnitType.ABSTRACT_TARGET)
    en_units_storage.add_unit('#3', (10, 9), UnitType.ABSTRACT_TARGET)

    # === TRANSPORTS ===
    transport_storage = UnitsStorage()
    transport_storage.add_unit('LT_1', (1, 1), UnitType.LAND_TRANSPORT)
    transport_storage.add_unit('LT_2', (1, 3), UnitType.LAND_TRANSPORT)
    transport_storage.add_unit('LT_3', (6, 9), UnitType.LAND_TRANSPORT)

    # емкость транспорта
    transport_capacity = transport_storage.get_units()[0][CAPACITY_KEY]
    # очки передвижения транспорта
    transport_mp = transport_storage.get_units()[0][MOVE_RANGE_KEY]

    transport_loads = generate_transport_loads(units_clusters, transport_capacity)

    transport_plans: List[TransportPlan] = []

    for transport_pos in transport_storage.get_units_pos():
        for target in en_units_storage.get_units():
            for idx, transport_load in transport_loads.items():
                for units_pos in transport_load:
                    units_4_transport = my_units_storage.get_units(units_pos)
                    positions_2_unload = find_unload_positions(grid, target, units_4_transport)
                    for position_2_unload in positions_2_unload:
                        best_order, cost, path = solve_transport_mission(
                            transport_pos=transport_pos,
                            transport_mp=transport_mp,
                            passengers=units_pos,
                            drop_zone=position_2_unload,
                            enemy_positions=en_units_storage.get_units_pos(),
                            grid=grid,
                            pf=pf
                        )
                        if best_order:
                            tp = TransportPlan(
                                transport=transport_storage.get_unit(transport_pos),
                                target=target,
                                passengers=my_units_storage.get_units(best_order),
                                path=path,
                                grid=grid,
                                pf=pf
                            )
                            if tp.utility > 20:
                                transport_plans.append(tp)

    optimizer = TransportPlanOptimizer(transport_plans)
    actual_plans, total_utility = optimizer.optimize(method='branch_and_bound', n_workers=4)
    #actual_plans, total_utility = optimizer.optimize(method='hybrid', n_workers=1)
    #actual_plans, total_utility = optimizer.optimize(method='auction')

    print(f'\n=================')
    for plan in actual_plans:
        print(str(plan))
    print(f'=== Total utility: {total_utility} ===')

# region   визуализация. Говнокод
    visualizer = HexVisualizer(grid)
    units_paths = dict()
    for plan in actual_plans:
        insertions = dict()
        units_2_load = dict()

        t_id, t_full_route = plan.to_path()
        plan.calculate_meeting_points()

        for step in t_full_route:
            wait_steps = 0
            for u_id, meeting_point in plan.meeting_points.items():
                if meeting_point == step:
                    if step not in units_2_load:
                        units_2_load[step] = list()
                    units_2_load[step].append(u_id)
                    wait_steps += 1
            for s in range(wait_steps):
                insertions[step] = wait_steps

        transport_route = list(t_full_route)
        for pos, waiting in insertions.items():
            for i in range(waiting):
                insert_after(transport_route, pos, pos)
            i = 0
            end_of_head_idx = transport_route.index(pos)
            head_len = len(transport_route[0: end_of_head_idx])
            while waiting > 0:
                passengers_id = units_2_load[pos]
                unit_id = units_2_load[pos][i]
                passenger = my_units_storage.get_unit_by_id(unit_id)

                if unit_id not in units_paths:
                    units_paths[unit_id] = list()

                for k in range(head_len):
                    units_paths[unit_id].append(passenger[POS_KEY])

                for j in range(waiting):
                    units_paths[unit_id].append(passenger[POS_KEY])
                i += 1
                waiting -= 1
                units_paths[unit_id].append(pos)

        units_paths[t_id] = transport_route
        # движение юнитов в транспортов
        for u_id, path in units_paths.items():
            if u_id == t_id:
                continue

            result = next((item for item in plan.passengers if item[ID_KEY] == u_id), None)
            if result:
                path_tail = transport_route[len(path):]
                path = path + path_tail
                units_paths[u_id] = path

    solution = {
        'assignments': [],
        'paths': dict()#units_paths
    }

    units = list(my_units_storage.get_units())
    units.extend(transport_storage.get_units())

    targets = en_units_storage.get_units()

    while True:
        restart = visualizer.animate_solution(solution, units, targets)
        if restart:
            for target in targets:
                target['current_hp'] = target['hp']
            continue
        else:
            break

    pygame.quit()
# endregion
    print('End!')
