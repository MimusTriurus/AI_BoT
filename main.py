from itertools import combinations
from itertools import permutations
from typing import Dict, List, Tuple, Optional

import pygame

from AI_BoT.common.constants import UnitType, ID_KEY, DAMAGE_KEY, POS_KEY, ATTACK_RANGE_KEY, MOVE_RANGE_KEY
from AI_BoT.common.units_loader import MultiUnitsLoader, UnitsStorage, units_data
from AI_BoT.transport_plan import TransportPlan
from AI_BoT.transport_plan_optimization import TransportPlanOptimizer
from common.helpers import find_unload_positions, insert_after, find_attack_positions_for_unit
from data_structures import *
from visualizer import HexVisualizer
from w9_pathfinding.envs import HexGrid, HexLayout
from w9_pathfinding.pf import IDAStar, AStar
from w9_pathfinding.mapf import CBS, SpaceTimeAStar, ReservationTable, MultiAgentAStar

UTILITY_THRESHOLD = 5


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
            if not grid.has_obstacle(p) and p != transport_pos:
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
                    if transport_pos == target_unit:
                        full_path_hexes.append(transport_pos)
                        continue
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
                        is_sequence_valid = False
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



scenario_path = "AI_BoT/scenarios/3.json"

if __name__ == '__main__':
    map_size = 22
    map_data = [[1] * map_size] * map_size

    grid = HexGrid(weights=map_data, edge_collision=True, layout=HexLayout.odd_q)
    pf = AStar(grid)

    my_units_storage = UnitsStorage()
    en_units_storage = UnitsStorage()
    transport_storage = UnitsStorage()

    storages = {
        "en_units_storage": en_units_storage,
        "my_units_storage": my_units_storage,
        "transports": transport_storage,
    }

    units_loader = MultiUnitsLoader(storages)
    units_loader.load_from_json(scenario_path)

    units_clusters = my_units_storage.get_clusters(grid)

    # емкость транспорта
    transport_capacity = units_data[UnitType.LAND_TRANSPORT][5]
    # очки передвижения транспорта
    transport_mp = units_data[UnitType.LAND_TRANSPORT][0]

    transport_loads = generate_transport_loads(units_clusters, transport_capacity)

    transport_plans: List[TransportPlan] = []

    for unit in my_units_storage.get_units():
        unit_pos = unit[POS_KEY]
        for target in en_units_storage.get_units():
            positions_2_attack = find_attack_positions_for_unit(
                grid,
                target[POS_KEY],
                unit[ATTACK_RANGE_KEY],
                0
            )
            for position_2_attack in positions_2_attack:
                best_order, cost, path = solve_transport_mission(
                    transport_pos=unit_pos,
                    transport_mp=unit[MOVE_RANGE_KEY],
                    passengers=[unit_pos],
                    drop_zone=position_2_attack,
                    enemy_positions=en_units_storage.get_units_pos(),
                    grid=grid,
                    pf=pf
                )
                if best_order:
                    unload_map = {
                        unit[ID_KEY]: position_2_attack
                    }

                    tp = TransportPlan(
                        transport=unit,
                        target=target,
                        passengers=[unit],
                        path=path,
                        unload_map=unload_map,
                        grid=grid,
                        pf=pf
                    )
                    if tp.utility > 0:
                        transport_plans.append(tp)

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
                            drop_zone = path[-1]
                            passengers = my_units_storage.get_units(best_order)
                            unload_map, total_damage = TransportPlan.solve_best_unload_configuration(
                                drop_zone=position_2_unload,
                                target_pos=target[POS_KEY],
                                passengers=passengers,
                                enemy_positions=en_units_storage.get_units_pos(),
                                other_obstacles=[],
                                grid=grid,
                                pf=pf
                            )
                            tp = TransportPlan(
                                transport=transport_storage.get_unit(transport_pos),
                                target=target,
                                passengers=passengers,
                                path=path,
                                unload_map=unload_map,
                                grid=grid,
                                pf=pf
                            )
                            if tp.utility > UTILITY_THRESHOLD:
                                transport_plans.append(tp)
    optimizer = TransportPlanOptimizer(transport_plans)

    # actual_plans, total_utility = optimizer.optimize(method='auction')
    # auction + local search
    actual_plans, total_utility = optimizer.optimize_hybrid()
    actual_plans, total_utility = optimizer.optimize_branch_and_bound(actual_plans, total_utility)

    print(f'\n=================')
    for plan in actual_plans:
        print(str(plan))
        print(f'-------')
    #print(f'=== Total utility: {total_utility} ===')

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
                path_2_target = [plan.unload_map[u_id]]
                path = path + path_tail + path_2_target
                units_paths[u_id] = path

    solution = {
        'assignments': [],
        #'paths': dict()
        'paths':  units_paths
    }

    assignment = {}
    for plan in actual_plans:
        target = plan.target
        t_id = target[ID_KEY]
        t_idx = next(i for i, obj in enumerate(en_units_storage.get_units()) if obj[ID_KEY] == t_id)

        for passenger in plan.passengers:
            u_id = passenger[ID_KEY]
            u_idx = next(i for i, obj in enumerate(my_units_storage.get_units()) if obj[ID_KEY] == u_id)
            solution['assignments'].append({
                "unit_idx": u_id,
                "target_idx": t_idx,
                "damage": passenger[DAMAGE_KEY]
            })

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
