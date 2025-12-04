from itertools import combinations
from itertools import permutations
from typing import Dict, List, Tuple, Optional

import pygame

from AI_BoT.common.constants import UnitType, ID_KEY, DAMAGE_KEY, POS_KEY, MAX_ATTACK_RANGE_KEY, MOVE_RANGE_KEY, \
    MIN_ATTACK_RANGE_KEY
from AI_BoT.common.units_loader import MultiUnitsLoader, UnitsStorage, units_data
from AI_BoT.transport_mission_solver import solve_transport_mission
from AI_BoT.transport_plan import TransportPlan
from AI_BoT.transport_plan_optimization import TransportPlanOptimizer
from common.helpers import find_unload_positions, insert_after, find_attack_positions_for_unit, \
    generate_transport_loads, get_units_could_unload
#from common.constants import *
from visualizer import HexVisualizer
from w9_pathfinding.envs import HexGrid, HexLayout
from w9_pathfinding.pf import IDAStar, AStar
from w9_pathfinding.mapf import CBS, SpaceTimeAStar, ReservationTable, MultiAgentAStar

UTILITY_THRESHOLD = 1

scenario_path = "AI_BoT/scenarios/lt+rl+am.json"
scenario_path = "AI_BoT/scenarios/lt+am.json"
scenario_path = "AI_BoT/scenarios/1.json"

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


    def is_unit_can_be_unloaded(u: dict) -> bool:
        return u[MOVE_RANGE_KEY] - 1 > 0
    # в кластеры включаем только тех юнитов которые могут выгрузиться и атаковать в один ход
    units_clusters = my_units_storage.get_clusters(grid, is_unit_can_be_unloaded)

    # емкость транспорта
    lt_data = units_data[UnitType.LAND_TRANSPORT]
    transport_capacity = lt_data[5]
    # очки передвижения транспорта
    transport_mp = lt_data[0]

    transport_plans: List[TransportPlan] = []
    # просчитываем планы атаки целей боевыми юнитами без использования транспортов
    for unit in my_units_storage.get_units():
        # todo: remove after tests
        continue
        unit_pos = unit[POS_KEY]
        for target in en_units_storage.get_units():
            # move_range == 0 т.к. мы пришли в точку атаки на своих двоих
            positions_2_attack = find_attack_positions_for_unit(
                grid=grid,
                target_pos=target[POS_KEY],
                move_range=0,
                max_weapon_range=unit[MAX_ATTACK_RANGE_KEY],
                min_weapon_range=unit[MIN_ATTACK_RANGE_KEY]
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
                    # точка выгрузки совпадает с точкой атаки - т.к. боевой юнит и есть транспорт
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
    # просчитываем планы атаки целей боевыми юнитами с использованием транспортов
    transport_loads = generate_transport_loads(units_clusters, transport_capacity)
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

    sorted_solution = sorted(transport_plans, key=lambda p: p.utility, reverse=True)
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

# region   визуализация. Лютый говнокод
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
