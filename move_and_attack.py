from typing import Dict, List, Tuple, Optional
from AI_BoT.clustering import complex_clustering
from AI_BoT.common.reachable_positions_calculator import PathInfo
from AI_BoT.data_structures import Unit
from AI_BoT.game_map import grid, pf_
from transport_mission_solver import solve_transport_mission, solve_move_mission
from transport_plan import TacticPlan
from common.helpers import find_unload_positions, find_attack_positions_for_unit, generate_transport_loads, get_units_by_positions
from w9_pathfinding.envs import HexGrid
from w9_pathfinding.pf import IDAStar, AStar
from w9_pathfinding.mapf import CBS, SpaceTimeAStar, ReservationTable, MultiAgentAStar

def calculate_move_and_attack_plans(
        my_units: List[Unit],
        enemy_units: List[Unit],
        reserved_positions: List[Tuple[int, int]],
        utility_threshold: float = 0.0,
        is_future: bool = False,
):
    plans: List[TacticPlan] = []

    for unit in my_units:
        for target in enemy_units:
            # move_range == 0 т.к. мы пришли в точку атаки на своих двоих
            positions_4_attack = find_attack_positions_for_unit(
                grid=grid,
                target_pos=target.pos,
                move_range=0,
                max_weapon_range=unit.wr[1],
                min_weapon_range=unit.wr[0]
            )
            for position_4_attack in positions_4_attack:
                path_info = solve_move_mission(
                    unit=unit,
                    pos_2_attack=position_4_attack,
                    enemies=enemy_units,
                    other_obstacles=reserved_positions,
                    grid=grid,
                    is_future=is_future
                )
                if path_info:
                    # точка выгрузки совпадает с точкой атаки - т.к. боевой юнит и есть транспорт
                    unload_map = {
                        unit.id: position_4_attack
                    }
                    tp = TacticPlan(
                        transport=unit,
                        target=target,
                        passengers=[unit],
                        path_info=path_info,
                        unload_map=unload_map,
                        grid=grid,
                        pf=pf_
                    )
                    if tp.utility > utility_threshold:
                        if is_future:
                            tp.unload_map = {}
                        plans.append(tp)
    return plans

def calculate_move_and_attack_transport_plans(
        my_units: Dict[Tuple[int, int], Unit],
        transport_units: List[Unit],
        enemy_units: List[Unit],
        reserved_positions: List[Tuple[int, int]],
        utility_threshold: float = 0.0,
        transport_mp_increaser: int = 1
):
    # region clustering
    def is_unit_can_be_unloaded(u: Unit) -> bool:
        return u.mp - 1 > 0

    predicate_4_clusterization = is_unit_can_be_unloaded if transport_mp_increaser == 1 else None
    # в кластеры включаем только тех юнитов которые могут выгрузиться и атаковать в один ход
    units_pos_clusters = complex_clustering(
        list(my_units.values()),
        grid,
        predicate_4_clusterization
    )
    # todo: КОСТЫЛЬ на время - удаляем кластер со всеми юнитами
    units_pos_clusters = units_pos_clusters[:-1]
    # endregion

    transport_plans = []

    for transport in transport_units:
        transport_pos = transport.pos
        transport_capacity = transport.capacity
        transport_loads = generate_transport_loads(units_pos_clusters, transport_capacity)
        transport_mp = transport.mp
        for target in enemy_units:
            for transport_load in transport_loads:
                for units_pos in transport_load:
                    units_4_transport = get_units_by_positions(my_units, units_pos)
                    positions_2_unload = find_unload_positions(grid, target, units_4_transport)
                    for position_2_unload in positions_2_unload:
                        best_order, cost, path = solve_transport_mission(
                            transport_pos=transport_pos,
                            transport_mp=transport_mp,
                            passengers=units_pos,
                            drop_zone=position_2_unload,
                            enemies=enemy_units,                    # для блокировки\резервирования гексов
                            reserved_positions=reserved_positions,  # для блокировки\резервирования гексов
                            grid=grid,
                            pf=pf_,
                            transport_mp_increaser=transport_mp_increaser
                        )
                        if best_order:
                            drop_zone = path[-1]
                            actual_path = path[:transport.mp + 1]
                            path_info = PathInfo(cost, actual_path)
                            passengers = get_units_by_positions(my_units, list(best_order))
                            unload_map, total_damage = TacticPlan.solve_best_unload_configuration(
                                drop_zone=drop_zone,
                                target_pos=target.pos,
                                passengers=passengers,
                                enemies=enemy_units,  # для блокировки гексов
                                grid=grid,
                                pf=pf_
                            )
                            tp = TacticPlan(
                                transport=transport,
                                target=target,
                                passengers=passengers,
                                path_info=path_info,
                                unload_map=unload_map,
                                grid=grid,
                                pf=pf_
                            )
                            if tp.utility > utility_threshold:
                                # мы планировали план транспортировки юнитов на более чем один ход, поэтому очищаем план выгрузки
                                # тех, кто мог атаковать цель на одном ходу мы уже обработали
                                if transport_mp_increaser > 1:
                                    tp.unload_map = {}
                                transport_plans.append(tp)

    return transport_plans