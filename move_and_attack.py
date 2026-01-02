from typing import Dict, List, Tuple, Optional
from AI_BoT.clustering import complex_clustering
from AI_BoT.game_map import grid, pf
from common.constants import ID_KEY, POS_KEY, MAX_ATTACK_RANGE_KEY, MOVE_RANGE_KEY, MIN_ATTACK_RANGE_KEY, CAPACITY_KEY
from transport_mission_solver import solve_transport_mission, solve_move_mission
from transport_plan import TransportPlan
from common.helpers import find_unload_positions, find_attack_positions_for_unit, generate_transport_loads, get_units_by_positions
from w9_pathfinding.envs import HexGrid
from w9_pathfinding.pf import IDAStar, AStar
from w9_pathfinding.mapf import CBS, SpaceTimeAStar, ReservationTable, MultiAgentAStar

def calculate_move_and_attack_plans(
        combat_units: list,
        enemy_units: list,
        utility_threshold: float = 0.0,
        mp_increaser: int = 1,
        other_obstacles: List[Tuple[int, int]] = []
):
    plans: List[TransportPlan] = []

    for unit in combat_units:
        # todo: remove after tests
        #continue
        unit_pos = unit[POS_KEY]
        for target in enemy_units:
            # move_range == 0 т.к. мы пришли в точку атаки на своих двоих
            positions_4_attack = find_attack_positions_for_unit(
                grid=grid,
                target_pos=target[POS_KEY],
                move_range=0,
                max_weapon_range=unit[MAX_ATTACK_RANGE_KEY],
                min_weapon_range=unit[MIN_ATTACK_RANGE_KEY]
            )
            for position_4_attack in positions_4_attack:
                path = solve_move_mission(
                    unit_pos,
                    unit[MOVE_RANGE_KEY] * mp_increaser,
                    position_4_attack,
                    enemy_units,
                    other_obstacles,
                    grid,
                    pf
                )
                if path:
                    actual_path = path[:unit[MOVE_RANGE_KEY] + 1]
                    # точка выгрузки совпадает с точкой атаки - т.к. боевой юнит и есть транспорт
                    unload_map = {
                        unit[ID_KEY]: position_4_attack
                    }
                    tp = TransportPlan(
                        transport=unit,
                        target=target,
                        passengers=[unit],
                        path=actual_path,
                        unload_map=unload_map,
                        grid=grid,
                        pf=pf
                    )
                    if tp.utility > utility_threshold:
                        if mp_increaser > 1:
                            tp.unload_map = {}
                        plans.append(tp)
    return plans

def calculate_move_and_attack_transport_plans(
        my_units: Dict[Tuple[int, int], dict],
        transport_units: list,
        enemy_units: list,
        utility_threshold: float = 0.0,
        transport_mp_increaser: int = 1,
        other_obstacles: List[Tuple[int, int]] = []
):
    # region clustering
    def is_unit_can_be_unloaded(u: dict) -> bool:
        return u[MOVE_RANGE_KEY] - 1 > 0

    # в кластеры включаем только тех юнитов которые могут выгрузиться и атаковать в один ход
    units_pos_clusters = complex_clustering(
        list(my_units.values()),
        grid,
        is_unit_can_be_unloaded
    )
    # todo: КОСТЫЛЬ на время - удаляем кластер со всеми юнитами
    units_pos_clusters = units_pos_clusters[:-1]
    # endregion

    transport_plans = []

    for transport in transport_units:
        transport_pos = transport[POS_KEY]
        transport_capacity = transport[CAPACITY_KEY]
        transport_loads = generate_transport_loads(units_pos_clusters, transport_capacity)
        transport_mp = transport[MOVE_RANGE_KEY]
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
                            enemies=enemy_units, # для блокировки гексов
                            grid=grid,
                            pf=pf,
                            transport_mp_increaser=transport_mp_increaser
                        )
                        if best_order:
                            drop_zone = path[-1]
                            actual_path = path[:transport[MOVE_RANGE_KEY] + 1]
                            passengers = get_units_by_positions(my_units, list(best_order))
                            unload_map, total_damage = TransportPlan.solve_best_unload_configuration(
                                drop_zone=drop_zone,
                                target_pos=target[POS_KEY],
                                passengers=passengers,
                                enemies=enemy_units,  # для блокировки гексов
                                grid=grid,
                                pf=pf
                            )
                            tp = TransportPlan(
                                transport=transport,
                                target=target,
                                passengers=passengers,
                                path=actual_path,
                                unload_map=unload_map,
                                grid=grid,
                                pf=pf
                            )
                            if tp.utility > utility_threshold:
                                # мы планировали план транспортировки юнитов на более чем один ход, поэтому очищаем план выгрузки
                                # тех, кто мог атаковать цель на одном ходу мы уже обработали
                                if transport_mp_increaser > 1:
                                    tp.unload_map = {}
                                transport_plans.append(tp)

    return transport_plans