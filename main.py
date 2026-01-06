import os
from typing import List

from AI_BoT.game_loop import game_loop
from AI_BoT.move_and_attack import (
    calculate_move_and_attack_plans,
    calculate_move_and_attack_transport_plans
)

from common.units_loader import MultiUnitsLoader
from AI_BoT.resource_manager import units_storages, my_units_storage, en_units_storage, transport_storage, get_free_units
from transport_plan import TacticPlan
from transport_plan_optimization import TransportPlanOptimizer

from w9_pathfinding.envs import HexGrid
from w9_pathfinding.pf import IDAStar, AStar
from w9_pathfinding.mapf import CBS, SpaceTimeAStar, ReservationTable, MultiAgentAStar

scenarios_dir_path = 'AI_BoT/scenarios/'
scenario_name = os.getenv('SCENARIO_NAME', '1')
MOVE_INCREASER = int(os.getenv('MOVE_INCREASER', 2))
UTILITY_THRESHOLD = int(os.getenv('UTILITY_THRESHOLD', 0))
TRANSPORT_UTILITY_THRESHOLD = int(os.getenv('TRANSPORT_UTILITY_THRESHOLD', 0))

if __name__ == '__main__':
    scenario_path = f'{scenarios_dir_path}{scenario_name}.json'

    units_loader = MultiUnitsLoader(units_storages)
    units_loader.load_from_json(scenario_path)

    attack_plans: List[TacticPlan] = []

    # планы на атаку целей боевыми юнитами без транспортов
    if my_units_storage.get_units():
        attack_plans += calculate_move_and_attack_plans(
            my_units=my_units_storage.get_units(),
            enemy_units=en_units_storage.get_units(),
            reserved_positions=[],
            utility_threshold=UTILITY_THRESHOLD
        )

    # просчитываем планы атаки целей боевыми юнитами с использованием транспортов
    if transport_storage.get_units():
        attack_plans += calculate_move_and_attack_transport_plans(
            my_units=my_units_storage.units,
            transport_units=transport_storage.get_units(),
            enemy_units=en_units_storage.get_units(),
            reserved_positions=[],
            utility_threshold=UTILITY_THRESHOLD,
        )

    attack_plans = sorted(attack_plans, key=lambda p: p.utility, reverse=True)
    optimizer = TransportPlanOptimizer(attack_plans)

    # actual_plans, total_utility = optimizer.optimize(method='auction')
    # auction + local search
    actual_plans, total_utility = optimizer.optimize_hybrid()
    #actual_plans, total_utility = optimizer.optimize_branch_and_bound(actual_plans, total_utility)

    #print(f'\n=================')
    #for plan in actual_plans:
    #    print(str(plan))
    #    print(f'-------')
    #print(f'=== Total utility: {total_utility} ===')

    unused_units, unused_transports, free_targets, reserved_positions = get_free_units(actual_plans)

    print(f"Unused combat units: {len(unused_units)}")
    print(f"Unused transports: {len(unused_transports)}")
    print(f"Alive targets: {len(free_targets)}")

    move_plans = []

    if unused_units:# and False:
        move_plans += calculate_move_and_attack_plans(
            my_units=list(unused_units.values()),
            enemy_units=free_targets,
            reserved_positions=reserved_positions,
            utility_threshold=UTILITY_THRESHOLD,
            is_future=True
        )
    if unused_transports:# and False:
        move_plans += calculate_move_and_attack_transport_plans(
            my_units=unused_units,
            transport_units=unused_transports,
            enemy_units=free_targets,
            reserved_positions=reserved_positions,
            utility_threshold=TRANSPORT_UTILITY_THRESHOLD,
            transport_mp_increaser=MOVE_INCREASER
        )

    move_plans = sorted(move_plans, key=lambda p: p.utility, reverse=True)
    optimizer = TransportPlanOptimizer(move_plans)
    actual_plans_2_move, total_utility_2_move = optimizer.optimize_hybrid()

    actual_plans += actual_plans_2_move

    game_loop(actual_plans)

    print('End!')
