import os
from typing import List

from AI_BoT.game_loop import game_loop
from AI_BoT.move_and_attack import (
    calculate_move_and_attack_for_units_plans,
    calculate_move_and_attack_for_units_plans_using_transports
)

from common.units_loader import MultiUnitsLoader
from resource_manager import units_storages, my_units_storage, en_units_storage, transport_storage, get_free_units
from transport_plan import TransportPlan
from transport_plan_optimization import TransportPlanOptimizer

from w9_pathfinding.envs import HexGrid
from w9_pathfinding.pf import IDAStar, AStar
from w9_pathfinding.mapf import CBS, SpaceTimeAStar, ReservationTable, MultiAgentAStar

UTILITY_THRESHOLD = 0

scenarios_dir_path = 'AI_BoT/scenarios/'
scenario_name = os.getenv('SCENARIO_NAME', '1')

if __name__ == '__main__':
    scenario_path = f'{scenarios_dir_path}{scenario_name}.json'

    units_loader = MultiUnitsLoader(units_storages)
    units_loader.load_from_json(scenario_path)

    transport_plans: List[TransportPlan] = []

    # планы на атаку целей юнитами без транспортов
    transport_plans += calculate_move_and_attack_for_units_plans(
        my_units_storage.get_units(),
        en_units_storage.get_units(),
        UTILITY_THRESHOLD
    )
    # просчитываем планы атаки целей боевыми юнитами с использованием транспортов
    transport_plans += calculate_move_and_attack_for_units_plans_using_transports(
        my_units_storage.units,
        transport_storage.get_units(),
        en_units_storage.get_units(),
        UTILITY_THRESHOLD
    )

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

    unused_units, unused_transports, free_targets = get_free_units(actual_plans)

    print(f"Unused combat units: {len(unused_units)}")
    print(f"Unused transports: {len(unused_transports)}")

    game_loop(actual_plans)

    print('End!')
