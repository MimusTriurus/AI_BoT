from typing import List, Tuple, Dict, Optional

from AI_BoT.data_structures import *
from w9_pathfinding.envs import HexGrid

def find_attack_positions_for_unit(grid: HexGrid, target_pos: Tuple, weapon_range: int, move_range: int = 0) -> List[Tuple]:
    visited = {target_pos}
    frontier = [target_pos]
    attack_range = weapon_range + move_range
    for _ in range(attack_range):
        next_frontier = []
        for node in frontier:
            for n, _ in grid.get_neighbors(node):
                if n not in visited and not grid.has_obstacle(n):
                    visited.add(n)
                    next_frontier.append(n)
        frontier = next_frontier

    visited.remove(target_pos)
    return list(visited)

def find_unload_positions(grid, target: Dict, units: List[Dict]) -> List:
    target_pos = target[POS_KEY]
    # Находим все возможные позиции атаки для каждого юнита
    unit_attack_positions = []
    for unit in units:
        positions = find_attack_positions_for_unit(
            grid,
            target_pos,
            unit[ATTACK_RANGE_KEY],
            unit[MOVE_RANGE_KEY] - 1,
        )
        unit_attack_positions.append(set(positions))

    # Находим пересечение - позиции, откуда ВСЕ юниты могут атаковать
    if not unit_attack_positions:
        return []

    common_positions = unit_attack_positions[0]
    for positions_set in unit_attack_positions[1:]:
        common_positions = common_positions.intersection(positions_set)

    if not common_positions:
        # Если нет общих позиций, берем позицию для юнита с наименьшей дальностью
        min_weapon_range = min(unit[ATTACK_RANGE_KEY] for unit in units)
        min_move_range = min(unit[MOVE_RANGE_KEY] for unit in units)
        min_attack_range = min(unit[ATTACK_RANGE_KEY] + unit[MOVE_RANGE_KEY] for unit in units)
        min_range_positions = []
        for unit in units:
            attack_range = unit[ATTACK_RANGE_KEY]# + unit[MOVE_RANGE_KEY]
            if attack_range == min_attack_range:
                min_range_positions = find_attack_positions_for_unit(
                    grid,
                    target_pos,
                    min_weapon_range,
                    min_move_range
                )
                break
        common_positions = min_range_positions

    return common_positions


def insert_after(lst, target, new_value):
    try:
        index = lst.index(target)
        lst.insert(index + 1, new_value)
    except ValueError:
        print("!")