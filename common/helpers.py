from itertools import combinations
from typing import List, Tuple, Dict, Optional

from AI_BoT.common.constants import *
from AI_BoT.data_structures import Unit
from w9_pathfinding.envs import HexGrid

def find_attack_positions_for_unit(
        grid: HexGrid,
        target_pos: Tuple[int, int],
        move_range: int,
        max_weapon_range: int,
        min_weapon_range: int = 0,
) -> List[Tuple[int, int]]:
    """
    Возвращает список позиций, из которых юнит сможет атаковать цель,
    если сможет до них добраться.
    BFS выполняется от цели, поэтому позиции не гарантированно достижимы,
    но геометрически подходят для атаки.
    """

    # Максимальная дистанция BFS = макс дальность атаки + движение
    max_bfs_range = max_weapon_range + move_range

    visited = {target_pos}
    frontier = [(target_pos, 0)]  # (клетка, уровень BFS)
    result = []

    while frontier:
        new_frontier = []
        for pos, dist in frontier:
            # Для всех позиций на корректной дистанции атаки — добавляем
            if min_weapon_range <= dist <= max_weapon_range + move_range:
                result.append(pos)

            # Не выходим за пределы BFS
            if dist == max_bfs_range:
                continue

            # Расширяем фронт
            for n, _ in grid.get_neighbors(pos):
                if n not in visited and not grid.has_obstacle(n):
                    visited.add(n)
                    new_frontier.append((n, dist + 1))

        frontier = new_frontier

    # Не включаем саму цель
    if target_pos in result:
        result.remove(target_pos)

    return result


def find_unload_positions(grid, target: Unit, units: List[Unit]) -> List[Tuple[int, int]]:
    target_pos = target.pos
    # Находим все возможные позиции атаки для каждого юнита
    unit_attack_positions = []
    for unit in units:
        #  - UNIT_MOVE_RANGE_AFTER_UNLOAD потому что мы потратили очко на погрузку
        positions = find_attack_positions_for_unit(
            grid=grid,
            target_pos=target_pos,
            move_range=unit.mp - UNIT_MOVE_RANGE_AFTER_UNLOAD,
            max_weapon_range=unit.wr[1],
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
        min_attack_range = min(unit.wr[1] for unit in units)

        min_range_positions = []
        for unit in units:
            attack_range = unit.wr[1]
            if attack_range == min_attack_range:
                min_range_positions = find_attack_positions_for_unit(
                    grid=grid,
                    target_pos=target_pos,
                    move_range=UNIT_MOVE_RANGE_AFTER_UNLOAD,
                    max_weapon_range=min_attack_range,
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

def generate_transport_loads(cluster_units: List[List[Tuple[int, int]]], transport_capacity: int):
    all_loads = []
    for units_positions in cluster_units:
        # генерируем все комбинации размера 1..T
        loads = []
        for r in range(1, min(transport_capacity, len(units_positions)) + 1):
            loads.extend(combinations(units_positions, r))
        # преобразуем из tuple в list
        all_loads.append(sorted([list(l) for l in loads], key=lambda x: -len(x)))
    return all_loads

# список юнитов которые могут быть загружены и разгружены за один ход
def get_units_could_unload(units: List[dict]) -> List[dict]:
    result: List[dict] = []
    for unit in units:
        if unit[MOVE_RANGE_KEY] > 1:
            result.append(unit)
    return result

def get_units_by_positions(units: Dict[Tuple[int, int], Unit], units_pos: List[Tuple[int, int]] = None) -> List[Unit]:
    if units_pos:
        results = []
        for u_p in units_pos:
            if u_p in units:
                results.append(units[u_p])
        return results
    else:
        return list(units.values())