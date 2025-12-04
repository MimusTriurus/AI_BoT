from itertools import combinations
from typing import List, Tuple, Dict, Optional

from AI_BoT.common.constants import *
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


def find_unload_positions(grid, target: Dict, units: List[Dict]) -> List[Tuple[int, int]]:
    target_pos = target[POS_KEY]
    # Находим все возможные позиции атаки для каждого юнита
    unit_attack_positions = []
    for unit in units:
        #  - UNIT_MOVE_RANGE_AFTER_UNLOAD потому что мы потратили очко на погрузку
        positions = find_attack_positions_for_unit(
            grid=grid,
            target_pos=target_pos,
            move_range=unit[MOVE_RANGE_KEY] - UNIT_MOVE_RANGE_AFTER_UNLOAD,
            max_weapon_range=unit[MAX_ATTACK_RANGE_KEY],
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
        min_attack_range = min(unit[MAX_ATTACK_RANGE_KEY] for unit in units)

        min_range_positions = []
        for unit in units:
            attack_range = unit[MAX_ATTACK_RANGE_KEY]
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

def generate_transport_loads(cluster_units: Dict[int, List[Tuple[int, int]]], transport_capacity: int):
    """
    cluster_units: dict {cluster_id: [(x1,y1), (x2,y2), ...]}
    T: вместимость транспорта (максимальное число юнитов)

    Возвращает:
        dict {cluster_id: [ [u1,u2], [u1], ... ] } - все возможные пакеты
        исключены юниты которые не могут двигаться после загрузки!
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

# список юнитов которые могут быть загружены и разгружены за один ход
def get_units_could_unload(units: List[dict]) -> List[dict]:
    result: List[dict] = []
    for unit in units:
        if unit[MOVE_RANGE_KEY] > 1:
            result.append(unit)
    return result