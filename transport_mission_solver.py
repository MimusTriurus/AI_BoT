from itertools import permutations
from typing import Tuple, List, Optional
from w9_pathfinding.envs import HexGrid
from w9_pathfinding.pf import IDAStar, AStar

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
        # --- Превращаем юнитов в препятствия ---

        # 1. Добавляем ПАССАЖИРОВ (своих). Потому как мы их забираем транспортом
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

        best_sequence = None
        min_mp_cost = float('inf')
        best_full_path = []

        # 3. ПЕРЕБОР КОМБИНАЦИЙ СБОРА ПАССАЖИРОВ (задача коммивояжёра)
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
                    # костыль для юнитов которые могут атаковать без транспорта.
                    # условно говоря юнит это транспорт, который перевозит пушку и не может ее никуда выгрузить и стреляет с точки выгрузки
                    if transport_pos == target_unit:
                        full_path_hexes.append(transport_pos)
                        break
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
                            cost = grid.calculate_cost(path) if isinstance(path, list) else getattr(path, 'cost', len(path))

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
                        cost_drop = grid.calculate_cost(path_drop)
                        total_cost = current_accumulated_cost + cost_drop

                        if total_cost <= transport_mp:
                            if total_cost < min_mp_cost:
                                min_mp_cost = total_cost
                                best_sequence = sequence
                                final_nodes = path_drop

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