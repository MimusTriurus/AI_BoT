"""
Система транспортных юнитов для тактической игры на гексах
Референс: Massive Assault 2
"""

import random
import math
from copy import deepcopy
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field

from visualizer import HexVisualizer
from w9_pathfinding.envs import HexGrid, HexLayout
from w9_pathfinding.pf import IDAStar, AStar
from w9_pathfinding.mapf import CBS, SpaceTimeAStar, ReservationTable, MultiAgentAStar

from game_state import *

@dataclass
class TransportSystem:
    """Система управления транспортом и загрузкой юнитов"""

    @staticmethod
    def can_load(transport: Dict, unit: Dict) -> bool:
        """Может ли транспорт загрузить юнит?"""
        # Проверяем, что это транспорт
        if transport.get(TYPE_KEY) != 'transport':
            return False

        # Проверяем вместимость
        cargo = transport.get(CARGO_KEY, [])
        capacity = transport.get(CAPACITY_KEY, 0)
        if len(cargo) >= capacity:
            return False

        # Нельзя загружать другие транспорты (как в MA2)
        if unit.get(TYPE_KEY) == 'transport':
            return False

        # Проверяем, что юнит не мертв
        if unit.get(HP_KEY, 0) <= 0:
            return False

        return True

    @staticmethod
    def load_unit(transport: Dict, unit: Dict) -> bool:
        """Загрузить юнит в транспорт"""
        if not TransportSystem.can_load(transport, unit):
            return False

        if CARGO_KEY not in transport:
            transport[CARGO_KEY] = []

        # Сохраняем ссылку на юнит
        transport[CARGO_KEY].append(unit[ID_KEY])

        # Помечаем юнит как загруженный
        unit['loaded_in'] = transport[ID_KEY]

        return True

    @staticmethod
    def unload_unit(transport: Dict, unit: Dict) -> bool:
        """Выгрузить юнит из транспорта"""
        if CARGO_KEY not in transport:
            return False

        unit_id = unit[ID_KEY]
        if unit_id not in transport[CARGO_KEY]:
            return False

        transport[CARGO_KEY].remove(unit_id)
        unit.pop('loaded_in', None)

        return True

    @staticmethod
    def get_cargo_units(transport: Dict, all_units: List[Dict]) -> List[Dict]:
        """Получить все юниты, загруженные в транспорт"""
        if CARGO_KEY not in transport:
            return []

        cargo_ids = transport[CARGO_KEY]
        return [u for u in all_units if u[ID_KEY] in cargo_ids]

    @staticmethod
    def is_loaded(unit: Dict) -> bool:
        """Проверить, загружен ли юнит"""
        return 'loaded_in' in unit


class GameState:
    """
    Refactored GameState:
      - apply_unit_action возвращает новый GameState с корректными deepcopy'ями только для изменяемых списков.
      - поддержка последовательного планирования полного хода (action per unit in turn order).
    """
    def __init__(self, my_units: List[Dict], enemy_units: List[Dict],
                 grid, mapf, pf, rt, current_player: int = 0):
        # копируем списки юнитов — предполагается, что caller передаёт НЕ-шаренные объекты,
        # но чтобы быть безопасным, делаем deepcopy здесь
        self.my_units = deepcopy(my_units)
        self.enemy_units = deepcopy(enemy_units)
        self.grid = grid
        self.mapf = mapf
        self.pf = pf
        self.rt = rt
        self.current_player = current_player

        # инициализация недостающих полей
        for i, unit in enumerate(self.my_units):
            unit.setdefault(HP_KEY, 3)
            unit.setdefault(ID_KEY, f'my_{i}')
            if unit.get(TYPE_KEY) == 'transport':
                unit.setdefault(CARGO_KEY, [])

        for i, unit in enumerate(self.enemy_units):
            unit.setdefault(HP_KEY, 3)
            unit.setdefault(ID_KEY, f'enemy_{i}')
            if unit.get(TYPE_KEY) == 'transport':
                unit.setdefault(CARGO_KEY, [])

    def get_current_units(self) -> List[Dict]:
        return self.my_units if self.current_player == 0 else self.enemy_units

    def get_enemy_units(self) -> List[Dict]:
        return self.enemy_units if self.current_player == 0 else self.my_units

    def get_all_units(self) -> List[Dict]:
        return self.my_units + self.enemy_units

    def get_occupied_positions(self) -> Set[Tuple[int, int]]:
        """Позиции занятые незагруженными юнитами (учитываем hp>0)."""
        positions = set()
        for unit in self.get_all_units():
            if unit.get(HP_KEY, 0) > 0 and not TransportSystem.is_loaded(unit):
                positions.add(unit[START_KEY])
        return positions

    def get_alive_unloaded_unit_indices(self) -> List[int]:
        """Возвращает индексы (в списке current_units) живых и незагруженных юнитов, в определённом планируемом порядке."""
        units = self.get_current_units()
        return [i for i, u in enumerate(units) if u.get(HP_KEY, 0) > 0 and not TransportSystem.is_loaded(u)]

    def hex_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Расстояние hex axial -> cube conversion, как было у тебя."""
        q1, r1 = pos1
        q2, r2 = pos2

        x1 = q1
        z1 = r1 - (q1 - (q1 & 1)) // 2
        y1 = -x1 - z1

        x2 = q2
        z2 = r2 - (q2 - (q2 & 1)) // 2
        y2 = -x2 - z2

        return (abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)) // 2

    # ========== ACTION GENERATION (исправлено: атаки по enemy_units) ==========
    def get_possible_actions(self, unit_idx: int) -> List[Dict]:
        """
        Список возможных действий для конкретного юнита (unit_idx — индекс в current_units).
        Возвращает список action dict.
        """
        units = self.get_current_units()
        if unit_idx >= len(units):
            return [{'type': 'wait', 'unit_idx': unit_idx}]
        unit = units[unit_idx]

        if unit.get(HP_KEY, 0) <= 0:
            return [{'type': 'wait', 'unit_idx': unit_idx}]

        # загруженные юниты не действуют, разве что могут стрелять
        if TransportSystem.is_loaded(unit):
            if unit.get(CAN_FIRE_LOADED_KEY, False):
                transport_id = unit.get(LOADED_IN_KEY)
                transport = next((u for u in units if u.get(ID_KEY) == transport_id), None)
                if transport:
                    return self._get_loaded_fire_actions(unit, transport, unit_idx)
            return [{'type': 'wait', 'unit_idx': unit_idx}]

        current_pos = unit[START_KEY]
        occupied = self.get_occupied_positions()
        actions = [{'type': 'wait', 'unit_idx': unit_idx}]

        # транспортные действия частично переиспользованы, но сформируем корректно
        if unit.get(TYPE_KEY) == 'transport':
            actions.extend(self._get_transport_actions(unit, unit_idx, units, self.get_all_units()))

        # соседние клетки
        neighbors = self.grid.get_neighbors(current_pos, include_self=False)
        enemy_units = self.get_enemy_units()
        # Комбо: движение + атака (цели — из enemy_units)
        for neighbor_pos, cost in neighbors:
            if self.grid.has_obstacle(neighbor_pos):
                continue
            if neighbor_pos in occupied:
                continue
            if self.hex_distance(current_pos, neighbor_pos) <= unit.get(MOVE_RANGE_KEY, 0):
                pos_used_4_attack = False
                for enemy_idx, enemy in enumerate(enemy_units):
                    if enemy.get(HP_KEY, 0) > 0 and not TransportSystem.is_loaded(enemy):
                        dist = self.hex_distance(neighbor_pos, enemy[START_KEY])
                        if dist <= unit.get(ATTACK_RANGE_KEY, 0):
                            if unit_idx == 2:
                                print('!')

                            actions.append({
                                'type': 'move_attack',
                                'unit_idx': unit_idx,
                                'to': neighbor_pos,
                                'target_idx': enemy_idx
                            })
                        pos_used_4_attack = True
                # обычное движение только если мы не можем атаковать с этой позиции
                #continue
                if not pos_used_4_attack:
                    actions.append({
                        'type': 'move',
                        'unit_idx': unit_idx,
                        'to': neighbor_pos
                    })

        # Движение в соседние клетки (если незанято)
        '''
        for neighbor_pos, cost in neighbors:
            if self.grid.has_obstacle(neighbor_pos):
                continue
            if neighbor_pos in occupied:
                continue
            if self.hex_distance(current_pos, neighbor_pos) <= unit.get(MOVE_RANGE_KEY, 0):
                actions.append({
                    'type': 'move',
                    'unit_idx': unit_idx,
                    'to': neighbor_pos
                })
        '''
        # Атака (на месте)
        if unit.get(DAMAGE_KEY, 0) > 0:
            for enemy_idx, enemy in enumerate(enemy_units):
                if enemy.get(HP_KEY, 0) > 0 and not TransportSystem.is_loaded(enemy):
                    dist = self.hex_distance(current_pos, enemy[START_KEY])
                    if dist <= unit.get(ATTACK_RANGE_KEY, 0):
                        actions.append({
                            'type': 'attack',
                            'unit_idx': unit_idx,
                            'target_idx': enemy_idx
                        })

        # Комбинированные (движение + загрузка) — для транспортов
        if unit.get(TYPE_KEY) == 'transport':
            for neighbor_pos, cost in neighbors:
                if self.grid.has_obstacle(neighbor_pos):
                    continue
                if self.hex_distance(current_pos, neighbor_pos) <= unit.get(MOVE_RANGE_KEY, 0):
                    for other_idx, other in enumerate(units):
                        if other_idx == unit_idx:
                            continue
                        if other.get(HP_KEY, 0) <= 0:
                            continue
                        if TransportSystem.is_loaded(other):
                            continue
                        if self.hex_distance(neighbor_pos, other[START_KEY]) == 0 and TransportSystem.can_load(unit, other):
                            actions.append({
                                'type': 'move_and_load',
                                'unit_idx': unit_idx,
                                'to': neighbor_pos,
                                'load_unit_idx': other_idx
                            })

        return actions if actions else [{'type': 'wait', 'unit_idx': unit_idx}]

    # transport helpers (практически как у тебя, слегка адаптировано)
    def _get_transport_actions(self, transport: Dict, transport_idx: int,
                               units: List[Dict], all_units: List[Dict]) -> List[Dict]:
        actions = []
        current_pos = transport[START_KEY]
        neighbors = self.grid.get_neighbors(current_pos, include_self=False)
        neighbor_positions = {pos for pos, _ in neighbors}
        neighbor_positions.add(current_pos)

        # load
        for other_idx, other in enumerate(units):
            if other.get(HP_KEY, 0) <= 0:
                continue
            if TransportSystem.is_loaded(other):
                continue
            if other[START_KEY] in neighbor_positions and TransportSystem.can_load(transport, other):
                actions.append({
                    'type': 'load',
                    'unit_idx': transport_idx,
                    'load_unit_idx': other_idx
                })

        # unload
        cargo_units = TransportSystem.get_cargo_units(transport, all_units)
        occupied = self.get_occupied_positions()
        for cargo_unit in cargo_units:
            cargo_id = cargo_unit.get(ID_KEY)
            for neighbor_pos, _ in neighbors:
                if self.grid.has_obstacle(neighbor_pos):
                    continue
                if neighbor_pos in occupied:
                    continue
                actions.append({
                    'type': 'unload',
                    'unit_idx': transport_idx,
                    'unload_unit_id': cargo_id,
                    'to': neighbor_pos
                })

        return actions

    def _get_loaded_fire_actions(self, unit: Dict, transport: Dict, unit_idx: int) -> List[Dict]:
        actions = []
        transport_pos = transport[START_KEY]
        enemy_units = self.get_enemy_units()
        for enemy_idx, enemy in enumerate(enemy_units):
            if enemy.get(HP_KEY, 0) <= 0:
                continue
            if TransportSystem.is_loaded(enemy):
                continue
            dist = self.hex_distance(transport_pos, enemy[START_KEY])
            if dist <= unit.get(ATTACK_RANGE_KEY, 0):
                actions.append({
                    'type': 'fire_from_transport',
                    'unit_idx': unit_idx,
                    'target_idx': enemy_idx
                })
        return actions if actions else [{'type': 'wait', 'unit_idx': unit_idx}]

    # ========== APPLY single unit action (возвращает новый GameState) ==========
    def apply_unit_action(self, unit_global_idx: int, action: Dict) -> 'GameState':
        """
        Применить действие для конкретного юнита в списке current_units.
        Возвращает новый GameState (копия).
        unit_global_idx - индекс юнита в списке get_current_units() для этой GameState.
        """
        # Создаём неглубокую копию (где списки юнитов копируются, чтобы состояния не пересекались)
        new_my = deepcopy(self.my_units)
        new_enemy = deepcopy(self.enemy_units)
        new_state = GameState(new_my, new_enemy, self.grid, self.mapf, self.pf, self.rt, self.current_player)

        units = new_state.get_current_units()
        enemy_units = new_state.get_enemy_units()
        all_units = new_state.get_all_units()

        if unit_global_idx >= len(units):
            return new_state

        unit = units[unit_global_idx]
        action_type = action.get('type')

        if action_type == 'move':
            unit[START_KEY] = action['to']

        elif action_type == 'attack':
            target_idx = action.get('target_idx')
            if target_idx is not None and target_idx < len(enemy_units):
                enemy_units[target_idx][HP_KEY] -= unit.get(DAMAGE_KEY, 0)

        elif action_type == 'move_attack':
            target_idx = action.get('target_idx')
            unit[START_KEY] = action['to']
            if target_idx is not None and target_idx < len(enemy_units):
                enemy_units[target_idx][HP_KEY] -= unit.get(DAMAGE_KEY, 0)

        elif action_type == 'load':
            load_idx = action.get('load_unit_idx')
            if load_idx is not None and load_idx < len(units):
                TransportSystem.load_unit(unit, units[load_idx])

        elif action_type == 'unload':
            unload_id = action.get('unload_unit_id')
            unload_unit = next((u for u in all_units if u.get(ID_KEY) == unload_id), None)
            if unload_unit:
                TransportSystem.unload_unit(unit, unload_unit)
                unload_unit[START_KEY] = action['to']

        elif action_type == 'move_and_load':
            unit[START_KEY] = action['to']
            load_idx = action.get('load_unit_idx')
            if load_idx is not None and load_idx < len(units):
                TransportSystem.load_unit(unit, units[load_idx])

        elif action_type == 'fire_from_transport':
            target_idx = action.get('target_idx')
            if target_idx is not None and target_idx < len(enemy_units):
                enemy_units[target_idx][HP_KEY] -= unit.get(DAMAGE_KEY, 0)

        elif action_type == 'wait':
            pass  # ничего

        # Возвращаем новый state (не переключаем игрока — это делается в apply_turn)
        return new_state

    # ========== APPLY TURN: применяет последовательность unit actions и переключает игрока ==========
    def apply_turn(self, unit_actions: List[Tuple[int, Dict]]) -> 'GameState':
        """
        unit_actions: список кортежей (unit_index_in_current_units, action_dict)
        Порядок действий должен соответствовать порядку, который планирует MCTS.
        После применения всех действий переключаем current_player.
        """
        state = self
        # применяем каждое действие по очереди (для текущего игрока)
        try:
            for action in unit_actions:
                unit_idx = action[0]
                state = state.apply_unit_action(unit_idx, action[1])
                # сохраняем current_player без авто-переключения
                state.current_player = self.current_player
        except Exception as e:
            print(e)

        # теперь переключаем игрока (ход завершён)
        new_state = GameState(state.my_units, state.enemy_units, state.grid, state.mapf, state.pf, state.rt,
                              current_player=1 - state.current_player)
        return new_state

    def is_terminal(self) -> bool:
        my_alive = sum(1 for u in self.my_units if u.get(HP_KEY, 0) > 0 and not TransportSystem.is_loaded(u))
        enemy_alive = sum(1 for u in self.enemy_units if u.get(HP_KEY, 0) > 0 and not TransportSystem.is_loaded(u))
        return my_alive == 0 or enemy_alive == 0

    def get_reward(self, for_player: int) -> float:
        my_alive = sum(1 for u in self.my_units if u.get(HP_KEY, 0) > 0 and not TransportSystem.is_loaded(u))
        enemy_alive = sum(1 for u in self.enemy_units if u.get(HP_KEY, 0) > 0 and not TransportSystem.is_loaded(u))

        my_hp = sum(u.get(HP_KEY, 0) for u in self.my_units if u.get(HP_KEY, 0) > 0)
        enemy_hp = sum(u.get(HP_KEY, 0) for u in self.enemy_units if u.get(HP_KEY, 0) > 0)

        my_transports_loaded = sum(1 for u in self.my_units if u.get(TYPE_KEY) == 'transport' and len(u.get(CARGO_KEY, [])) > 0)
        enemy_transports_loaded = sum(1 for u in self.enemy_units if u.get(TYPE_KEY) == 'transport' and len(u.get(CARGO_KEY, [])) > 0)

        if for_player == 0:
            if enemy_alive == 0:
                return 100.0
            if my_alive == 0:
                return -100.0
            reward = (my_hp - enemy_hp) * 0.5 + (my_alive - enemy_alive) * 5.0
            #reward += (my_transports_loaded - enemy_transports_loaded) * 2.0
            return reward
        else:
            if my_alive == 0:
                return 100.0
            if enemy_alive == 0:
                return -100.0
            reward = (enemy_hp - my_hp) * 0.5 + (enemy_alive - my_alive) * 5.0
            #reward += (enemy_transports_loaded - my_transports_loaded) * 2.0
            return reward


# ========== MCTS ==========
class MCTSNode:
    """
    Узел представляет состояние *внутри* планирования одного хода:
      - node.state: GameState
      - node.next_unit_index_ptr: индекс очередного юнита (в списке alive indices), для которого будем генерировать действия на этом уровне
    """
    def __init__(self, state: GameState, parent: Optional['MCTSNode'] = None,
                 action: Optional[Tuple[int, Dict]] = None, next_unit_ptr: int = 0, alive_unit_indices: Optional[List[int]] = None):
        self.state = state
        self.parent = parent
        self.action = action  # действие, приведшее в этот узел (unit_idx, action_dict)
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions: Optional[List[Dict]] = None
        self.next_unit_ptr = next_unit_ptr  # зачем на этом уровне — какой по счёту юнит действует
        # alive_unit_indices — индексы юнитов в current_units, порядок планирования
        if alive_unit_indices is None:
            self.alive_unit_indices = state.get_alive_unloaded_unit_indices()
        else:
            self.alive_unit_indices = alive_unit_indices

    def is_fully_expanded(self) -> bool:
        # если уже сгенерировали untried_actions и они пусты -> fully expanded
        if self.untried_actions is None:
            # Если next_unit_ptr == len(alive_unit_indices), значит мы уже спланировали все юниты — этот узел соответствует полному ходу
            if self.next_unit_ptr >= len(self.alive_unit_indices):
                self.untried_actions = []
                return True

            # Генерация действий только для текущего юнита в order
            unit_idx = self.alive_unit_indices[self.next_unit_ptr]
            self.untried_actions = self.state.get_possible_actions(unit_idx)
            # пометка: у каждой action должен быть скорректированный 'unit_idx' = unit_idx
            for a in self.untried_actions:
                a['unit_idx'] = unit_idx

            if not self.untried_actions:
                self.untried_actions = [{'type': 'wait', 'unit_idx': unit_idx}]

        return len(self.untried_actions) == 0

    def best_child(self, exploration_weight: float = 1.41) -> 'MCTSNode':
        choices = []
        for child in self.children:
            if child.visits > 0:
                exploitation = child.value / child.visits
                exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
                score = exploitation + exploration
                choices.append((score, child))

        if not choices:
            # если нет статистики — вернём случайного ребёнка (fallback)
            return random.choice(self.children)
        return max(choices, key=lambda x: x[0])[1]

    def expand(self) -> 'MCTSNode':
        # берем действие для текущего unit
        action = self.untried_actions.pop()
        unit_idx = action['unit_idx']
        # применяем действие локально (но не переключаем игрока) — эта apply_unit_action создаёт новый state
        next_state = self.state.apply_unit_action(unit_idx, action)
        # сохраняем current_player тот же (мы планируем ход, переключение будет на конце)
        next_state.current_player = self.state.current_player

        # продвигаем next_unit_ptr
        next_ptr = self.next_unit_ptr + 1
        child = MCTSNode(next_state, parent=self, action=(unit_idx, action), next_unit_ptr=next_ptr, alive_unit_indices=self.alive_unit_indices)
        self.children.append(child)
        return child


class MCTS:
    def __init__(self, iterations: int = 1000, exploration_weight: float = 1.41, rollout_depth: int = 6):
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.rollout_depth = rollout_depth

    def plan_turn(self, initial_state: GameState) -> List:
        """
        Планируем полный ход (действия для всех живых незагруженных юнитов).
        Возвращаем actions: список action dict в том порядке, в котором их следует применять.
        """
        # Root: ещё не выбрано ни одного unit action, next_unit_ptr = 0
        root = MCTSNode(
            initial_state,
            next_unit_ptr=0,
            alive_unit_indices=initial_state.get_alive_unloaded_unit_indices()
        )
        for _ in range(self.iterations):
            node = root
            start_player = node.state.current_player
            # Selection & Expansion
            # Продвигаемся вниз: пока узел полностью расширен и есть дети — выбираем лучший
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.exploration_weight)

            # Если не terminal на уровне хода (т.е. ещё есть unit'ы для планирования) — expand
            if not node.is_fully_expanded() and not node.state.is_terminal():
                node = node.expand()

            # Если node соответствует полному набору unit-акций (next_unit_ptr == len) — составим unit_actions и применим apply_turn
            # Иначе — симуляция будет начинаться с текущего частичного turn (node.state)
            if node.next_unit_ptr >= len(node.alive_unit_indices):
                # нужно собрать путь от root до node, чтобы получить последовательность (unit_idx, action)
                actions_seq = []
                cur = node
                while cur and cur.action is not None:
                    actions_seq.append(cur.action)  # (unit_idx, action_dict)
                    cur = cur.parent
                actions_seq.reverse()
                # actions_seq это список (unit_idx, action) — применим их как turn
                sim_state = node.state.apply_turn(actions_seq)
            else:
                # симуляция начнётся из node.state (частичный turn не завершён) — для корректности применим "wait" для незавершенных unit'ов,
                # либо можно сразу симулировать случайные оставшиеся actions и затем применить_turn; проще: в симуляции мы продолжим заполнение хода случайными действиями
                sim_state = deepcopy(node.state)

            # Simulation (playout) — чередуем игроков до глубины
            reward = self.simulate(sim_state, starting_player=start_player, max_depth=self.rollout_depth)

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent

        # После итераций — выбираем наиболее посещённого ребёнка на уровне root, и собираем полный набор действий
        # На root.children могут находиться узлы, каждый соответствует первому юниту; но нам нужно полное действие для всех юнитов.
        # Подход: рекурсивно выбрать наиболее посещённую ветку до конца.
        best_node = root
        actions = []
        while best_node.next_unit_ptr < len(best_node.alive_unit_indices):
            if not best_node.children:
                # нет расширений — нулевой fallback: выбираем первое действие 가능한
                unit_idx = best_node.alive_unit_indices[best_node.next_unit_ptr]
                possible = best_node.state.get_possible_actions(unit_idx)
                act = possible[0] if possible else {'type': 'wait', 'unit_idx': unit_idx}
                actions.append((unit_idx, act))
                # применяем локально
                new_state = best_node.state.apply_unit_action(unit_idx, act)
                new_state.current_player = best_node.state.current_player
                best_node = MCTSNode(new_state, parent=best_node, next_unit_ptr=best_node.next_unit_ptr + 1, alive_unit_indices=best_node.alive_unit_indices)
                continue

            # выбрать child с max visits
            best_child = max(best_node.children, key=lambda c: c.visits)
            actions.append(best_child.action)  # (unit_idx, action)
            best_node = best_child

        # actions в формате [(unit_idx, action_dict), ...] — вернём как список action_dict с корректными unit_idx
        # (Ты, возможно, предпочитаешь плоский список action_dict с unit_idx внутри)
        return actions

    def simulate(self, state: GameState, starting_player: int, max_depth: int = 10) -> float:
        """
        Симуляция со сменой игроков: на каждом шаге выбираем случайные/эвристические действия для каждого юнита в очереди.
        Возвращаем reward для starting_player (который был игроком, для которого мы планировали).
        """
        current_state = deepcopy(state)
        depth = 0
        while not current_state.is_terminal() and depth < max_depth:
            # Планируем полный ход для current_state.current_player простым способом:
            units_indices = current_state.get_alive_unloaded_unit_indices()
            unit_actions = []
            for unit_idx in units_indices:
                actions = current_state.get_possible_actions(unit_idx)
                if not actions:
                    act = {'type': 'wait', 'unit_idx': unit_idx}
                else:
                    act = self.rollout_policy(current_state)
                unit_actions.append((unit_idx, act))
                # применяем локально без переключения
                current_state = current_state.apply_unit_action(unit_idx, act)
                current_state.current_player = state.current_player  # сохраняем текущего игрока до конца хода

            # закончился ход — переключаем игрока
            current_state = current_state.apply_turn([(i, a) for (i, a) in unit_actions])
            depth += 1
        reward = current_state.get_reward(starting_player)
        return reward

    def rollout_policy(self, state):
        """
        Heuristic rollout:
        1) try killing blow
        2) try safe damage
        3) move toward nearest target
        4) fallback = wait
        """

        my_units = state.get_current_units()
        enemies = state.get_enemy_units()

        # --- 1. Если врагов нет → полный выигрыш, симуляция закончена ---
        alive_enemies = [e for e in enemies if e[HP_KEY] > 0]
        if not alive_enemies:
            return {"type": "wait"}

        # --- 2. Выбрать живого юнита ---
        alive_my_units = [(i, u) for i, u in enumerate(my_units) if u[HP_KEY] > 0]
        if not alive_my_units:
            return {"type": "wait"}  # по сути проигрыш

        unit_idx, unit = random.choice(alive_my_units)
        actions = state.get_possible_actions(unit_idx)
        if not actions:
            return {"type": "wait"}

        # --- 3. Фильтрация действий ---
        attack_actions = [a for a in actions if a["type"] == "attack"]
        move_attack_actions = [a for a in actions if a["type"] == "move_attack"]
        move_actions = [a for a in actions if a["type"] == "move"]

        # --- 4. Сначала ищем 100% kill ---
        killing_actions = []
        for a in attack_actions + move_attack_actions:
            target = enemies[a["target_idx"]]
            if target[HP_KEY] <= unit["damage"]:
                killing_actions.append(a)

        if killing_actions:
            return random.choice(killing_actions)

        # --- 5. Safe damage (мы ударим, назад не прилетит) ---
        safe_attacks = []
        for a in attack_actions + move_attack_actions:
            target = enemies[a["target_idx"]]
            target_damage = target.get("damage", 0)

            # столи бы под ответным огнём?
            if target_damage < unit["hp"]:  # он нас не убьёт
                safe_attacks.append(a)

        if safe_attacks:
            return random.choice(safe_attacks)

        # --- 6. Движение к ближайшему врагу ---
        if move_actions:
            # выбираем гекс, который минимизирует расстояние
            best_move = None
            best_dist = 999999

            for a in move_actions:
                new_pos = a["to"]
                dist = min(state.hex_distance(new_pos, e["start"]) for e in alive_enemies)
                if dist < best_dist:
                    best_dist = dist
                    best_move = a

            if best_move:
                return best_move

        # --- 7. Fallback: безопасней всего подождать ---
        return {"type": "wait"}


# === ДЕМОНСТРАЦИЯ ===
def demo():
    grid = HexGrid(weights=weights, edge_collision=True, layout=HexLayout.odd_q)
    pf = AStar(grid)

    mcts = MCTS(iterations=MCTS_ITERATIONS, rollout_depth=ROLLOUT_DEPTH)

    # Сравним оба режима планирования
    #for planning_mode in ['sequential', 'simultaneous']:
    for planning_mode in ['sequential']:
        print(f"\n{'=' * 60}")
        print(f"Режим планирования: {planning_mode.upper()}")
        print(f"{'=' * 60}")

        test_state = GameState(
            my_units,
            enemy_units,
            grid,
            None,
            None,
            None,
            current_player=0
        )

        # Симулируем несколько ходов
        for turn in range(TURNS):
            print(f"\n{'─' * 60}")
            print(f"ХОД {turn + 1}")
            print(f"{'─' * 60}")

            if test_state.is_terminal():
                print("Игра завершена!")
                break

            # Планируем ход для всех юнитов текущего игрока
            player = test_state.current_player
            player_name = "Мои войска" if player == 0 else "Враг"
            print(f"\n>>> {player_name} планирует ход...")

            best_turn = mcts.plan_turn(test_state)

            print(f"\nПлан действий ({len(best_turn)} действий):")
            for action in best_turn:
                action = action[1]
                action_desc = action.get('type', 'wait')
                if action_desc == 'load':
                    action_desc = f"load unit {action['load_unit_idx']}"
                elif action_desc == 'unload':
                    action_desc = f"unload to {action.get('to')}"
                elif action_desc == 'move':
                    action_desc = f"move to {action.get('to')}"
                elif action_desc == 'attack':
                    action_desc = f"attack enemy {action['target_idx']}"
                elif action_desc == 'move_attack':
                    action_desc = f"move to {action.get('to')} and attack enemy {action['target_idx']}"

                print(f"Unit {action.get('unit_idx', '?')}: {action_desc}")

            # Применяем все действия
            test_state = test_state.apply_turn(best_turn)

            # Выводим результат
            print(f"\nРезультат хода:")
            '''
            my_transport = test_state.my_units[0]
            cargo_count = len(my_transport.get(CARGO_KEY, []))
            print(f"  Транспорт: {my_transport[START_KEY]}, груз: {cargo_count}/{my_transport[CAPACITY_KEY]}")
            '''
            my_alive = sum(1 for u in test_state.my_units
                           if u[HP_KEY] > 0 and not TransportSystem.is_loaded(u))
            enemy_alive = sum(1 for u in test_state.enemy_units
                              if u[HP_KEY] > 0 and not TransportSystem.is_loaded(u))

            print(f"  Союзников на поле: {my_alive}")
            print(f"  Врагов на поле: {enemy_alive}")

            my_total_hp = sum(u[HP_KEY] for u in test_state.my_units if u[HP_KEY] > 0)
            enemy_total_hp = sum(u[HP_KEY] for u in test_state.enemy_units if u[HP_KEY] > 0)
            print(f"  Общее HP: союзники {my_total_hp}, враги {enemy_total_hp}")

        print(f"\n{'=' * 60}")
        print(f"ИТОГ ({planning_mode.upper()}):")
        print(f"{'=' * 60}")
        my_alive = sum(1 for u in test_state.my_units if u[HP_KEY] > 0)
        enemy_alive = sum(1 for u in test_state.enemy_units if u[HP_KEY] > 0)
        print(f"Союзников: {my_alive}, Врагов: {enemy_alive}")

        if enemy_alive == 0:
            print("🎉 ПОБЕДА!")
        elif my_alive == 0:
            print("💀 ПОРАЖЕНИЕ!")
        else:
            print("⚔️ Бой продолжается...")


if __name__ == "__main__":
    demo()