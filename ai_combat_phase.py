import pygame
from typing import List, Dict, Tuple, Optional, Set
from visualizer import HexVisualizer
from w9_pathfinding.envs import HexGrid, HexLayout
from w9_pathfinding.pf import IDAStar, AStar
from w9_pathfinding.mapf import CBS, SpaceTimeAStar, ReservationTable, MultiAgentAStar
from w9_pathfinding.visualization import animate_grid

class Solution:
    def __init__(self) -> None:
        self.assignments = list()
        self.paths = dict()
        # юниты (их id) имеющие цель но заблокированные на текущей итерации
        self.blocked_units = set()
        # юниты (их id) без цели
        self.unused_units = set()

        self.final_positions = set()
        self.attack_positions = set()

    def free_blocked_units(self):
        new_assignments = []
        for a in self.assignments:
            if a['unit_idx'] not in self.blocked_units:
                new_assignments.append(a)

        for bu_idx in self.blocked_units:
            del self.paths[bu_idx]

        self.unused_units.update(self.blocked_units)
        self.assignments = new_assignments


    def found(self) -> bool:
        return len(self.assignments) > 0

class Assignment:
    def __init__(self, t_idx, attack_pos, path, cost, score) -> None:
        self.target_idx = t_idx
        self.attack_pos = attack_pos
        self.path = path
        self.cost = cost
        self.score = score

class AssignmentList(List[Assignment]):
    def avgScore(self):
        if not self:
            return 0.0
        scores = [a.score for a in self if a.score is not None]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

class UnitAssignmentSolver:
    def __init__(self, grid, finder, reservation_table):
        self.grid = grid
        self.finder = finder
        self.rt = reservation_table

    def solve(self, units: List[Dict], targets: List[Dict]) -> Solution:
        possible_assignments = self._calculate_all_assignments(units, targets)

        best_solution: Solution = self._find_optimal_assignment(
            units, targets, possible_assignments
        )

        if best_solution.found():
            best_solution = self._handle_blocking_units(units, best_solution)
        return best_solution

    def _calculate_all_assignments(self, units: List[Dict], targets: List[Dict]) -> Dict:
        assignments = {}

        for u_idx, unit in enumerate(units):
            assignments[u_idx] = AssignmentList()

            for t_idx, target in enumerate(targets):
                attack_positions = self._get_attack_positions(
                    target['pos'],
                    unit['attack_range']
                )

                for attack_pos in attack_positions:
                    path = self._find_path(unit['start'], attack_pos, unit['move_range'])

                    if path:
                        cost = self.grid.calculate_cost(path)

                        if cost <= unit['move_range']:
                            score = self._calculate_assignment_score(
                                unit, target, path, cost
                            )
                            assignment = Assignment(t_idx, attack_pos, path, cost, score)
                            assignments[u_idx].append(assignment)
        # todo: на "подумать"
            #assignments[u_idx].sort(key=lambda x: x.score, reverse=True)
        #assignments = dict(sorted(assignments.items(), key=lambda item: item[1].avgScore(), reverse=True))

        return assignments

    def _calculate_assignment_score(
            self,
            unit: Dict,
            target: Dict,
            path: List,
            cost: int
    ) -> float:
        damage = unit['damage']
        hp = target['hp']
        value = target['value']

        score = 0.0

        if damage >= hp:
            score += 1000.0

        damage_ratio = min(damage / hp, 1.0)
        score += damage_ratio * 100.0

        score += value * 50.0

        distance_penalty = cost / unit['move_range']
        score -= distance_penalty * 10.0

        return score

    def _find_optimal_assignment(self, units, targets, possible_assignments):
        self.rt = type(self.rt)(self.grid)

        N = len(units)
        M = len(targets)

        solution = Solution()

        target_damage = {i: 0 for i in range(M)}

        target_priority = self._calculate_target_priority(units, targets, possible_assignments)

        assigned_units = set()

        while len(assigned_units) < N:
            all_closed = True
            for i in range(M):
                if target_damage[i] < targets[i][HP_KEY]:
                    all_closed = False
                    break

            if all_closed:
                break

            best_assignment = None
            best_value = -float('inf')

            for u_idx, unit in enumerate(units):
                if u_idx in assigned_units:
                    continue
                if u_idx == 4:
                    print('!')

                for a in possible_assignments[u_idx]:
                    t_idx = a.target_idx
                    attack_pos = a.attack_pos
                    target = targets[t_idx]

                    if target_damage[t_idx] >= target[HP_KEY]:
                        continue
                    if attack_pos in solution.attack_positions:
                        continue
                    if attack_pos in solution.final_positions:
                        continue
                    path = self._find_path_with_reservation(
                        unit['start'],
                        attack_pos,
                        unit['move_range']
                    )

                    if not path:
                        continue

                    cost = self.grid.calculate_cost(path)
                    if cost > unit['move_range']:
                        continue

                    value = self._evaluate_assignment_value(
                        unit,
                        target,
                        t_idx,
                        target_damage,
                        target_priority,
                        a.score
                    )
                    if value <= 0:
                        continue

                    if value > best_value:
                        best_value = value
                        best_assignment = {
                            'unit_idx': u_idx,
                            'target_idx': t_idx,
                            'attack_pos': attack_pos,
                            'path': path,
                            'damage': unit['damage']
                        }

            if best_assignment:
                u_idx = best_assignment['unit_idx']
                t_idx = best_assignment['target_idx']
                attack_pos = best_assignment['attack_pos']

                self.rt.add_path(best_assignment['path'], reserve_destination=True)

                assigned_units.add(u_idx)

                target_damage[t_idx] += best_assignment['damage']

                solution.assignments.append(best_assignment)
                solution.paths[u_idx] = best_assignment['path']

                solution.final_positions.add(attack_pos)
                solution.attack_positions.add(attack_pos)

            else:
                for u_idx in range(N):
                    if u_idx not in assigned_units:
                        solution.unused_units.add(u_idx)
                break

        for u_idx in range(N):
            if u_idx not in assigned_units:
                solution.unused_units.add(u_idx)

        return solution

    def _calculate_target_priority(
            self,
            units: List[Dict],
            targets: List[Dict],
            possible_assignments: Dict
    ) -> Dict[int, float]:
        priority = {}

        for t_idx, target in enumerate(targets):
            available_units = sum(
                1 for u_assignments in possible_assignments.values()
                if any(a.target_idx == t_idx for a in u_assignments)
            )
            total_damage = sum(
                units[u_idx]['damage']
                for u_idx, u_assignments in possible_assignments.items()
                if any(a.target_idx == t_idx for a in u_assignments)
            )
            can_kill = total_damage >= target['hp']
            priority[t_idx] = (
                    (1000.0 if can_kill else 0.0) +
                    target['value'] * 100.0 +
                    available_units * 10.0
            )

        return priority

    def _evaluate_assignment_value(
            self,
            unit,
            target,
            t_idx,
            target_damage,
            target_priority,
            base_score
    ):
        unit_damage = unit[DAMAGE_KEY]
        hp = target[HP_KEY]
        dmg = target_damage[t_idx]

        remaining_hp = hp - dmg

        if remaining_hp <= 0:
            return -999999.0

        if unit_damage > remaining_hp:
            return -5000.0

        value = 0.0

        if unit_damage == remaining_hp:
            value += 20000.0

        damage_ratio = (hp - remaining_hp) / hp
        value += 3000.0 * damage_ratio

        value += target_priority[t_idx]

        value += base_score

        return value

    def _handle_blocking_units(self, units: List[Dict], solution: Solution) -> Solution:
        for u_idx in solution.unused_units:
            unit = units[u_idx]
            current_pos = unit['start']

            is_blocking, blocked_units = self._is_position_blocking(current_pos, solution.paths)

            if is_blocking:
                retreat_pos = self._find_retreat_position(
                    current_pos,
                    unit['move_range'],
                    solution.paths
                )

                if retreat_pos:
                    path = self._find_path_with_reservation(current_pos, retreat_pos, unit['move_range'])
                    if path:
                        solution.paths[u_idx] = path
                        self.rt.add_path(path)
                else:
                    solution.blocked_units.update(blocked_units)
                    print(f"=== Retreat position not found. Blocked units: {' '.join(map(str, blocked_units))} ===")
        solution.free_blocked_units()
        return solution

    def _get_attack_positions(self, center: Tuple, max_radius: int) -> List[Tuple]:
        visited = {center}
        frontier = [center]

        for _ in range(max_radius):
            next_frontier = []
            for node in frontier:
                for n, _ in self.grid.get_neighbors(node):
                    if n not in visited and self.grid.weights[n[1]][n[0]] > 0:
                        visited.add(n)
                        next_frontier.append(n)
            frontier = next_frontier

        visited.remove(center)
        return list(visited)

    def _is_walkable(self, pos: Tuple) -> bool:
        return not self.grid.has_obstacle(pos)

    def _find_path(self, start: Tuple, goal: Tuple, max_unit_steps=1) -> Optional[List]:
        paths = self.finder.mapf([start], [goal])
        return paths[0] if paths else None

    def _find_path_with_reservation(self, start: Tuple, goal: Tuple, max_unit_steps=1) -> Optional[List]:
        paths = self.finder.mapf([start], [goal], reservation_table=self.rt, max_length=max_unit_steps)
        return paths[0] if paths else None

    def _is_position_blocking(self, pos: Tuple, paths: Dict) -> Tuple[bool, List]:
        blocked_units = set()
        for u_idx, path in paths.items():
            if pos in path:
                blocked_units.add(u_idx)
        return len(blocked_units) > 0, list(blocked_units)

    def _find_retreat_position(
            self,
            current_pos: Tuple,
            move_range: int,
            reserved_paths: Dict
    ) -> Optional[Tuple]:
        visited = {current_pos}
        frontier = [current_pos]

        for distance in range(move_range):
            next_frontier = []
            for node in frontier:
                for n, _ in self.grid.get_neighbors(node):
                    if n not in visited and self._is_walkable(n):
                        visited.add(n)
                        is_position_blocking, blocked_units = self._is_position_blocking(n, reserved_paths)
                        if not is_position_blocking:
                            return n

                        next_frontier.append(n)
            frontier = next_frontier

        return None

def make_animation(agents: list):
    anim = animate_grid(grid, agents)
    fname = "around_target.gif"
    anim.save(fname, fps=10, dpi=200)
    print(f"Animation saved to {fname}")


if __name__ == "__main__":
    weights = [
        # 0   1   2   3   4   5   6
        [ 1,  1,  1,  1,  1,  1,  1],  # 0
        [ 1,  1,  1,  1,  1,  1,  1],  # 1
        [ 1,  1,  1,  1,  1,  1,  1],  # 2
        [-1,  1, -1, -1, -1,  1, -1],  # 3
        [ 1,  1,  1, -1,  1,  1,  1],  # 4
        [ 1,  1,  1,  1,  1,  1,  1],  # 5
        [ 1,  1,  1,  1,  1,  1,  1],  # 6
    ]

    max_move_range = 7
    min_move_range = 3

    START_KEY = 'start'
    MOVE_RANGE_KEY = 'move_range'
    ATTACK_RANGE_KEY = 'attack_range'
    DAMAGE_KEY = 'damage'
    POS_KEY = 'pos'
    VALUE_KEY = 'value'
    HP_KEY = 'hp'

    units = [
        # group #1
        {START_KEY: (0, 0), MOVE_RANGE_KEY: max_move_range, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1},  # 1
        {START_KEY: (1, 1), MOVE_RANGE_KEY: max_move_range, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1}, # 0
        {START_KEY: (2, 4), MOVE_RANGE_KEY: max_move_range, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1}, # 2
        {START_KEY: (3, 0), MOVE_RANGE_KEY: max_move_range, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1}, # 3
        # blockers for group #1
        {START_KEY: (1, 2), MOVE_RANGE_KEY: min_move_range, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1}, # 4 -
        # group #2
        {START_KEY: (4, 4), MOVE_RANGE_KEY: max_move_range, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1},
        #{START_KEY: (5, 0), MOVE_RANGE_KEY: max_move_range, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1}, # 6
        {START_KEY: (6, 1), MOVE_RANGE_KEY: max_move_range, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1}, # 5 -
        # blockers for group #2
        {START_KEY: (5, 2), MOVE_RANGE_KEY: min_move_range, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1}, # 7 -
    ]

    targets = [
        {POS_KEY: (1, 5), VALUE_KEY: 0.1, HP_KEY: 3},
        {POS_KEY: (6, 6), VALUE_KEY: 10.2, HP_KEY: 3},
    ]

    grid = HexGrid(weights=weights, edge_collision=True, layout=HexLayout.odd_q)
    rt = ReservationTable(grid)
    #finder = MultiAgentAStar(grid)
    finder = CBS(grid)
    solver = UnitAssignmentSolver(grid, finder, rt)

    solution = solver.solve(units, targets)

    paths = solution.paths
    units_target = dict()
    for assignment in solution.assignments:
        unit_idx = assignment['unit_idx']
        target_idx = assignment['target_idx']
        units_target[unit_idx] = target_idx

    agents = []
    for u_idx, path in paths.items():
        agent = {
            'start': units[u_idx]['start'],
            'goal': path[-1],
            'path': path,
        }
        if u_idx in units_target:
            agent['target'] = targets[units_target[u_idx]][POS_KEY]
        agents.append(agent)

    make_animation(agents)

    for assignment in solution.assignments:
        unit_idx = assignment['unit_idx']
        target_idx = assignment['target_idx']
        path = assignment['path']
        print(f"Юнит {unit_idx} -> Цель {target_idx}, путь длиной {len(path) - 1}")

    for u_idx in solution.blocked_units:
        if u_idx in solution.paths:
            print(f"Юнит {u_idx} отходит в {solution.paths[u_idx][-1]}, путь длиной {len(solution.paths[u_idx]) - 1}")

    visualizer = HexVisualizer(grid)
    while True:
        restart = visualizer.animate_solution(solution.__dict__, units, targets)
        if restart:
            for target in targets:
                target['current_hp'] = target['hp']
            continue
        else:
            break

    pygame.quit()