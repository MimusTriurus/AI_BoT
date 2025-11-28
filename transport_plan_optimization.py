from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict
import heapq
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from functools import partial


class TransportPlanOptimizer:
    """
    Оптимизатор транспортных планов на основе аукциона задач.
    Максимизирует суммарную utility с учётом конфликтов по ресурсам.
    """

    def __init__(self, plans: List['TransportPlan']):
        """
        Args:
            plans: Список всех возможных TransportPlan
        """
        self.plans = plans
        self.selected_plans: List['TransportPlan'] = []
        self.total_utility = 0.0

    def optimize(self, method='hybrid', n_workers=None) -> Tuple[List['TransportPlan'], float]:
        """
        Находит оптимальный набор планов.

        Args:
            method: 'branch_and_bound', 'auction', 'hybrid', 'parallel_bnb'
            n_workers: Количество процессов для параллелизации (None = auto)

        Returns:
            (selected_plans, total_utility)
        """
        if method == 'branch_and_bound':
            return self._optimize_branch_and_bound()
        elif method == 'auction':
            return self._optimize_auction()
        elif method == 'hybrid':
            return self._optimize_hybrid()
        else:
            raise ValueError(f"Unknown method: {method}")

    # =========================================================================
    # ГИБРИДНЫЙ ПОДХОД: AUCTION + LOCAL SEARCH
    # =========================================================================

    def _optimize_hybrid(self) -> Tuple[List['TransportPlan'], float]:
        """
        Гибридный алгоритм:
        1. Auction для быстрого первичного решения
        2. Local Search для улучшения (2-opt, swap, insert)
        3. Simulated Annealing для выхода из локальных минимумов
        """
        print("=" * 70)
        print("ГИБРИДНАЯ ОПТИМИЗАЦИЯ: Auction → Local Search → Simulated Annealing")
        print("=" * 70)

        # Фаза 1: Auction (быстрое приближённое решение)
        print("\n[Фаза 1] Auction Algorithm...")
        initial_solution, initial_utility = self._optimize_auction()
        print(f"  Начальное решение: {len(initial_solution)} планов, utility = {initial_utility:.2f}")

        # Фаза 2: Local Search (жадное улучшение)
        print("\n[Фаза 2] Local Search (2-opt, swap, insert)...")
        improved_solution, improved_utility = self._local_search(
            initial_solution,
            initial_utility,
            max_iterations=100
        )
        print(f"  После Local Search: {len(improved_solution)} планов, utility = {improved_utility:.2f}")
        print(
            f"  Улучшение: +{improved_utility - initial_utility:.2f} ({((improved_utility / initial_utility - 1) * 100):.1f}%)")

        # Фаза 3: Simulated Annealing (выход из локальных минимумов)
        print("\n[Фаза 3] Simulated Annealing...")
        final_solution, final_utility = self._simulated_annealing(
            improved_solution,
            improved_utility,
            max_iterations=200,
            initial_temp=10.0,
            cooling_rate=0.95
        )
        print(f"  Финальное решение: {len(final_solution)} планов, utility = {final_utility:.2f}")
        print(
            f"  Общее улучшение: +{final_utility - initial_utility:.2f} ({((final_utility / initial_utility - 1) * 100):.1f}%)")

        print("\n" + "=" * 70 + "\n")

        return final_solution, final_utility

    def _local_search(self, initial_solution: List, initial_utility: float,
                      max_iterations: int = 100) -> Tuple[List, float]:
        """
        Локальный поиск с жадными улучшениями.
        Операторы: swap (замена плана), insert (добавление), remove (удаление).
        """
        current_solution = initial_solution[:]
        current_utility = initial_utility

        for iteration in range(max_iterations):
            improved = False

            # Оператор 1: SWAP - заменить один план на другой
            for i, old_plan in enumerate(current_solution):
                for new_plan in self.plans:
                    if new_plan in current_solution:
                        continue

                    # Проверяем, можно ли заменить
                    test_solution = current_solution[:]
                    test_solution[i] = new_plan

                    if self._is_valid_solution(test_solution):
                        test_utility = self._calculate_solution_utility(test_solution)

                        if test_utility > current_utility + 0.01:  # epsilon для численной стабильности
                            current_solution = test_solution
                            current_utility = test_utility
                            improved = True
                            break

                if improved:
                    break

            if improved:
                continue

            # Оператор 2: INSERT - добавить новый план
            for new_plan in self.plans:
                if new_plan in current_solution:
                    continue

                test_solution = current_solution + [new_plan]

                if self._is_valid_solution(test_solution):
                    test_utility = self._calculate_solution_utility(test_solution)

                    if test_utility > current_utility + 0.01:
                        current_solution = test_solution
                        current_utility = test_utility
                        improved = True
                        break

            if improved:
                continue

            # Оператор 3: REMOVE - удалить план (освободить ресурсы)
            for i in range(len(current_solution)):
                test_solution = current_solution[:i] + current_solution[i + 1:]

                # После удаления попробуем добавить более выгодный план
                best_insert = None
                best_utility = current_utility

                for new_plan in self.plans:
                    if new_plan in test_solution:
                        continue

                    candidate = test_solution + [new_plan]
                    if self._is_valid_solution(candidate):
                        candidate_utility = self._calculate_solution_utility(candidate)
                        if candidate_utility > best_utility:
                            best_utility = candidate_utility
                            best_insert = candidate

                if best_insert and best_utility > current_utility + 0.01:
                    current_solution = best_insert
                    current_utility = best_utility
                    improved = True
                    break

            if not improved:
                break  # Достигли локального максимума

        return current_solution, current_utility

    def _simulated_annealing(self, initial_solution: List, initial_utility: float,
                             max_iterations: int = 200,
                             initial_temp: float = 10.0,
                             cooling_rate: float = 0.95) -> Tuple[List, float]:
        """
        Simulated Annealing для выхода из локальных максимумов.
        """
        import random
        import math

        current_solution = initial_solution[:]
        current_utility = initial_utility

        best_solution = current_solution[:]
        best_utility = current_utility

        temperature = initial_temp

        for iteration in range(max_iterations):
            # Генерируем соседнее решение (случайная модификация)
            neighbor = self._generate_neighbor(current_solution)

            if not self._is_valid_solution(neighbor):
                continue

            neighbor_utility = self._calculate_solution_utility(neighbor)
            delta = neighbor_utility - current_utility

            # Критерий принятия решения
            if delta > 0:
                # Улучшение - всегда принимаем
                current_solution = neighbor
                current_utility = neighbor_utility

                if current_utility > best_utility:
                    best_solution = current_solution[:]
                    best_utility = current_utility
            else:
                # Ухудшение - принимаем с вероятностью exp(delta/T)
                acceptance_prob = math.exp(delta / temperature)
                if random.random() < acceptance_prob:
                    current_solution = neighbor
                    current_utility = neighbor_utility

            # Охлаждение
            temperature *= cooling_rate

        return best_solution, best_utility

    def _generate_neighbor(self, solution: List) -> List:
        """
        Генерирует соседнее решение (для Simulated Annealing).
        Операции: swap случайного плана, добавление, удаление.
        """
        import random

        neighbor = solution[:]
        operation = random.choice(['swap', 'insert', 'remove'])

        if operation == 'swap' and len(neighbor) > 0:
            # Заменить случайный план
            idx = random.randint(0, len(neighbor) - 1)
            available = [p for p in self.plans if p not in neighbor]
            if available:
                neighbor[idx] = random.choice(available)

        elif operation == 'insert':
            # Добавить случайный план
            available = [p for p in self.plans if p not in neighbor]
            if available:
                neighbor.append(random.choice(available))

        elif operation == 'remove' and len(neighbor) > 0:
            # Удалить случайный план
            idx = random.randint(0, len(neighbor) - 1)
            neighbor.pop(idx)

        return neighbor

    # =========================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # =========================================================================

    def _is_valid_solution(self, solution: List) -> bool:
        """Проверяет, валидно ли решение (нет конфликтов по ресурсам)."""
        used_transports = set()
        used_units = set()

        for plan in solution:
            transport_id = plan.transport['id']
            if transport_id in used_transports:
                return False
            used_transports.add(transport_id)

            for passenger in plan.passengers:
                unit_id = passenger['id']
                if unit_id in used_units:
                    return False
                used_units.add(unit_id)

        return True

    def _calculate_solution_utility(self, solution: List) -> float:
        """Вычисляет суммарную utility решения с учётом overkill."""
        target_hp_remaining = {}
        total_utility = 0.0

        # Инициализация HP целей
        for plan in solution:
            target_id = plan.target['id']
            if target_id not in target_hp_remaining:
                target_hp_remaining[target_id] = plan.target['hp']

        # Сортируем планы по utility (сначала выполняем лучшие)
        sorted_solution = sorted(solution, key=lambda p: p.utility, reverse=True)

        for plan in sorted_solution:
            target_id = plan.target['id']
            remaining_hp = target_hp_remaining[target_id]

            total_damage = sum(p['damage'] for p in plan.passengers)
            real_damage = min(total_damage, remaining_hp)

            if total_damage > 0:
                adjusted_utility = plan.utility * (real_damage / total_damage)
            else:
                adjusted_utility = 0.0

            total_utility += adjusted_utility
            target_hp_remaining[target_id] -= real_damage

        return max(0.1, total_utility)

    def _has_conflicts(self, plan, used_resources: Dict) -> bool:
        """Проверяет конфликты плана с использованными ресурсами."""
        transport_id = plan.transport['id']
        if transport_id in used_resources['transports']:
            return True

        for passenger in plan.passengers:
            unit_id = passenger['id']
            if unit_id in used_resources['units']:
                return True

        return False

    def _apply_plan(self, plan, used_resources: Dict) -> Dict:
        """Применяет план к ресурсам (возвращает новое состояние)."""
        new_used = {
            'transports': used_resources['transports'].copy(),
            'units': used_resources['units'].copy(),
            'targets': used_resources['targets'].copy()
        }

        new_used['transports'].add(plan.transport['id'])
        for passenger in plan.passengers:
            new_used['units'].add(passenger['id'])

        target_id = plan.target['id']
        target_remaining_hp = new_used['targets'].get(target_id, plan.target['hp'])

        total_damage = sum(p['damage'] for p in plan.passengers)
        real_damage = min(total_damage, target_remaining_hp)

        new_used['targets'][target_id] = target_remaining_hp - real_damage

        return new_used

    def _calculate_adjusted_utility(self, plan, used_resources: Dict) -> float:
        """Вычисляет скорректированную utility с учётом overkill."""
        target_id = plan.target['id']
        target_remaining_hp = used_resources['targets'].get(target_id, plan.target['hp'])

        total_damage = sum(p['damage'] for p in plan.passengers)
        real_damage = min(total_damage, target_remaining_hp)

        if total_damage > 0:
            return plan.utility * (real_damage / total_damage)
        return 0.0

    def _calculate_upper_bound(self, current_idx: int, current_utility: float,
                               used_resources: Dict, sorted_plans: List) -> float:
        """Оптимистичная оценка для отсечения (upper bound)."""
        remaining_utility = current_utility

        for i in range(current_idx, len(sorted_plans)):
            plan = sorted_plans[i]
            if not self._has_conflicts(plan, used_resources):
                remaining_utility += plan.utility

        return remaining_utility

    # =========================================================================
    # БАЗОВЫЕ АЛГОРИТМЫ (из предыдущей версии)
    # =========================================================================

    def _optimize_branch_and_bound(self) -> Tuple[List['TransportPlan'], float]:
        """Последовательный Branch & Bound (из предыдущей версии)."""
        sorted_plans = sorted(self.plans, key=lambda p: p.utility, reverse=True)

        best_solution = []
        best_utility = 0.0

        def bnb(current_idx: int, current_solution: List,
                current_utility: float, used_resources: Dict):
            nonlocal best_solution, best_utility

            if current_idx >= len(sorted_plans):
                if current_utility > best_utility:
                    best_utility = current_utility
                    best_solution = current_solution[:]
                return

            upper_bound = self._calculate_upper_bound(
                current_idx, current_utility, used_resources, sorted_plans
            )
            if upper_bound <= best_utility:
                return

            plan = sorted_plans[current_idx]

            if not self._has_conflicts(plan, used_resources):
                new_used = self._apply_plan(plan, used_resources)
                adjusted_utility = self._calculate_adjusted_utility(plan, used_resources)

                bnb(current_idx + 1, current_solution + [plan],
                    current_utility + adjusted_utility, new_used)

            bnb(current_idx + 1, current_solution, current_utility, used_resources)

        initial_resources = {
            'transports': set(),
            'units': set(),
            'targets': defaultdict(int)
        }

        bnb(0, [], 0.0, initial_resources)
        return best_solution, best_utility

    def _optimize_auction(self) -> Tuple[List['TransportPlan'], float]:
        """Auction Algorithm (из предыдущей версии)."""
        plans_by_transport = defaultdict(list)
        for plan in self.plans:
            transport_id = plan.transport['id']
            plans_by_transport[transport_id].append(plan)

        selected = []
        used_units = set()
        target_hp_remaining = {}

        for plan in self.plans:
            target_id = plan.target['id']
            if target_id not in target_hp_remaining:
                target_hp_remaining[target_id] = plan.target['hp']

        max_iterations = 10
        epsilon = 0.01

        for iteration in range(max_iterations):
            improved = False

            for transport_id, transport_plans in plans_by_transport.items():
                current_plan = None
                for plan in selected:
                    if plan.transport['id'] == transport_id:
                        current_plan = plan
                        break

                best_plan = None
                best_net_utility = 0.0 if current_plan is None else current_plan.utility

                for plan in transport_plans:
                    units_available = True
                    for passenger in plan.passengers:
                        unit_id = passenger['id']
                        if unit_id in used_units:
                            if current_plan is None:
                                units_available = False
                                break
                            if passenger not in current_plan.passengers:
                                units_available = False
                                break

                    if not units_available:
                        continue

                    target_id = plan.target['id']
                    remaining_hp = target_hp_remaining[target_id]

                    if current_plan and current_plan.target['id'] == target_id:
                        total_dmg_current = sum(p['damage'] for p in current_plan.passengers)
                        remaining_hp += min(total_dmg_current, plan.target['hp'] - remaining_hp)

                    total_damage = sum(p['damage'] for p in plan.passengers)
                    real_damage = min(total_damage, remaining_hp)

                    if total_damage > 0:
                        adjusted_utility = plan.utility * (real_damage / total_damage)
                    else:
                        adjusted_utility = 0.0

                    if adjusted_utility > best_net_utility + epsilon:
                        best_net_utility = adjusted_utility
                        best_plan = plan
                        improved = True

                if best_plan != current_plan:
                    if current_plan:
                        selected.remove(current_plan)
                        for passenger in current_plan.passengers:
                            used_units.discard(passenger['id'])

                        target_id = current_plan.target['id']
                        total_dmg = sum(p['damage'] for p in current_plan.passengers)
                        target_hp_remaining[target_id] += min(total_dmg, current_plan.target['hp'])

                    if best_plan:
                        selected.append(best_plan)
                        for passenger in best_plan.passengers:
                            used_units.add(passenger['id'])

                        target_id = best_plan.target['id']
                        total_dmg = sum(p['damage'] for p in best_plan.passengers)
                        real_dmg = min(total_dmg, target_hp_remaining[target_id])
                        target_hp_remaining[target_id] -= real_dmg

            if not improved:
                break

        total_utility = self._calculate_solution_utility(selected)
        return selected, total_utility
