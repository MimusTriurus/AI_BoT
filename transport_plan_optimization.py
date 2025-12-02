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

    # =========================================================================
    # ГИБРИДНЫЙ ПОДХОД: AUCTION + LOCAL SEARCH
    # =========================================================================

    def optimize_hybrid(self) -> Tuple[List['TransportPlan'], float]:
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

    def _check_auction_candidate_conflict(
            self,
            candidate_plan: 'TransportPlan',
            global_used_units: set,
            global_occupied_hexes: set,
            my_held_units: set,  # Ресурсы, которые можно переиспользовать (от текущего плана)
            my_occupied_hexes: set  # Гексы, которые я сейчас занимаю (и могу переиспользовать)
    ) -> bool:
        """
        Проверяет, конфликтует ли кандидатский план с ГЛОБАЛЬНЫМ состоянием,
        исключая ресурсы, которые уже использует этот транспорт (my_...).
        Возвращает True, если есть конфликт (план невалиден).
        """

        # 1. КОНФЛИКТ ЮНИТОВ
        for passenger in candidate_plan.passengers:
            u_id = passenger['id']
            # Конфликт, если юнит занят (в global_used_units) И он не является моим юнитом (not in my_held_units)
            if u_id in global_used_units and u_id not in my_held_units:
                return True  # Конфликт: Юнит занят другим транспортом

        # 2. КОНФЛИКТ ПОЗИЦИЙ (Транспорт + Десант) 🗺️
        for hex_pos in candidate_plan.occupied_hexes_set:
            # Конфликт, если гекс занят (в global_occupied_hexes) И он не является моим текущим гексом
            if hex_pos in global_occupied_hexes and hex_pos not in my_occupied_hexes:
                return True  # Конфликт: Гекс занят другим планом

        return False  # Конфликтов нет

    def _local_search(
            self,
            initial_solution: List,
            initial_utility: float,
            max_iterations: int = 100
    ) -> Tuple[List, float]:
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

    def _simulated_annealing(
            self,
            initial_solution: List,
            initial_utility: float,
            max_iterations: int = 200,
            initial_temp: float = 10.0,
            cooling_rate: float = 0.95
    ) -> Tuple[List, float]:
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

    def _is_valid_solution(self, solution: List['TransportPlan']) -> bool:
        """
        Проверяет, является ли данное полное решение валидным,
        учитывая: Транспорты, Юниты, и ВСЕ занятые гексы (транспорт + десант).
        """
        used_transports = set()
        used_units = set()
        # Теперь отслеживает ВСЕ занятые гексы (позиция транспорта + десант)
        all_occupied_hexes = set()

        for plan in solution:
            # 1. КОНФЛИКТ ТРАНСПОРТОВ (по ID)
            transport_id = plan.transport['id']
            if transport_id in used_transports:
                return False
            used_transports.add(transport_id)

            # 2. КОНФЛИКТ ЮНИТОВ (по ID)
            for passenger in plan.passengers:
                unit_id = passenger['id']
                if unit_id in used_units:
                    return False
                used_units.add(unit_id)

            # 3. КОНФЛИКТ ПОЗИЦИЙ (Транспорт + Десант) 🗺️
            # Проверяем, есть ли пересечение между гексами, которые займет план, и уже занятыми
            if not plan.occupied_hexes_set.isdisjoint(all_occupied_hexes):
                return False  # Пересечение с уже занятыми клетками

            all_occupied_hexes.update(plan.occupied_hexes_set)

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

    def _apply_plan(self, plan, used_resources: Dict) -> Dict:
        """
        Применяет план к ресурсам (возвращает новое состояние).
        """
        new_used = {
            'transports': used_resources['transports'].copy(),
            'units': used_resources['units'].copy(),
            'targets': used_resources['targets'].copy(),
            'occupied_hexes': used_resources['occupied_hexes'].copy()  # <--- Копируем множество гексов
        }

        # Занимаем транспорт
        new_used['transports'].add(plan.transport['id'])

        # Занимаем юнитов
        for passenger in plan.passengers:
            new_used['units'].add(passenger['id'])

        # Занимаем ВСЕ гексы (Транспорт + Десант) <--- НОВОЕ
        new_used['occupied_hexes'].update(plan.occupied_hexes_set)

        # Обновляем HP цели
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

    def _has_conflicts(self, plan, used_resources: Dict) -> bool:
        """
        Проверяет конфликты плана с использованными ресурсами:
        1. Транспорт (ID).
        2. Юниты (ID).
        3. ВСЕ Пространственные конфликты (Транспорт + Десант). 🗺️
        """

        # 1. Конфликт Транспортов (ID)
        transport_id = plan.transport['id']
        if transport_id in used_resources['transports']:
            return True

        # 2. Конфликт Юнитов (ID)
        for passenger in plan.passengers:
            unit_id = passenger['id']
            if unit_id in used_resources['units']:
                return True

        # 3. Конфликт Позиций (Транспорт + Десант) 🗺️
        # plan.occupied_hexes_set содержит path[-1] И все точки выгрузки
        # Проверяем пересечение с уже занятыми гексами
        if not plan.occupied_hexes_set.isdisjoint(used_resources['occupied_hexes']):
            return True

        return False

    def _calculate_upper_bound(
            self,
            current_idx: int,
            current_utility: float,
            used_resources: Dict,
            sorted_plans: List
    ) -> float:
        """
        Оптимистичная оценка для отсечения (upper bound).
        Теперь учитывает, что планы не могут занимать одни и те же клетки выгрузки.
        """
        remaining_utility = current_utility

        # Жадный добор оставшихся планов
        # (Обратите внимание: это упрощенная оценка. Мы проверяем конфликт только
        # с УЖЕ принятыми ресурсами, но не проверяем конфликты между кандидатами в хвосте.
        # Это допустимо для Upper Bound, так как это релаксация задачи).
        for i in range(current_idx, len(sorted_plans)):
            plan = sorted_plans[i]
            if not self._has_conflicts(plan, used_resources):
                remaining_utility += plan.utility

        return remaining_utility

    def optimize_branch_and_bound(
            self,
            initial_bnb_solution: List['TransportPlan'],
            initial_bnb_utility: float
    ) -> Tuple[List['TransportPlan'], float]:
        """
        Последовательный Branch & Bound с полной проверкой пространственных конфликтов.
        """
        # Сортировка по убыванию полезности для быстрого нахождения хороших решений
        sorted_plans = sorted(self.plans, key=lambda p: p.utility, reverse=True)

        best_solution = initial_bnb_solution[:]
        best_utility = initial_bnb_utility

        def bnb(
                current_idx: int,
                current_solution: List,
                current_utility: float,
                used_resources: Dict
        ):
            nonlocal best_solution, best_utility

            # Базовый случай: Прошли все планы
            if current_idx >= len(sorted_plans):
                if current_utility > best_utility:
                    best_utility = current_utility
                    best_solution = current_solution[:]
                return

            # Отсечение (Pruning)
            upper_bound = self._calculate_upper_bound(
                current_idx, current_utility, used_resources, sorted_plans
            )
            if upper_bound <= best_utility:
                return

            plan = sorted_plans[current_idx]

            # ВЕТВЬ 1: Берем текущий план (если нет конфликтов)
            if not self._has_conflicts(plan, used_resources):
                new_used = self._apply_plan(plan, used_resources)

                # Важно: пересчитываем utility с учетом реального HP (Overkill)
                adjusted_utility = self._calculate_adjusted_utility(plan, used_resources)

                bnb(current_idx + 1,
                    current_solution + [plan],
                    current_utility + adjusted_utility,
                    new_used)

            # ВЕТВЬ 2: Пропускаем текущий план
            bnb(current_idx + 1, current_solution, current_utility, used_resources)

        initial_resources = {
            'transports': set(),
            'units': set(),
            'targets': defaultdict(int),
            'occupied_hexes': set()  # <--- ИСПОЛЬЗУЕМ МНОЖЕСТВО ВСЕХ ГЕКСОВ
        }

        bnb(0, [], 0.0, initial_resources)

        return best_solution, best_utility


    def _optimize_auction(self) -> Tuple[List['TransportPlan'], float]:
        """
        Auction Algorithm with full Spatial Conflict Resolution (DRY version).
        Optimizes for: Net Utility, Unit Availability, Target HP, and Spatial Conflicts.
        """

        # Группировка планов по ID транспорта
        plans_by_transport = defaultdict(list)
        for plan in self.plans:
            plans_by_transport[plan.transport['id']].append(plan)

        # --- ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ ---
        selected_plans_map: Dict[str, 'TransportPlan'] = {}
        used_units: set = set()
        # ИЗМЕНЕНО: Отслеживаем ВСЕ занятые гексы (транспорт + десант)
        all_occupied_hexes: set = set()

        target_hp_remaining = {}
        for plan in self.plans:
            t_id = plan.target['id']
            if t_id not in target_hp_remaining:
                target_hp_remaining[t_id] = plan.target['hp']

        # (TransportID, TargetID) -> RealDamage
        real_damage_ledger = {}

        max_iterations = 10
        epsilon = 0.001

        for iteration in range(max_iterations):
            improved = False

            # Перебираем транспорты (агентов аукциона)
            for transport_id, candidate_plans in plans_by_transport.items():

                # 1. Анализ текущего плана (что можно переиспользовать)
                current_plan = selected_plans_map.get(transport_id)

                current_net_utility = 0.0
                current_real_damage = 0.0

                # Ресурсы, которые можно ПЕРЕИСПОЛЬЗОВАТЬ
                my_held_units = set()
                my_occupied_hexes = set()

                if current_plan:
                    # Получаем реальный урон из леджера
                    current_real_damage = real_damage_ledger.get((transport_id, current_plan.target['id']), 0.0)
                    total_pot = sum(p['damage'] for p in current_plan.passengers)

                    if total_pot > 0:
                        ratio = current_real_damage / total_pot
                        current_net_utility = current_plan.utility * ratio

                    # Запоминаем свои текущие ресурсы для переиспользования:
                    my_held_units = {p['id'] for p in current_plan.passengers}
                    my_occupied_hexes = current_plan.occupied_hexes_set

                # Инициализируем лучшего кандидата текущим состоянием
                best_plan = current_plan
                best_net_utility = current_net_utility
                best_real_damage_forecast = current_real_damage

                # 2. Поиск лучшего кандидата
                for plan in candidate_plans:

                    # --- ПРОВЕРКА КОНФЛИКТОВ (DRY) ---
                    if self._check_auction_candidate_conflict(
                            plan,
                            used_units,
                            all_occupied_hexes,
                            my_held_units,
                            my_occupied_hexes
                    ):
                        continue  # План невалиден

                    # --- Б. В, Г. РАСЧЕТ UTILITY (Уникальная логика Аукциона) ---
                    target_id = plan.target['id']
                    hp_snapshot = target_hp_remaining[target_id]

                    # Если я уже атакую эту цель, временно возвращаем мой урон для честного расчета
                    if current_plan and current_plan.target['id'] == target_id:
                        hp_snapshot += real_damage_ledger.get((transport_id, target_id), 0.0)

                    total_damage_potential = sum(p['damage'] for p in plan.passengers)
                    real_damage_forecast = min(total_damage_potential, hp_snapshot)

                    if total_damage_potential > 0:
                        adjusted_utility = plan.utility * (real_damage_forecast / total_damage_potential)
                    else:
                        adjusted_utility = 0.0

                    # --- Д. СРАВНЕНИЕ ---
                    if adjusted_utility > best_net_utility + epsilon:
                        best_net_utility = adjusted_utility
                        best_plan = plan
                        best_real_damage_forecast = real_damage_forecast
                        improved = True

                # 3. ПРИМЕНЕНИЕ (COMMIT/ROLLBACK)
                if best_plan != current_plan:

                    # ROLLBACK (Откат старого плана)
                    if current_plan:
                        # 1. Возвращаем HP
                        prev_tid = current_plan.target['id']
                        restored_dmg = real_damage_ledger.pop((transport_id, prev_tid), 0.0)
                        target_hp_remaining[prev_tid] += restored_dmg

                        # 2. Освобождаем юнитов
                        for p in current_plan.passengers:
                            used_units.discard(p['id'])

                        # 3. Освобождаем ВСЕ гексы (транспорт + десант) <--- ОБНОВЛЕНО
                        for hex_pos in current_plan.occupied_hexes_set:
                            all_occupied_hexes.discard(hex_pos)

                        del selected_plans_map[transport_id]

                    # COMMIT (Применение нового плана)
                    if best_plan:
                        new_tid = best_plan.target['id']

                        # 1. Занимаем юнитов
                        for p in best_plan.passengers:
                            used_units.add(p['id'])

                        # 2. Занимаем ВСЕ гексы (транспорт + десант) <--- ОБНОВЛЕНО
                        all_occupied_hexes.update(best_plan.occupied_hexes_set)

                        # 3. Отнимаем HP
                        actual_dmg = min(best_real_damage_forecast, target_hp_remaining[new_tid])
                        target_hp_remaining[new_tid] -= actual_dmg
                        real_damage_ledger[(transport_id, new_tid)] = actual_dmg

                        selected_plans_map[transport_id] = best_plan

            if not improved:
                break

        selected_list = list(selected_plans_map.values())
        total_utility = self._calculate_solution_utility(selected_list)
        return selected_list, total_utility
