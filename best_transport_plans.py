import random
from typing import List, Dict, Tuple, Set, Optional
import time
from dataclasses import dataclass
from copy import deepcopy
from data_structures import *

@dataclass
class Solution:
    """Класс для хранения решения"""
    plans: List['TransportPlan']
    total_utility: float
    used_transports: Set[str]
    used_units: Set[str]


def select_optimal_plans_exact(plans: List['TransportPlan'], timeout: int = 30) -> List['TransportPlan']:
    """
    Точный алгоритм с использованием backtracking с отсечениями.

    Args:
        plans: Список TransportPlan для оптимизации
        timeout: Максимальное время выполнения в секундах

    Returns:
        Оптимальный список TransportPlan
    """
    if not plans:
        return []

    # Сортируем планы по убыванию utility для лучшего отсечения
    sorted_plans = sorted(plans, key=lambda x: x.utility, reverse=True)

    best_solution = Solution([], 0.0, set(), set())
    start_time = time.time()

    def backtrack(idx: int, current: Solution):
        nonlocal best_solution

        # Проверка таймаута
        if time.time() - start_time > timeout:
            return

        # Если дошли до конца или не можем улучшить решение
        if idx >= len(sorted_plans):
            if current.total_utility > best_solution.total_utility:
                best_solution = deepcopy(current)
            return

        # Отсечение: если даже взяв все оставшиеся планы, не можем улучшить
        remaining_utility = sum(p.utility for p in sorted_plans[idx:])
        if current.total_utility + remaining_utility <= best_solution.total_utility:
            return

        current_plan = sorted_plans[idx]
        transport_id = current_plan.transport[ID_KEY]
        unit_ids = {p[ID_KEY] for p in current_plan.passengers}

        # Проверяем, можно ли добавить текущий план
        can_add = (transport_id not in current.used_transports and
                   not (unit_ids & current.used_units))

        # Ветвь с добавлением плана
        if can_add:
            # Добавляем план
            current.plans.append(current_plan)
            current.total_utility += current_plan.utility
            current.used_transports.add(transport_id)
            current.used_units.update(unit_ids)

            backtrack(idx + 1, current)

            # Откатываем изменения
            current.plans.pop()
            current.total_utility -= current_plan.utility
            current.used_transports.remove(transport_id)
            current.used_units.difference_update(unit_ids)

        # Ветвь без добавления плана
        backtrack(idx + 1, current)

    backtrack(0, best_solution)
    return best_solution.plans


def select_optimal_plans_beam_search(plans: List['TransportPlan'],
                                     beam_width: int = 100,
                                     max_iterations: int = 1000) -> List['TransportPlan']:
    """
    Алгоритм поиска по лучу (Beam Search) для больших наборов данных.

    Args:
        plans: Список TransportPlan для оптимизации
        beam_width: Ширина луча (количество лучших решений на каждом шаге)
        max_iterations: Максимальное количество итераций

    Returns:
        Лучший найденный список TransportPlan
    """
    if not plans:
        return []

    # Сортируем планы по убыванию utility
    sorted_plans = sorted(plans, key=lambda x: x.utility, reverse=True)

    # Инициализируем лучшие решения
    best_solutions = [Solution([], 0.0, set(), set())]

    for plan in sorted_plans[:max_iterations]:
        transport_id = plan.transport[ID_KEY]
        unit_ids = {p[ID_KEY] for p in plan.passengers}

        new_solutions = []

        for sol in best_solutions:
            # Рассматриваем решение без добавления текущего плана
            new_solutions.append(deepcopy(sol))

            # Рассматриваем решение с добавлением текущего плана (если возможно)
            if (transport_id not in sol.used_transports and
                    not (unit_ids & sol.used_units)):
                new_sol = deepcopy(sol)
                new_sol.plans.append(plan)
                new_sol.total_utility += plan.utility
                new_sol.used_transports.add(transport_id)
                new_sol.used_units.update(unit_ids)
                new_solutions.append(new_sol)

        # Сортируем по utility и выбираем beam_width лучших
        best_solutions = sorted(new_solutions,
                                key=lambda x: x.total_utility,
                                reverse=True)[:beam_width]

    return best_solutions[0].plans if best_solutions else []


def select_optimal_plans_genetic(plans: List['TransportPlan'],
                                 population_size: int = 50,
                                 generations: int = 100,
                                 mutation_rate: float = 0.1) -> List['TransportPlan']:
    """
    Генетический алгоритм для оптимизации выбора планов.

    Args:
        plans: Список TransportPlan для оптимизации
        population_size: Размер популяции
        generations: Количество поколений
        mutation_rate: Вероятность мутации

    Returns:
        Лучший найденный список TransportPlan
    """
    if not plans or len(plans) == 0:
        return []

    def calculate_fitness(solution_mask: List[bool]) -> float:
        """Рассчитывает fitness решения"""
        total_utility = 0
        used_transports = set()
        used_units = set()

        for i, selected in enumerate(solution_mask):
            if selected:
                plan = plans[i]
                transport_id = plan.transport[ID_KEY]
                unit_ids = {p[ID_KEY] for p in plan.passengers}

                # Проверяем конфликты
                if (transport_id in used_transports or
                        (unit_ids & used_units)):
                    return 0  # Недопустимое решение

                total_utility += plan.utility
                used_transports.add(transport_id)
                used_units.update(unit_ids)

        return total_utility

    def create_random_solution() -> List[bool]:
        """Создает случайное решение"""
        solution = [False] * len(plans)
        used_transports = set()
        used_units = set()

        # Пытаемся добавить планы в случайном порядке
        indices = list(range(len(plans)))
        random.shuffle(indices)

        for idx in indices:
            plan = plans[idx]
            transport_id = plan.transport[ID_KEY]
            unit_ids = {p[ID_KEY] for p in plan.passengers}

            if (transport_id not in used_transports and
                    not (unit_ids & used_units)):
                solution[idx] = True
                used_transports.add(transport_id)
                used_units.update(unit_ids)

        return solution

    def crossover(parent1: List[bool], parent2: List[bool]) -> List[bool]:
        """Скрещивание двух родителей"""
        child = [False] * len(plans)
        used_transports = set()
        used_units = set()

        # Берем гены от обоих родителей
        for i in range(len(plans)):
            if parent1[i] or parent2[i]:
                plan = plans[i]
                transport_id = plan.transport[ID_KEY]
                unit_ids = {p[ID_KEY] for p in plan.passengers}

                if (transport_id not in used_transports and
                        not (unit_ids & used_units)):
                    child[i] = True
                    used_transports.add(transport_id)
                    used_units.update(unit_ids)

        return child

    def mutate(solution: List[bool]) -> List[bool]:
        """Мутация решения"""
        mutated = solution.copy()

        if random.random() < mutation_rate:
            # Пытаемся добавить случайный план
            idx = random.randint(0, len(plans) - 1)
            if not mutated[idx]:
                plan = plans[idx]
                transport_id = plan.transport[ID_KEY]
                unit_ids = {p[ID_KEY] for p in plan.passengers}

                # Проверяем конфликты
                used_transports = set()
                used_units = set()
                for i, selected in enumerate(mutated):
                    if selected:
                        used_transports.add(plans[i].transport[ID_KEY])
                        used_units.update({p[ID_KEY] for p in plans[i].passengers})

                if (transport_id not in used_transports and
                        not (unit_ids & used_units)):
                    mutated[idx] = True

        return mutated

    # Инициализация популяции
    population = [create_random_solution() for _ in range(population_size)]

    for generation in range(generations):
        # Оценка fitness
        fitness_scores = [calculate_fitness(ind) for ind in population]

        # Селекция (турнирная)
        new_population = []
        for _ in range(population_size):
            # Выбираем 2 случайных индивида
            idx1, idx2 = random.sample(range(population_size), 2)
            if fitness_scores[idx1] > fitness_scores[idx2]:
                new_population.append(population[idx1])
            else:
                new_population.append(population[idx2])

        # Скрещивание и мутация
        population = []
        for i in range(0, population_size, 2):
            if i + 1 < population_size:
                child1 = crossover(new_population[i], new_population[i + 1])
                child2 = crossover(new_population[i + 1], new_population[i])
                population.append(mutate(child1))
                population.append(mutate(child2))

        # Элитизм - сохраняем лучшее решение
        best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        population[0] = new_population[best_idx]

    # Находим лучшее решение
    best_solution = max(population, key=calculate_fitness)

    # Преобразуем в список планов
    result = []
    for i, selected in enumerate(best_solution):
        if selected:
            result.append(plans[i])

    return result


def optimize_transport_operations(plans: List['TransportPlan'],
                                  method: str = "auto",
                                  **kwargs) -> List['TransportPlan']:
    """
    Основная функция оптимизации транспортных операций.

    Args:
        plans: Список всех возможных TransportPlan
        method: Метод оптимизации ("exact", "beam", "genetic", "greedy", "auto")
        **kwargs: Дополнительные параметры для методов

    Returns:
        Список выбранных TransportPlan с максимальной utility
    """
    if not plans:
        return []

    # Автоматический выбор метода в зависимости от размера задачи
    if method == "auto":
        if len(plans) <= 20:
            method = "exact"
        elif len(plans) <= 100:
            method = "beam"
        else:
            method = "genetic"

    if method == "exact":
        return select_optimal_plans_exact(plans, **kwargs)
    elif method == "beam":
        return select_optimal_plans_beam_search(plans, **kwargs)
    elif method == "genetic":
        return select_optimal_plans_genetic(plans, **kwargs)
    elif method == "greedy":
        # Используем улучшенный жадный алгоритм
        return select_optimal_plans_greedy_improved(plans)
    else:
        raise ValueError(f"Unknown method: {method}")


def select_optimal_plans_greedy_improved(plans: List['TransportPlan']) -> List['TransportPlan']:
    """
    Улучшенный жадный алгоритм с несколькими эвристиками.
    """
    if not plans:
        return []

    # Пробуем разные стратегии сортировки
    strategies = [
        # Сортировка по utility
        lambda x: x.utility,
        # Сортировка по utility на единицу capacity
        lambda x: x.utility / (len(x.passengers) + 1),
        # Сортировка по utility на транспорт
        lambda x: x.utility / (1 + len(x.passengers) * 0.1),
    ]

    best_result = []
    best_utility = 0

    for strategy in strategies:
        sorted_plans = sorted(plans, key=strategy, reverse=True)

        selected_plans = []
        used_transports = set()
        used_units = set()
        total_utility = 0

        for plan in sorted_plans:
            transport_id = plan.transport[ID_KEY]
            plan_unit_ids = {passenger[ID_KEY] for passenger in plan.passengers}

            if (transport_id not in used_transports and
                    not (plan_unit_ids & used_units)):
                selected_plans.append(plan)
                used_transports.add(transport_id)
                used_units.update(plan_unit_ids)
                total_utility += plan.utility

        if total_utility > best_utility:
            best_utility = total_utility
            best_result = selected_plans

    return best_result