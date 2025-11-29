import random
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import time

from examples.data_structures import *


class GeneticTransportOptimizer:
    def __init__(self,
                 transport_plans: List['TransportPlan'],
                 population_size: int = 100,
                 generations: int = 200,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elite_size: int = 10):
        """
        Args:
            transport_plans: список всех возможных TransportPlan
            population_size: размер популяции
            generations: количество поколений
            mutation_rate: вероятность мутации
            crossover_rate: вероятность скрещивания
            elite_size: количество лучших особей для элитизма
        """
        self.transport_plans = transport_plans
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size

        # Группируем планы по транспортам
        self.plans_by_transport = self._group_plans_by_transport()
        self.transport_ids = list(self.plans_by_transport.keys())

        # Предварительно вычисляем данные для ускорения
        self.plan_utilities = {}
        self.plan_units = {}

        for transport_id, plans in self.plans_by_transport.items():
            for plan_idx, plan in enumerate(plans):
                plan_key = (transport_id, plan_idx)
                self.plan_utilities[plan_key] = plan.utility
                self.plan_units[plan_key] = self._extract_unit_ids(plan)

    def _group_plans_by_transport(self) -> Dict[str, List['TransportPlan']]:
        """Группирует планы по ID транспортов"""
        plans_by_transport = defaultdict(list)
        for plan in self.transport_plans:
            transport_id = plan.transport[ID_KEY]
            plans_by_transport[transport_id].append(plan)
        return dict(plans_by_transport)

    def _extract_unit_ids(self, plan: 'TransportPlan') -> Set[str]:
        """Извлекает ID всех юнитов из плана"""
        unit_ids = set()

        # Юниты транспорта
        if plan.transport and ID_KEY in plan.transport:
            unit_ids.add(plan.transport[ID_KEY])

        # Пассажиры
        for passenger in plan.passengers:
            if ID_KEY in passenger:
                unit_ids.add(passenger[ID_KEY])

        return unit_ids

    def create_individual(self) -> List[Tuple[str, int]]:
        """
        Создает случайную особь (решение)
        Формат: [(transport_id, plan_index), ...] для выбранных планов
        """
        individual = []
        for transport_id in self.transport_ids:
            plans = self.plans_by_transport[transport_id]
            if plans:
                # Выбираем случайный план или -1 (не использовать транспорт)
                plan_idx = random.randint(-1, len(plans) - 1)
                individual.append((transport_id, plan_idx))
            else:
                individual.append((transport_id, -1))
        return individual

    def calculate_fitness(self, individual: List[Tuple[str, int]]) -> float:
        """Вычисляет приспособленность особи"""
        total_utility = 0.0
        used_units = set()

        for transport_id, plan_idx in individual:
            if plan_idx >= 0:  # План выбран
                plan_key = (transport_id, plan_idx)

                # Проверяем конфликты юнитов
                plan_units = self.plan_units[plan_key]
                if used_units & plan_units:  # Есть пересечение множеств
                    return 0.0  # Жесткое ограничение - конфликт

                used_units.update(plan_units)
                total_utility += self.plan_utilities[plan_key]

        return total_utility

    def batch_calculate_fitness(self, population: List[List[Tuple[str, int]]]) -> List[float]:
        """Пакетное вычисление приспособленности (без параллелизации)"""
        return [self.calculate_fitness(ind) for ind in population]

    def select_parents(self, population: List[List[Tuple[str, int]]], fitnesses: List[float]) -> List[
        List[Tuple[str, int]]]:
        """Турнирный отбор родителей"""
        selected = []
        population_with_fitness = list(zip(population, fitnesses))

        for _ in range(len(population)):
            # Выбираем 3 случайных особи и берем лучшую
            contestants = random.sample(population_with_fitness, min(3, len(population_with_fitness)))
            best_individual = max(contestants, key=lambda x: x[1])[0]
            selected.append(best_individual)
        return selected

    def crossover(self, parent1: List[Tuple[str, int]], parent2: List[Tuple[str, int]]) -> Tuple[
        List[Tuple[str, int]], List[Tuple[str, int]]]:
        """Одноточечное скрещивание"""
        if random.random() < self.crossover_rate and len(parent1) > 1:
            point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutate(self, individual: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Мутация особи"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                transport_id, _ = mutated[i]
                plans = self.plans_by_transport[transport_id]
                if plans:
                    # Случайно меняем план или отключаем транспорт
                    mutated[i] = (transport_id, random.randint(-1, len(plans) - 1))
        return mutated

    def optimize(self, time_limit_ms: int = 100) -> Tuple[List[Tuple[str, int]], float]:
        """Основной метод оптимизации"""
        start_time = time.time()
        time_limit = time_limit_ms / 1000.0  # конвертируем в секунды

        # Инициализация популяции
        population = [self.create_individual() for _ in range(self.population_size)]
        best_individual = None
        best_fitness = 0.0

        for generation in range(self.generations):
            # Проверка времени
            if time.time() - start_time > time_limit:
                print(f"Прерывание по времени: {generation} поколений")
                break

            # Оценка приспособленности
            fitnesses = self.batch_calculate_fitness(population)

            # Обновление лучшего решения
            current_best_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[current_best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()
                print(f"Поколение {generation}: улучшено до {best_fitness}")

            # Элитизм - сохраняем лучших особей
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            elites = [population[i] for i in elite_indices]

            # Отбор родителей
            parents = self.select_parents(population, fitnesses)

            # Создание нового поколения
            new_population = elites.copy()

            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1))
                if len(new_population) < self.population_size:
                    new_population.append(self.mutate(child2))

            population = new_population

            # Ранняя остановка если нашли идеальное решение
            if best_fitness >= self._get_max_possible_utility():
                print(f"Ранняя остановка: достигнута максимальная utility")
                break

        # Если не нашли решение, возвращаем лучшее из последнего поколения
        if best_individual is None and population:
            fitnesses = self.batch_calculate_fitness(population)
            best_idx = np.argmax(fitnesses)
            best_individual = population[best_idx]
            best_fitness = fitnesses[best_idx]

        return best_individual, best_fitness

    def _get_max_possible_utility(self) -> float:
        """Оценка максимально возможной utility (для ранней остановки)"""
        max_utility = 0.0
        for transport_id, plans in self.plans_by_transport.items():
            if plans:
                max_plan_utility = max(plan.utility for plan in plans)
                max_utility += max_plan_utility
        return max_utility

    def decode_solution(self, individual: List[Tuple[str, int]]) -> List['TransportPlan']:
        """Декодирует решение в список TransportPlan"""
        solution = []
        for transport_id, plan_idx in individual:
            if plan_idx >= 0:
                plan = self.plans_by_transport[transport_id][plan_idx]
                solution.append(plan)
        return solution


# Основная функция для использования
def optimize_transport_plans(
        transport_plans: List['TransportPlan'],
        time_limit_ms: int = 50,
        population_size: int = 50
) -> List['TransportPlan']:
    """
    Основная функция для оптимизации транспортных планов

    Args:
        transport_plans: список всех возможных TransportPlan
        time_limit_ms: ограничение по времени в миллисекундах
        population_size: размер популяции генетического алгоритма

    Returns:
        Список оптимальных TransportPlan для выполнения
    """

    if not transport_plans:
        return []

    # Определяем сложность задачи
    num_transports = len(set(plan.transport[ID_KEY] for plan in transport_plans))
    total_plans = len(transport_plans)

    print(f"Оптимизация: {num_transports} транспортов, {total_plans} планов")

    # Настраиваем параметры в зависимости от размера задачи
    if num_transports > 20 or total_plans > 100:
        population_size = min(100, population_size * 2)
        generations = 100
        print("Используем быстрые настройки для большой задачи")
    else:
        generations = 200
        print("Используем детальные настройки для маленькой задачи")

    optimizer = GeneticTransportOptimizer(
        transport_plans=transport_plans,
        population_size=population_size,
        generations=generations,
        mutation_rate=0.15,
        crossover_rate=0.7,
        elite_size=5
    )

    best_individual, best_fitness = optimizer.optimize(time_limit_ms)

    if best_individual is None:
        print("Решение не найдено")
        return []

    solution = optimizer.decode_solution(best_individual)
    print(f"Найдено решение: utility={best_fitness}, планов={len(solution)}")

    return solution


# Утилиты для работы с результатами
def validate_solution(solution: List['TransportPlan']) -> Tuple[bool, str]:
    """Проверяет валидность решения (отсутствие конфликтов юнитов)"""
    used_units = set()

    for plan in solution:
        # Проверяем транспорт
        if plan.transport and ID_KEY in plan.transport:
            unit_id = plan.transport[ID_KEY]
            if unit_id in used_units:
                return False, f"Конфликт транспорта: {unit_id}"
            used_units.add(unit_id)

        # Проверяем пассажиров
        for passenger in plan.passengers:
            if ID_KEY in passenger:
                unit_id = passenger[ID_KEY]
                if unit_id in used_units:
                    return False, f"Конфликт пассажира: {unit_id}"
                used_units.add(unit_id)

    return True, "Решение валидно"


def calculate_total_utility(solution: List['TransportPlan']) -> float:
    """Вычисляет общую utility решения"""
    return sum(plan.utility for plan in solution)


def get_solution_statistics(solution: List['TransportPlan']) -> Dict[str, any]:
    """Возвращает статистику по решению"""
    total_utility = calculate_total_utility(solution)
    transports_used = len(solution)
    total_passengers = sum(len(plan.passengers) for plan in solution)
    targets = set(plan.target.get(ID_KEY, str(plan.target)) for plan in solution)

    is_valid, validity_msg = validate_solution(solution)

    return {
        'total_utility': total_utility,
        'transports_used': transports_used,
        'total_passengers': total_passengers,
        'unique_targets': len(targets),
        'is_valid': is_valid,
        'validity_message': validity_msg
    }


# Альтернативная упрощенная версия для очень больших наборов данных
def greedy_optimize_transport_plans(transport_plans: List['TransportPlan']) -> List['TransportPlan']:
    """
    Жадный алгоритм для быстрой оптимизации (альтернатива генетическому)
    Полезно когда генетический алгоритм слишком медленный
    """
    # Сортируем планы по убыванию utility
    sorted_plans = sorted(transport_plans, key=lambda x: x.utility, reverse=True)

    solution = []
    used_units = set()
    used_transports = set()

    for plan in sorted_plans:
        transport_id = plan.transport[ID_KEY]

        # Проверяем, не использован ли уже транспорт
        if transport_id in used_transports:
            continue

        # Проверяем конфликты юнитов
        plan_units = set()
        if plan.transport and ID_KEY in plan.transport:
            plan_units.add(plan.transport[ID_KEY])
        for passenger in plan.passengers:
            if ID_KEY in passenger:
                plan_units.add(passenger[ID_KEY])

        if not (used_units & plan_units):  # Нет конфликтов
            solution.append(plan)
            used_units.update(plan_units)
            used_transports.add(transport_id)

    return solution