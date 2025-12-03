import math
import random
from typing import List, Tuple, Dict, Any
from w9_pathfinding.envs import HexGrid, HexLayout
from w9_pathfinding.pf import IDAStar, AStar
from w9_pathfinding.mapf import CBS, SpaceTimeAStar, ReservationTable, MultiAgentAStar

pf: AStar
grid: HexGrid

def kmeans_clustering_units_only(units: List[Dict], grid: Any, pf: Any, k: int = None, max_iters: int = 100,
                                 min_cluster_size: int = 2) -> Dict[Tuple, List[Dict]]:
    """
    Кластеризация юнитов на гексогональном поле методом K-Means без учета целей

    Args:
        units: Список юнитов с позициями
        grid: Объект гексогонального поля
        pf: Pathfinder (A*)
        k: Количество кластеров (если None, вычисляется автоматически)
        max_iters: Максимальное количество итераций
        min_cluster_size: Минимальный размер кластера

    Returns:
        Словарь {центроид_кластера: список_юнитов_в_кластере}
    """

    def hex_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Вычисление расстояния между двумя гексами через A*"""
        try:
            path = pf.find_path(pos1, pos2)
            if path:
                return len(path) - 1  # Длина пути без стартовой позиции
            else:
                # Если путь не найден, используем манхэттенское расстояние
                return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        except:
            # Запасной вариант если A* не работает
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def find_medoid(points: List[Tuple]) -> Tuple:
        """Находит точку с минимальной суммой расстояний до всех других точек (медоид)"""
        if not points:
            return None

        min_total_distance = float('inf')
        best_point = points[0]

        for candidate in points:
            total_distance = 0
            for point in points:
                if point != candidate:
                    total_distance += hex_distance(candidate, point)

            if total_distance < min_total_distance:
                min_total_distance = total_distance
                best_point = candidate

        return best_point

    # Извлекаем позиции юнитов
    unit_positions = [unit[START_KEY] for unit in units]

    # Если нет юнитов, возвращаем пустой словарь
    if not unit_positions:
        return {}

    # Если k не задано, вычисляем на основе количества юнитов
    if k is None:
        k = calculate_optimal_k_units(unit_positions, grid, pf, min_cluster_size)

    # Если k получилось больше чем юнитов, корректируем
    k = min(k, len(unit_positions))

    # Инициализация центроидов методом K-Means++
    def kmeans_plus_plus(points: List[Tuple], k: int) -> List[Tuple]:
        if k >= len(points):
            return points.copy()

        centroids = [random.choice(points)]

        for _ in range(1, k):
            distances = []
            for point in points:
                min_dist = min(hex_distance(point, centroid) for centroid in centroids)
                distances.append(min_dist)

            # Преобразуем расстояния в вероятности
            total = sum(distances)
            if total > 0:
                probabilities = [d / total for d in distances]
            else:
                probabilities = [1.0 / len(points)] * len(points)

            # Выбираем следующий центроид
            next_centroid = random.choices(points, weights=probabilities)[0]
            centroids.append(next_centroid)

        return centroids

    # Инициализируем центроиды
    centroids = kmeans_plus_plus(unit_positions, k)

    # Основной цикл K-Means
    for iteration in range(max_iters):
        # Шаг 1: Назначение точек кластерам
        clusters = {centroid: [] for centroid in centroids}

        for unit in units:
            unit_pos = unit[START_KEY]
            min_distance = float('inf')
            closest_centroid = None

            for centroid in centroids:
                distance = hex_distance(unit_pos, centroid)
                if distance < min_distance:
                    min_distance = distance
                    closest_centroid = centroid

            if closest_centroid is not None:
                clusters[closest_centroid].append(unit)

        # Шаг 2: Пересчет центроидов (находим медоиды для каждого кластера)
        new_centroids = []
        empty_clusters = []

        for centroid, cluster_units in clusters.items():
            if len(cluster_units) < min_cluster_size:
                # Слишком маленький кластер - помечаем для удаления
                empty_clusters.append(centroid)
                continue

            # Находим медоид (точку с минимальной суммой расстояний)
            cluster_positions = [unit[START_KEY] for unit in cluster_units]
            new_centroid = find_medoid(cluster_positions)
            new_centroids.append(new_centroid)

        # Если есть пустые кластеры, перераспределяем их точки
        if empty_clusters:
            for empty_centroid in empty_clusters:
                empty_units = clusters[empty_centroid]
                for unit in empty_units:
                    unit_pos = unit[START_KEY]
                    min_distance = float('inf')
                    closest_centroid = None

                    for centroid in new_centroids:
                        distance = hex_distance(unit_pos, centroid)
                        if distance < min_distance:
                            min_distance = distance
                            closest_centroid = centroid

                    # Находим кластер с этим центроидом и добавляем юнита
                    for i, centroid in enumerate(new_centroids):
                        if centroid == closest_centroid:
                            # Обновляем кластер
                            cluster_key = list(clusters.keys())[
                                list(clusters.values()).index(clusters[centroid])] if centroid in clusters else centroid
                            if cluster_key in clusters:
                                clusters[cluster_key].append(unit)

        # Проверка сходимости
        centroids_set_old = set(centroids)
        centroids_set_new = set(new_centroids)

        if centroids_set_old == centroids_set_new:
            break

        centroids = new_centroids

    # Формируем финальный результат
    return create_final_clusters(centroids, units, min_cluster_size)


def calculate_optimal_k_units(unit_positions: List[Tuple], grid: Any, pf: Any, min_cluster_size: int = 2) -> int:
    """Вычисляет оптимальное количество кластеров для юнитов методом локтя"""
    if len(unit_positions) <= 1:
        return 1

    def hex_distance_simple(pos1, pos2):
        """Упрощенная функция расстояния для внутренних вычислений"""
        try:
            path = pf.find_path(pos1, pos2)
            return len(path) - 1 if path else abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        except:
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    # Максимальное количество кластеров - не больше чем юнитов делённых на минимальный размер кластера
    max_k = min(5, len(unit_positions) // min_cluster_size)
    max_k = max(1, max_k)  # Как минимум 1 кластер

    distortions = []
    k_values = range(1, max_k + 1)

    for k in k_values:
        if k == 1:
            # Для одного кластера вычисляем суммарное квадратичное отклонение
            centroid = find_medoid_simple(unit_positions, hex_distance_simple)
            distortion = sum(hex_distance_simple(p, centroid) ** 2 for p in unit_positions)
        else:
            # Упрощенный K-Means для вычисления distortion
            temp_centroids = random.sample(unit_positions, k)

            for _ in range(10):  # Несколько итераций
                # Назначение точек кластерам
                cluster_assignments = {}
                for p in unit_positions:
                    min_dist = float('inf')
                    closest_centroid = temp_centroids[0]
                    for c in temp_centroids:
                        dist = hex_distance_simple(p, c)
                        if dist < min_dist:
                            min_dist = dist
                            closest_centroid = c
                    if closest_centroid not in cluster_assignments:
                        cluster_assignments[closest_centroid] = []
                    cluster_assignments[closest_centroid].append(p)

                # Пересчет центроидов (медоидов)
                new_centroids = []
                for centroid, cluster_points in cluster_assignments.items():
                    if cluster_points:
                        # Находим медоид
                        new_centroid = find_medoid_simple(cluster_points, hex_distance_simple)
                        new_centroids.append(new_centroid)
                    else:
                        new_centroids.append(centroid)

                temp_centroids = new_centroids

            # Вычисляем distortion
            distortion = 0
            for centroid, cluster_points in cluster_assignments.items():
                for p in cluster_points:
                    distortion += hex_distance_simple(p, centroid) ** 2

        distortions.append(distortion)

    # Находим "локоть" - точку, где уменьшение distortion замедляется
    if len(distortions) > 2:
        # Вычисляем относительные уменьшения
        reductions = []
        for i in range(1, len(distortions)):
            if distortions[i - 1] > 0:
                reduction_percent = (distortions[i - 1] - distortions[i]) / distortions[i - 1]
                reductions.append(reduction_percent)
            else:
                reductions.append(0)

        # Ищем точку, где выгода от добавления кластера становится небольшой
        threshold = 0.3  # 30% улучшение считается значимым
        for i, reduction in enumerate(reductions):
            if reduction < threshold:
                return i + 1  # +1 потому что k начинается с 1

        # Если все уменьшения значительные, возвращаем последний k
        return len(distortions)

    # По умолчанию возвращаем разумное значение
    return min(2, len(unit_positions))


def find_medoid_simple(points: List[Tuple], distance_func) -> Tuple:
    """Упрощенный поиск медоида"""
    if not points:
        return None

    min_total_distance = float('inf')
    best_point = points[0]

    for candidate in points:
        total_distance = 0
        for point in points:
            if point != candidate:
                total_distance += distance_func(candidate, point)

        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_point = candidate

    return best_point


def create_final_clusters(centroids: List[Tuple], units: List[Dict], min_cluster_size: int = 2) -> Dict[
    Tuple, List[Dict]]:
    """Создает финальное распределение юнитов по кластерам"""

    def hex_distance_simple(pos1, pos2):
        """Упрощенная функция расстояния для финального распределения"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    final_clusters = {}

    # Для каждого центроида создаем список юнитов
    for centroid in centroids:
        final_clusters[centroid] = []

    # Распределяем юниты по ближайшим центроидам
    for unit in units:
        unit_pos = unit[START_KEY]
        min_distance = float('inf')
        closest_centroid = None

        for centroid in centroids:
            distance = hex_distance_simple(unit_pos, centroid)
            if distance < min_distance:
                min_distance = distance
                closest_centroid = centroid

        if closest_centroid is not None:
            final_clusters[closest_centroid].append(unit)

    # Объединяем слишком маленькие кластеры
    clusters_to_remove = []
    for centroid, cluster_units in list(final_clusters.items()):
        if len(cluster_units) < min_cluster_size:
            # Находим ближайший кластер для объединения
            min_centroid_distance = float('inf')
            nearest_centroid = None

            for other_centroid in final_clusters:
                if other_centroid != centroid and other_centroid not in clusters_to_remove:
                    distance = hex_distance_simple(centroid, other_centroid)
                    if distance < min_centroid_distance:
                        min_centroid_distance = distance
                        nearest_centroid = other_centroid

            if nearest_centroid is not None:
                # Объединяем кластеры
                final_clusters[nearest_centroid].extend(cluster_units)
                clusters_to_remove.append(centroid)

    # Удаляем объединенные кластеры
    for centroid in clusters_to_remove:
        if centroid in final_clusters:
            del final_clusters[centroid]

    return final_clusters


# Пример использования с вашими данными:
def example_usage():
    # Ваши данные
    units = [
        {START_KEY: (0, 0), MOVE_RANGE_KEY: 5, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1},
        {START_KEY: (1, 0), MOVE_RANGE_KEY: 5, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1},
        {START_KEY: (1, 1), MOVE_RANGE_KEY: 5, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1},
        {START_KEY: (1, 2), MOVE_RANGE_KEY: 5, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1},
        {START_KEY: (1, 6), MOVE_RANGE_KEY: 5, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1},
    ]

    # Создаем простую карту 7x7 без препятствий
    map_data = [[1 for _ in range(7)] for _ in range(7)]

    grid = HexGrid(weights=map_data, edge_collision=True, layout=HexLayout.odd_q)
    pf = AStar(grid)

    # Кластеризация только по юнитам с минимальным размером кластера = 2
    clusters = kmeans_clustering_units_only(units, grid, pf, min_cluster_size=2)

    # Вывод результатов
    print("Результаты кластеризации юнитов:")
    for centroid, cluster_units in clusters.items():
        print(f"Кластер с центром в {centroid}:")
        print(f"  Количество юнитов: {len(cluster_units)}")
        for unit in cluster_units:
            print(f"  - Юнит на позиции {unit[START_KEY]}")
        print()


# Константы
START_KEY = "start"
MOVE_RANGE_KEY = "move_range"
ATTACK_RANGE_KEY = "attack_range"
DAMAGE_KEY = "damage"

if __name__ == '__main__':
    example_usage()