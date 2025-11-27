import math
import random
from typing import List, Tuple, Dict, Any
from typing import List, Dict, Tuple, Optional, Set
from w9_pathfinding.envs import HexGrid, HexLayout
from w9_pathfinding.pf import IDAStar, AStar
from w9_pathfinding.mapf import CBS, SpaceTimeAStar, ReservationTable, MultiAgentAStar

pf: AStar
grid: HexGrid

def hex_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    path = pf.find_path(pos1, pos2)
    return grid.calculate_cost(path)


def kmeans_clustering(units: List[Dict], targets: List[Dict], grid: Any, k: int = None, max_iters: int = 100) -> Dict[
    Tuple, List[Dict]]:
    """
    Кластеризация юнитов на гексогональном поле методом K-Means с учетом расстояний A*

    Args:
        units: Список юнитов с позициями
        targets: Список целей для учета в кластеризации
        grid: Объект гексогонального поля
        k: Количество кластеров (если None, вычисляется автоматически)
        max_iters: Максимальное количество итераций

    Returns:
        Словарь {центроид_кластера: список_юнитов_в_кластере}
    """

    def find_medoid(points: List[Tuple], grid: Any) -> Tuple:
        """Находит точку с минимальной суммой расстояний до всех других точек (медоид)"""
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

    # Если k не задано, вычисляем на основе количества юнитов и целей
    if k is None:
        k = max(1, min(len(units) // 2, len(unit_positions)))
        k = min(k, len(unit_positions))  # Не больше чем точек

    # Инициализация центроидов методом K-Means++
    def kmeans_plus_plus(points: List[Tuple], k: int) -> List[Tuple]:
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
    if len(unit_positions) <= k:
        centroids = unit_positions.copy()
    else:
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

        # Шаг 2: Пересчет центроидов
        new_centroids = []
        for centroid, cluster_units in clusters.items():
            if not cluster_units:
                # Если кластер пуст, оставляем старый центроид
                new_centroids.append(centroid)
            else:
                # Находим медоид (точку с минимальной суммой расстояний)
                cluster_positions = [unit[START_KEY] for unit in cluster_units]
                new_centroid = find_medoid(cluster_positions, grid)
                new_centroids.append(new_centroid)

        # Проверка сходимости
        if set(new_centroids) == set(centroids):
            break

        centroids = new_centroids

    # Формируем результат
    final_clusters = {}
    for centroid in centroids:
        cluster_units = []
        for unit in units:
            unit_pos = unit[START_KEY]
            min_distance = float('inf')
            closest_centroid = None

            for c in centroids:
                distance = hex_distance(unit_pos, c)
                if distance < min_distance:
                    min_distance = distance
                    closest_centroid = c

            if closest_centroid == centroid:
                cluster_units.append(unit)

        final_clusters[centroid] = cluster_units

    return final_clusters


def adaptive_kmeans_with_targets(units: List[Dict], targets: List[Dict], grid: Any, pf: Any) -> Dict[Tuple, List[Dict]]:
    """
    Адаптивная кластеризация с учетом целей и их ценности
    """
    # Группируем цели по регионам
    target_positions = [target[POS_KEY] for target in targets]
    target_values = [target.get(VALUE_KEY, 1.0) for target in targets]

    # Вычисляем оптимальное количество кластеров на основе целей
    if targets:
        # Используем метод локтя для определения k
        k = calculate_optimal_k(target_positions, grid, pf, max_k=min(5, len(units)))
    else:
        k = max(1, len(units) // 2)

    # Кластеризуем юнитов
    clusters = kmeans_clustering(units, targets, grid, k)

    # Оптимизируем распределение с учетом целей
    return optimize_clusters_with_targets(clusters, targets, grid, pf)


def calculate_optimal_k(points: List[Tuple], grid: Any, pf: Any, max_k: int = 5) -> int:
    """Вычисляет оптимальное количество кластеров методом локтя"""
    if len(points) <= 1:
        return 1

    distortions = []
    k_values = range(1, min(max_k + 1, len(points) + 1))

    for k in k_values:
        if k == 1:
            # Для одного кластера вычисляем суммарное квадратичное отклонение
            centroid = points[0]  # Просто берем первую точку как центроид
            distortion = sum(hex_distance(p, centroid) ** 2 for p in points)
        else:
            # Используем упрощенный K-Means для вычисления distortion
            temp_centroids = random.sample(points, k)
            for _ in range(10):  # Несколько итераций
                clusters = {c: [] for c in temp_centroids}
                for p in points:
                    min_dist = float('inf')
                    closest = temp_centroids[0]
                    for c in temp_centroids:
                        dist = hex_distance(p, c)
                        if dist < min_dist:
                            min_dist = dist
                            closest = c
                    clusters[closest].append(p)

                new_centroids = []
                for centroid, cluster_points in clusters.items():
                    if cluster_points:
                        # Находим медоид
                        new_centroid = min(cluster_points,
                                           key=lambda x: sum(hex_distance(x, p) for p in cluster_points))
                        new_centroids.append(new_centroid)
                    else:
                        new_centroids.append(centroid)

                temp_centroids = new_centroids

            # Вычисляем distortion
            distortion = 0
            for centroid, cluster_points in clusters.items():
                for p in cluster_points:
                    distortion += hex_distance(p, centroid) ** 2

        distortions.append(distortion)

    # Находим "локоть" - точку, где уменьшение distortion замедляется
    if len(distortions) > 1:
        reductions = []
        for i in range(1, len(distortions)):
            reduction = distortions[i - 1] - distortions[i]
            reductions.append(reduction)

        if reductions:
            # Ищем точку, где относительное уменьшение становится маленьким
            max_reduction = max(reductions)
            for i, reduction in enumerate(reductions):
                if reduction < max_reduction * 0.3:  # Порог 30% от максимального уменьшения
                    return i + 1  # +1 потому что k начинается с 1

    return min(3, len(points))


def optimize_clusters_with_targets(clusters: Dict, targets: List[Dict], grid: Any, pf: Any) -> Dict:
    """Оптимизирует кластеры с учетом близости к целям"""
    target_positions = [target[POS_KEY] for target in targets]

    optimized_clusters = {}

    for centroid, cluster_units in clusters.items():
        if not cluster_units:
            continue

        # Находим среднюю позицию юнитов в кластере
        unit_positions = [unit[START_KEY] for unit in cluster_units]
        avg_row = sum(pos[0] for pos in unit_positions) / len(unit_positions)
        avg_col = sum(pos[1] for pos in unit_positions) / len(unit_positions)

        # Ищем ближайшую цель к средней позиции
        if target_positions:
            closest_target = min(target_positions,
                                 key=lambda pos: hex_distance((avg_row, avg_col), pos))

            # Смещаем центроид в сторону ближайшей цели (но не слишком далеко)
            new_centroid_row = int((avg_row + closest_target[0]) / 2)
            new_centroid_col = int((avg_col + closest_target[1]) / 2)

            # Проверяем, что новая позиция валидна
            if not grid.has_obstacle((new_centroid_row, new_centroid_col)):
                optimized_clusters[(new_centroid_row, new_centroid_col)] = cluster_units
            else:
                optimized_clusters[centroid] = cluster_units
        else:
            optimized_clusters[centroid] = cluster_units

    return optimized_clusters


# Пример использования:
def example_usage():
    # Ваши данные
    map_data = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]

    units = [
        {START_KEY: (0, 0), MOVE_RANGE_KEY: 5, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1},
        {START_KEY: (1, 0), MOVE_RANGE_KEY: 5, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1},
        {START_KEY: (1, 1), MOVE_RANGE_KEY: 5, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1},
        {START_KEY: (1, 2), MOVE_RANGE_KEY: 5, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1},
        {START_KEY: (1, 6), MOVE_RANGE_KEY: 5, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1},
    ]

    targets = [
        {POS_KEY: (5, 1), VALUE_KEY: 0.5, HP_KEY: 2},
        {POS_KEY: (6, 5), VALUE_KEY: 0.5, HP_KEY: 2},
    ]

    grid = HexGrid(weights=map_data, edge_collision=True, layout=HexLayout.odd_q)
    pf = AStar(grid)

    # Кластеризация
    clusters = adaptive_kmeans_with_targets(units, targets, grid, pf)

    # Вывод результатов
    for centroid, cluster_units in clusters.items():
        print(f"Кластер с центром в {centroid}:")
        for unit in cluster_units:
            print(f"  - Юнит на позиции {unit[START_KEY]}")
        print()


# Константы (должны быть определены в вашем коде)
START_KEY = "start"
MOVE_RANGE_KEY = "move_range"
ATTACK_RANGE_KEY = "attack_range"
DAMAGE_KEY = "damage"
POS_KEY = "pos"
VALUE_KEY = "value"
HP_KEY = "hp"

if __name__ == '__main__':
    example_usage()