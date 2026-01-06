import heapq
from dataclasses import dataclass
from typing import Tuple, Set, Dict, List, Optional

from w9_pathfinding.envs import HexGrid

@dataclass
class PathInfo:
    cost: float
    path: List[Tuple[int, int]]


class ReachabilityCache:
    def __init__(
            self,
            start: Tuple[int, int],
            grid: HexGrid
    ):
        self.start = start
        self.grid = grid
        self._cache = None  # Dict[pos, PathInfo]
        self._max_computed_mp = 0
        self._distance_cache = {}  # Кэш расстояний для get_closest_reachable

    def get_reachable_(self, max_mp: int) -> Dict[Tuple[int, int], PathInfo]:
        if self._cache is None or max_mp > self._max_computed_mp:
            # Нужен пересчёт с новым (большим) лимитом
            self._cache = self._calculate_full(max_mp)
            self._max_computed_mp = max_mp
            self._distance_cache.clear()  # Инвалидируем кэш расстояний

        # Фильтруем по запрошенному max_mp
        return {
            pos: info
            for pos, info in self._cache.items()
            if info.cost <= max_mp
        }

    def get_reachable(self, max_mp: int) -> Dict[Tuple[int, int], PathInfo]:
        need_recalc = self._cache is None or max_mp > self._max_computed_mp

        if need_recalc:
            self._cache = self._calculate_full(max_mp)
            self._max_computed_mp = max_mp
            self._distance_cache.clear()

        reachable = {}
        for pos, info in self._cache.items():
            if info.cost <= max_mp:
                reachable[pos] = info

        return reachable

    def get_path(
            self,
            target: Tuple[int, int],
            max_mp: int,
            ) -> Optional[PathInfo]:
        reachable = self.get_reachable(max_mp)

        if not reachable:
            return None

        if target in reachable:
            return reachable[target]

        return None

    def get_closest_reachable(
            self,
            target: Tuple[int, int],
            max_mp: int,
            metric: str = 'euclidean'
    ) -> Optional[PathInfo]:
        reachable = self.get_reachable(max_mp)

        if not reachable:
            return None

        # Если цель достижима - возвращаем прямой путь
        if target in reachable:
            return reachable[target]

        # Выбираем функцию расстояния
        distance_func = self._get_distance_function(metric)

        # Ищем ближайшую достижимую позицию
        closest_pos = None
        min_distance = float('inf')

        for pos, info in reachable.items():
            # Кэшируем расстояния для ускорения повторных вызовов
            cache_key = (pos, target, metric)

            if cache_key not in self._distance_cache:
                self._distance_cache[cache_key] = distance_func(pos, target)

            distance = self._distance_cache[cache_key]

            if distance < min_distance:
                min_distance = distance
                closest_pos = pos

        return reachable[closest_pos] if closest_pos else None
    '''
    def get_best_advance_toward(
            self,
            target: Tuple[int, int],
            max_mp: int,
            weight_distance: float = 0.7,
            weight_mp_efficiency: float = 0.3
    ) -> Optional[PathInfo]:
        """
        Найти оптимальную позицию для продвижения к цели с учётом эффективности

        Args:
            target: Целевая позиция
            max_mp: Доступные очки движения
            weight_distance: Вес важности близости к цели (0-1)
            weight_mp_efficiency: Вес важности экономии MP (0-1)

        Returns:
            PathInfo для оптимальной позиции

        Отличие от get_closest_reachable:
            Учитывает не только расстояние, но и стоимость пути.
            Может выбрать позицию чуть дальше, но с меньшим расходом MP.

        Пример:
            Позиция A: ближе к цели на 1 клетку, стоимость 9 MP
            Позиция B: дальше на 1 клетку, стоимость 5 MP

            При weight_distance=0.9: выберет A (приоритет близости)
            При weight_distance=0.5: может выбрать B (баланс)
        """
        reachable = self.get_reachable(max_mp)

        if not reachable:
            return None

        # Если цель достижима - возвращаем прямой путь
        if target in reachable:
            return reachable[target]

        # Нормализуем веса
        total_weight = weight_distance + weight_mp_efficiency
        weight_distance /= total_weight
        weight_mp_efficiency /= total_weight

        # Вычисляем метрики для нормализации
        distance_func = self._get_distance_function('euclidean')

        max_distance = 0
        min_distance = float('inf')

        for pos in reachable.keys():
            dist = distance_func(pos, target)
            max_distance = max(max_distance, dist)
            min_distance = min(min_distance, dist)

        distance_range = max_distance - min_distance or 1

        # Ищем позицию с лучшим score
        best_pos = None
        best_score = float('inf')

        for pos, info in reachable.items():
            # Нормализованное расстояние до цели (0 = близко, 1 = далеко)
            distance = distance_func(pos, target)
            norm_distance = (distance - min_distance) / distance_range

            # Нормализованная стоимость пути (0 = дёшево, 1 = дорого)
            norm_cost = info.cost / max_mp if max_mp > 0 else 0

            # Комбинированный score (чем меньше - тем лучше)
            score = (
                    weight_distance * norm_distance +
                    weight_mp_efficiency * norm_cost
            )

            if score < best_score:
                best_score = score
                best_pos = pos

        return reachable[best_pos] if best_pos else None

    def get_advance_options(
            self,
            target: Tuple[int, int],
            max_mp: int,
            top_n: int = 5
    ) -> List[Tuple[PathInfo, Dict[str, float]]]:
        """
        Получить несколько лучших вариантов продвижения к цели

        Args:
            target: Целевая позиция
            max_mp: Доступные очки движения
            top_n: Количество вариантов для возврата

        Returns:
            Список кортежей (PathInfo, metrics) отсортированный по качеству

        metrics содержит:
            - 'distance_to_target': расстояние до цели
            - 'mp_cost': стоимость пути
            - 'mp_efficiency': процент использованных MP
            - 'score': общая оценка (меньше = лучше)

        Использование:
            Для AI или UI, чтобы показать игроку несколько вариантов
            продвижения с разными trade-off'ами.
        """
        reachable = self.get_reachable(max_mp)

        if not reachable:
            return []

        # Если цель достижима - она единственный вариант
        if target in reachable:
            return [(reachable[target], {
                'distance_to_target': 0.0,
                'mp_cost': reachable[target].cost,
                'mp_efficiency': reachable[target].cost / max_mp if max_mp > 0 else 0,
                'score': 0.0
            })]

        distance_func = self._get_distance_function('euclidean')

        # Вычисляем метрики для всех позиций
        options = []

        for pos, info in reachable.items():
            distance = distance_func(pos, target)
            mp_efficiency = info.cost / max_mp if max_mp > 0 else 0

            # Score: комбинация расстояния и эффективности
            # Меньше = лучше
            score = distance * 0.7 + mp_efficiency * 0.3

            metrics = {
                'distance_to_target': distance,
                'mp_cost': info.cost,
                'mp_efficiency': mp_efficiency,
                'score': score
            }

            options.append((info, metrics))

        # Сортируем по score и берём top_n
        options.sort(key=lambda x: x[1]['score'])

        return options[:top_n]
    '''
    def _get_distance_function(self, metric: str):
        """Возвращает функцию для вычисления расстояния"""
        if metric == 'euclidean':
            return lambda p1, p2: (
                                          (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
                                  ) ** 0.5

        elif metric == 'manhattan':
            return lambda p1, p2: abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        elif metric == 'hex':
            # Hexagonal distance (для offset coordinates)
            def hex_distance(p1, p2):
                x1, y1 = p1
                x2, y2 = p2
                dx = x2 - x1
                dy = y2 - y1

                # Для offset hex coordinates
                if (y1 % 2 != y2 % 2):
                    if dx > 0:
                        dx -= 0.5
                    elif dx < 0:
                        dx += 0.5

                return abs(dx) + abs(dy) + abs(dx - dy)

            return hex_distance

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _calculate_full(self, max_mp: int) -> Dict[Tuple[int, int], PathInfo]:
        reachable = {}
        queue = [(0, self.start, [self.start])]
        visited = {self.start: 0}

        while queue:
            cost, pos, path = heapq.heappop(queue)

            if cost > visited.get(pos, float('inf')):
                continue

            reachable[pos] = PathInfo(cost=cost, path=path)

            if cost >= max_mp:
                continue

            neighbors = self.grid.get_neighbors(pos, include_self=False)

            for neighbor, _ in neighbors:
                # todo: можно удалить в принципе. get_neighbors возвращает результат с учетом препятствий
                if self.grid.has_obstacle(neighbor):
                    continue

                step_cost = self.grid.calculate_cost([pos, neighbor])
                new_cost = cost + step_cost

                if new_cost > max_mp:
                    continue

                if new_cost < visited.get(neighbor, float('inf')):
                    visited[neighbor] = new_cost
                    heapq.heappush(queue, (new_cost, neighbor, path + [neighbor]))
        # todo: юнит с 1 ОД может двигаться по песку. НУЖНО подумать
        if len(reachable) == 1:
            neighbors = self.grid.get_neighbors(self.start, include_self=False)
            for neighbor, _ in neighbors:
                reachable[neighbor] = PathInfo(cost=1, path=[self.start, neighbor])

        return reachable