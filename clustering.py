import random
from collections import defaultdict, deque
from w9_pathfinding.envs import HexGrid, HexLayout
from w9_pathfinding.pf import IDAStar, AStar
from w9_pathfinding.mapf import CBS, SpaceTimeAStar, ReservationTable, MultiAgentAStar

START_KEY = "start"
MOVE_RANGE_KEY = "move_range"
ATTACK_RANGE_KEY = "attack_range"
DAMAGE_KEY = "damage"
POS_KEY = "pos"
VALUE_KEY = "value"
HP_KEY = "hp"
# ───────────────────────────────────────────────────────────────────────────────
# HEX DISTANCE FOR odd-q axial
# ───────────────────────────────────────────────────────────────────────────────
#pf: AStar

# --- безопасная A* метрика ---
def hex_distance_astar(pf, a, b):
    try:
        path = pf.find_path(a, b)
        return len(path)
    except Exception:
        return 999999  # очень далеко


# --- округление центров обратно в гексы ---
def round_to_hex(center):
    # Центр может быть float, но нам нужен ближайший гекс
    return (round(center[0]), round(center[1]))


# --- KMEANS++ ИНИЦИАЛИЗАЦИЯ ---
def kmeanspp_init(pf, units, k):
    """Инициализация центров по KMeans++ с A*-расстоянием"""
    centers = []

    # 1. Первый центр — случайный
    centers.append(random.choice(units))

    # 2. Каждый следующий — с вероятностью ∝ расстоянию²
    for _ in range(1, k):
        # вычисляем минимальную дистанцию до любого уже выбранного центра
        dists = []
        for u in units:
            md = min(hex_distance_astar(pf, u, c) for c in centers)
            dists.append(md * md)

        total = sum(dists)
        if total == 0:
            centers.append(random.choice(units))
            continue

        # рулетка
        r = random.uniform(0, total)
        acc = 0
        for u, d in zip(units, dists):
            acc += d
            if acc >= r:
                centers.append(u)
                break

    return centers


# --- ОСНОВНОЙ АЛГОРИТМ ---
def kmeans_hex(pf, units_pos, k, max_iter=20):
    # 1. KMeans++ выбор центров
    centers = kmeanspp_init(pf, units_pos, k)

    # 2. Основной цикл
    for _ in range(max_iter):
        clusters = defaultdict(list)

        # assign
        for u in units_pos:
            dists = [hex_distance_astar(pf, u, c) for c in centers]
            ci = dists.index(min(dists))
            if isinstance(u, dict):
                u = u[POS_KEY]
            clusters[ci].append(u)

        # recompute
        new_centers = []
        for i in range(k):
            if not clusters[i]:
                new_centers.append(centers[i])  # пустой кластер — пропускаем
                continue
            xs = [p[0] for p in clusters[i]]
            ys = [p[1] for p in clusters[i]]

            # среднее → возможно float
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)

            # округляем к ближайшему гексу
            new_centers.append(round_to_hex((cx, cy)))

        # check convergence
        if all(centers[i] == new_centers[i] for i in range(k)):
            break

        centers = new_centers

    return centers, clusters


# ============================================================
#       SSE: СУММА КВАДРАТОВ ДИСТАНЦИЙ В КЛАСТЕРАХ
# ============================================================

def compute_sse(clusters, centers):
    sse = 0
    for cid, units in clusters.items():
        c = centers[cid]
        for u in units:
            d = hex_distance_astar(u, c)
            sse += d * d
    return sse


# ============================================================
#   ПОИСК "ЛОКТЯ": МАКС. ОТКЛОНЕНИЕ ОТ ПРЯМОЙ МЕЖДУ 1 И K_MAX
# ============================================================

def find_elbow_point(sse_list):
    """
    sse_list: [(k, sse), ...]
    Метод: расстояние каждой точки до прямой (k1, sse1) - (kN, sseN)
    """
    k1, sse1 = sse_list[0]
    kN, sseN = sse_list[-1]

    # прямая
    dk = kN - k1
    ds = sseN - sse1

    # расстояние каждой точки от прямой
    best_k = k1
    best_dist = -1

    for k, sse in sse_list:
        # формула расстояния от точки до прямой в 2D
        num = abs(ds * (k - k1) - dk * (sse - sse1))
        den = (dk * dk + ds * ds) ** 0.5
        dist = num / den if den != 0 else 0

        if dist > best_dist:
            best_dist = dist
            best_k = k

    return best_k


# ============================================================
#         ОБЩИЙ МЕТОД: НАХОДИМ ОПТИМАЛЬНОЕ K
# ============================================================

def find_optimal_k(units, k_max=8):
    """
    Возвращает:
    - optimal_k
    - centers
    - clusters
    - sse_list
    """
    sse_list = []

    for k in range(1, k_max + 1):
        centers, clusters = kmeans_hex(units, k)
        sse = compute_sse(clusters, centers)
        sse_list.append((k, sse))

    # ищем локоть
    optimal_k = find_elbow_point(sse_list)

    # финальная кластеризация
    centers, clusters = kmeans_hex(units, optimal_k)

    return optimal_k, centers, clusters, sse_list



# ───────────────────────────────────────────────────────────────────────────────
# REAL PATH DISTANCE USING ASTAR
# ───────────────────────────────────────────────────────────────────────────────

def true_path_distance(pf, a, b):
    """Returns A* path length or large value if unreachable."""
    path = pf.find_path(a, b)
    if not path:
        return 99999
    return len(path) - 1


# ───────────────────────────────────────────────────────────────────────────────
# NEIGHBORHOOD CLUSTERING (CONNECTED COMPONENTS)
# ───────────────────────────────────────────────────────────────────────────────

def cluster_by_proximity(pf, units, grid, max_range=2):
    visited = set()
    units_set = set(units)
    clusters = []

    for u in units:
        if u in visited:
            continue

        q = [u]
        visited.add(u)
        cluster = [u]

        while q:
            cur = q.pop()
            for (nx, ny), cost in grid.get_neighbors(cur):
                if (nx, ny) in units_set and (nx, ny) not in visited:
                    if hex_distance_astar(pf, (nx, ny), cur) <= max_range:
                        visited.add((nx, ny))
                        q.append((nx, ny))
                        cluster.append((nx, ny))

        clusters.append(cluster)

    return clusters


# ───────────────────────────────────────────────────────────────────────────────
# CLUSTERING AROUND TARGETS (ATTRACTION)
# ───────────────────────────────────────────────────────────────────────────────

def cluster_around_targets(pf, units, targets, influence_radius=7):
    clusters = defaultdict(list)

    for u in units:
        best_target = None
        best_score = 1e9

        for t in targets:
            tpos = t["pos"]
            dist = true_path_distance(pf, u, tpos)
            score = dist - t["value"] * 3.0  # ценность притягивает

            if score < best_score:
                best_score = score
                best_target = tpos

        clusters[best_target].append(u)

    return clusters


# ───────────────────────────────────────────────────────────────────────────────
# SOFT CLUSTERING — юнит может входить в несколько кластеров
# ───────────────────────────────────────────────────────────────────────────────

def soft_clustering(units, centers, pf, move_range=10):
    result = defaultdict(list)

    for u in units:
        for ci, c in enumerate(centers):
            d = true_path_distance(pf, u, (int(c[0]), int(c[1])))
            if d <= move_range:
                result[ci].append(u)

    return result


# ───────────────────────────────────────────────────────────────────────────────
# ASCII VISUALIZATION OF CLUSTERS
# ───────────────────────────────────────────────────────────────────────────────

def visualize_ascii(map_grid, units, clusters, cluster_symbols=None):
    """
    map_grid[y][x] values: 1 = walkable, -1 = obstacle
    clusters: dict {cluster_id: [(x,y), ...]}
    """

    height = len(map_grid)
    width = len(map_grid[0])

    # default symbols
    if cluster_symbols is None:
        cluster_symbols = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # map: build empty canvas
    canvas = [[("." if map_grid[y][x] != -1 else "#")
               for x in range(width)] for y in range(height)]

    # draw clusters
    for cid, positions in clusters.items():
        symbol = cluster_symbols[cid % len(cluster_symbols)]
        for (x, y) in positions:
            canvas[y][x] = symbol

    # draw units border if needed
    for (x, y) in units:
        if canvas[y][x] == '.':
            canvas[y][x] = '@'

    # print
    text = []
    for y in range(height):
        # offset for odd-q hex grid
        prefix = " " if (y & 1) else ""
        row = prefix + " ".join(canvas[y])
        text.append(row)

    return "\n".join(text)

if __name__ == '__main__':
    map_data_small = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]
    map_data = [[1] * 40] * 40

    units = [
        {POS_KEY: (0, 0), MOVE_RANGE_KEY: 5, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1},
        {POS_KEY: (1, 0), MOVE_RANGE_KEY: 5, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1},
        #{POS_KEY: (1, 1), MOVE_RANGE_KEY: 5, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1},
        {POS_KEY: (1, 2), MOVE_RANGE_KEY: 5, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1},
        {POS_KEY: (1, 6), MOVE_RANGE_KEY: 5, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1},
        {POS_KEY: (6, 6), MOVE_RANGE_KEY: 5, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1},
        {POS_KEY: (5, 6), MOVE_RANGE_KEY: 5, ATTACK_RANGE_KEY: 1, DAMAGE_KEY: 1},
    ]

    targets = [
        {POS_KEY: (5, 1), VALUE_KEY: 0.5, HP_KEY: 2},
        {POS_KEY: (6, 5), VALUE_KEY: 0.5, HP_KEY: 2},
    ]

    transports = [
        {POS_KEY: (1, 1), VALUE_KEY: 0.5, HP_KEY: 2},
        {POS_KEY: (1, 3), VALUE_KEY: 0.5, HP_KEY: 2},
    ]

    grid = HexGrid(weights=map_data, edge_collision=True, layout=HexLayout.odd_q)
    pf = AStar(grid)

    units_pos = [u[POS_KEY] for u in units]

    # 1. K-Means
    centers, km_clusters = kmeans_hex(pf, units_pos, k=3)

    #optimal_k, centers, clusters, sse_curve = find_optimal_k(units, k_max=10)

    # 2. Соседние группы
    prox_clusters = cluster_by_proximity(pf, units_pos, grid, max_range=2)
    prox_clusters = {i: v for i, v in enumerate(prox_clusters)}
    # 3. Кластеры вокруг целей
    target_clusters = cluster_around_targets(pf, units_pos, targets)

    transport_clusters = cluster_around_targets(pf, units_pos, transports)

    # 4. Soft-clustering по move_range
    soft = soft_clustering(units_pos, centers, pf, move_range=6)

    # 5. ASCII визуализация
    print(visualize_ascii(map_data, units_pos, prox_clusters))
