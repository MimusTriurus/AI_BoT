import copy
from typing import List

from w9_pathfinding.envs import HexGrid, HexLayout
from w9_pathfinding.pf import IDAStar, AStar
from w9_pathfinding.mapf import CBS, SpaceTimeAStar, ReservationTable, MultiAgentAStar

def merge_pairs(matrix):
    merged = []
    for i in range(0, len(matrix), 2):
        row_a = matrix[i]
        row_b = matrix[i + 1]

        new_row = []
        for a, b in zip(row_a, row_b):
            new_row.append(a if a != 0 else b)
        merged.append(new_row)

    return merged


def make_square_(matrix, fill_value=-1):
    n_rows = len(matrix)
    n_cols = max(len(row) for row in matrix)
    size = max(n_rows, n_cols)

    square = []
    for row in matrix:
        # дополняем строку до size
        padded = row + [fill_value] * (size - len(row))
        square.append(padded)

    # если строк меньше, чем size — добавляем новые строки
    while len(square) < size:
        square.append([fill_value] * size)

    return square

def make_square(matrix, fill_value=-1):
    n_rows = len(matrix)
    n_cols = max(len(row) for row in matrix)
    size = max(n_rows, n_cols)

    # --- 1. Симметрично дополняем строки до size ---
    padded_rows = []
    for row in matrix:
        diff = size - len(row)
        left = diff // 2
        right = diff - left
        padded_rows.append([fill_value] * left + row + [fill_value] * right)

    # --- 2. Симметрично добавляем недостающие строки ---
    diff_rows = size - len(padded_rows)
    top = diff_rows // 2
    bottom = diff_rows - top

    # строки сверху
    for _ in range(top):
        padded_rows.insert(0, [fill_value] * size)

    # строки снизу
    for _ in range(bottom):
        padded_rows.append([fill_value] * size)

    return padded_rows



def parse(text):
    rows = text.split('\n')
    width = len(rows[0].split(',')) - 1
    nums = [int(x.strip()) for x in text.split(",") if x.strip()]
    return [nums[i:i + width] for i in range(0, len(nums), width)]



def read_map_data(file_path: str) -> str:
    with open(file_path) as f:
        return f.read()

'''
	None		= 0,
	Road		= 1,
	Plain		= 2, // aka Ground
	Difficult	= 3, // Desert + Ice + Swamp + River
	Forest		= 4, // Former Swamp
	Unpassable	= 5, // Mountain + Chasm
	Sea         = 6,
'''

passing_data = {
    0: -1,          # None
    1: 1,           # Road
    2: 1,           # Plain
    3: 2,           # Difficult - Desert + Ice + Swamp + River
    4: 1,           # Former Swamp
    5: -1,          # Unpassable - Mountain + Chasm
    6: -1,          # Sea
}

def refine_hex_values(input_matrix: List[List[int]]) -> List[List[int]]:
    n_rows = len(input_matrix)

    refined_matrix = copy.deepcopy(input_matrix)
    for r in range(n_rows):
        for c in range(n_rows):
            current_value = input_matrix[r][c]
            refined_matrix[r][c] = passing_data[current_value]

    return refined_matrix

map_size = 22
map_data_ = [[1] * map_size] * map_size

map_name = 'micronesia'
map_name = 'polygon'

map_str_data = read_map_data(f'AI_BoT/data/maps/{map_name}/passability_map.txt')
matrix = parse(map_str_data)
map_data = merge_pairs(matrix)
map_data = make_square(map_data, fill_value=6)
map_of_passage = refine_hex_values(map_data)

grid = HexGrid(weights=map_of_passage, edge_collision=True, layout=HexLayout.odd_q)
pf_ = AStar(grid)

path = pf_.find_path((4,11), (4,10))
cost = grid.calculate_cost(path)
path = pf_.find_path((4,10), (4,11))
cost = grid.calculate_cost(path)
print(cost)
