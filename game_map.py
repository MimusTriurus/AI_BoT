from w9_pathfinding.envs import HexGrid, HexLayout
from w9_pathfinding.pf import IDAStar, AStar
from w9_pathfinding.mapf import CBS, SpaceTimeAStar, ReservationTable, MultiAgentAStar

map_size = 22
map_data = [[1] * map_size] * map_size

grid = HexGrid(weights=map_data, edge_collision=True, layout=HexLayout.odd_q)
pf = AStar(grid)