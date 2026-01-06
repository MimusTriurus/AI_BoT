import json
from typing import List, Dict, Tuple, Optional, Set
from AI_BoT.clustering import cluster_by_proximity, kmeans_hex, soft_clustering
from AI_BoT.common.reachable_positions_calculator import ReachabilityCache
from AI_BoT.common.units_data_loader import UnitsDataLoader

from AI_BoT.common.helpers import get_units_by_positions
from AI_BoT.common.constants import *
from AI_BoT.common.constants import UnitType
from AI_BoT.data_structures import Unit
from AI_BoT.game_map import grid

from w9_pathfinding.pf import IDAStar, AStar
from w9_pathfinding.envs import HexGrid

u_data_loader = UnitsDataLoader(UnitType)
units_data = u_data_loader.load('AI_BoT/data/units_data.json')

def make_unit(u_id, pos, unit_type: UnitType):
    unit_data = units_data[UnitType(unit_type.value)]

    unit = Unit(
        id=u_id,
        pos=pos,
        mp=unit_data[0],
        wd=(unit_data[1], 0),
        wr=(unit_data[2][0], unit_data[2][1]),
        hp=unit_data[3],
        value=unit_data[4],
        capacity=unit_data[5],
        inside_transport=False,
        unit_type=unit_type
    )
    unit.reachability_cache = ReachabilityCache(pos, grid)

    return unit

class UnitsStorage:
    def __init__(self):
        self.units: Dict[Tuple[int, int], Unit] = dict()

    def add_unit(
            self,
            u_id: str,
            pos: Tuple[int, int],
            unit_type: UnitType,
            new_hp = None,
            new_value = None
    ) -> Unit:
        unit = make_unit(u_id, pos, unit_type)
        if new_hp:
            unit.hp = new_hp
        if new_value:
            unit.value = new_value
        self.units[pos] = unit
        return unit

    def get_units_pos(self) -> List[Tuple[int, int]]:
        return list(self.units.keys())

    def get_units(self, units_pos: List[Tuple[int, int]] = None) -> List[Unit]:
        return get_units_by_positions(self.units, units_pos)

    def get_unit(self, pos: Tuple[int, int]):
        return self.units[pos]

    def get_unit_by_id(self, u_id: str) -> Optional[Unit]:
        for p, u in self.units.items():
            if u.id == u_id:
                return u
        return None

    def get_clusters(self, grid, filter_predicate = None):
        units_poses = [u.pos for u in self.units.values() if filter_predicate(u)] if filter_predicate else self.get_units_pos()
        if not units_poses:
            return dict()
        pf = AStar(grid)
        centers, clusters = kmeans_hex(pf, units_poses, k=3)
        # кластеризация по степени близости
        # prox_clusters = cluster_by_proximity(pf, units_poses, grid, max_range=2)

        # кластеризация с перекрытием
        clusters = soft_clustering(units_poses, centers, pf, move_range=5)
        # все юниты
        clusters['all'] = units_poses

        return list(clusters.values())

    def clear(self):
        self.units.clear()

class MultiUnitsLoader:
    def __init__(self, storages: dict):
        self.storages = storages

    def load_from_json(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        for storage_name, units in data.items():
            if storage_name not in self.storages:
                raise ValueError(f"Unknown storage: {storage_name}")

            storage = self.storages[storage_name]

            for unit in units:
                unit_id = unit["id"]
                pos = tuple(unit["pos"])
                #unit_type = getattr(UnitType, unit["type"])
                unit_type = UnitType[unit["type"]]
                kwargs = {}
                if "new_hp" in unit:
                    kwargs["new_hp"] = unit["new_hp"]
                if "new_value" in unit:
                    kwargs["new_value"] = unit["new_value"]

                storage.add_unit(unit_id, pos, unit_type, **kwargs)
