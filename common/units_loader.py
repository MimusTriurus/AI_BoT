import json
from typing import List, Dict, Tuple, Optional, Set
from AI_BoT.clustering import cluster_by_proximity, kmeans_hex, soft_clustering
from AI_BoT.common.units_data_loader import UnitsDataLoader
from w9_pathfinding.pf import IDAStar, AStar
from w9_pathfinding.envs import HexGrid
from AI_BoT.common.constants import *
from AI_BoT.common.constants import UnitType

u_data_loader = UnitsDataLoader(UnitType)
units_data = u_data_loader.load('AI_BoT/data/units_data.json')

def make_unit(u_id, pos, unit_type: UnitType):
    unit_data = units_data[UnitType(unit_type.value)]
    unit = {
        ID_KEY: u_id,
        POS_KEY: pos,
        MOVE_RANGE_KEY:         unit_data[0],
        DAMAGE_KEY:             unit_data[1],
        MIN_ATTACK_RANGE_KEY:   unit_data[2][0],
        MAX_ATTACK_RANGE_KEY:   unit_data[2][1],
        HP_KEY:                 unit_data[3],
        VALUE_KEY:              unit_data[4],
        CAPACITY_KEY:           unit_data[5],
    }
    return unit

class UnitsStorage:
    def __init__(self):
        self.units = dict()

    def add_unit(
            self,
            u_id: str,
            pos: Tuple,
            unit_type: UnitType,
            new_hp = None,
            new_value = None
    ) -> dict:
        unit = make_unit(u_id, pos, unit_type)
        if new_hp:
            unit[HP_KEY] = new_hp
        if new_value:
            unit[VALUE_KEY] = new_value
        self.units[pos] = unit
        return unit

    def get_units_pos(self):
        return list(self.units.keys())

    def get_units(self, units_pos: List[Tuple[int, int]] = None) -> List:
        if units_pos:
            results = []
            for u_p in units_pos:
                if u_p in self.units:
                    results.append(self.units[u_p])
            return results
        else:
            return list(self.units.values())

    def get_unit(self, pos: Tuple):
        return self.units[pos]

    def get_unit_by_id(self, u_id: str) -> Optional[dict]:
        for p, u in self.units.items():
            if u[ID_KEY] == u_id:
                return u
        return None

    def get_clusters(self, grid, filter_predicate = None):
        units_poses = [u[POS_KEY] for u in self.units.values() if filter_predicate(u)] if filter_predicate else self.get_units_pos()
        if not units_poses:
            return dict()
        pf = AStar(grid)
        centers, clusters = kmeans_hex(pf, units_poses, k=3)
        # кластеризация по степени близости
        # prox_clusters = cluster_by_proximity(pf, units_poses, grid, max_range=2)
        # кластеризация с перекрытием
        clusters = soft_clustering(units_poses, centers, pf, move_range=5)
        return clusters

    def clear(self):
        self.units.clear()

class MultiUnitsLoader:
    def __init__(self, storages: dict):
        """
        storages: словарь {имя_storage: объект UnitsStorage}
        """
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
