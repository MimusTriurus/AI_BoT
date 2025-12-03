import json
from typing import List, Dict, Tuple, Optional, Set
from AI_BoT.clustering import cluster_by_proximity, kmeans_hex, soft_clustering
from w9_pathfinding.pf import IDAStar, AStar
from w9_pathfinding.envs import HexGrid
from AI_BoT.common.constants import *
from AI_BoT.common.constants import UnitType

# m - move,
# w - weapon
# d - damage,
# r - weapon range,
# h - hp,
# v - value,
# c - capacity (transport)
units_data = {
    #                                 m    d   r   h   v   c
    UnitType.TANK:                  ( 2 ,  2,  1,  5,  2,  0 ),
    UnitType.LAND_TRANSPORT:        ( 5,   0,  0,  3,  1,  2 ),
    UnitType.ABSTRACT_TARGET:       ( 0 ,  0,  0,  4,  9,  0 ),
    UnitType.LAV:                   ( 1 ,  1,  1,  4,  1,  0 ),
    UnitType.SCORCHER:              ( 1 ,  2,  2,  4,  2,  0 ),
}


def make_unit(u_id, pos, unit_type: UnitType):
    unit_data = units_data[UnitType(unit_type.value)]
    unit = {
        ID_KEY: u_id,
        POS_KEY: pos,
        MOVE_RANGE_KEY:     unit_data[0],
        DAMAGE_KEY:         unit_data[1],
        ATTACK_RANGE_KEY:   unit_data[2],
        HP_KEY:             unit_data[3],
        VALUE_KEY:          unit_data[4],
        CAPACITY_KEY:       unit_data[5],
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

    def get_clusters(self, grid):
        units_poses = self.get_units_pos()
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
