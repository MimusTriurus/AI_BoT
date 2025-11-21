from typing import List, Tuple, Dict


class PickupPlan:
    def __init__(self, unit_idx, unit_pos, meeting_point, unit_meeting_point=None):
        self.unit_idx = unit_idx
        self.unit_pos = unit_pos
        self.meeting_point = meeting_point
        self.unit_meeting_point = unit_meeting_point


class TransportMission:
    def __init__(self, transport_idx, pickup_sequence: List[PickupPlan], full_route: List[Tuple[int, int]]):
        self.transport_idx = transport_idx
        self.pickup_sequence = pickup_sequence
        self.full_route = full_route


def expand_transport_and_units_paths(mission: TransportMission):
    route = mission.full_route
    pickups = mission.pickup_sequence

    # итоговые пути
    transport_path: List[Tuple[int, int]] = []
    unit_paths: Dict[int, List[Tuple[int, int]]] = {p.unit_idx: [p.unit_pos] for p in pickups}

    loaded = set()
    pickup_index = 0
    transport_i = 0

    while transport_i < len(route):

        cur_pos = route[transport_i]
        transport_path.append(cur_pos)

        # все незагруженные юниты ждут
        for p in pickups:
            if p.unit_idx not in loaded:
                unit_paths[p.unit_idx].append(unit_paths[p.unit_idx][-1])

        # проверка — пришли ли мы к следующему месту погрузки
        if pickup_index < len(pickups):
            p = pickups[pickup_index]
            if cur_pos == p.meeting_point:

                # loading tick
                transport_path.append(cur_pos)

                # загружаемый юнит перемещается на транспорт
                uid = p.unit_idx
                unit_paths[uid].append(cur_pos)
                loaded.add(uid)

                # остальные незагруженные юниты ждут ещё один тик
                for pp in pickups:
                    if pp.unit_idx not in loaded:
                        unit_paths[pp.unit_idx].append(unit_paths[pp.unit_idx][-1])

                pickup_index += 1

        transport_i += 1

    return transport_path, unit_paths


# ---------------------------
# Тест — ровно твои данные
# ---------------------------
if __name__ == "__main__":
    mission = TransportMission(
        0,
        [
            PickupPlan(0, (1,0), (0,0)),
            PickupPlan(1, (0,1), (0,0)),
        ],
        [(0,0),(1,1),(2,2),(3,3),(4,4)]
    )

    t, u = expand_transport_and_units_paths(mission)

    print("Transport path:")
    print(t)
    print()
    for k,v in u.items():
        print(f"Unit {k} path: {v}")
