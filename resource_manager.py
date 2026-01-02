from typing import List, Tuple
from AI_BoT.common.constants import *
from AI_BoT.common.units_loader import UnitsStorage
from AI_BoT.transport_plan import TransportPlan

my_units_storage = UnitsStorage()
en_units_storage = UnitsStorage()
transport_storage = UnitsStorage()

units_storages = {
    "en_units_storage": en_units_storage,
    "my_units_storage": my_units_storage,
    "transports": transport_storage,
}

# получаем свободные ресурсы
def get_free_units(actual_plans: List[TransportPlan]) -> Tuple:
    used_units = set()
    used_transports = set()
    used_targets = set()
    busy_hexes = set()

    targets_hp = {}

    for plan in actual_plans:
        busy_hexes.update(plan.occupied_hexes_set)
        target_id = plan.target[ID_KEY]
        if target_id not in targets_hp:
            targets_hp[target_id] = plan.target[HP_KEY]

        targets_hp[target_id] -= plan.actual_damage_contribution

        used_targets.add(plan.target[ID_KEY])
        used_transports.add(plan.transport[ID_KEY])
        for passenger in plan.passengers:
            used_units.add(passenger[ID_KEY])

    # Находим свободные ресурсы
    all_units = my_units_storage.get_units()
    all_transports = transport_storage.get_units()
    all_targets = en_units_storage.get_units()

    unused_units = {u[POS_KEY]: u for u in all_units if u[ID_KEY] not in used_units}
    unused_transports = [t for t in all_transports if t[ID_KEY] not in used_transports]

    free_targets = [u for u in all_targets if targets_hp[u[ID_KEY]] > 0]

    return unused_units, unused_transports, free_targets