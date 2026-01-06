from typing import List, Tuple
from AI_BoT.common.constants import *
from AI_BoT.common.units_loader import UnitsStorage
from AI_BoT.transport_plan import TacticPlan

my_units_storage = UnitsStorage()
en_units_storage = UnitsStorage()
transport_storage = UnitsStorage()

units_storages = {
    "en_units_storage": en_units_storage,
    "my_units_storage": my_units_storage,
    "transports": transport_storage,
}

# получаем свободные ресурсы
def get_free_units(actual_plans: List[TacticPlan]) -> Tuple:
    '''
    Возвращает незадействованые боевые юниты, транспорты, живые цели и зарезервированые позиции
    '''
    used_units = set()
    used_transports = set()
    used_targets = set()
    # не учитывает, что транспорт может отойти с точки выгрузки если у него остались очки движения
    busy_hexes = set()

    targets_hp = {}

    for en_unit in en_units_storage.get_units():
        targets_hp[en_unit.id] = en_unit.hp

    for plan in actual_plans:
        busy_hexes.update(plan.occupied_hexes_set)
        target_id = plan.target.id
        #if target_id not in targets_hp:
        #    targets_hp[target_id] = plan.target.hp

        targets_hp[target_id] -= plan.actual_damage_contribution

        used_targets.add(plan.target.id)
        used_transports.add(plan.transport.id)
        for passenger in plan.passengers:
            used_units.add(passenger.id)

    # Находим свободные ресурсы
    all_units = my_units_storage.get_units()
    all_transports = transport_storage.get_units()
    all_targets = en_units_storage.get_units()

    unused_units = {u.pos: u for u in all_units if u.id not in used_units}
    unused_transports = [t for t in all_transports if t.id not in used_transports]
    alive_targets = []
    if targets_hp:
        alive_targets = [u for u in all_targets if u.id in targets_hp and targets_hp[u.id] > 0]

    return unused_units, unused_transports, alive_targets, busy_hexes