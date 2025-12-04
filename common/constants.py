from enum import Enum

POS_KEY = 'pos'
ID_KEY = 'id'
TYPE_KEY = 'type'
MOVE_RANGE_KEY = 'move_range'
MAX_ATTACK_RANGE_KEY = 'max_attack_range'
MIN_ATTACK_RANGE_KEY = 'min_attack_range'
DAMAGE_KEY = 'damage'
VALUE_KEY = 'value'
HP_KEY = 'hp'
CAPACITY_KEY = 'capacity'

class UnitType(Enum):
    TANK                = 0
    LAND_TRANSPORT      = 1
    ABSTRACT_TARGET     = 2
    LAV                 = 3
    SCORCHER            = 4
    AMPHIBIA            = 5
    ROCKET_LAUNCHER     = 6

UNIT_MOVE_RANGE_AFTER_UNLOAD = 1