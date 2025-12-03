from enum import Enum

POS_KEY = 'pos'
ID_KEY = 'id'
TYPE_KEY = 'type'
MOVE_RANGE_KEY = 'move_range'
ATTACK_RANGE_KEY = 'attack_range'
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