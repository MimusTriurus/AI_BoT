# Константы
START_KEY = 'start'
MOVE_RANGE_KEY = 'move_range'
ATTACK_RANGE_KEY = 'attack_range'
DAMAGE_KEY = 'damage'
HP_KEY = 'hp'
ID_KEY = 'id'
TYPE_KEY = 'type'
CAPACITY_KEY = 'capacity'
CARGO_KEY = 'cargo'
CAN_FIRE_LOADED_KEY = 'can_fire_loaded'  # Может ли груз стрелять из транспорта

weights = [
    # 0   1   2   3   4   5   6   7   8   9   10
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 0
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 1
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 2
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 3
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 4
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 5
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 6
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 7
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 8
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 9
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 10
]

my_units = [
    {
        START_KEY: (0, 0),
        MOVE_RANGE_KEY: 1,
        ATTACK_RANGE_KEY: 1,
        DAMAGE_KEY: 2,
        TYPE_KEY: 'infantry',
        HP_KEY: 3
    },
    {
        START_KEY: (0, 3),
        MOVE_RANGE_KEY: 1,
        ATTACK_RANGE_KEY: 1,
        DAMAGE_KEY: 2,
        TYPE_KEY: 'infantry',
        HP_KEY: 3
    },
    {
        START_KEY: (1, 3),
        MOVE_RANGE_KEY: 1,
        ATTACK_RANGE_KEY: 1,
        DAMAGE_KEY: 2,
        TYPE_KEY: 'infantry',
        HP_KEY: 3
    }
]

#del my_units[1]

enemy_units = [
    {
        START_KEY: (1, 1),
        MOVE_RANGE_KEY: 1,
        ATTACK_RANGE_KEY: 1,
        DAMAGE_KEY: 1,
        TYPE_KEY: 'tank',
        HP_KEY: 6
    }
]

MCTS_ITERATIONS = 500
TURNS = 1
ROLLOUT_DEPTH = 6
