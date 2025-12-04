import json

class UnitsDataLoader:
    def __init__(self, enum_cls):
        self.enum_cls = enum_cls

    def load(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        units_data = {}

        for unit_name, params in data.items():
            unit_type = self.enum_cls[unit_name]

            move = params["move"]
            damage = params["damage"]
            r_min, r_max = params["range"]
            hp = params["hp"]
            value = params["value"]
            capacity = params["capacity"]

            units_data[unit_type] = (move, damage, (r_min, r_max), hp, value, capacity)

        return units_data
