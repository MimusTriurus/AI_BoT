import pygame

from AI_BoT.common.constants import POS_KEY, ID_KEY, DAMAGE_KEY
from AI_BoT.common.helpers import insert_after
from AI_BoT.game_map import grid
from AI_BoT.visualizer import HexVisualizer
from resource_manager import my_units_storage, en_units_storage, transport_storage

def game_loop(actual_plans: list):
    visualizer = HexVisualizer(grid)
    units_paths = dict()
    for plan in actual_plans:
        insertions = dict()
        units_2_load = dict()

        t_id, t_full_route = plan.to_path()
        plan.calculate_meeting_points()
        t_start_pos = t_full_route[0]
        for step in t_full_route:
            wait_steps = 0

            for u_id, meeting_point in plan.meeting_points.items():
                if meeting_point == step:# and meeting_point != t_start_pos:
                    if step not in units_2_load:
                        units_2_load[step] = list()
                    units_2_load[step].append(u_id)
                    #if meeting_point == t_start_pos:
                    #    continue
                    wait_steps += 1

            for s in range(wait_steps):
                insertions[step] = wait_steps

        transport_route = list(t_full_route)
        for pos, waiting in insertions.items():
            for i in range(waiting):
                if t_start_pos == pos:
                    continue
                insert_after(transport_route, pos, pos)
            i = 0
            end_of_head_idx = transport_route.index(pos)
            head_len = len(transport_route[0: end_of_head_idx])
            while waiting > 0:
                passengers_id = units_2_load[pos]
                unit_id = units_2_load[pos][i]
                passenger = my_units_storage.get_unit_by_id(unit_id)

                if unit_id not in units_paths:
                    units_paths[unit_id] = list()
                passenger_pos = passenger[POS_KEY]

                for k in range(head_len):
                    units_paths[unit_id].append(passenger_pos)

                for j in range(waiting):
                    if t_start_pos == passenger_pos:
                        continue
                    units_paths[unit_id].append(passenger_pos)
                i += 1
                waiting -= 1
                units_paths[unit_id].append(pos)

        units_paths[t_id] = transport_route
        # движение юнитов в транспортов
        for u_id, path in units_paths.items():
            # боевой юнит транспортирует сам себя
            if u_id == t_id:
                continue

            result = next((item for item in plan.passengers if item[ID_KEY] == u_id), None)
            if result:
                path_tail = transport_route[len(path):]
                path_2_target = []
                if plan.unload_map.get(u_id):
                    path_2_target = [plan.unload_map[u_id]]
                path = path + path_tail + path_2_target
                units_paths[u_id] = path

    solution = {
        'assignments': [],
        #'paths': dict()
        'paths':  units_paths
    }

    assignment = {}
    for plan in actual_plans:
        target = plan.target
        t_id = target[ID_KEY]
        t_idx = next(i for i, obj in enumerate(en_units_storage.get_units()) if obj[ID_KEY] == t_id)

        for passenger in plan.passengers:
            u_id = passenger[ID_KEY]
            u_idx = next(i for i, obj in enumerate(my_units_storage.get_units()) if obj[ID_KEY] == u_id)
            solution['assignments'].append({
                "unit_idx": u_id,
                "target_idx": t_idx,
                "damage": passenger[DAMAGE_KEY]
            })

    units = list(my_units_storage.get_units())
    units.extend(transport_storage.get_units())

    targets = en_units_storage.get_units()

    while True:
        restart = visualizer.animate_solution(solution, units, targets)
        if restart:
            for target in targets:
                target['current_hp'] = target['hp']
            continue
        else:
            break

    pygame.quit()