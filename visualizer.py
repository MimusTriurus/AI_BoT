# спизжено
import pygame
import math
from typing import List, Dict, Tuple
from enum import Enum


class AnimationState(Enum):
    IDLE = 0
    MOVING = 1
    ATTACKING = 2
    DYING = 3


class HexVisualizer:
    def _hex_to_cartesian(self, hex_x, hex_y):
        layout = self.grid.layout

        if layout <= 1:
            y = hex_y * math.sqrt(3) / 2
            if hex_y % 2 == layout:
                x = hex_x
            else:
                x = hex_x + 0.5
        else:
            x = hex_x * math.sqrt(3) / 2
            if hex_x % 2 == layout - 2:
                y = hex_y
            else:
                y = hex_y + 0.5

        return x, y

    def __init__(self, grid, width=1200, height=800):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Hex Tactical Visualization")
        self.clock = pygame.time.Clock()

        self.grid = grid
        self.width = width
        self.height = height

        self.hex_size = 40
        self.hex_width = self.hex_size * 2
        self.hex_height = math.sqrt(3) * self.hex_size

        self.offset_x = 100
        self.offset_y = 100

        # Цвета
        self.COLORS = {
            'background': (20, 20, 30),
            'hex_border': (60, 60, 80),
            'hex_walkable': (40, 40, 60),
            'hex_obstacle': (80, 40, 40),
            'hex_highlight': (80, 80, 120),
            'unit': (100, 150, 255),
            'unit_selected': (150, 200, 255),
            'target': (255, 80, 80),
            'target_damaged': (200, 60, 60),
            'target_destroyed': (255, 255, 255),
            'path': (100, 200, 100),
            'attack_range': (255, 100, 100, 100),
            'text': (220, 220, 220),
            'hp_bar_bg': (60, 60, 60),
            'hp_bar': (80, 200, 80),
            'damage_text': (255, 200, 0)
        }

        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 36)

        self.animation_speed = 2.0
        self.animation_state = {}
        self.damage_effects = []

        self.running = True
        self.paused = False
        self.show_grid = True
        self.show_coords = True

    def hex_to_pixel(self, hex_pos):
        col, row = hex_pos  # hex_x, hex_y
        cart_x, cart_y = self._hex_to_cartesian(col, row)
        pixel_x = cart_x * self.hex_width + self.offset_x
        pixel_y = cart_y * self.hex_height + self.offset_y

        return (pixel_x, pixel_y)

    def draw_hexagon(self, center: Tuple[float, float], color: Tuple[int, int, int],
                     border_color: Tuple[int, int, int] = None, width: int = 0):
        points = []
        for i in range(6):
            angle = math.pi / 3 * i
            x = center[0] + self.hex_size * math.cos(angle)
            y = center[1] + self.hex_size * math.sin(angle)
            points.append((x, y))

        pygame.draw.polygon(self.screen, color, points, width)

        if border_color:
            pygame.draw.polygon(self.screen, border_color, points, 2)

    def draw_grid(self):
        rows = len(self.grid.weights)
        cols = len(self.grid.weights[0])

        for row in range(rows):
            for col in range(cols):
                center = self.hex_to_pixel((row, col))

                color = (
                    self.COLORS['hex_obstacle']
                    if self.grid.has_obstacle((row, col))
                    else self.COLORS['hex_walkable']
                )

                self.draw_hexagon(center, color, self.COLORS['hex_border'])

                if self.show_coords:
                    text = self.font_small.render(f"{row},{col}", True, self.COLORS['text'])
                    text_rect = text.get_rect(center=center)
                    self.screen.blit(text, text_rect)

    def draw_path(self, path: List[Tuple[int, int]], color=None):
        if len(path) < 2:
            return

        color = color or self.COLORS['path']

        for i in range(len(path) - 1):
            start = self.hex_to_pixel(path[i])
            end = self.hex_to_pixel(path[i + 1])
            pygame.draw.line(self.screen, color, start, end, 3)

        for pos in path:
            center = self.hex_to_pixel(pos)
            pygame.draw.circle(self.screen, color, (int(center[0]), int(center[1])), 4)

    def draw_unit(self, pos: Tuple[float, float], unit_data: Dict, animated_pos=None):
        center = animated_pos if animated_pos else self.hex_to_pixel(pos)

        is_selected = unit_data.get('selected', False)
        color = self.COLORS['unit_selected'] if is_selected else self.COLORS['unit']

        pygame.draw.circle(self.screen, color,
                           (int(center[0]), int(center[1])), 15)
        pygame.draw.circle(self.screen, self.COLORS['hex_border'],
                           (int(center[0]), int(center[1])), 15, 2)

        text = self.font_small.render(str(unit_data.get('id', '?')),
                                      True, (255, 255, 255))
        text_rect = text.get_rect(center=center)
        self.screen.blit(text, text_rect)

    def draw_target(self, pos: Tuple[int, int], target_data: Dict, target_destroyed=False):
        center = self.hex_to_pixel(pos)

        hp_ratio = target_data.get('current_hp', target_data['hp']) / target_data['hp']
        if hp_ratio < 0.5:
            color = self.COLORS['target_damaged']
        else:
            color = self.COLORS['target']

        if target_destroyed:
            color = self.COLORS['target_destroyed']

        size = 20
        rect = pygame.Rect(center[0] - size / 2, center[1] - size / 2, size, size)
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, self.COLORS['hex_border'], rect, 2)

        hp_bar_width = 30
        hp_bar_height = 4
        hp_bar_x = center[0] - hp_bar_width / 2
        hp_bar_y = center[1] + 18

        pygame.draw.rect(self.screen, self.COLORS['hp_bar_bg'],
                         (hp_bar_x, hp_bar_y, hp_bar_width, hp_bar_height))

        current_hp = target_data.get('current_hp', target_data['hp'])
        hp_width = hp_bar_width * (current_hp / target_data['hp'])
        pygame.draw.rect(self.screen, self.COLORS['hp_bar'],
                         (hp_bar_x, hp_bar_y, hp_width, hp_bar_height))

        hp_text = self.font_small.render(f"{int(current_hp)}/{target_data['hp']}",
                                         True, self.COLORS['text'])
        text_rect = hp_text.get_rect(center=(center[0], center[1] + 30))
        self.screen.blit(hp_text, text_rect)

    def draw_attack_effect(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int],
                           progress: float):
        start = self.hex_to_pixel(from_pos)
        end = self.hex_to_pixel(to_pos)

        current_x = start[0] + (end[0] - start[0]) * progress
        current_y = start[1] + (end[1] - start[1]) * progress

        pygame.draw.circle(self.screen, (255, 200, 0),
                           (int(current_x), int(current_y)), 6)

        if progress > 0.1:
            trail_length = 5
            for i in range(trail_length):
                t = max(0, progress - (i + 1) * 0.05)
                tx = start[0] + (end[0] - start[0]) * t
                ty = start[1] + (end[1] - start[1]) * t
                alpha = 255 - i * 50
                pygame.draw.circle(self.screen, (255, 200, 0, alpha),
                                   (int(tx), int(ty)), 4 - i)

    def draw_damage_numbers(self, dt: float):
        """Рисует всплывающий урон"""
        for effect in self.damage_effects[:]:
            effect['lifetime'] -= dt

            if effect['lifetime'] <= 0:
                self.damage_effects.remove(effect)
                continue

            alpha = 1.0 - (effect['lifetime'] / effect['max_lifetime'])
            pos = self.hex_to_pixel(effect['pos'])
            y_offset = -30 * alpha

            text = self.font_large.render(f"-{effect['damage']}",
                                          True, self.COLORS['damage_text'])
            text_rect = text.get_rect(center=(pos[0], pos[1] + y_offset))

            text.set_alpha(int(255 * (1 - alpha)))
            self.screen.blit(text, text_rect)

    def draw_info_panel(self, solution: Dict, units: List[Dict], targets: List[Dict]):
        panel_x = self.width - 250
        panel_y = 10

        panel_rect = pygame.Rect(panel_x - 10, panel_y - 10, 240, 300)
        pygame.draw.rect(self.screen, (30, 30, 40), panel_rect)
        pygame.draw.rect(self.screen, self.COLORS['hex_border'], panel_rect, 2)

        y = panel_y

        title = self.font_medium.render("Tactical Info", True, self.COLORS['text'])
        self.screen.blit(title, (panel_x, y))
        y += 35

        info_lines = [
            f"Units: {len(units)}",
            f"Targets: {len(targets)}",
            f"Assignments: {len(solution.get('assignments', []))}",
            f"Blocked: {len(solution.get('blocked_units', []))}",
        ]

        for line in info_lines:
            text = self.font_small.render(line, True, self.COLORS['text'])
            self.screen.blit(text, (panel_x, y))
            y += 25

        y += 20
        controls = [
            "Controls:",
            "SPACE - Play/Pause",
            "R - Reset",
            "G - Toggle Grid",
            "C - Toggle Coords",
            "ESC - Exit"
        ]

        for line in controls:
            text = self.font_small.render(line, True, self.COLORS['text'])
            self.screen.blit(text, (panel_x, y))
            y += 20

    def animate_solution(self, solution: Dict, units: List[Dict], targets: List[Dict]):
        for u_idx, path in solution["paths"].items():
            self.animation_state[u_idx] = {
                'state': AnimationState.IDLE,
                'path': path,
                'path_index': 0,
                'progress': 0.0,
                #'target_idx': assignment['target_idx']
            }
            for assignment in solution['assignments']:
                if assignment['unit_idx'] == u_idx:
                    self.animation_state[u_idx]['target_idx'] = assignment['target_idx']
                    break

        for target in targets:
            if 'current_hp' not in target:
                target['current_hp'] = target['hp']

        animation_phase = 0

        while self.running:
            dt = self.clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        return True
                    elif event.key == pygame.K_g:
                        self.show_grid = not self.show_grid
                    elif event.key == pygame.K_c:
                        self.show_coords = not self.show_coords

            if not self.paused:
                all_idle = True

                for u_idx, anim in self.animation_state.items():
                    if anim['state'] == AnimationState.MOVING:
                        all_idle = False
                        anim['progress'] += self.animation_speed * dt

                        if anim['progress'] >= 1.0:
                            anim['progress'] = 0.0
                            anim['path_index'] += 1

                            if anim['path_index'] >= len(anim['path']) - 1:
                                anim['state'] = AnimationState.IDLE
                                animation_phase = 1

                    elif anim['state'] == AnimationState.ATTACKING:
                        all_idle = False
                        anim['progress'] += dt * 2.0

                        if anim['progress'] >= 1.0:
                            if 'target_idx' in anim:
                                t_idx = anim['target_idx']
                                target = targets[t_idx]
                                unit = units[u_idx]

                                target['current_hp'] -= unit['damage']

                                self.damage_effects.append({
                                    'pos': target['pos'],
                                    'damage': unit['damage'],
                                    'lifetime': 1.0,
                                    'max_lifetime': 1.0
                                })
                            anim['state'] = AnimationState.IDLE
                            anim['progress'] = 0.0

                if animation_phase == 0 and all_idle:
                    for anim in self.animation_state.values():
                        if anim['path_index'] < len(anim['path']) - 1:
                            anim['state'] = AnimationState.MOVING

                elif animation_phase == 1 and all_idle:
                    for anim in self.animation_state.values():
                        if anim['progress'] == 0.0:
                            anim['state'] = AnimationState.ATTACKING
                    animation_phase = 2

            self.screen.fill(self.COLORS['background'])

            if self.show_grid:
                self.draw_grid()

            for u_idx, path in solution['paths'].items():
                self.draw_path(path)

            for target in targets:
                target_destroyed = target['current_hp'] <= 0
                self.draw_target(target['pos'], target, target_destroyed)

            for u_idx, unit in enumerate(units):
                if u_idx in self.animation_state:
                    anim = self.animation_state[u_idx]

                    if anim['state'] == AnimationState.MOVING:
                        current_idx = anim['path_index']
                        start_pos = self.hex_to_pixel(anim['path'][current_idx])
                        end_pos = self.hex_to_pixel(anim['path'][current_idx + 1])

                        x = start_pos[0] + (end_pos[0] - start_pos[0]) * anim['progress']
                        y = start_pos[1] + (end_pos[1] - start_pos[1]) * anim['progress']

                        unit_data = {'id': u_idx}
                        self.draw_unit(unit['start'], unit_data, (x, y))

                    elif anim['state'] == AnimationState.ATTACKING:
                        if 'target_idx' in anim:
                            unit_pos = anim['path'][-1]
                            target_pos = targets[anim['target_idx']]['pos']
                            self.draw_attack_effect(unit_pos, target_pos, anim['progress'])

                        unit_data = {'id': u_idx}
                        self.draw_unit(anim['path'][-1], unit_data)

                    else:
                        pos = anim['path'][anim['path_index']]
                        unit_data = {'id': u_idx}
                        self.draw_unit(pos, unit_data)
                else:
                    unit_data = {'id': u_idx}
                    self.draw_unit(unit['start'], unit_data)

            self.draw_damage_numbers(dt)

            self.draw_info_panel(solution, units, targets)

            if self.paused:
                pause_text = self.font_large.render("PAUSED", True, (255, 255, 0))
                pause_rect = pause_text.get_rect(center=(self.width // 2, 50))
                self.screen.blit(pause_text, pause_rect)

            pygame.display.flip()

        return False