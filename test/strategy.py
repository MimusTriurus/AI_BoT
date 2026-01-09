"""
Полная интегрированная система ИИ для Massive Assault 2
с географией, нейтральными странами и динамическими фронтами

Включает:
- ✅ Географическую систему с путями через нейтралов
- ✅ Динамическое создание/удаление фронтов
- ✅ GOAP планирование с учетом географии
- ✅ MCTS для выбора оптимальной стратегии
- ✅ Исправленные closure bugs
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple, Set
from enum import Enum
import math
import random
import copy
import heapq


# ============================================================================
#  ENUMS & CONSTANTS
# ============================================================================

class Allegiance(Enum):
    PLAYER = "player"
    ENEMY = "enemy"
    NEUTRAL = "neutral"


class TerrainType(Enum):
    LAND = "land"
    WATER = "water"
    MOUNTAIN = "mountain"


PHASE_ORDER = ["reveal", "guerrilla", "buy", "move_attack"]


def is_under_attack(state: GameState) -> bool:
    """Проверить, атакованы ли наши страны"""
    return any(c.under_attack for c in state.countries.values()
              if c.allegiance == Allegiance.PLAYER)


# ============================================================================
#  DATA STRUCTURES
# ============================================================================

@dataclass
class CountryData:
    """Статическая информация о стране"""
    country_id: str
    name: str
    neighbors: List[str]
    terrain: TerrainType
    base_income: int
    strategic_importance: int


@dataclass
class CountryState:
    """Динамическое состояние страны"""
    country_id: str
    allegiance: Allegiance
    bank: int
    income: int
    guerrilla_points: int
    fronts: List[str]
    occupied: bool = False
    under_attack: bool = False  # Страна подвергается вторжению


@dataclass
class FrontState:
    """Состояние фронта"""
    front_id: str
    my_strength: int
    enemy_strength: int
    strategic_value: int
    my_countries: List[str]
    enemy_countries: List[str]
    terrain_modifier: float = 1.0

    def is_active(self) -> bool:
        return self.my_strength > 0 and self.enemy_strength > 0


@dataclass
class Action:
    name: str
    front_id: Optional[str] = None
    country_id: Optional[str] = None


# ============================================================================
#  WORLD MAP - ГЕОГРАФИЯ
# ============================================================================

class WorldMap:
    """Карта мира с географией"""

    def __init__(self, countries: Dict[str, CountryData]):
        self.countries = countries
        self._path_cache: Dict[Tuple[str, str], Optional[List[str]]] = {}

    def get_neighbors(self, country_id: str) -> List[str]:
        if country_id not in self.countries:
            return []
        return self.countries[country_id].neighbors

    def are_neighbors(self, country1: str, country2: str) -> bool:
        if country1 not in self.countries or country2 not in self.countries:
            return False
        return country2 in self.countries[country1].neighbors

    def find_path(self, from_country: str, to_country: str,
                  country_states: Dict[str, CountryState],
                  allow_enemy_target: bool = True) -> Optional[List[str]]:
        """
        BFS поиск пути через контролируемые/нейтральные территории
        """
        if from_country == to_country:
            return [from_country]

        if from_country not in self.countries or to_country not in self.countries:
            return None

        # Проверяем кэш
        cache_key = (from_country, to_country)
        if cache_key in self._path_cache:
            cached = self._path_cache[cache_key]
            if cached and self._is_path_valid(cached, country_states):
                return cached

        # BFS
        queue = [(from_country, [from_country])]
        visited = {from_country}

        while queue:
            current, path = queue.pop(0)

            for neighbor in self.countries[current].neighbors:
                if neighbor in visited:
                    continue

                neighbor_state = country_states.get(neighbor)
                if not neighbor_state:
                    continue

                # Целевая страна
                if neighbor == to_country:
                    if allow_enemy_target or neighbor_state.allegiance != Allegiance.ENEMY:
                        result = path + [neighbor]
                        self._path_cache[cache_key] = result
                        return result
                    continue

                # Промежуточные страны: только свои или нейтральные
                if neighbor_state.allegiance == Allegiance.ENEMY:
                    continue

                new_path = path + [neighbor]
                visited.add(neighbor)
                queue.append((neighbor, new_path))

        self._path_cache[cache_key] = None
        return None

    def _is_path_valid(self, path: List[str],
                      country_states: Dict[str, CountryState]) -> bool:
        """Проверить, что путь всё ещё валиден"""
        for i in range(len(path) - 1):
            country = path[i]
            next_country = path[i + 1]

            # Промежуточные страны не должны быть вражескими
            if i < len(path) - 1:
                state = country_states.get(country)
                if state and state.allegiance == Allegiance.ENEMY:
                    return False

            # Проверяем соседство
            if not self.are_neighbors(country, next_country):
                return False

        return True

    def get_reachable_enemies(self, my_country: str,
                             country_states: Dict[str, CountryState],
                             max_distance: int = 4) -> List[str]:
        """Найти достижимых врагов"""
        reachable = []

        for country_id, state in country_states.items():
            if state.allegiance == Allegiance.ENEMY:
                path = self.find_path(my_country, country_id, country_states)
                if path and len(path) <= max_distance:
                    reachable.append(country_id)

        return reachable

    def invalidate_cache(self):
        """Очистить кэш путей при изменении контроля стран"""
        self._path_cache.clear()


# ============================================================================
#  FRONT MANAGEMENT - ДИНАМИЧЕСКОЕ СОЗДАНИЕ
# ============================================================================

def generate_front_id(my_countries: List[str], enemy_countries: List[str]) -> str:
    """Генерировать уникальный ID для фронта"""
    my_sorted = sorted(my_countries)
    enemy_sorted = sorted(enemy_countries)
    return f"front_{'_'.join(my_sorted)}_vs_{'_'.join(enemy_sorted)}"


def create_or_update_fronts(state) -> Dict[str, FrontState]:
    """
    Динамически создавать или обновлять фронты
    """
    new_fronts: Dict[str, FrontState] = {}

    # Группируем страны
    my_countries = [cid for cid, c in state.countries.items()
                   if c.allegiance == Allegiance.PLAYER]
    enemy_countries = [cid for cid, c in state.countries.items()
                      if c.allegiance == Allegiance.ENEMY]

    # Находим все возможные фронты
    front_pairs: Dict[Tuple[str, str], Dict] = {}

    for my_country in my_countries:
        reachable = state.world_map.get_reachable_enemies(my_country, state.countries)

        for enemy_country in reachable:
            path = state.world_map.find_path(my_country, enemy_country, state.countries)

            if path and len(path) <= 3:
                key = (my_country, enemy_country)
                front_pairs[key] = {
                    'my_countries': [my_country],
                    'enemy_countries': [enemy_country],
                    'path': path
                }

    # Создаем фронты
    for (my_c, enemy_c), data in front_pairs.items():
        my_list = data['my_countries']
        enemy_list = data['enemy_countries']

        front_id = generate_front_id(my_list, enemy_list)

        # Обновляем существующий или создаем новый
        if front_id in state.fronts:
            new_fronts[front_id] = state.fronts[front_id]
        else:
            # Вычисляем стратегическую ценность
            strategic_value = 0
            for eid in enemy_list:
                if eid in state.world_map.countries:
                    strategic_value += state.world_map.countries[eid].strategic_importance

            # Модификатор местности
            terrain_modifier = 1.0
            for eid in enemy_list:
                if eid in state.world_map.countries:
                    terrain = state.world_map.countries[eid].terrain
                    if terrain == TerrainType.MOUNTAIN:
                        terrain_modifier = min(terrain_modifier, 0.5)
                    elif terrain == TerrainType.WATER:
                        terrain_modifier = min(terrain_modifier, 0.7)

            new_fronts[front_id] = FrontState(
                front_id=front_id,
                my_strength=0,
                enemy_strength=0,
                strategic_value=strategic_value,
                my_countries=my_list,
                enemy_countries=enemy_list,
                terrain_modifier=terrain_modifier
            )

    # Оставляем только активные или потенциальные фронты
    return new_fronts


def can_deploy_to_front(country_id: str, front: FrontState,
                        world_map: WorldMap,
                        countries: Dict[str, CountryState]) -> bool:
    """Проверить, может ли страна развернуть войска на фронте"""
    if country_id not in countries:
        return False

    country_state = countries[country_id]
    if country_state.allegiance != Allegiance.PLAYER:
        return False

    # Участвует напрямую
    if country_id in front.my_countries:
        return True

    # Граничит с участником
    for front_country in front.my_countries:
        if world_map.are_neighbors(country_id, front_country):
            return True

    return False


# ============================================================================
#  GAME STATE
# ============================================================================

@dataclass
class GameState:
    world_map: WorldMap
    fronts: Dict[str, FrontState]
    countries: Dict[str, CountryState]
    turn: int
    phase: str


def has_any_guerrilla_points(state: GameState) -> bool:
    """Можно ли использовать guerrilla - если есть очки И кто-то атакован"""
    has_points = any(c.guerrilla_points > 0 for c in state.countries.values()
                    if c.allegiance == Allegiance.PLAYER)
    return has_points and is_under_attack(state)


def next_phase(phase: str, state: GameState) -> str:
    idx = PHASE_ORDER.index(phase)
    while True:
        idx = (idx + 1) % len(PHASE_ORDER)
        candidate = PHASE_ORDER[idx]
        if candidate == "guerrilla" and not has_any_guerrilla_points(state):
            continue
        return candidate


# ============================================================================
#  ACTIONS & SIMULATION
# ============================================================================

def apply_primitive_action(state: GameState, action: Action) -> GameState:
    new_state = copy.deepcopy(state)

    if action.name == "reveal_allies" and action.front_id in new_state.fronts:
        f = new_state.fronts[action.front_id]
        f.my_strength += 2

    elif action.name == "guerrilla_buy" and action.front_id in new_state.fronts and action.country_id in new_state.countries:
        f = new_state.fronts[action.front_id]
        c = new_state.countries[action.country_id]

        if not can_deploy_to_front(action.country_id, f, new_state.world_map, new_state.countries):
            return new_state

        cost = 2
        if c.guerrilla_points >= cost:
            c.guerrilla_points -= cost
            f.my_strength += 2

    elif action.name == "buy_unit" and action.front_id in new_state.fronts and action.country_id in new_state.countries:
        f = new_state.fronts[action.front_id]
        c = new_state.countries[action.country_id]

        if not can_deploy_to_front(action.country_id, f, new_state.world_map, new_state.countries):
            return new_state

        cost = 3
        if c.bank >= cost:
            c.bank -= cost
            f.my_strength += 3

    elif action.name == "attack_front" and action.front_id in new_state.fronts:
        f = new_state.fronts[action.front_id]
        effective_attack = int(f.my_strength * f.terrain_modifier)

        if effective_attack > f.enemy_strength:
            f.my_strength -= f.enemy_strength // 2
            f.enemy_strength = 0

            # Захват вражеских стран
            for enemy_country_id in f.enemy_countries:
                if enemy_country_id in new_state.countries:
                    enemy_country = new_state.countries[enemy_country_id]
                    enemy_country.allegiance = Allegiance.PLAYER
                    enemy_country.occupied = True
                    # Инвалидируем кэш путей
                    new_state.world_map.invalidate_cache()
        else:
            f.enemy_strength -= effective_attack // 2
            f.my_strength = 0

            # Отмечаем наши страны как атакованные
            for my_country_id in f.my_countries:
                if my_country_id in new_state.countries:
                    new_state.countries[my_country_id].under_attack = True

    return new_state


def apply_plan(state: GameState, plan: List[Action]) -> GameState:
    """
    Применить план действий за одну фазу
    """
    new_state = copy.deepcopy(state)
    for act in plan:
        new_state = apply_primitive_action(new_state, act)

    new_state.phase = next_phase(new_state.phase, new_state)
    new_state.fronts = create_or_update_fronts(new_state)

    return new_state


def execute_full_turn(state: GameState, plans_by_phase: Dict[str, List[Action]]) -> GameState:
    """
    Выполнить полный ход игрока (все фазы) + ход врага

    plans_by_phase: {
        "reveal": [Action(...)],
        "guerrilla": [...],
        "buy": [...],
        "move_attack": [...]
    }
    """
    new_state = copy.deepcopy(state)

    # Сброс флагов атаки перед началом хода
    for c in new_state.countries.values():
        c.under_attack = False

    # Проходим все фазы в порядке
    for phase in PHASE_ORDER:
        new_state.phase = phase

        # Пропускаем guerrilla если нет вторжения
        if phase == "guerrilla" and not has_any_guerrilla_points(new_state):
            continue

        # Применяем план для текущей фазы
        plan = plans_by_phase.get(phase, [])
        for action in plan:
            new_state = apply_primitive_action(new_state, action)

        # Обновляем фронты после каждой фазы
        new_state.fronts = create_or_update_fronts(new_state)

    # Ход врага
    new_state = simulate_enemy_turn(new_state)
    new_state.turn += 1
    new_state.phase = "reveal"  # Начинаем новый ход с reveal

    # Финальное обновление фронтов
    new_state.fronts = create_or_update_fronts(new_state)

    return new_state


def simulate_enemy_turn(state: GameState) -> GameState:
    """Симуляция хода врага"""
    new_state = copy.deepcopy(state)

    # Сброс флагов атаки
    for c in new_state.countries.values():
        c.under_attack = False

    for fid, f in new_state.fronts.items():
        if f.enemy_strength > f.my_strength * 1.2:
            # Враг атакует
            effective_attack = int(f.enemy_strength * f.terrain_modifier)
            if effective_attack > f.my_strength:
                f.enemy_strength -= f.my_strength // 2
                f.my_strength = 0

                # Враг захватил наши страны
                for my_country_id in f.my_countries:
                    if my_country_id in new_state.countries:
                        my_country = new_state.countries[my_country_id]
                        my_country.allegiance = Allegiance.ENEMY
                        new_state.world_map.invalidate_cache()
            else:
                f.my_strength -= effective_attack // 2
                f.enemy_strength = 0

                # Мы отразили атаку, но страны были под угрозой
                for my_country_id in f.my_countries:
                    if my_country_id in new_state.countries:
                        new_state.countries[my_country_id].under_attack = True
        elif f.enemy_strength < f.my_strength:
            # Враг усиливается
            f.enemy_strength += 2
        else:
            # Поддержание паритета
            f.enemy_strength += 1

    return new_state


def evaluate_state(state: GameState) -> float:
    score = 0.0

    for f in state.fronts.values():
        control = f.my_strength - f.enemy_strength
        score += control * f.strategic_value

        if f.enemy_strength > f.my_strength * 2:
            score -= f.strategic_value * 5

        if f.enemy_strength == 0 and f.my_strength > 0:
            score += f.strategic_value * 3

    for cid, c in state.countries.items():
        if c.allegiance == Allegiance.PLAYER:
            score += c.bank * 0.5
            score += c.income * 0.3
            score += c.guerrilla_points * 0.2

            if cid in state.world_map.countries:
                score += state.world_map.countries[cid].strategic_importance * 2

    return score


# ============================================================================
#  GOAP PLANNING (С ИСПРАВЛЕННЫМИ ЗАМЫКАНИЯМИ)
# ============================================================================

@dataclass
class GoapAction:
    name: str
    cost: int
    apply_fn: Callable[[GameState], GameState]
    front_id: Optional[str] = None
    country_id: Optional[str] = None


def goap_plan(start: GameState,
              goal_pred: Callable[[GameState], bool],
              actions: List[GoapAction],
              max_depth: int = 3,
              max_nodes: int = 64) -> List[Action]:
    """GOAP планировщик"""
    counter = 0
    frontier: List[Tuple[float, int, GameState, List[Action]]] = []
    heapq.heappush(frontier, (0.0, counter, copy.deepcopy(start), []))

    best_plan: List[Action] = []
    best_score = float('-inf')

    while frontier and counter < max_nodes:
        cost, _, state, plan = heapq.heappop(frontier)

        if goal_pred(state):
            return plan

        if len(plan) >= max_depth:
            current_score = evaluate_state(state)
            if current_score > best_score:
                best_score = current_score
                best_plan = plan
            continue

        for ga in actions:
            new_state = ga.apply_fn(state)
            new_plan = plan + [Action(
                name=ga.name,
                front_id=ga.front_id,
                country_id=ga.country_id
            )]
            counter += 1

            heuristic = evaluate_state(new_state)
            priority = cost + ga.cost - heuristic * 0.1

            heapq.heappush(frontier, (priority, counter, new_state, new_plan))

    return best_plan


def goap_plans_for_state(state: GameState) -> List[List[Action]]:
    """
    Генерация планов с учетом географии
    ⭐ ВСЕ ЗАМЫКАНИЯ ИСПРАВЛЕНЫ
    """
    plans: List[List[Action]] = []

    if state.phase == "reveal":
        actions: List[GoapAction] = []
        for fid in state.fronts.keys():
            def make_apply(front_id: str) -> Callable[[GameState], GameState]:
                def _apply(st: GameState) -> GameState:
                    return apply_primitive_action(st, Action(name="reveal_allies", front_id=front_id))
                return _apply

            actions.append(GoapAction(
                name="reveal_allies",
                cost=1,
                apply_fn=make_apply(fid),
                front_id=fid
            ))

        if not actions:
            return [[]]

        base_score = evaluate_state(state)

        def goal(s: GameState) -> bool:
            return evaluate_state(s) >= base_score + 2.0

        plan = goap_plan(state, goal, actions, max_depth=2)
        if plan:
            plans.append(plan)

    elif state.phase == "guerrilla":
        actions: List[GoapAction] = []

        for cid, c in state.countries.items():
            if c.allegiance != Allegiance.PLAYER or c.guerrilla_points < 2:
                continue

            for fid, f in state.fronts.items():
                if not can_deploy_to_front(cid, f, state.world_map, state.countries):
                    continue

                def make_apply(front_id: str, country_id: str) -> Callable[[GameState], GameState]:
                    def _apply(st: GameState) -> GameState:
                        return apply_primitive_action(st, Action(
                            name="guerrilla_buy",
                            front_id=front_id,
                            country_id=country_id
                        ))
                    return _apply

                actions.append(GoapAction(
                    name="guerrilla_buy",
                    cost=1,
                    apply_fn=make_apply(fid, cid),
                    front_id=fid,
                    country_id=cid
                ))

        if actions:
            total_points = sum(c.guerrilla_points for c in state.countries.values()
                             if c.allegiance == Allegiance.PLAYER)

            def goal(s: GameState) -> bool:
                current = sum(c.guerrilla_points for c in s.countries.values()
                            if c.allegiance == Allegiance.PLAYER)
                return current < total_points - 2

            plan = goap_plan(state, goal, actions, max_depth=3)
            if plan:
                plans.append(plan)

    elif state.phase == "buy":
        actions: List[GoapAction] = []

        for cid, c in state.countries.items():
            if c.allegiance != Allegiance.PLAYER:
                continue

            for fid, f in state.fronts.items():
                if not can_deploy_to_front(cid, f, state.world_map, state.countries):
                    continue

                def make_apply(front_id: str, country_id: str) -> Callable[[GameState], GameState]:
                    def _apply(st: GameState) -> GameState:
                        cs = st.countries[country_id]
                        if cs.guerrilla_points >= 2:
                            return apply_primitive_action(st, Action(
                                name="guerrilla_buy",
                                front_id=front_id,
                                country_id=country_id
                            ))
                        elif cs.bank >= 3:
                            return apply_primitive_action(st, Action(
                                name="buy_unit",
                                front_id=front_id,
                                country_id=country_id
                            ))
                        return st
                    return _apply

                actions.append(GoapAction(
                    name="buy_or_guerrilla",
                    cost=1,
                    apply_fn=make_apply(fid, cid),
                    front_id=fid,
                    country_id=cid
                ))

        if actions:
            base_score = evaluate_state(state)

            def goal(s: GameState) -> bool:
                return evaluate_state(s) >= base_score + 2.0

            plan = goap_plan(state, goal, actions, max_depth=3)
            if plan:
                plans.append(plan)

    elif state.phase == "move_attack":
        actions: List[GoapAction] = []

        for fid, f in state.fronts.items():
            if f.my_strength <= 0:
                continue

            def make_apply(front_id: str) -> Callable[[GameState], GameState]:
                def _apply(st: GameState) -> GameState:
                    return apply_primitive_action(st, Action(
                        name="attack_front",
                        front_id=front_id
                    ))
                return _apply

            actions.append(GoapAction(
                name="attack_front",
                cost=1,
                apply_fn=make_apply(fid),
                front_id=fid
            ))

        if actions:
            base_score = evaluate_state(state)

            def goal(s: GameState) -> bool:
                return evaluate_state(s) > base_score

            plan = goap_plan(state, goal, actions, max_depth=2)
            if plan:
                plans.append(plan)

    return plans if plans else [[]]


# ============================================================================
#  MCTS
# ============================================================================

class MCTSNode:
    def __init__(self, state: GameState, parent: Optional[MCTSNode] = None, plan: Optional[List[Action]] = None):
        self.state = state
        self.parent = parent
        self.plan = plan
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self.untried_plans: List[List[Action]] = goap_plans_for_state(state)

    def is_fully_expanded(self) -> bool:
        return len(self.untried_plans) == 0

    def best_child(self, c_param: float = 1.4) -> MCTSNode:
        if not self.children:
            return self

        choices = []
        for child in self.children:
            if child.visits == 0:
                uct = float('inf')
            else:
                exploitation = child.value / child.visits
                exploration = c_param * math.sqrt(math.log(self.visits + 1) / child.visits)
                uct = exploitation + exploration
            choices.append((uct, child))

        return max(choices, key=lambda x: x[0])[1]

    def expand(self) -> MCTSNode:
        if not self.untried_plans:
            return self

        plan = self.untried_plans.pop()
        next_state = apply_plan(self.state, plan)
        child = MCTSNode(next_state, parent=self, plan=plan)
        self.children.append(child)
        return child

    def backpropagate(self, reward: float):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def is_terminal(self) -> bool:
        return len(self.state.fronts) == 0 or self.state.turn >= 20


def rollout(state: GameState, max_depth: int = 10) -> float:
    """Улучшенный rollout с эвристикой"""
    current_state = copy.deepcopy(state)
    depth = 0

    while depth < max_depth:
        if len(current_state.fronts) == 0 or current_state.turn >= 20:
            break

        plans = goap_plans_for_state(current_state)
        if not plans:
            break

        if len(plans) == 1:
            plan = plans[0]
        else:
            if random.random() < 0.3:
                plan = random.choice(plans)
            else:
                plan_scores = [(evaluate_state(apply_plan(current_state, p)), p) for p in plans]
                plan = max(plan_scores, key=lambda x: x[0])[1]

        current_state = apply_plan(current_state, plan)
        depth += 1

    return evaluate_state(current_state)


def plan_full_turn(state: GameState, iterations: int = 300, verbose: bool = False) -> Dict[str, List[Action]]:
    """
    Планирование полного хода (все фазы)

    Возвращает словарь с планами для каждой фазы:
    {
        "reveal": [Action(...)],
        "guerrilla": [...],  # может быть пустым
        "buy": [...],
        "move_attack": [...]
    }
    """
    plans_by_phase: Dict[str, List[Action]] = {
        "reveal": [],
        "guerrilla": [],
        "buy": [],
        "move_attack": []
    }

    current_state = copy.deepcopy(state)
    current_state.phase = "reveal"

    if verbose:
        print(f"\n{'='*70}")
        print(f"ПЛАНИРОВАНИЕ ПОЛНОГО ХОДА (ход {current_state.turn})")
        print(f"{'='*70}\n")

    # Планируем каждую фазу последовательно
    for phase in PHASE_ORDER:
        current_state.phase = phase

        # Пропускаем guerrilla если нет вторжения
        if phase == "guerrilla" and not has_any_guerrilla_points(current_state):
            if verbose:
                print(f"⏭️  Фаза {phase}: пропущена (нет вторжения)")
            continue

        if verbose:
            print(f"\n🎯 Фаза: {phase}")
            print(f"   Оценка состояния: {evaluate_state(current_state):.2f}")

        # Генерируем планы для фазы
        phase_plans = goap_plans_for_state(current_state)

        if not phase_plans or not phase_plans[0]:
            if verbose:
                print(f"   ⚠️  Нет доступных действий")
            continue

        # Выбираем лучший план для фазы
        if len(phase_plans) == 1:
            best_plan = phase_plans[0]
        else:
            # Оцениваем каждый план
            plan_scores = []
            for plan in phase_plans:
                if not plan:
                    continue
                test_state = copy.deepcopy(current_state)
                for action in plan:
                    test_state = apply_primitive_action(test_state, action)
                score = evaluate_state(test_state)
                plan_scores.append((score, plan))

            if plan_scores:
                best_plan = max(plan_scores, key=lambda x: x[0])[1]
            else:
                best_plan = []

        if best_plan:
            plans_by_phase[phase] = best_plan

            if verbose:
                print(f"   ✅ План выбран ({len(best_plan)} действий):")
                for i, action in enumerate(best_plan, 1):
                    info = f"      {i}. {action.name}"
                    if action.front_id:
                        info += f" @ {action.front_id}"
                    if action.country_id:
                        info += f" from {action.country_id}"
                    print(info)

            # Применяем план и обновляем состояние для следующей фазы
            for action in best_plan:
                current_state = apply_primitive_action(current_state, action)
            current_state.fronts = create_or_update_fronts(current_state)

    if verbose:
        # Симулируем полный ход для предварительного просмотра
        preview_state = execute_full_turn(state, plans_by_phase)
        print(f"\n📊 ПРЕДВАРИТЕЛЬНЫЙ РЕЗУЛЬТАТ ХОДА:")
        print(f"   Начальная оценка: {evaluate_state(state):.2f}")
        print(f"   Конечная оценка: {evaluate_state(preview_state):.2f}")
        print(f"   Изменение: {evaluate_state(preview_state) - evaluate_state(state):+.2f}")
        print(f"\n{'='*70}\n")

    return plans_by_phase


def mcts(root_state: GameState, iterations: int = 500, verbose: bool = False) -> List[Action]:
    """MCTS с поддержкой географии - для одной фазы"""
    root = MCTSNode(root_state)

    for i in range(iterations):
        node = root

        while not node.is_terminal() and node.is_fully_expanded() and node.children:
            node = node.best_child()

        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()

        reward = rollout(node.state)
        node.backpropagate(reward)

        if verbose and (i + 1) % 100 == 0:
            best_value = max((c.value/c.visits if c.visits > 0 else 0) for c in root.children) if root.children else 0
            print(f"Iteration {i + 1}/{iterations}, Root visits: {root.visits}, Best value: {best_value:.2f}")

    if not root.children:
        plans = goap_plans_for_state(root_state)
        if not plans:
            return []
        plan_scores = [(evaluate_state(apply_plan(root_state, p)), p) for p in plans]
        return max(plan_scores, key=lambda x: x[0])[1]

    best = max(root.children, key=lambda c: c.visits)

    if verbose:
        print(f"\nBest plan: visits={best.visits}, avg_value={best.value/best.visits:.2f}")

    return best.plan or []


def plan_full_turn_mcts(state: GameState, iterations_per_phase: int = 200, verbose: bool = False) -> Dict[str, List[Action]]:
    """
    Планирование полного хода с использованием MCTS для каждой фазы

    Для каждой фазы запускает MCTS для выбора оптимального плана,
    затем применяет план и переходит к следующей фазе.

    Args:
        state: Текущее состояние игры
        iterations_per_phase: Количество итераций MCTS для каждой фазы
        verbose: Выводить отладочную информацию

    Returns:
        Словарь с планами для каждой фазы
    """
    plans_by_phase: Dict[str, List[Action]] = {
        "reveal": [],
        "guerrilla": [],
        "buy": [],
        "move_attack": []
    }

    current_state = copy.deepcopy(state)
    current_state.phase = "reveal"

    if verbose:
        print(f"\n{'='*70}")
        print(f"ПЛАНИРОВАНИЕ ПОЛНОГО ХОДА С MCTS (ход {current_state.turn})")
        print(f"{'='*70}\n")

    # Планируем каждую фазу последовательно с MCTS
    for phase in PHASE_ORDER:
        current_state.phase = phase

        # Пропускаем guerrilla если нет вторжения
        if phase == "guerrilla" and not has_any_guerrilla_points(current_state):
            if verbose:
                print(f"⏭️  Фаза {phase}: пропущена (нет вторжения)")
            continue

        if verbose:
            print(f"\n🎯 Фаза: {phase}")
            print(f"   Оценка состояния: {evaluate_state(current_state):.2f}")
            print(f"   Запуск MCTS ({iterations_per_phase} итераций)...")

        # Используем MCTS для выбора плана
        best_plan = mcts(current_state, iterations=iterations_per_phase, verbose=False)

        if best_plan:
            plans_by_phase[phase] = best_plan

            if verbose:
                print(f"   ✅ План выбран ({len(best_plan)} действий):")
                for i, action in enumerate(best_plan, 1):
                    info = f"      {i}. {action.name}"
                    if action.front_id:
                        info += f" @ {action.front_id}"
                    if action.country_id:
                        info += f" from {action.country_id}"
                    print(info)

            # Применяем план и обновляем состояние для следующей фазы
            for action in best_plan:
                current_state = apply_primitive_action(current_state, action)
            current_state.fronts = create_or_update_fronts(current_state)
        else:
            if verbose:
                print(f"   ⚠️  MCTS не нашел план")

    if verbose:
        # Симулируем полный ход для предварительного просмотра
        preview_state = execute_full_turn(state, plans_by_phase)
        print(f"\n📊 ПРЕДВАРИТЕЛЬНЫЙ РЕЗУЛЬТАТ ХОДА:")
        print(f"   Начальная оценка: {evaluate_state(state):.2f}")
        print(f"   Конечная оценка: {evaluate_state(preview_state):.2f}")
        print(f"   Изменение: {evaluate_state(preview_state) - evaluate_state(state):+.2f}")
        print(f"\n{'='*70}\n")

    return plans_by_phase


# ============================================================================
#  DEMO
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("СИСТЕМА ИИ ДЛЯ MASSIVE ASSAULT 2 - ПЛАНИРОВАНИЕ ПОЛНОГО ХОДА")
    print("="*70)

    # Создание мира
    countries_data = {
        "A":  CountryData("A",  "Альфа",   ["N1", "B"], TerrainType.LAND, 3, 5),
        "B":  CountryData("B",  "Бета",    ["A", "Y"],  TerrainType.LAND, 2, 4),
        "N1": CountryData("N1", "Нейтрал", ["A", "X"],  TerrainType.LAND, 1, 3),
        "X":  CountryData("X",  "Икс",     ["N1", "Y"], TerrainType.MOUNTAIN, 4, 7),
        "Y":  CountryData("Y",  "Игрек",   ["B", "X"],  TerrainType.LAND, 3, 6),
    }

    world_map = WorldMap(countries_data)

    countries_state = {
        "A":  CountryState("A", Allegiance.PLAYER, 15, 3, 2, []),
        "B":  CountryState("B", Allegiance.PLAYER, 10, 2, 4, []),
        "N1": CountryState("N1", Allegiance.NEUTRAL, 2, 1, 0, []),
        "X":  CountryState("X", Allegiance.ENEMY, 8, 4, 0, []),
        "Y":  CountryState("Y", Allegiance.ENEMY, 6, 3, 0, []),
    }

    state = GameState(
        world_map=world_map,
        fronts={},
        countries=countries_state,
        turn=0,
        phase="reveal"
    )

    # Создаем начальные фронты с войсками
    state.fronts = create_or_update_fronts(state)

    # Добавляем войска на фронты для демонстрации
    for i, (fid, front) in enumerate(state.fronts.items()):
        if i == 0:  # Главный фронт с преимуществом врага
            front.my_strength = 4
            front.enemy_strength = 7
        elif i == 1:  # Второй фронт с нашим преимуществом
            front.my_strength = 5
            front.enemy_strength = 3
        else:  # Остальные сбалансированные
            front.my_strength = 3
            front.enemy_strength = 3

    print(f"\n📍 КАРТА МИРА:")
    print("-" * 70)
    for cid, cdata in world_map.countries.items():
        cstate = countries_state[cid]
        allegiance_symbol = {"player": "🟢", "enemy": "🔴", "neutral": "⚪"}
        print(f"{allegiance_symbol[cstate.allegiance.value]} {cdata.name} ({cid})")
        print(f"   Соседи: {', '.join(cdata.neighbors)}")
        if cstate.allegiance == Allegiance.PLAYER:
            print(f"   Ресурсы: bank={cstate.bank}, guerrilla={cstate.guerrilla_points}")
        print(f"   Важность: {cdata.strategic_importance}/10")

    print(f"\n⚔️  НАЧАЛЬНОЕ СОСТОЯНИЕ:")
    print("-" * 70)
    print(f"Ход: {state.turn}")
    print(f"Оценка: {evaluate_state(state):.2f}\n")

    print(f"Фронты:")
    for fid, f in state.fronts.items():
        balance = "⚖️" if abs(f.my_strength - f.enemy_strength) <= 1 else "⚠️"
        my_bar = "█" * f.my_strength
        enemy_bar = "█" * f.enemy_strength
        print(f"\n  {balance} {fid}")
        print(f"     Участники: {', '.join(f.my_countries)} vs {', '.join(f.enemy_countries)}")
        print(f"     Наши:  {my_bar} ({f.my_strength})")
        print(f"     Враги: {enemy_bar} ({f.enemy_strength})")
        print(f"     Ценность: {f.strategic_value}, Модификатор: {f.terrain_modifier}x")

    # ========================================================================
    # ДЕМОНСТРАЦИЯ 1: GOAP (быстрое, эвристическое планирование)
    # ========================================================================
    print("\n" + "="*70)
    print("МЕТОД 1: GOAP (эвристическое планирование)")
    print("="*70)

    plans_goap = plan_full_turn(state, verbose=True)
    result_goap = execute_full_turn(state, plans_goap)

    print(f"\n📊 РЕЗУЛЬТАТ GOAP:")
    print(f"   Оценка: {evaluate_state(state):.2f} → {evaluate_state(result_goap):.2f}")
    print(f"   Изменение: {evaluate_state(result_goap) - evaluate_state(state):+.2f}")

    # ========================================================================
    # ДЕМОНСТРАЦИЯ 2: MCTS (медленнее, но более оптимально)
    # ========================================================================
    print("\n" + "="*70)
    print("МЕТОД 2: MCTS (поиск в дереве возможностей)")
    print("="*70)

    plans_mcts = plan_full_turn_mcts(state, iterations_per_phase=150, verbose=True)
    result_mcts = execute_full_turn(state, plans_mcts)

    print(f"\n📊 РЕЗУЛЬТАТ MCTS:")
    print(f"   Оценка: {evaluate_state(state):.2f} → {evaluate_state(result_mcts):.2f}")
    print(f"   Изменение: {evaluate_state(result_mcts) - evaluate_state(state):+.2f}")

    # ========================================================================
    # СРАВНЕНИЕ
    # ========================================================================
    print("\n" + "="*70)
    print("СРАВНЕНИЕ МЕТОДОВ")
    print("="*70)

    goap_improvement = evaluate_state(result_goap) - evaluate_state(state)
    mcts_improvement = evaluate_state(result_mcts) - evaluate_state(state)

    print(f"\n{'Метод':<20} {'Улучшение':>15} {'Победитель':>15}")
    print("-" * 52)
    print(f"{'GOAP':<20} {goap_improvement:>+15.2f} {'🏆' if goap_improvement >= mcts_improvement else ''}")
    print(f"{'MCTS':<20} {mcts_improvement:>+15.2f} {'🏆' if mcts_improvement > goap_improvement else ''}")

    if abs(goap_improvement - mcts_improvement) < 1.0:
        print(f"\n⚖️  Методы дали примерно одинаковый результат")
    elif mcts_improvement > goap_improvement:
        print(f"\n✅ MCTS нашел лучшее решение (+{mcts_improvement - goap_improvement:.2f} очков)")
    else:
        print(f"\n✅ GOAP оказался эффективнее (+{goap_improvement - mcts_improvement:.2f} очков)")

    # ========================================================================
    # СИМУЛЯЦИЯ ИГРЫ
    # ========================================================================
    print(f"\n\n{'='*70}")
    print("СИМУЛЯЦИЯ ИГРЫ (3 хода с MCTS)")
    print("="*70)

    current = copy.deepcopy(state)
    for turn_num in range(3):
        print(f"\n{'─'*70}")
        print(f"ХОД {turn_num + 1}")
        print(f"{'─'*70}")

        plans = plan_full_turn_mcts(current, iterations_per_phase=100, verbose=False)
        current = execute_full_turn(current, plans)

        print(f"Оценка: {evaluate_state(current):.2f}")
        print(f"Наши страны: {sum(1 for c in current.countries.values() if c.allegiance == Allegiance.PLAYER)}/5")
        print(f"Вражеские страны: {sum(1 for c in current.countries.values() if c.allegiance == Allegiance.ENEMY)}/5")
        print(f"Активных фронтов: {len([f for f in current.fronts.values() if f.is_active()])}")

        # Проверка победы/поражения
        player_countries = sum(1 for c in current.countries.values() if c.allegiance == Allegiance.PLAYER)
        enemy_countries = sum(1 for c in current.countries.values() if c.allegiance == Allegiance.ENEMY)

        if enemy_countries == 0:
            print("\n🎉 ПОБЕДА! Все вражеские страны захвачены!")
            break
        elif player_countries == 0:
            print("\n💀 ПОРАЖЕНИЕ! Все наши страны потеряны!")
            break

    print("\n" + "="*70)
    print("ЗАВЕРШЕНО")
    print("="*70)
    print("="*70)
    print("СИСТЕМА ИИ ДЛЯ MASSIVE ASSAULT 2 - ПЛАНИРОВАНИЕ ПОЛНОГО ХОДА")
    print("="*70)

    # Создание мира
    countries_data = {
        "A": CountryData("A", "Альфа", ["N1", "B"], TerrainType.LAND, 3, 5),
        "B": CountryData("B", "Бета", ["A", "Y"], TerrainType.LAND, 2, 4),
        "N1": CountryData("N1", "Нейтрал", ["A", "X"], TerrainType.LAND, 1, 3),
        "X": CountryData("X", "Икс", ["N1", "Y"], TerrainType.MOUNTAIN, 4, 7),
        "Y": CountryData("Y", "Игрек", ["B", "X"], TerrainType.LAND, 3, 6),
    }

    world_map = WorldMap(countries_data)

    countries_state = {
        "A": CountryState("A", Allegiance.PLAYER, 15, 3, 2, []),
        "B": CountryState("B", Allegiance.PLAYER, 10, 2, 4, []),
        "N1": CountryState("N1", Allegiance.NEUTRAL, 2, 1, 0, []),
        "X": CountryState("X", Allegiance.ENEMY, 8, 4, 0, []),
        "Y": CountryState("Y", Allegiance.ENEMY, 6, 3, 0, []),
    }

    state = GameState(
        world_map=world_map,
        fronts={},
        countries=countries_state,
        turn=0,
        phase="reveal"
    )

    # Создаем начальные фронты с войсками
    state.fronts = create_or_update_fronts(state)

    # Добавляем войска на фронты для демонстрации
    for i, (fid, front) in enumerate(state.fronts.items()):
        if i == 0:  # Главный фронт с преимуществом врага
            front.my_strength = 4
            front.enemy_strength = 7
        elif i == 1:  # Второй фронт с нашим преимуществом
            front.my_strength = 5
            front.enemy_strength = 3
        else:  # Остальные сбалансированные
            front.my_strength = 3
            front.enemy_strength = 3

    print(f"\n📍 КАРТА МИРА:")
    print("-" * 70)
    for cid, cdata in world_map.countries.items():
        cstate = countries_state[cid]
        allegiance_symbol = {"player": "🟢", "enemy": "🔴", "neutral": "⚪"}
        print(f"{allegiance_symbol[cstate.allegiance.value]} {cdata.name} ({cid})")
        print(f"   Соседи: {', '.join(cdata.neighbors)}")
        if cstate.allegiance == Allegiance.PLAYER:
            print(f"   Ресурсы: bank={cstate.bank}, guerrilla={cstate.guerrilla_points}")
        print(f"   Важность: {cdata.strategic_importance}/10")

    print(f"\n⚔️  НАЧАЛЬНОЕ СОСТОЯНИЕ:")
    print("-" * 70)
    print(f"Ход: {state.turn}")
    print(f"Оценка: {evaluate_state(state):.2f}\n")

    print(f"Фронты:")
    for fid, f in state.fronts.items():
        balance = "⚖️" if abs(f.my_strength - f.enemy_strength) <= 1 else "⚠️"
        my_bar = "█" * f.my_strength
        enemy_bar = "█" * f.enemy_strength
        print(f"\n  {balance} {fid}")
        print(f"     Участники: {', '.join(f.my_countries)} vs {', '.join(f.enemy_countries)}")
        print(f"     Наши:  {my_bar} ({f.my_strength})")
        print(f"     Враги: {enemy_bar} ({f.enemy_strength})")
        print(f"     Ценность: {f.strategic_value}, Модификатор: {f.terrain_modifier}x")

    # Планирование полного хода
    print("\n" + "="*70)
    plans = plan_full_turn(state, verbose=True)

    # Выполнение полного хода
    print("\n🎬 ВЫПОЛНЕНИЕ ХОДА...")
    print("-" * 70)

    result_state = execute_full_turn(state, plans)

    # Результаты
    print(f"\n📊 РЕЗУЛЬТАТЫ ПОСЛЕ ХОДА:")
    print("-" * 70)
    print(f"Ход: {result_state.turn}")
    print(f"Начальная оценка: {evaluate_state(state):.2f}")
    print(f"Конечная оценка: {evaluate_state(result_state):.2f}")
    print(f"Изменение: {evaluate_state(result_state) - evaluate_state(state):+.2f}\n")

    print(f"Фронты после хода:")
    for fid, f in result_state.fronts.items():
        my_bar = "█" * f.my_strength
        enemy_bar = "█" * f.enemy_strength
        status = "✅ ПОБЕДА" if f.enemy_strength == 0 else "⚔️  БОЙ" if f.my_strength > 0 else "❌ ПОРАЖЕНИЕ"
        print(f"\n  {status} {fid}")
        print(f"     Наши:  {my_bar} ({f.my_strength})")
        print(f"     Враги: {enemy_bar} ({f.enemy_strength})")

    print(f"\nСтатус стран:")
    for cid, c in result_state.countries.items():
        if c.allegiance == Allegiance.PLAYER:
            status = "🟢"
            extra = f", bank={c.bank}, guerrilla={c.guerrilla_points}"
            if c.under_attack:
                status += " ⚠️ ПОД АТАКОЙ"
        elif c.allegiance == Allegiance.ENEMY:
            status = "🔴"
            extra = ""
        else:
            status = "⚪"
            extra = ""

        print(f"  {status} {cid}{extra}")

    # Демонстрация нескольких ходов
    print(f"\n\n{'='*70}")
    print("СИМУЛЯЦИЯ НЕСКОЛЬКИХ ХОДОВ")
    print("="*70)

    current = copy.deepcopy(state)
    for turn_num in range(3):
        print(f"\n--- ХОД {turn_num + 1} ---")
        plans = plan_full_turn(current, verbose=False)
        current = execute_full_turn(current, plans)

        print(f"Оценка: {evaluate_state(current):.2f}")
        print(f"Наши страны: {sum(1 for c in current.countries.values() if c.allegiance == Allegiance.PLAYER)}")
        print(f"Вражеские страны: {sum(1 for c in current.countries.values() if c.allegiance == Allegiance.ENEMY)}")
        print(f"Активных фронтов: {len([f for f in current.fronts.values() if f.is_active()])}")

    print("\n" + "="*70)
    print("ЗАВЕРШЕНО")
    print("="*70)