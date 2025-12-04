import json
import socket
import math
from typing import List, Dict, Optional, Any

import requests  # shared moderator HTTP client

from env import SupermarketEnv  # (still unused but harmless)
from utilsCopy import *
from utilsCopy import can_interact_default

# ============================================================
# PhaseMonitor (unchanged logic; works per PLAYER_ID)
# ============================================================

class PhaseMonitor:
    """
    Read-only phase validator for 1 or more agents.
    Tracks whether each agent has completed:
      - container
      - shopping
      - checkout
      - exit
    Does NOT alter control flow. Just records + logs.
    """

    def __init__(self, num_players=1, verbose=True):
        self.num_players = num_players
        self.verbose = verbose

        self.phase_status = {
            pid: {"container": False, "shopping": False, "checkout": False, "exit": False}
            for pid in range(num_players)
        }
        self.prev_out = {pid: None for pid in range(num_players)}

    def get_player(self, obs, player_id):
        return obs["players"][player_id]

    def get_owned_basket(self, obs, player_id):
        for b in obs.get("baskets", []):
            if b.get("owner") == player_id:
                return b
        return None

    def get_owned_cart(self, obs, player_id):
        p = self.get_player(obs, player_id)
        idx = p.get("curr_cart", -1)
        if idx != -1 and idx < len(obs.get("carts", [])):
            return obs["carts"][idx]
        return None

    def get_container(self, obs, player_id):
        cart = self.get_owned_cart(obs, player_id)
        if cart is not None:
            return cart, "cart"
        basket = self.get_owned_basket(obs, player_id)
        if basket is not None:
            return basket, "basket"
        return None, None

    def container_is_correct(self, prev_out, out, player_id):
        obs = out["observation"]
        p = self.get_player(obs, player_id)
        n_items = len(p.get("shopping_list", []))
        if n_items <= 6:
            basket = self.get_owned_basket(obs, player_id)
            return basket is not None
        else:
            return p.get("curr_cart", -1) != -1

    def pickup_succeeded(self, prev_out, out, food, player_id):
        if prev_out is None:
            return False

        prev_obs = prev_out["observation"]
        obs = out["observation"]

        # shelf quantity drop
        for s_prev, s_cur in zip(prev_obs.get("shelves", []), obs.get("shelves", [])):
            if s_prev.get("food") == food and s_cur.get("food") == food:
                if s_cur.get("quantity", 0) < s_prev.get("quantity", 0):
                    return True

        # counter quantity drop
        for c_prev, c_cur in zip(prev_obs.get("counters", []), obs.get("counters", [])):
            if c_prev.get("food") == food and c_cur.get("food") == food:
                if c_prev.get("quantity") is not None and c_cur.get("quantity") is not None:
                    if c_cur["quantity"] < c_prev["quantity"]:
                        return True

        # holding_food
        p = self.get_player(obs, player_id)
        if p.get("holding_food") == food:
            return True

        # in container
        container, _ = self.get_container(obs, player_id)
        if container and food in container.get("contents", []):
            return True

        return False

    def quantities_match_list(self, out, player_id):
        obs = out["observation"]
        p = self.get_player(obs, player_id)

        target_items = p.get("shopping_list", [])
        target_quants = p.get("list_quant", [])

        container, _ = self.get_container(obs, player_id)
        if container is None:
            return False

        got_items = container.get("contents", [])
        got_quants = container.get("contents_quant", [])

        return {i: q for i, q in zip(got_items, got_quants)} == \
               {i: q for i, q in zip(target_items, target_quants)}

    def checkout_succeeded(self, prev_out, out, player_id):
        obs = out["observation"]
        p = self.get_player(obs, player_id)

        if p.get("budget", 100) != 100:
            return True

        container, _ = self.get_container(obs, player_id)
        if container and len(container.get("purchased_contents", [])) > 0:
            return True

        return False

    def exit_succeeded(self, out, player_id):
        obs = out["observation"]
        p = obs["players"][player_id]
        return p["position"][0] < 0

    def update_prev(self, out, player_id):
        self.prev_out[player_id] = out

    def mark_container_phase(self, prev_out, out, player_id):
        ok = self.container_is_correct(prev_out, out, player_id)
        self.phase_status[player_id]["container"] = ok
        if self.verbose:
            print(f"[P{player_id} CHECK] container: {ok}")
        return ok

    def mark_pickup(self, prev_out, out, food, player_id):
        ok = self.pickup_succeeded(prev_out, out, food, player_id)
        if self.verbose:
            print(f"[P{player_id} CHECK] pickup {food}: {ok}")
        return ok

    def mark_shopping_phase(self, out, player_id):
        ok = self.quantities_match_list(out, player_id)
        self.phase_status[player_id]["shopping"] = ok
        if self.verbose:
            print(f"[P{player_id} CHECK] shopping quantities correct: {ok}")
        return ok

    def mark_checkout_phase(self, prev_out, out, player_id):
        ok = self.checkout_succeeded(prev_out, out, player_id)
        self.phase_status[player_id]["checkout"] = ok
        if self.verbose:
            print(f"[P{player_id} CHECK] checkout: {ok}")
        return ok

    def mark_exit_phase(self, out, player_id):
        ok = self.exit_succeeded(out, player_id)
        self.phase_status[player_id]["exit"] = ok
        if self.verbose:
            print(f"[P{player_id} CHECK] exit: {ok}")
        return ok

    def summary(self):
        for pid in range(self.num_players):
            st = self.phase_status[pid]
            print(
                f"[P{pid} STATUS] "
                f"container={st['container']} shopping={st['shopping']} "
                f"checkout={st['checkout']} exit={st['exit']}"
            )

# ============================================================
# localize_state (unchanged)
# ============================================================

def localize_state(output: dict, player_id: int) -> dict:
    """
    Make observation['players'][0] = controlled player.
    """
    obs = output.get("observation", {})
    players = obs.get("players", [])
    if not players or player_id >= len(players):
        return output

    new_players = [players[player_id]] + [p for i, p in enumerate(players) if i != player_id]
    new_output = dict(output)
    new_obs = dict(obs)
    new_obs["players"] = new_players
    new_output["observation"] = new_obs
    return new_output

# ============================================================
# NormRiskAssessor
# ============================================================

class NormRiskAssessor:
    PERSONAL_SPACE_THRESH = 0.9
    COLLISION_THRESH = 0.8
    UNATTENDED_DIST = 5.0
    UNATTENDED_STEPS = 5

    # ---- NEW: cart / basket return zone (interaction area) ----
    CONTAINER_X_MIN = 0.1
    CONTAINER_X_MAX = 5.0
    CONTAINER_Y_MIN = 15.0
    CONTAINER_Y_MAX = 18.5
    # -----------------------------------------------------------

    def __init__(self, player_id: int):
        self.player_id = player_id
        self.unattended_counter = 0

    def _dist(self, a, b) -> float:
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def _in_container_zone(self, pos) -> bool:
        x, y = pos
        return (
            self.CONTAINER_X_MIN <= x <= self.CONTAINER_X_MAX
            and self.CONTAINER_Y_MIN <= y <= self.CONTAINER_Y_MAX
        )

    def _predict_next_pos(self, ppos, direction_int, action: str, speed=0.15):
        dir_map = {
            "NORTH": (0, -1, 0),
            "SOUTH": (0,  1, 1),
            "EAST":  (1,  0, 2),
            "WEST":  (-1, 0, 3),
        }
        if action not in dir_map:
            return ppos, direction_int

        dx, dy, dir_int = dir_map[action]
        if direction_int != dir_int:
            return ppos, dir_int
        return [ppos[0] + speed * dx, ppos[1] + speed * dy], dir_int

    def _owned_container_pos(self, obs):
        p = obs["players"][0]
        cart_idx = p.get("curr_cart", -1)
        if cart_idx != -1 and cart_idx < len(obs.get("carts", [])):
            return obs["carts"][cart_idx]["position"]

        for b in obs.get("baskets", []):
            if b.get("owner") == self.player_id:
                return b["position"]
        return None

    def _register_zone_risk(self, obs, next_pos) -> bool:
        for r in obs.get("registers", []):
            fake_player = {
                "position": next_pos,
                "width": 0.6,
                "height": 0.4,
                "direction": 3  # WEST
            }
            if can_interact_default(r, fake_player, range=0.5):
                for op in obs["players"][1:]:
                    if can_interact_default(r, op, range=0.5):
                        return True
        return False

    def _exit_zone_risk(self, obs, next_pos) -> bool:
        ex = exit  # assuming global exit position is defined elsewhere
        if self._dist(next_pos, ex) < 0.8:
            for op in obs["players"][1:]:
                if self._dist(op["position"], ex) < 0.8:
                    return True
        return False

    def _personal_space_risk(self, obs, next_pos) -> bool:
        return any(
            self._dist(next_pos, op["position"]) < self.PERSONAL_SPACE_THRESH
            for op in obs["players"][1:]
        )

    def _player_collision_risk(self, obs, next_pos) -> bool:
        return any(
            self._dist(next_pos, op["position"]) < self.COLLISION_THRESH
            for op in obs["players"][1:]
        )

    def _unattended_risk(self, obs) -> bool:
        p = obs["players"][0]
        cont_pos = self._owned_container_pos(obs)
        if cont_pos is None:
            self.unattended_counter = 0
            return False

        if p.get("curr_cart", -1) != -1:
            self.unattended_counter = 0
            return False

        d = self._dist(p["position"], cont_pos)
        self.unattended_counter = self.unattended_counter + 1 if d >= self.UNATTENDED_DIST else 0
        return self.unattended_counter >= (self.UNATTENDED_STEPS - 1)

    def _wrong_item_or_too_many_risk(self, obs, food):
        if food is None:
            return []
        p = obs["players"][0]
        target_dict = {i: q for i, q in zip(p.get("shopping_list", []), p.get("list_quant", []))}

        flags = []
        if food not in target_dict:
            flags.append("ADHERE_TO_LIST_RISK")

        got = {}
        cart_idx = p.get("curr_cart", -1)
        if cart_idx != -1:
            cart = obs["carts"][cart_idx]
            got = {i: q for i, q in zip(cart.get("contents", []), cart.get("contents_quant", []))}
        else:
            basket = next(
                (b for b in obs.get("baskets", []) if b.get("owner") == self.player_id),
                None
            )
            if basket:
                got = {
                    i: q for i, q in zip(
                        basket.get("contents", []), basket.get("contents_quant", [])
                    )
                }

        if food in target_dict and got.get(food, 0) >= target_dict[food]:
            flags.append("TOOK_TOO_MANY_RISK")

        return flags

    def _wall_collision_risk(self, next_pos):
        x, y = next_pos
        # approximate map bounds
        return (x <= 0.1 or x >= 19.0 or y <= 2.2 or y >= 24.2)

    def _object_collision_risk(self, obs, next_pos, phase=None, goal=None, food=None):
        """
        Geometric object collision risk.

        Special rule:
          - In SHOPPING phase, when moving toward a specific shelf/counter
            (goal is not a walkway / None), we *suppress* object collision.
            This lets the moderator treat it as a pure coordination problem
            (only PLAYER / PERSONAL_SPACE risks remain).
        """

        # 0) Do not consider object collision inside the container zone
        if self._in_container_zone(next_pos):
            return False

        # 1) If we're in shopping and actively approaching a target (shelf/counter),
        #    relax object collision so we don't locally veto the moderator's
        #    decision about who yields at the shelf.
        #
        #    Goals like "walkway", "east_walkway", "west_walkway" are just transit,
        #    not a specific shelf/counter.
        if phase == "shopping" and goal not in (None, "walkway", "east_walkway", "west_walkway"):
            return False

        # 2) Normal object collision against shelves/counters/registers/carts/baskets
        obstacles = []
        obstacles += obs.get("shelves", [])
        obstacles += obs.get("counters", [])
        obstacles += obs.get("registers", [])
        obstacles += obs.get("carts", [])
        obstacles += obs.get("baskets", [])

        for o in obstacles:
            op = o.get("position")
            if op is None:
                continue
            if self._dist(next_pos, op) < 0.5:  # body buffer
                return True
        return False
    

    def assess(self, prev_out, out, baseline_action, phase, goal=None, food=None):
        obs = out["observation"]
        p = obs["players"][0]

        next_pos, _ = self._predict_next_pos(
            ppos=p["position"],
            direction_int=p.get("direction", 0),
            action=baseline_action
        )

        flags = []

        # --- multi-agent / social risks ---
        if baseline_action in ["NORTH", "SOUTH", "EAST", "WEST"]:
            if self._player_collision_risk(obs, next_pos):
                flags.append("PLAYER_COLLISION_RISK")
            elif self._personal_space_risk(obs, next_pos):
                flags.append("PERSONAL_SPACE_RISK")

        if phase == "checkout" and baseline_action in ["NORTH", "SOUTH", "EAST", "WEST"]:
            if self._register_zone_risk(obs, next_pos):
                flags.append("WAIT_FOR_CHECKOUT_RISK")

        if phase == "exit" and baseline_action in ["NORTH", "SOUTH", "EAST", "WEST"]:
            if self._exit_zone_risk(obs, next_pos):
                flags.append("BLOCKING_EXIT_RISK")

        # unattended cart/basket
        if self._unattended_risk(obs):
            flags.append("UNATTENDED_CONTAINER_RISK")

        # wrong item / too many
        if baseline_action == "INTERACT" and food is not None:
            flags.extend(self._wrong_item_or_too_many_risk(obs, food))

        # --- object / wall collision ---
        if baseline_action in ["NORTH", "SOUTH", "EAST", "WEST"]:
            # 1) never consider object collision in 'container' phase
            if phase == "container":
                pass
            else:
                # 2) special-case: leaving the container zone toward the walkway
                ignore_objects_here = (
                    phase == "shopping"
                    and goal == "walkway"
                    and self._in_container_zone(p["position"])
                )

                if not ignore_objects_here:
                    wall_risk = self._wall_collision_risk(next_pos)
                    obj_risk = self._object_collision_risk(
                        obs, next_pos, phase=phase, goal=goal, food=food
                    )
                    if wall_risk or obj_risk:
                        flags.append("OBJECT_COLLISION_RISK")
                        
        return list(dict.fromkeys(flags))


# ============================================================
# Shared AI Moderator Client
# ============================================================

COORDINATION_RISKS = {
    "PLAYER_COLLISION_RISK",
    "PERSONAL_SPACE_RISK",
    "BLOCKING_EXIT_RISK",
    "WAIT_FOR_CHECKOUT_RISK",
}

class SharedModeratorClient:
    """
    Talks to ONE shared moderator server at /moderate.

    The server:
      - Decides which player should yield based on the situation
        (e.g., who is closer to exit/register/shelf).
      - Returns a horizon ONLY for the yielding player, containing
        NOPs and/or detour moves.
      - For the other player, returns no plan (client uses baseline).
      - Also handles single-agent (object/unattended) risks.
    """

    def __init__(self, enabled=True, horizon_k=5, verbose=False):
        self.enabled = enabled
        self.horizon_k = horizon_k
        self.verbose = verbose
        self.url = "http://localhost:8000/moderate"
        self.cached_plan: List[str] = []

    def query_plan(self, obs: dict, player_id: int, phase: str,
                   baseline_action: str, risk_flags: List[str],
                   goal=None, food=None) -> List[str]:
        """
        Returns a horizon list IF this player is designated yielder;
        otherwise returns [] meaning "no special plan, use baseline".
        """
        if not self.enabled:
            return []

        needs_coordination = any(r in COORDINATION_RISKS for r in risk_flags)

        payload = {
            "player_id": player_id,
            "phase": phase,
            "baseline_action": baseline_action,
            "risk_flags": risk_flags,
            "observation": obs,
            "goal": goal,
            "food_target": food,
            "needs_coordination": needs_coordination,
        }

        params = {"request_horizon_k": self.horizon_k}

        try:
            resp = requests.post(self.url, params=params, json=payload, timeout=2.0)
            data = resp.json()
        except Exception as e:
            if self.verbose:
                print("[MODERATOR ERROR]", e)
            return []

        yield_player = data.get("yield_player", None)
        # If this agent is not the yielder, no horizon
        if yield_player is None or int(yield_player) != player_id:
            return []

        override_actions = data.get("override_actions", {})
        plan = override_actions.get(str(player_id)) or override_actions.get(player_id)
        if not plan:
            return []
        return plan

    def get_action(self, obs: dict, player_id: int, phase: str,
                   baseline_action: str, risk_flags: List[str],
                   goal=None, food=None) -> str:
        """
        Horizon cache:
          - If cache non-empty -> consume one.
          - Otherwise query server. If no plan -> use baseline this step.
        """
        if not self.enabled:
            return baseline_action

        # Use cached horizon if available
        if self.cached_plan:
            return self.cached_plan.pop(0)

        plan = self.query_plan(
            obs=obs, player_id=player_id, phase=phase,
            baseline_action=baseline_action, risk_flags=risk_flags,
            goal=goal, food=food
        )

        if not plan:
            return baseline_action

        self.cached_plan = plan
        return self.cached_plan.pop(0)

    def clear_plan(self):
        self.cached_plan = []


# ============================================================
# Safe step wrapper
# ============================================================

PLAYER_ID = 0  # SET TO 1 IN THE SECOND COPY

risk_assessor = NormRiskAssessor(player_id=PLAYER_ID)
moderator = SharedModeratorClient(enabled=True, horizon_k=5, verbose=False)

_prev_out_for_risk: Optional[dict] = None
_last_out_for_risk: Optional[dict] = None
TIMESTEP = 0  # kept for reference/debug if needed

def safe_step(sock_game, baseline_action: str, phase: str, goal=None, food=None) -> dict:
    """
    1) If we have a cached horizon, try to follow it (re-checking only local safety).
    2) Otherwise:
       - assess baseline
       - if no risks: execute baseline
       - if risks: query moderator for a horizon (maybe this agent is yielder)
       - for non-coordination risks, re-check chosen action; if still locally risky, fallback NOP
       - for pure coordination risks, trust moderator (no veto).
    """
    global _prev_out_for_risk, _last_out_for_risk, TIMESTEP

    if _last_out_for_risk is None:
        out = step(sock_game, baseline_action)
        _prev_out_for_risk = None
        _last_out_for_risk = out
        TIMESTEP += 1
        return out

    obs = _last_out_for_risk["observation"]

    # -------------------------------------------------------
    # 0) Try to follow cached horizon if still *locally* safe
    # -------------------------------------------------------
    if moderator.cached_plan:
        planned = moderator.cached_plan[0]  # peek
        planned_flags = risk_assessor.assess(
            prev_out=_prev_out_for_risk,
            out=_last_out_for_risk,
            baseline_action=planned,
            phase=phase, goal=goal, food=food
        )

        # separate coord vs non-coord flags
        non_coord_planned = [f for f in planned_flags if f not in COORDINATION_RISKS]

        # If there are no non-coordination risks, we allow the planned step.
        if not non_coord_planned:
            planned = moderator.cached_plan.pop(0)
            out = step(sock_game, planned)
            _prev_out_for_risk = _last_out_for_risk
            _last_out_for_risk = out
            TIMESTEP += 1
            return out
        else:
            # plan has local safety problems -> drop it
            moderator.clear_plan()

    # -------------------------------------------------------
    # 1) Normal baseline -> risk -> moderator flow
    # -------------------------------------------------------
    baseline_flags = risk_assessor.assess(
        prev_out=_prev_out_for_risk,
        out=_last_out_for_risk,
        baseline_action=baseline_action,
        phase=phase, goal=goal, food=food
    )

    # Debug print you already added:
    print(f"[P{PLAYER_ID}] phase={phase} baseline={baseline_action} flags={baseline_flags}")

    chosen = baseline_action

    if baseline_flags:
        chosen = moderator.get_action(
            obs=obs, player_id=PLAYER_ID, phase=phase,
            baseline_action=baseline_action, risk_flags=baseline_flags,
            goal=goal, food=food
        )

        # Only care about *non-coordination* baseline risks for veto
        baseline_non_coord = [f for f in baseline_flags if f not in COORDINATION_RISKS]

        if baseline_non_coord:
            # Re-check chosen action, but again only look at non-coordination flags
            override_flags = risk_assessor.assess(
                prev_out=_prev_out_for_risk,
                out=_last_out_for_risk,
                baseline_action=chosen,
                phase=phase, goal=goal, food=food
            )
            override_non_coord = [f for f in override_flags if f not in COORDINATION_RISKS]

            if override_non_coord:
                # Moderator plan is locally unsafe (walls/objects/etc) -> veto
                moderator.clear_plan()
                chosen = "NOP"

    # -------------------------------------------------------
    # 2) Execute chosen action
    # -------------------------------------------------------
    out = step(sock_game, chosen)
    _prev_out_for_risk = _last_out_for_risk
    _last_out_for_risk = out
    TIMESTEP += 1
    return out


# ============================================================
# step() to env (localized)
# ============================================================

def step(sock_game, action: str):
    action = f"{PLAYER_ID} " + action
    sock_game.send(str.encode(action))
    output = recv_socket_data(sock_game)
    if output == b'':
        print("Game has ended.")
        return None
    output = json.loads(output)
    return localize_state(output, PLAYER_ID)

# ============================================================
# Task logic (step -> safe_step)
# ============================================================

def get_container(output: dict, socket_game) -> dict:
    player = output['observation']['players'][0]
    player_pos = player['position']
    if player_pos[0] > WEST_WALKWAY['x_min'] or player_pos[1] < register_y_max:
        raise ValueError("Player is not in the correct position to get a container.")
    shopping_list = player['shopping_list']
    rel_pos_fn = rel_pos_basket_return if len(shopping_list) <= 6 else rel_pos_cart_return

    rel_pos = discretize(rel_pos_fn(output))
    while rel_pos != (0.0, 0.0):
        if rel_pos[0] > 0:
            action = "EAST"
        elif rel_pos[0] < 0:
            action = "WEST"
        else:
            action = "NORTH" if rel_pos[1] < 0 else "SOUTH"
        output = safe_step(socket_game, action, phase="container", goal="container_return")
        rel_pos = discretize(rel_pos_fn(output))

    output = safe_step(socket_game, "INTERACT", phase="container", goal="container_return")
    output = safe_step(socket_game, "INTERACT", phase="container", goal="container_return")
    return output


def get_to_walkway(output: dict, socket_game) -> dict:
    rel_pos = discretize(rel_pos_walkway(output))
    step_count = 0
    while rel_pos != (0.0, 0.0):
        action = "EAST" if rel_pos[0] > 0 else "WEST"
        output = safe_step(socket_game, action, phase="shopping", goal="walkway")
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to reach walkway.")
        rel_pos = discretize(rel_pos_walkway(output))
    return output


def get_to_aisle(output: dict, socket_game, aisle: str) -> dict:
    output = get_to_walkway(output, socket_game)
    rel_pos = rel_pos_aisle(output, aisle)
    step_count = 0
    while rel_pos != (0, 0):
        action = "SOUTH" if rel_pos[1] > 0 else "NORTH"
        output = safe_step(socket_game, action, phase="shopping", goal=f"{aisle}_aisle")
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to reach aisle.")
        rel_pos = rel_pos_aisle(output, aisle)
    return output


def get_to_food_shelf(output: dict, socket_game, shelf: str) -> dict:
    output = get_to_aisle(output, socket_game, shelf)
    rel_pos = discretize(rel_pos_shelf(output, shelf))
    step_count = 0
    while rel_pos != (0, 0):
        if rel_pos[1] > 0:
            action = "SOUTH"
        elif rel_pos[1] < 0:
            action = "NORTH"
        else:
            action = "EAST" if rel_pos[0] > 0 else "WEST"
        output = safe_step(socket_game, action, phase="shopping", goal=shelf)
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to reach shelf.")
        rel_pos = discretize(rel_pos_shelf(output, shelf))
    return output


def get_to_food_counter(output: dict, socket_game, food: str) -> dict:
    output = get_to_aisle(output, socket_game, "brie cheese")
    rel_pos_walkway = rel_pos_east_walkway(output)
    step_count = 0
    while rel_pos_walkway != (0.0, 0.0):
        action = "EAST" if rel_pos_walkway[0] > 0 else "WEST"
        output = safe_step(socket_game, action, phase="shopping", goal="east_walkway")
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to reach east walkway.")
        rel_pos_walkway = rel_pos_east_walkway(output)

    rel_pos = discretize(rel_pos_counter(output, food))
    step_count = 0
    while rel_pos != (0.0, 0.0):
        if rel_pos[0] > 0:
            action = "EAST"
        elif rel_pos[0] < 0:
            action = "WEST"
        else:
            action = "SOUTH" if rel_pos[1] > 0 else "NORTH"
        output = safe_step(socket_game, action, phase="shopping", goal=f"{food}_counter")
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to reach counter.")
        rel_pos = discretize(rel_pos_counter(output, food))
    return output


def get_food_from_shelf(output: dict, socket_game, shelf: str) -> dict:
    output = get_to_food_shelf(output, socket_game, shelf)
    player = output['observation']['players'][0]

    if player['curr_cart'] != -1:
        output = safe_step(socket_game, "TOGGLE_CART", phase="shopping")

    rel_pos = rel_pos_shelf(output, shelf)
    step_count = 0
    while rel_pos != (0.0, 0.0):
        output = safe_step(socket_game, "NORTH", phase="shopping", goal=shelf)
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to reach shelf for interaction.")
        rel_pos = rel_pos_shelf(output, shelf)

    output = safe_step(socket_game, "INTERACT", phase="shopping", food=shelf, goal=shelf)
    return output


def get_food_from_counter(output: dict, socket_game, food: str) -> dict:
    output = get_to_food_counter(output, socket_game, food)
    player = output['observation']['players'][0]

    if player['curr_cart'] != -1:
        output = safe_step(socket_game, "TOGGLE_CART", phase="shopping")

    counter_obj = next((obj for obj in output['observation']['counters'] if obj['food'] == food), None)
    if counter_obj is None:
        raise ValueError(f"Counter with food {food} not found in observation")

    rel_pos = rel_pos_counter(output, food)
    step_count = 0
    while rel_pos != (0.0, 0.0):
        output = safe_step(socket_game, "EAST", phase="shopping", goal=f"{food}_counter")
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to reach counter for interaction.")
        rel_pos = rel_pos_counter(output, food)

    output = safe_step(socket_game, "INTERACT", phase="shopping", food=food, goal=f"{food}_counter")
    output = safe_step(socket_game, "INTERACT", phase="shopping", food=food, goal=f"{food}_counter")
    return output


def put_food_in_container(output: dict, socket_game, food: str) -> dict:
    p = output["observation"]["players"][0]
    target_dict = {i: q for i, q in zip(p.get("shopping_list", []), p.get("list_quant", []))}
    required = target_dict.get(food, 0)

    current = 0
    cart_index = p.get("curr_cart", -1)
    if cart_index != -1:
        cart = output["observation"]["carts"][cart_index]
        got = {i: q for i, q in zip(cart.get("contents", []), cart.get("contents_quant", []))}
        current = got.get(food, 0)
    else:
        basket = next(
            (b for b in output["observation"].get("baskets", []) if b.get("owner") == PLAYER_ID),
            None
        )
        if basket:
            got = {i: q for i, q in zip(basket.get("contents", []), basket.get("contents_quant", []))}
            current = got.get(food, 0)

    if current >= required and required > 0:
        # Already satisfied; avoid TookTooMany
        return output

    if food in ["prepared foods", "fresh fish"]:
        output = get_food_from_counter(output, socket_game, food)
    else:
        output = get_food_from_shelf(output, socket_game, food)

    cart_index = output['observation']['players'][0]['curr_cart']
    if cart_index != -1:
        cart = output['observation']['carts'][cart_index]
        cart_dir = INT_TO_DIRECTION[cart['direction']].name
    else:
        cart_dir = None

    player = output['observation']['players'][0]
    if player['holding_food'] == food and cart_dir is not None:
        output = safe_step(socket_game, "INTERACT", phase="shopping", food=food)  # close dialog
        output = safe_step(socket_game, cart_dir, phase="shopping")
        output = safe_step(socket_game, "INTERACT", phase="shopping", food=food)
        output = safe_step(socket_game, "TOGGLE_CART", phase="shopping")

    output = safe_step(socket_game, "INTERACT", phase="shopping", food=food)  # close any dialog
    return output


def get_to_register(output: dict, socket_game) -> dict:
    cart_index = output['observation']['players'][0]['curr_cart']
    output = get_to_aisle(output, socket_game, "sausage")

    rel_pos_walkway = rel_pos_west_walkway(output)
    step_count = 0
    while rel_pos_walkway != (0.0, 0.0):
        action = "EAST" if rel_pos_walkway[0] > 0 else "WEST"
        output = safe_step(socket_game, action, phase="checkout", goal="west_walkway")
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to reach walkway for checkout.")
        rel_pos_walkway = rel_pos_west_walkway(output)

    output = get_to_aisle(output, socket_game, "brie cheese" if cart_index != -1 else "sausage")

    rel_pos_reg = discretize(rel_pos_register(output))
    step_count = 0
    while rel_pos_reg != (0.0, 0.0):
        if rel_pos_reg[0] > 0:
            action = "EAST"
        elif rel_pos_reg[0] < 0:
            action = "WEST"
        else:
            action = "SOUTH" if rel_pos_reg[1] > 0 else "NORTH"
        output = safe_step(socket_game, action, phase="checkout", goal="register")
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to reach register for checkout.")
        rel_pos_reg = discretize(rel_pos_register(output))

    return output


def checkout(output: dict, socket_game) -> dict:
    cart_index = output['observation']['players'][0]['curr_cart']
    output = get_to_register(output, socket_game)

    if cart_index != -1:
        cart = output['observation']['carts'][cart_index]
        cart_dir = INT_TO_DIRECTION[cart['direction']].name
        output = safe_step(socket_game, "TOGGLE_CART", phase="checkout")
    else:
        cart_dir = None

    rel_pos_reg = rel_pos_register(output)
    step_count = 0
    while rel_pos_reg != (0.0, 0.0):
        output = safe_step(socket_game, "WEST", phase="checkout", goal="register")
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to face register for checkout.")
        rel_pos_reg = rel_pos_register(output)

    output = safe_step(socket_game, "INTERACT", phase="checkout", goal="register")
    output = safe_step(socket_game, "INTERACT", phase="checkout", goal="register")

    if cart_dir is not None:
        output = safe_step(socket_game, cart_dir, phase="checkout")
        output = safe_step(socket_game, "TOGGLE_CART", phase="checkout")

    return output


def exit_supermarket(output: dict, socket_game) -> dict:
    rel_pos = discretize(rel_pos_exit(output))
    step_count = 0
    while rel_pos != (0.0, 0.0):
        if rel_pos[1] > 0:
            action = "SOUTH"
        elif rel_pos[1] < 0:
            action = "NORTH"
        else:
            action = "EAST" if rel_pos[0] > 0 else "WEST"
        output = safe_step(socket_game, action, phase="exit", goal="exit")
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to reach exit.")
        rel_pos = discretize(rel_pos_exit(output))
    return output

# ============================================================
# Main loop
# ============================================================

if __name__ == "__main__":

    print("action_commands: ", ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT'])

    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    NUM_PLAYERS = 2
    monitor = PhaseMonitor(num_players=NUM_PLAYERS, verbose=False)

    successes = 0

    for episode in range(10):

        output = step(sock_game, "RESET")
        print(f"[P{PLAYER_ID}] Shopping List: ", output['observation']['players'][0]['shopping_list'])

        # seed globals for safe_step
        _last_out_for_risk = output
        _prev_out_for_risk = None
        TIMESTEP = 0
        moderator.clear_plan()

        monitor.update_prev(output, PLAYER_ID)

        try:
            # container
            prev_output = output
            output = get_container(output, sock_game)
            monitor.mark_container_phase(prev_output, output, PLAYER_ID)

            # shopping
            shopping_list = output['observation']['players'][0]['shopping_list']
            for food in shopping_list:
                prev_output = output
                output = put_food_in_container(output, sock_game, food)
                monitor.mark_pickup(prev_output, output, food, PLAYER_ID)

            monitor.mark_shopping_phase(output, PLAYER_ID)

            # checkout
            prev_output = output
            output = checkout(output, sock_game)
            monitor.mark_checkout_phase(prev_output, output, PLAYER_ID)

            # exit
            output = exit_supermarket(output, sock_game)
            monitor.mark_exit_phase(output, PLAYER_ID)

        except RuntimeError as e:
            print(f"[P{PLAYER_ID}] Episode {episode} failed: {e}")
            monitor.summary()
            moderator.clear_plan()
            continue

        successes += 1
        print(f"[P{PLAYER_ID}] Episode {episode} succeeded.")
        monitor.summary()
        moderator.clear_plan()

    print(f"[P{PLAYER_ID}] Success rate: {successes}/10")
    sock_game.close()
