from typing import Dict, List, Optional, Any
from fastapi import FastAPI
from pydantic import BaseModel
import math

app = FastAPI()

class AgentState(BaseModel):
    player_id: int
    phase: str
    baseline_action: str
    risk_flags: List[str]
    observation: Dict[str, Any]
    goal: Optional[str] = None
    food_target: Optional[str] = None
    needs_coordination: Optional[bool] = False

class ModerationResponse(BaseModel):
    """
    yield_player:
        - Which (global) player_id should follow the override horizon (yield/wait/detour).
        - None if agent should just use its baseline.
    override_actions:
        - Map from player_id -> list of actions (only present for yield_player).
    """
    yield_player: Optional[int] = None
    override_actions: Dict[int, List[str]]
    reason: str

MOVE_ACTIONS = {"NORTH", "SOUTH", "EAST", "WEST"}

CROWD_RISKS = {
    "PLAYER_COLLISION_RISK",
    "PERSONAL_SPACE_RISK",
    "BLOCKING_EXIT_RISK",
    "WAIT_FOR_CHECKOUT_RISK",
}

LOCAL_ONLY_RISKS = {
    "OBJECT_COLLISION_RISK",
    "UNATTENDED_CONTAINER_RISK",
}

# Approx map bounds (from your description)
MIN_X, MAX_X = 0.1, 19.0
MIN_Y, MAX_Y = 2.2, 24.2

def has_any(agent: AgentState, risks: set) -> bool:
    return any(r in risks for r in agent.risk_flags)

def _dist(a, b) -> float:
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def choose_yield_player(agent: AgentState) -> Optional[int]:
    """
    Decide which GLOBAL player_id should yield, based on localized observation.

    Observation convention (from client):
      players[0] = self (agent.player_id)
      players[1] = other (1 - agent.player_id)

    We compute distances for each GLOBAL id and break ties by player_id.
    """
    obs = agent.observation
    players = obs.get("players", [])
    if len(players) < 2:
        return None

    self_id = agent.player_id
    other_id = 1 - agent.player_id

    self_pos = players[0].get("position", [0.0, 0.0])   # self (local)
    other_pos = players[1].get("position", [0.0, 0.0])  # other (local)

    phase = agent.phase

    # Build a dict of distances per GLOBAL player id
    d_map: Dict[int, float] = {}

    # 1) Exit: approximate exit at x≈0 (use same y band)
    if phase == "exit":
        exit_point = [0.0, (self_pos[1] + other_pos[1]) / 2.0]
        d_map[self_id] = abs(self_pos[0] - exit_point[0])
        d_map[other_id] = abs(other_pos[0] - exit_point[0])

    # 2) Checkout: nearest register
    elif phase == "checkout" and obs.get("registers"):
        regs = obs["registers"]

        def dist_to_nearest_reg(pos):
            return min(_dist(pos, r.get("position", pos)) for r in regs)

        d_map[self_id] = dist_to_nearest_reg(self_pos)
        d_map[other_id] = dist_to_nearest_reg(other_pos)

    # 3) Fallback: whoever is closer to the midpoint between them
    else:
        mid = [(self_pos[0] + other_pos[0]) / 2.0, (self_pos[1] + other_pos[1]) / 2.0]
        d_map[self_id] = _dist(self_pos, mid)
        d_map[other_id] = _dist(other_pos, mid)

    # Decide priority in GLOBAL id space
    min_dist = min(d_map.values())
    # All players with minimal distance
    candidates = [pid for pid, d in d_map.items() if abs(d - min_dist) < 1e-6]
    # Tie-break by smallest player_id
    priority_global = min(candidates)

    # The *other* global player yields (we have exactly 2 players: 0 and 1)
    yield_global = 1 - priority_global
    return yield_global


def _opposite_action(action: str) -> Optional[str]:
    mapping = {
        "NORTH": "SOUTH",
        "SOUTH": "NORTH",
        "EAST": "WEST",
        "WEST": "EAST",
    }
    return mapping.get(action)

def _safe_direction_away_from_other(self_pos, other_pos) -> str:
    """
    Pick a single cardinal direction that tends to move self away from other.
    Very simple: move along the axis with greater distance.
    """
    dx = self_pos[0] - other_pos[0]
    dy = self_pos[1] - other_pos[1]

    if abs(dx) >= abs(dy):
        return "EAST" if dx > 0 else "WEST"
    else:
        return "SOUTH" if dy > 0 else "NORTH"

def _avoid_wall(action: str, pos) -> str:
    """
    If the chosen action would push us into a wall region,
    try to pick a perpendicular detour instead. Very rough heuristic.
    """
    x, y = pos
    # basic wall danger checks; if action is into wall, rotate turn
    if action == "WEST" and x <= MIN_X + 0.5:
        return "NORTH" if y > (MIN_Y + MAX_Y) / 2.0 else "SOUTH"
    if action == "EAST" and x >= MAX_X - 0.5:
        return "NORTH" if y > (MIN_Y + MAX_Y) / 2.0 else "SOUTH"
    if action == "NORTH" and y <= MIN_Y + 0.5:
        return "EAST" if x > (MIN_X + MAX_X) / 2.0 else "WEST"
    if action == "SOUTH" and y >= MAX_Y - 0.5:
        return "EAST" if x > (MIN_X + MAX_X) / 2.0 else "WEST"
    return action  # looks ok

def _object_detour_plan(agent: AgentState, k: int) -> List[str]:
    """
    For OBJECT_COLLISION_RISK:
      - Try to step opposite the baseline direction (back off),
        then do NOPs.
      - If baseline isn't a move, just NOP.

    IMPORTANT: observation is localized, so players[0] is self.
    """
    if agent.baseline_action not in MOVE_ACTIONS:
        return ["NOP"] * k

    players = agent.observation.get("players", [])
    if not players:
        return ["NOP"] * k

    self_pos = players[0].get("position", [0.0, 0.0])  # self
    back = _opposite_action(agent.baseline_action)
    if back is None:
        return ["NOP"] * k

    back = _avoid_wall(back, self_pos)
    plan = [back] + ["NOP"] * (k - 1)
    return plan

def _unattended_detour_plan(agent: AgentState, k: int) -> List[str]:
    """
    For UNATTENDED_CONTAINER_RISK:
      - Move one step toward own container (cart or basket) if we can locate it.
      - Otherwise just NOP.

    Observation is localized, but basket 'owner' is a global id.
    """
    obs = agent.observation
    players = obs.get("players", [])
    if not players:
        return ["NOP"] * k

    # self is local index 0
    self_pos = players[0].get("position", [0.0, 0.0])
    p = players[0]

    # try to find owned cart (owned by this global player_id)
    cont_pos = None
    cart_idx = p.get("curr_cart", -1)
    if cart_idx != -1 and cart_idx < len(obs.get("carts", [])):
        cont_pos = obs["carts"][cart_idx]["position"]
    else:
        for b in obs.get("baskets", []):
            if b.get("owner") == agent.player_id:
                cont_pos = b["position"]
                break

    if cont_pos is None:
        return ["NOP"] * k

    dx = cont_pos[0] - self_pos[0]
    dy = cont_pos[1] - self_pos[1]
    if abs(dx) >= abs(dy):
        action = "EAST" if dx > 0 else "WEST"
    else:
        action = "SOUTH" if dy > 0 else "NORTH"

    action = _avoid_wall(action, self_pos)
    return [action] + ["NOP"] * (k - 1)

def _crowd_detour_plan(agent: AgentState, k: int) -> List[str]:
    """
    For crowding risks:
      - If there's another player, step slightly away from them (side-step),
        then NOPs.
      - If we don't have the other player, just NOP.

    Observation is localized: players[0] is self, players[1] is other.
    """
    obs = agent.observation
    players = obs.get("players", [])
    if len(players) < 2:
        return ["NOP"] * k

    self_pos = players[0].get("position", [0.0, 0.0])  # self
    other_pos = players[1].get("position", [0.0, 0.0])  # other

    action = _safe_direction_away_from_other(self_pos, other_pos)
    action = _avoid_wall(action, self_pos)
    return [action] + ["NOP"] * (k - 1)

@app.post("/moderate", response_model=ModerationResponse)
def moderate(
    agent: AgentState,
    request_horizon_k: int = 5
):
    """
    Single-agent call, but observation contains both players (localized on the client).

    Policy:
      - If local-only risk (object/unattended):
            -> the caller yields with a local detour strategy (back off or move toward container).
      - If crowding risk:
            -> pick which GLOBAL player_id should yield based on heuristic
               (first to exit/register/etc. goes first).
            -> yielding player gets a small detour horizon (step aside/away + NOPs).
            -> priority player keeps baseline (no override).
      - Otherwise: baseline ok (no horizon).
    """
    risk_set = set(agent.risk_flags)
    obs = agent.observation

    # No risks -> baseline ok
    if not risk_set:
        return ModerationResponse(
            yield_player=None,
            override_actions={},
            reason="baseline ok: no risks"
        )

    # Local-only risks (walls/objects/unattended) -> caller detours/yields
    if has_any(agent, LOCAL_ONLY_RISKS):
        if "OBJECT_COLLISION_RISK" in risk_set:
            plan = _object_detour_plan(agent, request_horizon_k)
            reason = "local object collision; caller backs off/detours"
        elif "UNATTENDED_CONTAINER_RISK" in risk_set:
            plan = _unattended_detour_plan(agent, request_horizon_k)
            reason = "unattended container; caller moves toward container"
        else:
            plan = ["NOP"] * request_horizon_k
            reason = "local risk; caller yields with NOPs"

        return ModerationResponse(
            yield_player=agent.player_id,
            override_actions={agent.player_id: plan},
            reason=reason
        )

    # Crowding / social norms
    if has_any(agent, CROWD_RISKS) and agent.baseline_action in MOVE_ACTIONS:
        yield_player = choose_yield_player(agent)
        if yield_player is None:
            # can't decide -> conservative: caller yields with detour
            plan = _crowd_detour_plan(agent, request_horizon_k)
            return ModerationResponse(
                yield_player=agent.player_id,
                override_actions={agent.player_id: plan},
                reason="crowding risk; fallback caller detours"
            )

        # If THIS agent is the yielder, we give it a detour horizon.
        if agent.player_id == yield_player:
            plan = _crowd_detour_plan(agent, request_horizon_k)
            return ModerationResponse(
                yield_player=yield_player,
                override_actions={yield_player: plan},
                reason=f"crowding risk; player {yield_player} yields with detour"
            )
        else:
            # This agent is priority: no override, keep baseline
            return ModerationResponse(
                yield_player=yield_player,  # for logging / debugging
                override_actions={},
                reason=f"crowding risk; player {yield_player} yields with detour"
            )

    # Other risks (e.g. adhere-to-list) should be handled locally by the agent.
    return ModerationResponse(
        yield_player=None,
        override_actions={},
        reason="non-critical or locally handled risks; baseline ok"
    )
