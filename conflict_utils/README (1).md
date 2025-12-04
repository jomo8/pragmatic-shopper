
# Multi-Agent Supermarket + AI Moderator (Socket Env)

This project runs 2 scripted shopper agents in a supermarket environment
(communicating via a socket server) and coordinates them using a **shared moderator**.

The current moderator is a simple FastAPI service that returns override action
horizons to avoid collisions, personal-space violations, and checkout/exit
blocking. Later, we can swap this heuristic logic out and call an AI model
instead while keeping the same API.

---

## Components

### 1. Environment

- Python: `socket_env.py` (not included here, but assumed to be the env server).
- Start it like:

  ```bash
  python socket_env.py --num_player 2 --mode 1 --render_number
  ```

- Listens on `127.0.0.1:9000`.

The env accepts messages like:

- `"0 NORTH"`, `"1 INTERACT"`, `"0 TOGGLE_CART"`, etc.
- Responds with a JSON blob containing:
  - `observation`: includes `players`, `shelves`, `counters`, `registers`, `carts`, `baskets`, etc.
  - `players[i].shopping_list`, `players[i].list_quant`, `players[i].position`, `direction`, `curr_cart`, etc.

---

### 2. Agent Script: `project-test-agent.py`

Each agent runs the same script, with a different `PLAYER_ID`:

```python
PLAYER_ID = 0  # in one copy
# PLAYER_ID = 1  # in the second copy
```

Key responsibilities:

1. **Phase management + validation (`PhaseMonitor`)**

   Tracks four high-level phases for each player:

   - `container` – get cart or basket, depending on shopping list length.
   - `shopping` – navigate the store and pick up all items.
   - `checkout` – go to register and pay.
   - `exit` – leave the store.

   `PhaseMonitor` only observes and logs; it does **not** change actions.

2. **Local state localization**

   ```python
   def localize_state(output: dict, player_id: int) -> dict:
       # make observation['players'][0] = the controlled player
   ```

   This ensures that, from the script’s perspective:

   - `players[0]` is always “me”.
   - `players[1]` is the other agent.

3. **NormRiskAssessor**

   This class computes **risk flags** for a candidate action:

   - Social / coordination risks:
     - `PLAYER_COLLISION_RISK`
     - `PERSONAL_SPACE_RISK`
     - `WAIT_FOR_CHECKOUT_RISK`
     - `BLOCKING_EXIT_RISK`
   - Local-only risks:
     - `OBJECT_COLLISION_RISK` (walls, shelves, carts, etc.)
     - `UNATTENDED_CONTAINER_RISK` (cart/basket left too far behind)
   - Shopping list adherence:
     - `ADHERE_TO_LIST_RISK`
     - `TOOK_TOO_MANY_RISK`

   Special behavior:
   - Inside the **container return zone** (basket/cart area), object collisions
     are ignored to let agents maneuver more easily.
   - During **shopping**, when moving toward a specific shelf/counter
     (`goal` is not a walkway), `OBJECT_COLLISION_RISK` is **suppressed** so
     that the shared moderator can decide who yields at the shelf, instead of
     the local agent vetoing moves.

4. **SharedModeratorClient**

   The agent sends one step’s context to a shared moderator:

   ```python
   POST http://localhost:8000/moderate?request_horizon_k=5
   ```

   Payload (JSON):

   ```jsonc
   {
     "player_id": 0,
     "phase": "shopping",
     "baseline_action": "EAST",
     "risk_flags": ["PLAYER_COLLISION_RISK"],
     "observation": { ... full obs ... },
     "goal": "banana",
     "food_target": "banana",
     "needs_coordination": true
   }
   ```

   The moderator responds with:

   ```jsonc
   {
     "yield_player": 0,
     "override_actions": {
       "0": ["NORTH", "NOP", "NOP", "NOP", "NOP"]
     },
     "reason": "crowding risk; player 0 yields with detour"
   }
   ```

   The agent then:

   - Caches the override horizon locally (`["NORTH", "NOP", ...]`).
   - Uses those actions for subsequent steps until the horizon is exhausted.
   - On each step, re-checks local safety and can veto override actions for **local-only** risks.

5. **`safe_step(...)`**

   This is the main wrapper around sending actions to the env:

   - If there is a cached moderator plan, try to execute the next planned action
     if it passes local safety checks (no OBJECT_COLLISION, etc.).
   - Otherwise:
     1. Compute **baseline action** (e.g., go toward a shelf).
     2. Run `NormRiskAssessor.assess(...)` to get `risk_flags`.
     3. If no risks → execute baseline.
     4. If there are risks:
        - Ask the moderator for a horizon (who yields + detour plan).
        - If risks are **coordination-only**: trust the moderator.
        - If there are **local-only** risks:
          - Re-run `assess` on the override action.
          - If still locally unsafe → veto and send `"NOP"` instead.

6. **High-level task logic**

   Functions such as:

   - `get_container`
   - `get_to_walkway`, `get_to_aisle`
   - `get_to_food_shelf`, `get_to_food_counter`
   - `put_food_in_container`
   - `get_to_register`, `checkout`
   - `exit_supermarket`

   Each uses `safe_step(...)` instead of calling the env directly.
   They also pass appropriate `phase` and `goal` values so risk assessment
   & moderation behave correctly.

7. **Main loop**

   In `if __name__ == "__main__":`:

   - Connect to env socket: `127.0.0.1:9000`.
   - For each episode:
     - Send `"RESET"` to env.
     - Run through container → shopping → checkout → exit.
     - Use `PhaseMonitor` to log success per phase.

To run two agents, use two processes:

```bash
# Terminal 1: environment
python socket_env.py --num_player 2 --mode 1 --render_number

# Terminal 2: Player 0
PLAYER_ID=0  # inside project-test-agent.py
python project-test-agent.py

# Terminal 3: Player 1
PLAYER_ID=1  # in a copy or param
python project-test-agent-v2.py
```

---

### 3. Moderator: `moderator_server.py`

This is a small FastAPI app that the agents call once per step.

#### Data models

```python
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
    yield_player: Optional[int] = None
    override_actions: Dict[int, List[str]]
    reason: str
```

The **shape of these models is the contract** between agents and moderator.

#### Current heuristic policy

1. If **no risks** → baseline is fine:

   ```python
   if not risk_set:
       return ModerationResponse(
           yield_player=None,
           override_actions={},
           reason="baseline ok: no risks"
       )
   ```

2. If **local-only risks**:

   - `OBJECT_COLLISION_RISK` → back off & NOP horizon.
   - `UNATTENDED_CONTAINER_RISK` → move toward own cart/basket & NOP horizon.

   The caller always yields in this case:

   ```python
   return ModerationResponse(
       yield_player=agent.player_id,
       override_actions={agent.player_id: plan},
       reason=reason
   )
   ```

3. If **crowding/social risks** (collision, personal space, blocking):

   - Decide a global `yield_player` (0 or 1) with `choose_yield_player(...)`.
   - Yielder gets a detour horizon; priority player gets no override and keeps baseline.

4. Otherwise: fall back to baseline (agent handles it locally).

---

## How to Run

### 1. Start the environment

```bash
python socket_env.py --num_player 2 --mode 1 --render_number
```

### 2. Start the moderator

In another terminal:

```bash
uvicorn moderator_server:app --reload --port 8000
```

### 3. Start the two agents

```bash
# Terminal for player 0
# Ensure PLAYER_ID = 0 in project-test-agent.py
python project-test-agent.py

# Terminal for player 1
# Either use a second copy of the file or a variant with PLAYER_ID = 1
python project-test-agentv2.py
```

---

## Integrating a Real AI Moderator

Right now, `moderator_server.py` is a **fake moderator** (pure heuristic).
To integrate a real AI model, keep these things **unchanged**:

- The FastAPI route + URL:
  - `POST http://localhost:8000/moderate?request_horizon_k=K`
- Request schema:
  - `AgentState` fields (`player_id`, `phase`, `baseline_action`, `risk_flags`,
    `observation`, `goal`, `food_target`, `needs_coordination`).
- Response schema:
  - `ModerationResponse` with:
    - `yield_player: Optional[int]`
    - `override_actions: Dict[int, List[str]]`
    - `reason: str`
- The agent-side client `SharedModeratorClient` in `project-test-agent.py`.

You only change **how** the moderator decides the plan.

### Where to change

Open `moderator_server.py` and edit the function:

```python
@app.post("/moderate", response_model=ModerationResponse)
def moderate(
    agent: AgentState,
    request_horizon_k: int = 5
):
    ...
```

Everything inside this function can be replaced with an AI call, as long as you return a valid `ModerationResponse`.

### Example: Calling an AI Model Inside `moderate`

```python
@app.post("/moderate", response_model=ModerationResponse)
def moderate(
    agent: AgentState,
    request_horizon_k: int = 5
):
    # 1) Construct payload for AI
    ai_request = {
        "player_id": agent.player_id,
        "phase": agent.phase,
        "baseline_action": agent.baseline_action,
        "risk_flags": agent.risk_flags,
        "observation": agent.observation,
        "goal": agent.goal,
        "food_target": agent.food_target,
        "needs_coordination": agent.needs_coordination,
        "request_horizon_k": request_horizon_k,
    }

    # 2) Call your AI service here (pseudo-code)
    # ai_response = call_your_ai(ai_request)

    # Expected AI response:
    # {
    #   "yield_player": 0 or 1 or null,
    #   "override_actions": { "0": ["NORTH", "NOP", ...] },
    #   "reason": "AI decided player 0 yields"
    # }

    # 3) Convert AI response into ModerationResponse
    return ModerationResponse(
        yield_player=ai_response["yield_player"],
        override_actions=ai_response.get("override_actions", {}),
        reason=ai_response.get("reason", "AI moderator response")
    )
```

As long as the AI responds in this format, you do **not** need to modify the agents.

### Example: Using `/moderate` as a Proxy to Another AI Service

If you already have an AI service that implements this API, the FastAPI
app can just forward the request:

```python
@app.post("/moderate", response_model=ModerationResponse)
def moderate(
    agent: AgentState,
    request_horizon_k: int = 5
):
    payload = agent.dict()
    payload["request_horizon_k"] = request_horizon_k

    ai_resp = requests.post(
        "http://your-ai-service/moderate",
        json=payload,
        timeout=2.0
    ).json()

    return ModerationResponse(**ai_resp)
```

In this setup:

- The agents talk to `http://localhost:8000/moderate`.
- The local FastAPI app forwards to `http://your-ai-service/...`.
- The remote AI service is responsible for returning a valid `ModerationResponse`.

---

## Notes / Caveats

- The `observation` object sent to the moderator is **localized**:
  - `players[0]` is always the calling player (`agent.player_id`).
  - `players[1]` is the other player (global id `1 - agent.player_id`).
- The moderator must reason in **global player id** space when choosing `yield_player`
  (0 or 1), not local index 0/1.
- If you add more players in the future, you’ll need to:
  - Extend the schema and conventions.
  - Update the moderator’s yielding logic (`choose_yield_player`) accordingly.

---
