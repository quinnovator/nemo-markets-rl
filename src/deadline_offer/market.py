"""The one-shot trade-deadline market and its reward.

Our club (the buyer) wants a player. The selling club has a hidden asking
price r. We make one offer p, or walk away. If p >= r the deal happens and we
gain V - p, where V is what the player is worth to us. Otherwise nothing
happens.

The prompt shows the player's public value B, our value V, and some market
signals. The rules that turn those signals into the seller's asking-price
distribution are hidden from the model; it can only learn them from reward.

This module is standard-library only. The Gym resources server, the data
generator and the evaluator all import it, so the market and the reward are
defined in exactly one place.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass

# ---------------------------------------------------------------------------
# Hidden market structure (never shown to the model)
# ---------------------------------------------------------------------------
SELLER_MODE_MULT = {
    "rebuilding": 0.75,  # motivated sellers
    "middling": 1.00,
    "fringe contender": 1.30,  # reluctant sellers, need to be overpaid
}
RIVAL_BIDDER_MULT = 0.10  # each rival bidder lifts the ask 10%
DAYS_LEFT_MULT = 0.015  # sellers soften as the deadline approaches
CONTROL_YEAR_MULT = 0.06  # each extra year of club control lifts the ask 6%
SIGMA_FRAC = 0.12  # spread of the ask, as a fraction of its mean

POSITIONS = [
    "left-handed reliever",
    "right-handed reliever",
    "starting pitcher",
    "corner infielder",
    "middle infielder",
    "outfielder",
    "catcher",
]

NO_DEAL_EPS = 0.01  # if the best possible EV is under 1% of B, walking away is correct
OPTIMUM_GRID = 2000  # offers searched over (0, V] to find the optimum


@dataclass
class Task:
    task_id: str
    position: str
    public_value: float  # B: consensus value, $M
    our_value: float  # V: our private value (fit + need), $M
    seller_mode: str
    rival_bidders: int
    days_left: int
    control_years: int  # years of control beyond this season (0 = rental)

    def ask_mean(self) -> float:
        m = SELLER_MODE_MULT[self.seller_mode]
        m *= 1 + RIVAL_BIDDER_MULT * self.rival_bidders
        m *= 1 + DAYS_LEFT_MULT * (self.days_left - 1)
        m *= 1 + CONTROL_YEAR_MULT * self.control_years
        return self.public_value * m

    def ask_sigma(self) -> float:
        return SIGMA_FRAC * self.ask_mean()


def sample_task(rng: random.Random, split: str = "train", idx: int = 0) -> Task:
    """Draw one task. `split` is "train", "test" (same distribution, different
    seed) or "shift" (4-5 rival bidders instead of 0-3)."""
    public_value = round(rng.uniform(4.0, 30.0), 1)
    need = rng.uniform(0.9, 2.0)
    rivals = rng.randint(4, 5) if split == "shift" else rng.randint(0, 3)
    return Task(
        task_id=f"{split}-{idx:06d}",
        position=rng.choice(POSITIONS),
        public_value=public_value,
        our_value=round(public_value * need, 1),
        seller_mode=rng.choice(list(SELLER_MODE_MULT)),
        rival_bidders=rivals,
        days_left=rng.randint(1, 14),
        control_years=rng.choice([0, 0, 1, 2, 3]),
    )


# ---------------------------------------------------------------------------
# Expected surplus and the optimum
# ---------------------------------------------------------------------------
def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def p_accept(task: Task, offer: float) -> float:
    """Probability the seller's hidden ask is at or below `offer`."""
    return _normal_cdf((offer - task.ask_mean()) / task.ask_sigma())


def expected_surplus(task: Task, offer: float | None) -> float:
    """S(p) = (V - p) * P(accept). `offer=None` means walk away (0 surplus)."""
    if offer is None:
        return 0.0
    return (task.our_value - offer) * p_accept(task, offer)


_OPTIMUM_CACHE: dict[tuple, tuple[float | None, float]] = {}


def optimal_offer(task: Task, steps: int = OPTIMUM_GRID) -> tuple[float | None, float]:
    """Grid-search the best offer over (0, V]. Returns (offer or None, expected
    surplus). Cached, because every GRPO group scores the same task many times."""
    key = (*asdict(task).values(), steps)
    if key not in _OPTIMUM_CACHE:
        _OPTIMUM_CACHE[key] = _best_offer(task, steps, lambda p: expected_surplus(task, p))
    return _OPTIMUM_CACHE[key]


def _best_offer(task: Task, steps: int, ev) -> tuple[float | None, float]:
    best_offer, best_ev = None, 0.0
    for i in range(1, steps + 1):
        offer = task.our_value * i / steps
        value = ev(offer)
        if value > best_ev:
            best_offer, best_ev = offer, value
    return best_offer, best_ev


# ---------------------------------------------------------------------------
# Fixed rules, for comparison with the model
# ---------------------------------------------------------------------------
def midpoint_offer(task: Task) -> float:
    """Offer halfway between the public value and our value: (B + V) / 2."""
    return (task.public_value + task.our_value) / 2


def textbook_offer(task: Task, steps: int = OPTIMUM_GRID) -> float | None:
    """The optimum for a buyer who knows the shape of the market but none of
    the hidden rules: it assumes the ask is centred on B with the same spread."""
    mu, sd = task.public_value, SIGMA_FRAC * task.public_value
    return _best_offer(task, steps, lambda p: (task.our_value - p) * _normal_cdf((p - mu) / sd))[0]


# ---------------------------------------------------------------------------
# Prompt and parsing
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are the general manager of a baseball club at the trade deadline. "
    "You make exactly ONE offer for a player, or walk away. The seller accepts "
    "if your offer is at least their private asking price; otherwise no deal. "
    "If the deal happens you gain (your value - your offer). Maximize expected gain. "
    "All values are in $M of surplus value. "
    'Reply with JSON only: {"action": "offer", "amount": <number>} or {"action": "walk"}.'
)


def render_prompt(task: Task) -> str:
    if task.control_years == 0:
        control = "a rental (no control beyond this season)"
    else:
        control = f"{task.control_years} extra year(s) of club control"
    return (
        f"Target: {task.position}, {control}.\n"
        f"Public consensus value: {task.public_value:.1f}\n"
        f"Your internal value (fit + need): {task.our_value:.1f}\n"
        f"Seller: a {task.seller_mode} club.\n"
        f"Rival bidders known to be in on this player: {task.rival_bidders}\n"
        f"Days until the deadline: {task.days_left}\n"
        "Make your move."
    )


def to_messages(task: Task) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_prompt(task)},
    ]


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_action(text: str) -> tuple[bool, float | None]:
    """Returns (valid, offer or None for a walk). Reads the last JSON object in
    the text, so stray reasoning before the answer doesn't break parsing."""
    matches = _JSON_RE.findall(text or "")
    if not matches:
        return False, None
    try:
        obj = json.loads(matches[-1])
    except json.JSONDecodeError:
        return False, None
    action = str(obj.get("action", "")).lower()
    if action == "walk":
        return True, None
    if action == "offer":
        try:
            amount = float(obj["amount"])
        except (KeyError, TypeError, ValueError):
            return False, None
        if not math.isfinite(amount) or amount < 0:
            return False, None
        return True, amount
    return False, None


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
def reward(task: Task, completion_text: str) -> float:
    """Share of the best possible expected surplus, clipped to [-1, 1].

     1.0  the best move (the optimal offer, or walking when no deal is worth it)
     0.0  walking away from a deal worth making
    <0    overpaying (clipped at -1)
    -1.0  no parseable action

    Dividing by each task's optimum puts every trade on the same scale, so a
    $30M star doesn't drown out a $4M reliever.
    """
    valid, offer = parse_action(completion_text)
    if not valid:
        return -1.0
    _, best_ev = optimal_offer(task)
    floor = NO_DEAL_EPS * task.public_value
    ev = 0.0 if offer is None else expected_surplus(task, offer)
    if best_ev <= floor:
        # No deal worth making: walking (or a lowball that is as good as one)
        # is correct, and overpaying is penalized on the same scale.
        return 1.0 if ev >= 0 else max(-1.0, 1.0 + ev / floor)
    return max(-1.0, min(1.0, ev / best_ev))


def regret(task: Task, completion_text: str) -> float:
    """Expected surplus left on the table versus the optimum, as a share of B."""
    _, best = optimal_offer(task)
    valid, offer = parse_action(completion_text)
    got = expected_surplus(task, offer) if valid else 0.0
    return (best - got) / task.public_value


def task_to_json(task: Task) -> dict:
    return asdict(task)


def task_from_json(d: dict) -> Task:
    return Task(**d)
