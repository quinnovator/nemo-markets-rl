import random

import pytest

from deadline_offer import market


def task(i: int = 0, split: str = "train") -> market.Task:
    return market.sample_task(random.Random(i), split, i)


def offer(amount: float) -> str:
    return f'{{"action": "offer", "amount": {amount}}}'


WALK = '{"action": "walk"}'


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (WALK, (True, None)),
        ('{"action": "offer", "amount": 12.5}', (True, 12.5)),
        ('{"action": "OFFER", "amount": "7"}', (True, 7.0)),
        ('thinking {"a": 1} then {"action": "offer", "amount": 7}', (True, 7.0)),
        ('{"action": "offer", "amount": -3}', (False, None)),
        ('{"action": "offer", "amount": "NaN"}', (False, None)),
        ('{"action": "offer"}', (False, None)),
        ('{"action": "hold"}', (False, None)),
        ("no json here", (False, None)),
        ("", (False, None)),
    ],
)
def test_parse_action(text, expected):
    assert market.parse_action(text) == expected


def test_optimal_offer_scores_one_and_bounds_hold():
    for i in range(200):
        t = task(i)
        best, _ = market.optimal_offer(t)
        assert market.reward(t, WALK if best is None else offer(best)) == 1.0
        assert -1.0 <= market.reward(t, offer(1.5 * t.our_value)) <= 1.0
        assert market.reward(t, "garbage") == -1.0


def test_walking_scores_zero_when_a_deal_is_worth_making():
    t = next(
        t for t in (task(i) for i in range(100)) if market.optimal_offer(t)[1] > market.NO_DEAL_EPS * t.public_value
    )
    assert market.reward(t, WALK) == 0.0


def test_overpaying_is_negative():
    t = task(5)
    assert market.reward(t, offer(t.our_value + 10)) < 0


def test_hidden_rules_move_the_ask():
    base = market.task_to_json(task(3)) | {
        "rival_bidders": 0,
        "seller_mode": "middling",
        "days_left": 1,
        "control_years": 0,
    }
    middling = market.task_from_json(base)
    assert middling.ask_mean() == pytest.approx(middling.public_value)
    assert market.task_from_json(base | {"seller_mode": "rebuilding"}).ask_mean() == pytest.approx(
        0.75 * middling.ask_mean()
    )
    assert market.task_from_json(base | {"seller_mode": "fringe contender"}).ask_mean() == pytest.approx(
        1.30 * middling.ask_mean()
    )
    assert market.task_from_json(base | {"rival_bidders": 2}).ask_mean() == pytest.approx(1.20 * middling.ask_mean())
    assert market.task_from_json(base | {"days_left": 11}).ask_mean() == pytest.approx(1.15 * middling.ask_mean())
    assert market.task_from_json(base | {"control_years": 3}).ask_mean() == pytest.approx(1.18 * middling.ask_mean())


def test_shift_split_has_more_rivals():
    rng = random.Random(0)
    assert {market.sample_task(rng, "shift").rival_bidders for _ in range(200)} == {4, 5}
    assert {market.sample_task(rng, "test").rival_bidders for _ in range(200)} == {0, 1, 2, 3}


def test_post_example_card():
    """The card shown in the post is test task 3."""
    from deadline_offer import data

    t = next(t for i, t in enumerate(data.tasks("test")) if i == 3)
    assert (t.public_value, t.our_value, t.seller_mode, t.rival_bidders, t.days_left, t.control_years) == (
        17.1,
        32.0,
        "fringe contender",
        1,
        9,
        2,
    )
    assert market.render_prompt(t).startswith("Target: left-handed reliever, 2 extra year(s) of club control.")
