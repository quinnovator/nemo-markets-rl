"""The published evaluations in results/ reproduce the numbers in the post."""

import json
from pathlib import Path

import pytest

from deadline_offer import analysis, market

RESULTS = Path(__file__).resolve().parents[1] / "results"
RUNS = ("base", "grpo1000")
SPLITS = analysis.SPLITS

# (EV capture, 95% CI low, 95% CI high), as reported
REPORTED = {
    ("base", "test"): (0.2120, 0.170, 0.258),
    ("base", "shift"): (0.1762, 0.133, 0.215),
    ("grpo1000", "test"): (0.9298, 0.918, 0.941),
    ("grpo1000", "shift"): (0.7806, 0.727, 0.826),
}


@pytest.fixture(scope="module")
def replies():
    return {(run, split): analysis.load(RESULTS / run / f"{split}.jsonl") for run in RUNS for split in SPLITS}


@pytest.mark.parametrize(("run", "split"), REPORTED)
def test_ev_capture_and_interval(replies, run, split):
    capture, lo, hi = REPORTED[(run, split)]
    r = replies[(run, split)]
    assert len(r) == 2400 and len({x.task_index for x in r}) == 300
    assert analysis.ev_capture(r) == pytest.approx(capture, abs=5e-5)
    assert analysis.bootstrap_ci(r) == pytest.approx((lo, hi), abs=5e-4)


@pytest.mark.parametrize("run", RUNS)
@pytest.mark.parametrize("split", SPLITS)
def test_saved_scores_match_a_fresh_rescore(run, split):
    """Every saved reply rescores to the same reward and expected surplus."""
    tasks = analysis.eval_tasks(split)
    for line in (RESULTS / run / f"{split}.jsonl").read_text().splitlines():
        r = json.loads(line)
        task = tasks[r["_ng_task_index"]]
        valid, offer = market.parse_action(r["text"])
        assert (valid, offer) == (r["valid_action"], r["offer"])
        assert market.reward(task, r["text"]) == pytest.approx(r["reward"], abs=1e-12)
        ev = market.expected_surplus(task, offer) if valid else 0.0
        assert ev == pytest.approx(r["expected_surplus"], abs=1e-12)
        assert market.optimal_offer(task)[1] == pytest.approx(r["optimal_surplus"], abs=1e-12)


@pytest.mark.parametrize("run", RUNS)
@pytest.mark.parametrize("split", SPLITS)
def test_summary_files_agree(replies, run, split):
    summary = json.loads((RESULTS / run / f"{split}_summary.json").read_text())
    assert summary["ev_capture"] == pytest.approx(analysis.ev_capture(replies[(run, split)]), abs=1e-12)


def test_shifted_market_breakdown(replies):
    r = replies[("grpo1000", "shift")]
    by_rivals = analysis.capture_by(r, lambda x: x.task.rival_bidders)
    assert by_rivals == pytest.approx({4: 0.857, 5: 0.656}, abs=5e-4)
    assert analysis.shortfall(r)["rebuilding, under"] == pytest.approx(0.862, abs=5e-4)
    assert not any(x.offer is None or x.expected_surplus < 0 for x in r)


def test_fixed_rules_and_available_surplus():
    assert analysis.rule_capture("test", market.midpoint_offer) == pytest.approx(0.7532, abs=5e-5)
    assert analysis.rule_capture("shift", market.midpoint_offer) == pytest.approx(0.8678, abs=5e-5)
    assert analysis.rule_capture("test", market.textbook_offer) == pytest.approx(0.690, abs=5e-4)
    assert analysis.rule_capture("shift", market.textbook_offer) == pytest.approx(0.527, abs=5e-4)
    assert analysis.available_surplus("test") == pytest.approx(697.5, abs=0.05)
    assert analysis.available_surplus("shift") == pytest.approx(325.2, abs=0.05)
