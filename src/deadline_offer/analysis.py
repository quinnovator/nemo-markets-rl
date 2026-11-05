"""Score evaluation outputs: the numbers reported in the post.

An evaluation file holds one JSON line per sampled reply, with the task's index
in its split (`_ng_task_index`), the reply, and its scored expected surplus.
Tasks are rebuilt from the split's seed, so results carry no copy of the data.

    python -m deadline_offer.analysis results/base results/grpo1000
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from . import data, market

SPLITS = ("test", "shift")
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261103


@dataclass
class Reply:
    task_index: int
    task: market.Task
    offer: float | None
    valid: bool
    reward: float
    expected_surplus: float
    optimal_surplus: float
    optimal_offer: float | None


def eval_tasks(split: str, n: int = data.EVAL_TASKS) -> list[market.Task]:
    return list(islice(data.tasks(split), n))


def load(path: Path) -> list[Reply]:
    """Read one evaluation file and re-attach each reply's hidden task."""
    split = path.stem
    tasks = eval_tasks(split)
    replies = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        replies.append(
            Reply(
                task_index=r["_ng_task_index"],
                task=tasks[r["_ng_task_index"]],
                offer=r["offer"],
                valid=r["valid_action"],
                reward=r["reward"],
                expected_surplus=r["expected_surplus"],
                optimal_surplus=r["optimal_surplus"],
                optimal_offer=r["optimal_offer"],
            )
        )
    return replies


def ev_capture(replies: list[Reply]) -> float:
    """Summed expected surplus over summed optimal expected surplus."""
    return sum(r.expected_surplus for r in replies) / sum(r.optimal_surplus for r in replies)


def bootstrap_ci(
    replies: list[Reply], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """95% interval for EV capture, resampling whole tasks (a task's eight
    replies are not independent trades)."""
    by_task: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in replies:
        by_task[r.task_index][0] += r.expected_surplus
        by_task[r.task_index][1] += r.optimal_surplus
    ids = list(by_task)
    rng = random.Random(seed)
    stats = []
    for _ in range(resamples):
        sample = [rng.choice(ids) for _ in ids]
        stats.append(sum(by_task[i][0] for i in sample) / sum(by_task[i][1] for i in sample))
    stats.sort()
    return stats[int(0.025 * resamples)], stats[int(0.975 * resamples) - 1]


def capture_by(replies: list[Reply], key: Callable[[Reply], object]) -> dict:
    groups: dict[object, list[Reply]] = defaultdict(list)
    for r in replies:
        groups[key(r)].append(r)
    return {k: ev_capture(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def shortfall(replies: list[Reply]) -> dict[str, float]:
    """Where the missing surplus goes: share of the total shortfall by seller
    type and by whether the offer was under or over the optimal offer."""
    parts: dict[str, float] = defaultdict(float)
    for r in replies:
        if r.offer is None:
            side = "walk"
        else:
            side = "under" if r.optimal_offer is not None and r.offer < r.optimal_offer else "over"
        parts[f"{r.task.seller_mode}, {side}"] += r.optimal_surplus - r.expected_surplus
    total = sum(parts.values())
    return dict(sorted(((k, v / total) for k, v in parts.items()), key=lambda kv: -kv[1]))


RULES: dict[str, Callable[[market.Task], float | None]] = {
    "midpoint (B+V)/2": market.midpoint_offer,
    "textbook optimum": market.textbook_offer,
}


def rule_capture(split: str, rule: Callable[[market.Task], float | None]) -> float:
    tasks = eval_tasks(split)
    got = sum(market.expected_surplus(t, rule(t)) for t in tasks)
    return got / sum(market.optimal_offer(t)[1] for t in tasks)


def available_surplus(split: str) -> float:
    """The whole market's best possible expected surplus, $M over the eval tasks."""
    return sum(market.optimal_offer(t)[1] for t in eval_tasks(split))


TRAINING_COLUMNS = {
    "reward": "reward",
    "reward_std": "total_reward/stddev",
    "valid_rate": "deadline_offer_simple_agent/valid_action/mean",
    "mixed_groups": "baseline_reward/pct_mixed",
    "expected_surplus": "deadline_offer_simple_agent/expected_surplus/mean",
    "optimal_surplus": "deadline_offer_simple_agent/optimal_surplus/mean",
    "loss": "dr_grpo_loss",
    "grad_norm": "grad_norm",
    "lr": "lr",
    "gen_kl_error": "gen_kl_error",
    "mean_completion_tokens": "mean_gen_tokens_per_sample",
}


def training_curve(metrics_jsonl: Path, out_csv: Path) -> None:
    """Reduce NeMo RL's logged metrics to one CSV row per training step."""
    import csv

    steps: dict[int, dict] = {}
    for line in metrics_jsonl.read_text().splitlines():
        r = json.loads(line)
        if r["prefix"] == "train":
            row = steps.setdefault(r["step"], {"step": r["step"]})
            row.update({k: r["metrics"][src] for k, src in TRAINING_COLUMNS.items() if src in r["metrics"]})
        elif r["prefix"] == "timing/train":
            steps.setdefault(r["step"], {"step": r["step"]})["step_seconds"] = r["metrics"]["total_step_time"]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, ["step", *TRAINING_COLUMNS, "step_seconds"])
        writer.writeheader()
        for step in sorted(steps):
            writer.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in steps[step].items()})


def summarize(run: Path) -> dict:
    out = {}
    for split in SPLITS:
        replies = load(run / f"{split}.jsonl")
        lo, hi = bootstrap_ci(replies)
        out[split] = {
            "tasks": len({r.task_index for r in replies}),
            "replies": len(replies),
            "ev_capture": ev_capture(replies),
            "ev_capture_95ci": [lo, hi],
            "mean_reward": statistics.mean(r.reward for r in replies),
            "valid_rate": statistics.mean(r.valid for r in replies),
            "walks": sum(r.valid and r.offer is None for r in replies),
            "negative_ev": sum(r.expected_surplus < 0 for r in replies),
            "ev_capture_by_rivals": capture_by(replies, lambda r: r.task.rival_bidders),
            "ev_capture_by_seller": capture_by(replies, lambda r: r.task.seller_mode),
            "shortfall": shortfall(replies),
        }
    return out


def baselines() -> dict:
    return {
        split: {
            "available_surplus_musd": available_surplus(split),
            **{name: rule_capture(split, rule) for name, rule in RULES.items()},
        }
        for split in SPLITS
    }


def report(runs: list[Path]) -> str:
    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    lines = [
        "| model | split | EV capture | 95% CI | mean reward | valid | walks | negative EV |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    detail = []
    for run in runs:
        summary = summarize(run)
        for split, s in summary.items():
            lo, hi = s["ev_capture_95ci"]
            lines.append(
                f"| {run.name} | {split} | {pct(s['ev_capture'])} | {pct(lo)}–{pct(hi)} | "
                f"{s['mean_reward']:.3f} | {pct(s['valid_rate'])} | {s['walks']} | {s['negative_ev']} |"
            )
            by_rivals = ", ".join(f"{k}: {pct(v)}" for k, v in s["ev_capture_by_rivals"].items())
            top = ", ".join(f"{k} {pct(v)}" for k, v in list(s["shortfall"].items())[:3])
            detail.append(f"- {run.name} / {split}: by rival bidders {by_rivals}; shortfall {top}")
    lines.append("")
    lines.append("| fixed rule | split | EV capture |")
    lines.append("| --- | --- | ---: |")
    base = baselines()
    for split, b in base.items():
        for name in RULES:
            lines.append(f"| {name} | {split} | {pct(b[name])} |")
    lines.append("")
    lines.extend(detail)
    lines.append("")
    lines.append(
        "Available expected surplus over the eval tasks: "
        + ", ".join(f"{split} ${b['available_surplus_musd']:.1f}M" for split, b in base.items())
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+", help="directories holding test.jsonl and shift.jsonl")
    parser.add_argument("--json", action="store_true", help="print full summaries as JSON")
    args = parser.parse_args()
    if args.json:
        print(json.dumps({"runs": {r.name: summarize(r) for r in args.runs}, "rules": baselines()}, indent=2))
    else:
        print(report(args.runs))


if __name__ == "__main__":
    main()
