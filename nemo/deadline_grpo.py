"""Training entry point: NeMo RL's GRPO with two changes and two log hooks.

Runs inside the NeMo RL container. `prepare.py` copies this file to the root
of the NeMo RL checkout, so the config can name `deadline_grpo.CosineWithWarmup`.

Changes to the objective:
- `DrGRPOLoss` divides the summed token loss by (responses x 128) instead of
  by the actual number of response tokens (Dr. GRPO), so long and short
  replies get equal weight per token.
- `CosineWithWarmup` is a linear warmup followed by cosine decay to zero.

Log hooks, written to $DEADLINE_RUN_DIR:
- metrics.jsonl: every scalar NeMo RL logs, one line per (step, prefix).
- rollouts.jsonl: every scored training rollout. Training stops if the first
  batch has no valid action, which means the integration is broken.
"""

import json
import math
import numbers
import os
import time
from functools import partial
from pathlib import Path

from nemo_rl.algorithms.loss import ClippedPGLossFn
from torch.optim.lr_scheduler import LambdaLR

MAX_COMPLETION_TOKENS = 128


def cosine_multiplier(step: int, *, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


class CosineWithWarmup(LambdaLR):
    def __init__(self, optimizer, total_steps: int, warmup_steps: int):
        super().__init__(optimizer, partial(cosine_multiplier, total_steps=total_steps, warmup_steps=warmup_steps))


class DrGRPOLoss(ClippedPGLossFn):
    """NeMo's clipped policy-gradient loss, renormalized for Dr. GRPO.

    NeMo's token-level loss divides by the batch's real response-token count.
    Rescaling by tokens / (responses * 128) replaces only that denominator and
    keeps NeMo's clipping and diagnostics. The recipe has no KL or NLL terms.
    """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if not cfg.token_level_loss or cfg.reference_policy_kl_penalty != 0:
            raise ValueError("Dr. GRPO here requires token-level loss and zero KL")
        if cfg.positive_example_nll_weight != 0:
            raise ValueError("Dr. GRPO here does not include an NLL term")

    def __call__(self, next_token_logprobs, data, global_valid_seqs, global_valid_toks):
        loss, metrics = super().__call__(next_token_logprobs, data, global_valid_seqs, global_valid_toks)
        loss = loss * global_valid_toks / (global_valid_seqs * MAX_COMPLETION_TOKENS)
        metrics["dr_grpo_loss"] = loss.detach().item()
        return loss, metrics


def _run_dir() -> Path:
    return Path(os.environ["DEADLINE_RUN_DIR"])


def _log_metrics_to_file():
    from nemo_rl.utils.logger import Logger

    original = Logger.log_metrics

    def log_metrics(self, metrics, step, prefix="", *args, **kwargs):
        scalars = {k: float(v) for k, v in metrics.items() if isinstance(v, numbers.Real)}
        with (_run_dir() / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"time": time.time(), "step": step, "prefix": prefix, "metrics": scalars}) + "\n")
        return original(self, metrics, step, prefix, *args, **kwargs)

    Logger.log_metrics = log_metrics


def _log_rollouts_to_file(grpo):
    original = grpo.run_async_nemo_gym_rollout
    batch = 0

    def rollout(*args, **kwargs):
        nonlocal batch
        result = original(*args, **kwargs)
        records = [
            json.loads(row[0])
            for key, table in result.rollout_metrics.items()
            if "full_result" in key
            for row in table.data
        ]
        with (_run_dir() / "rollouts.jsonl").open("a") as f:
            for record in records:
                f.write(json.dumps({"batch": batch, **record}) + "\n")
        if batch == 0 and not any(r.get("valid_action") for r in records):
            raise RuntimeError("The first rollout batch has no valid actions; inspect rollouts.jsonl")
        batch += 1
        return result

    grpo.run_async_nemo_gym_rollout = rollout


def main():
    import nemo_rl.algorithms.grpo as grpo

    _log_metrics_to_file()
    grpo.ClippedPGLossFn = DrGRPOLoss
    _log_rollouts_to_file(grpo)
    from examples.nemo_gym.run_grpo_nemo_gym import main as run

    run()


if __name__ == "__main__":
    main()
