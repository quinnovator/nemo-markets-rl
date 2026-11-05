# nemo-markets-rl

Can a small model learn pricing rules it is never told? This repository builds
a one-shot trade-deadline market where the seller's asking price follows hidden
rules, wraps it as a [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) environment,
and trains [Nemotron 3 Nano 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)
against it with GRPO on [NeMo RL](https://github.com/NVIDIA-NeMo/RL).

It accompanies the post
[Can models learn hidden pricing rules in a market economy?](https://undueattention.com/posts/teaching-hidden-market-dynamics/)
and contains everything needed to check its numbers and rerun the experiment.

## Results

Share of the best possible expected surplus captured on the first 300 tasks of
each split, eight sampled replies per task, with 95% bootstrap intervals over tasks.

| | held-out trades (0–3 rival bidders) | shifted market (4–5 rival bidders) |
| --- | ---: | ---: |
| Nemotron 3 Nano 4B, untrained | 21.2% (17.0–25.8) | 17.6% (13.3–21.5) |
| after 1,000 GRPO steps | **93.0%** (91.8–94.1) | 78.1% (72.7–82.6) |
| fixed rule: offer (B + V) / 2 | 75.3% | **86.8%** |
| fixed rule: textbook optimum | 69.0% | 52.7% |

Training lifts the model from 21% to 93% on trades like the ones it practised
on, past both fixed rules. With more rival bidders than it ever saw, it falls
to 78% (86% with four rivals, 66% with five) and loses to the simple midpoint
rule; 86% of what it misses there is offering too little to rebuilding clubs.
The trained model never walked away and never made a negative-EV offer.

## The market

The model sees a player card and makes one offer, or walks away:

```text
Target: left-handed reliever, 2 extra year(s) of club control.
Public consensus value: 17.1
Your internal value (fit + need): 32.0
Seller: a fringe contender club.
Rival bidders known to be in on this player: 1
Days until the deadline: 9
Make your move.
```

```json
{"action": "offer", "amount": 29.0}
```

The seller accepts if the offer `p` is at least its hidden ask, which is drawn
from a normal distribution around the public value `B`, moved by rules the
model is never shown:

| signal | effect on the seller's ask |
| --- | --- |
| rebuilding / middling / fringe-contender seller | ×0.75 / ×1.00 / ×1.30 |
| each rival bidder | +10% |
| each day before the deadline | +1.5% |
| each extra year of club control | +6% |
| spread | standard deviation = 12% of the center |

The reward is the offer's expected surplus `(V − p) · P(accept)` divided by the
best achievable expected surplus for that card, clipped to [−1, 1]: 1 for the
best offer, 0 for walking away from a deal worth making, negative for
overpaying, −1 for an unparseable reply. Using the analytic expectation instead
of a sampled accept/reject gives GRPO a noiseless signal. All of it is in
[`src/deadline_offer/market.py`](src/deadline_offer/market.py), standard library only.

## Check the published numbers

No GPU needed. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest
uv run python -m deadline_offer.analysis results/base results/grpo1000
```

The tests regenerate every data split from its seed and check it byte for byte
against the data the runs used, rescore every saved reply in `results/`, and
check every number in the table above. The analysis prints the table along
with the breakdowns quoted in the post. `uv run python -m deadline_offer.data`
writes the four splits to `data/` if you want to look at them.

## Rerun the experiment

Training runs in NVIDIA's NeMo RL container (`nvcr.io/nvidia/nemo-rl:v0.7.0`,
pinned by digest) on [Modal](https://modal.com): one H100 for training, one
L40S for evaluation. You need a Modal account; nothing else is installed locally.

```bash
uv sync --extra modal
uv run modal setup

# 1. Smoke test: five training steps
uv run modal run --detach nemo/modal_app.py::train --run smoke-001 --steps 5

# 2. The full run: 1,000 steps of training, then merge and held-out evaluation
uv run modal run --detach nemo/modal_app.py::reproduce --run grpo1000-repro --steps 1000

# 3. The untrained baseline on the same tasks
uv run modal run --detach nemo/modal_app.py::evaluate --run grpo1000-repro --base

# 4. Fetch and score
uv run python nemo/collect.py --run grpo1000-repro
uv run python nemo/collect.py --run grpo1000-repro --base
uv run python -m deadline_offer.analysis results/grpo1000-repro-base results/grpo1000-repro
```

The published run trained for 72 minutes (median 4.1 s per step) and took
85 minutes end to end, including merge and evaluation. Runs go to the Modal volume `nemo-markets-rl` (set
`NEMO_MARKETS_VOLUME` to change it); every run needs a new name, and existing
runs are never overwritten or resumed.

Expect your numbers to land near the published ones, not on them: sampling is
at temperature 1, and GPU kernels are not bitwise deterministic.

### Training setup

| setting | value |
| --- | --- |
| base model | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, reasoning off |
| adapter | LoRA rank 32, alpha 64, dropout 0, on `q/k/v/o_proj`, `up/down_proj`, Mamba `in_proj` (71 modules) |
| batch | 8 cards × 8 replies = 64 completions per step, one optimizer update per batch |
| advantage | reward minus the group mean; no std scaling, no leave-one-out |
| objective | clipped policy gradient (ε = 0.2), Dr. GRPO normalization (÷ replies × 128), no KL |
| optimizer | AdamW, lr 5 × 10⁻⁶, 10 warmup steps then cosine to zero, weight decay 0.1, grad clip 0.1 |
| generation | temperature 1.0; top-p 1.0 in training and 0.95 (the model's default) in evaluation; up to 128 tokens in training and 256 in evaluation |
| steps, seed | 1,000, 3407 |

## Layout

```text
src/deadline_offer/      standard-library Python package
  market.py              the market, the hidden rules, the reward
  data.py                seeded train / validation / test / shift splits, with their hashes
  analysis.py            EV capture, bootstrap intervals, breakdowns, fixed rules
nemo/                    the training pipeline (runs in the NeMo RL container)
  modal_app.py           Modal functions: train, merge, evaluate, reproduce
  launch.py              prepare a run, check the objective, train, keep logs
  prepare.py             install the environment into NeMo RL, write data and config
  deadline_grpo.py       Dr. GRPO loss, LR schedule, metric and rollout logging
  gym/                   the NeMo Gym resources server and its config
  check_objective.py     preflight: the loss and schedule match their formulas
  audit_weights.py       preflight: the loaded base weights match the checkpoint
  resident.py            keep policy and vLLM resident on one GPU
  export.py              merge the LoRA adapter into the base model
  evaluate.py            score a model on the 300 test and shift tasks with vLLM
  collect.py             download a finished run into results/
results/
  base/                  the untrained model's evaluation
  grpo1000/              the trained model's evaluation, config, manifest, audits, training.csv
tests/
```

## How the pipeline fits together

`modal_app.py::train` runs `launch.py` in the NeMo RL container. `prepare.py`
copies `market.py` and the Gym server in [`nemo/gym/`](nemo/gym/) into NeMo
Gym, so training scores rollouts with the same reward code as everything else.
It also writes the train and validation splits, checking them against the
published hashes, and builds the NeMo RL config from NVIDIA's GRPO and NeMo Gym
example recipes, with every setting above set explicitly.

Before training, `launch.py` runs two checks. `check_objective.py` compares
the loss's value and gradient with the Dr. GRPO formula written out by hand.
`audit_weights.py` compares every tensor the policy loaded with the original
checkpoint. In this container, 42 frozen Mamba tensors (`dt_bias` and
`out_proj.weight` in 21 layers) come back altered after initialization; the
audit restores exactly those from the checkpoint and stops on any other
difference. Its report is saved with each run (see
[`results/grpo1000/base_weight_audit.json`](results/grpo1000/base_weight_audit.json)).

By default NeMo RL moves the policy to CPU and puts vLLM to sleep every time
it switches between generating and training. `resident.py` keeps both on the
GPU and only drops the caches a weight refit invalidates, which brought a step
from about 15 s to about 4 s. Pass `--lifecycle stock` to train without the patch.

After training, `export.py` merges the last LoRA checkpoint into the base
weights, and `evaluate.py` samples eight replies for each of the first 300
test and shift tasks with vLLM.

## Notes and limitations

- The reward is expected surplus, not a sampled outcome, so it measures
  pricing, not luck. It is also capped at 1, so once an offer is close to the
  best one there is little gradient left to sharpen it.
- One training seed. A task's eight replies are not eight independent trades,
  which is why the intervals resample whole tasks.
- The market is synthetic, one-shot, and has five fixed rules.
- `results/` holds the published runs' original artifacts, from an earlier
  layout of this code; [`results/README.md`](results/README.md) maps their names
  to this one.

## License

Apache 2.0. The data is synthetic and generated by `deadline_offer.data`.
