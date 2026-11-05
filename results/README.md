# Published results

Both evaluations score the first 300 tasks of the test and shift splits, eight
sampled replies per task. Each line of `test.jsonl` / `shift.jsonl` is one
reply: its task's index in the split (`_ng_task_index`), the raw text, the
parsed offer, and its reward and expected surplus. Tasks are not stored here;
`deadline_offer.analysis` regenerates them from the split seeds.

| directory | model | Modal run |
| --- | --- | --- |
| `base/` | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, untrained | `nemotron4b-baseline-001` (`eval-base`) |
| `grpo1000/` | the same model after 1,000 GRPO steps | `nemotron4b-grpo1000-002` |

`grpo1000/` also holds the run's own records:

- `config.yaml`: the full NeMo RL config the run trained with.
- `manifest.json`: model, seed, data hashes, the NeMo RL commit and the GPU.
  Its `matched` and `differences` fields compare the run with an earlier
  implementation of the same recipe that is not part of this repository.
- `base_weight_audit.json`: all 263 base tensors checked; the 42 known Mamba
  tensors restored, no other mismatch.
- `merge_manifest.json`: the merge of the step-1,000 adapter (71 LoRA modules).
- `training.csv`: one row per training step, reduced from NeMo RL's logged
  metrics: mean reward, valid-reply rate, share of groups with mixed rewards,
  loss, gradient norm, learning rate, generation/training KL error, reply
  length and step time.
- `sampling.txt`: the vLLM sampling parameters used for evaluation.

The run predates this repository's layout, so its config and manifest use
earlier names for two things: the scheduler `deadline_experiment.ReproductionScheduler`
(now `deadline_grpo.CosineWithWarmup`, the same schedule) and the lifecycle
`resident-fast-clean` (now `--lifecycle resident`, the same patch).
`nemo/prepare.py` reproduces the config exactly apart from run paths and the
scheduler's name.
