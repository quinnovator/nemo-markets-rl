"""Score a model on the first 300 test and shift tasks, eight replies each.

    python nemo/evaluate.py --model <hf id or merged dir> --output <dir>

Runs with vLLM on one GPU. Reasoning is off, temperature 1.0, up to 256
tokens per reply. Writes <split>.jsonl (one scored line per reply, keyed by
`_ng_task_index` into the split), <split>_summary.json, and sampling.txt.
`python -m deadline_offer.analysis <dir>` turns them into the reported numbers.
"""

import argparse
import copy
import json
import statistics
import sys
from itertools import islice
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from deadline_offer import data, market  # noqa: E402

REPLIES_PER_TASK = 8


def main():
    from vllm import LLM

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=data.EVAL_TASKS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        mamba_ssm_cache_dtype="float32",
    )
    tokenizer = llm.get_tokenizer()
    sampling = llm.get_default_sampling_params()
    sampling.temperature = 1.0
    sampling.max_tokens = 256
    sampling.n = REPLIES_PER_TASK
    (args.output / "sampling.txt").write_text(repr(sampling) + "\n")

    for split in ("test", "shift"):
        tasks = list(islice(data.tasks(split), args.limit))
        prompts = [
            tokenizer.apply_chat_template(
                market.to_messages(task), tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for task in tasks
        ]
        responses = llm.generate(prompts, copy.deepcopy(sampling))
        scored = []
        for i, (task, response) in enumerate(zip(tasks, responses, strict=True)):
            if len(response.outputs) != REPLIES_PER_TASK:
                raise RuntimeError(f"Expected {REPLIES_PER_TASK} replies per task")
            best_offer, best_ev = market.optimal_offer(task)
            for j, reply in enumerate(response.outputs):
                valid, offer = market.parse_action(reply.text)
                scored.append(
                    {
                        "_ng_task_index": i,
                        "_ng_rollout_index": j,
                        "text": reply.text,
                        "reward": market.reward(task, reply.text),
                        "valid_action": valid,
                        "offer": offer,
                        "optimal_offer": best_offer,
                        "expected_surplus": market.expected_surplus(task, offer) if valid else 0.0,
                        "optimal_surplus": best_ev,
                        "finish_reason": reply.finish_reason,
                    }
                )
        with (args.output / f"{split}.jsonl").open("w") as f:
            for record in scored:
                f.write(json.dumps(record) + "\n")
        summary = {
            "n_tasks": len(tasks),
            "n_rollouts": len(scored),
            "mean_reward": statistics.mean(r["reward"] for r in scored),
            "ev_capture": sum(r["expected_surplus"] for r in scored) / sum(r["optimal_surplus"] for r in scored),
            "valid_rate": statistics.mean(r["valid_action"] for r in scored),
            "negative_ev_count": sum(r["expected_surplus"] < 0 for r in scored),
            "walk_count": sum(r["valid_action"] and r["offer"] is None for r in scored),
        }
        (args.output / f"{split}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(split, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
