"""Run the experiment on Modal: train, merge and evaluate.

    uv run modal run --detach nemo/modal_app.py::reproduce --run grpo1000-repro --steps 1000
    uv run python nemo/collect.py --run grpo1000-repro

Training uses NVIDIA's NeMo RL container, pinned by digest, on one H100.
Merging and evaluation use vLLM on an L40S. Runs are written to a Modal volume
(`nemo-markets-rl` unless $NEMO_MARKETS_VOLUME says otherwise); model weights
are cached in the `huggingface-cache` volume.
"""

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
# nvcr.io/nvidia/nemo-rl:v0.7.0 (linux/amd64), pinned by digest
NEMO_RL_IMAGE = "nvcr.io/nvidia/nemo-rl@sha256:9392cc2cc07140f6a4cbd30f31ebffd0a54de3ded25c5a3b49505407a779e4f9"
VLLM_VERSION = "0.21.0"
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
TRAIN_GPU = os.environ.get("NEMO_GPU", "H100")
EVAL_GPU = "L40S"

app = modal.App("nemo-markets-rl")
runs = modal.Volume.from_name(os.environ.get("NEMO_MARKETS_VOLUME", "nemo-markets-rl"), create_if_missing=True)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
VOLUMES = {"/vol": runs, "/root/.cache/huggingface": hf_cache}


def _with_code(image: modal.Image) -> modal.Image:
    ignore = ["**/__pycache__", "**/*.pyc"]
    return image.add_local_dir(REPO / "src", "/workspace/src", ignore=ignore).add_local_dir(
        REPO / "nemo", "/workspace/nemo", ignore=ignore
    )


train_image = _with_code(modal.Image.from_registry(NEMO_RL_IMAGE).entrypoint([]))
eval_image = _with_code(
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}")
)


def _run(*command: str) -> None:
    try:
        subprocess.run(command, check=True)
    finally:
        runs.commit()


@app.function(image=train_image, gpu=TRAIN_GPU, cpu=16, memory=65536, volumes=VOLUMES, timeout=24 * 3600)
def train(run: str, steps: int, lifecycle: str = "resident"):
    """GRPO training. A run of 5 steps is a cheap end-to-end smoke test."""
    _run("python", "/workspace/nemo/launch.py", "--run", run, "--steps", str(steps), "--lifecycle", lifecycle)


@app.function(image=eval_image, cpu=8, memory=65536, volumes=VOLUMES, timeout=1800)
def merge(run: str):
    """Merge the run's last saved adapter into the base model, in <run>/merged."""
    root = Path("/vol") / run
    adapters = list((root / "checkpoints").rglob("adapter_config.json"))
    if not adapters:
        raise RuntimeError(f"No adapter checkpoint under {root / 'checkpoints'}")
    adapter = max(adapters, key=lambda p: p.stat().st_mtime).parent
    _run("python", "/workspace/nemo/export.py", str(adapter), str(root / "merged"))


@app.function(image=eval_image, gpu=EVAL_GPU, cpu=8, memory=49152, volumes=VOLUMES, timeout=3600)
def evaluate(run: str, base: bool = False, limit: int = 300):
    """Evaluate <run>/merged into <run>/eval, or the untrained model into <run>/eval-base."""
    root = Path("/vol") / run
    model = BASE_MODEL if base else str(root / "merged")
    output = root / ("eval-base" if base else "eval")
    _run("python", "/workspace/nemo/evaluate.py", "--model", model, "--output", str(output), "--limit", str(limit))


@app.function(image=modal.Image.debian_slim(), volumes={"/vol": runs}, timeout=24 * 3600)
def reproduce(run: str, steps: int = 1000, lifecycle: str = "resident"):
    """Train, merge and evaluate in sequence. Progress, failures and the final
    summaries are written to <run>.pipeline.json on the volume."""
    if Path(run).name != run or steps <= 0:
        raise ValueError("Use a single directory name and a positive step count")
    status_path = Path("/vol") / f"{run}.pipeline.json"
    if status_path.exists() or (Path("/vol") / run).exists():
        raise ValueError(f"Run {run} already exists; choose a new name")

    def status(stage: str, **extra):
        runs.reload()
        value = {
            "run": run,
            "steps": steps,
            "lifecycle": lifecycle,
            "stage": stage,
            "updated_utc": datetime.now(UTC).isoformat(),
            **extra,
        }
        status_path.write_text(json.dumps(value, indent=2) + "\n")
        runs.commit()
        print(json.dumps(value), flush=True)

    stage = "training"
    try:
        status(stage)
        train.remote(run=run, steps=steps, lifecycle=lifecycle)
        stage = "merging"
        status(stage)
        merge.remote(run=run)
        stage = "evaluating"
        status(stage)
        evaluate.remote(run=run)
        runs.reload()
        summaries = {
            split: json.loads((Path("/vol") / run / "eval" / f"{split}_summary.json").read_text())
            for split in ("test", "shift")
        }
        status("complete", results=summaries)
        return summaries
    except Exception as exc:
        status("failed", failed_stage=stage, error=str(exc))
        raise
