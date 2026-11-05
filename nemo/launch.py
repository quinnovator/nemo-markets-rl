"""Prepare a run, check the objective, and train. Runs inside the NeMo RL container.

    python nemo/launch.py --run my-run --steps 1000

Everything goes to $RUNS_DIR/<run> (default /vol, the Modal volume): config,
manifest, train.log, gpu.csv (sampled every second), metrics.jsonl,
rollouts.jsonl, checkpoints and status.json. Logs are kept even if training fails.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from prepare import LIFECYCLES, prepare

RUNS_DIR = Path(os.environ.get("RUNS_DIR", "/vol"))
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="new run name; existing runs are never reused")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lifecycle", choices=LIFECYCLES, default="resident")
    args = parser.parse_args()
    if Path(args.run).name != args.run:
        raise ValueError("--run must be a single directory name")

    rl_root = Path(importlib.util.find_spec("nemo_rl").origin).resolve().parent.parent
    run_dir = RUNS_DIR / args.run
    config = prepare(rl_root, run_dir, args.steps, args.lifecycle)
    _record_runtime(rl_root, run_dir / "manifest.json")

    env = os.environ | {
        "PYTHONPATH": str(rl_root),
        "PYTHONUNBUFFERED": "1",
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
        "RAY_DEDUP_LOGS": "0",
        "DEADLINE_RUN_DIR": str(run_dir),
    }
    subprocess.run([sys.executable, str(HERE / "check_objective.py")], cwd=rl_root, env=env, check=True)

    command = [sys.executable, str(rl_root / "deadline_grpo.py"), "--config", str(config)]
    print("Running:", command, flush=True)
    with (run_dir / "gpu.csv").open("w") as gpu_log, (run_dir / "train.log").open("w") as log:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,memory.used,utilization.gpu,power.draw",
                "--format=csv,nounits",
                "-l",
                "1",
            ],
            stdout=gpu_log,
        )
        process = subprocess.Popen(
            command, cwd=rl_root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            exit_code = process.wait()
        finally:
            monitor.terminate()
            monitor.wait()
    (run_dir / "status.json").write_text(json.dumps({"exit_code": exit_code}) + "\n")
    if exit_code:
        raise RuntimeError(f"NeMo RL exited {exit_code}; logs are in {run_dir}")


def _record_runtime(rl_root: Path, manifest_path: Path) -> None:
    """Add the container's NeMo RL commit, submodules and GPU to the manifest."""

    def output(*cmd):
        return subprocess.check_output(cmd, cwd=rl_root, text=True).strip()

    manifest = json.loads(manifest_path.read_text())
    manifest["nemo_rl_commit"] = output("git", "rev-parse", "HEAD")
    manifest["submodules"] = output("git", "submodule", "status")
    manifest["gpu"] = output("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
