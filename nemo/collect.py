"""Download a finished run from the Modal volume into results/.

    uv run python nemo/collect.py --run grpo1000-repro          # trained run
    uv run python nemo/collect.py --run grpo1000-repro --base   # its untrained eval

Fetches the evaluation files, the manifest, config and audit reports, and
reduces metrics.jsonl to training.csv. Prints the pipeline status first.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from deadline_offer.analysis import training_curve  # noqa: E402

EVAL_FILES = ("test.jsonl", "shift.jsonl", "test_summary.json", "shift_summary.json", "sampling.txt")
RUN_FILES = ("manifest.json", "config.yaml", "base_weight_audit.json", "merged/merge_manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--base", action="store_true", help="fetch <run>/eval-base, the untrained model's eval")
    parser.add_argument("--dest", type=Path, default=REPO / "results")
    args = parser.parse_args()
    if Path(args.run).name != args.run:
        raise ValueError("--run must be a single directory name")
    volume = modal.Volume.from_name(os.environ.get("NEMO_MARKETS_VOLUME", "nemo-markets-rl"))

    def read(remote: str) -> bytes:
        return b"".join(volume.read_file(remote))

    if not args.base:
        try:
            status = json.loads(read(f"{args.run}.pipeline.json"))
            print(f"{args.run}: {status['stage']}")
            if status["stage"] != "complete":
                print(status.get("error", ""))
                return
        except (FileNotFoundError, modal.exception.NotFoundError):
            print(f"{args.run}: no pipeline status (trained with `train` directly?)")

    out = args.dest / (f"{args.run}-base" if args.base else args.run)
    out.mkdir(parents=True, exist_ok=True)
    eval_dir = "eval-base" if args.base else "eval"
    for name in EVAL_FILES:
        (out / name).write_bytes(read(f"{args.run}/{eval_dir}/{name}"))
    if not args.base:
        for name in RUN_FILES:
            (out / Path(name).name).write_bytes(read(f"{args.run}/{name}"))
        metrics = out / "metrics.jsonl"
        metrics.write_bytes(read(f"{args.run}/metrics.jsonl"))
        training_curve(metrics, out / "training.csv")
        metrics.unlink()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
