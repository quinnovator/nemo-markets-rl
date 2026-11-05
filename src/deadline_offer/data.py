"""Generate the train / validation / test / shift splits.

Each row is a NeMo Gym dataset row: the prompt (`responses_create_params`) plus
the hidden task, which the resources server needs to score a reply. Seeds are
fixed, so every run sees identical tasks; `SHA256` pins the exact bytes the
published runs trained and evaluated on.

    python -m deadline_offer.data data/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Iterator
from pathlib import Path

from . import market

# name: (task distribution, seed, rows)
SPLITS = {
    "train": ("train", 1, 20000),
    "validation": ("train", 2, 500),
    "test": ("test", 1234, 1000),
    "shift": ("shift", 999, 1000),
}

SHA256 = {
    "train": "0518940f8ab8beb53df8f8f216a9034e59c2fd5219b6bf8cbf78554d36a546fa",
    "validation": "3b7fa1a7aab93032837cd582ce0ba7fd950efd93c58f163dd296c494876a0f95",
    "test": "7a5be0b08a898e7ee29a7ec7125c93ebbb15f01834fc2a0825a1872eacd9c39c",
    "shift": "e2ffe3ce1f2158776355ebc92d7b3c8e0bb10205a7d4a216dafe85934142c589",
}

EVAL_TASKS = 300  # the published evals use the first 300 tasks of test and shift


def tasks(split: str) -> Iterator[market.Task]:
    kind, seed, n = SPLITS[split]
    rng = random.Random(seed)
    for i in range(n):
        yield market.sample_task(rng, kind, i)


def row(task: market.Task) -> dict:
    return {
        "responses_create_params": {
            "input": market.to_messages(task),
            "temperature": 1.0,
            "max_output_tokens": 256,
        },
        "task": market.task_to_json(task),
    }


def lines(split: str) -> Iterator[str]:
    for task in tasks(split):
        yield json.dumps(row(task)) + "\n"


def write_split(split: str, directory: Path) -> Path:
    """Write one split and check its bytes against the published hash."""
    path = directory / f"{split}.jsonl"
    digest = hashlib.sha256()
    with path.open("w") as f:
        for line in lines(split):
            f.write(line)
            digest.update(line.encode())
    if digest.hexdigest() != SHA256[split]:
        raise RuntimeError(f"{path} does not match the published {split} split")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path, nargs="?", default=Path("data"))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = write_split(split, args.directory)
        print(f"{SPLITS[split][2]:>6} rows -> {path}")


if __name__ == "__main__":
    main()
