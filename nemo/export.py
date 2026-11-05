"""Merge a NeMo Automodel LoRA checkpoint into the base model for evaluation.

    python nemo/export.py <adapter dir> <output dir>

The output is a plain Hugging Face checkpoint: the base model's own config and
code files, with each LoRA update B·A·(alpha/r) added to its weight in FP32 and
cast back to BF16. The adapter must have exactly the 71 targets and shapes in
expected_adapter_shapes.json, and at least one nonzero B, or nothing is written.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

EXPECTED_SHAPES = Path(__file__).with_name("expected_adapter_shapes.json")


def export(adapter_dir: Path, destination: Path) -> dict:
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    if any(config.get(k) for k in ("use_dora", "use_rslora", "rank_pattern", "alpha_pattern")):
        raise ValueError("Only plain fixed-rank LoRA is supported")
    base = Path(snapshot_download(config["base_model_name_or_path"]))
    adapters = load_file(str(adapter_dir / "adapter_model.safetensors"))
    shapes = {key.split("layers.", 1)[1]: list(value.shape) for key, value in adapters.items()}
    if shapes != json.loads(EXPECTED_SHAPES.read_text()):
        raise ValueError("Adapter targets or shapes differ from the expected 71-target LoRA")

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in adapters.items():
        name = key.removeprefix("base_model.model.")
        for part in ("A", "B"):
            suffix = f".lora_{part}.weight"
            if name.endswith(suffix):
                pairs.setdefault(name.removesuffix(suffix) + ".weight", {})[part] = value
                break
        else:
            raise ValueError(f"Unexpected adapter tensor: {key}")
    if not pairs or any(set(pair) != {"A", "B"} for pair in pairs.values()):
        raise ValueError("Missing LoRA tensor pairs")
    if not any(torch.count_nonzero(pair["B"]) for pair in pairs.values()):
        raise ValueError("Every LoRA B weight is zero: there is no learned update to export")

    destination.mkdir(parents=True, exist_ok=False)
    for path in base.iterdir():
        if path.is_file() and path.suffix in (".json", ".py", ".jinja", ".txt", ".model"):
            shutil.copy2(path, destination / path.name)
    scale = config["lora_alpha"] / config["r"]
    merged = set()
    for shard in sorted(base.glob("*.safetensors")):
        weights = load_file(str(shard))
        for name, pair in pairs.items():
            if name not in weights:
                continue
            delta = (pair["B"].float() @ pair["A"].float()) * scale
            if delta.shape != weights[name].shape or not torch.isfinite(delta).all():
                raise ValueError(f"Invalid LoRA update for {name}")
            weights[name] = (weights[name].float() + delta).to(weights[name].dtype)
            merged.add(name)
        save_file(weights, str(destination / shard.name), metadata={"format": "pt"})
        del weights
    if merged != set(pairs):
        raise ValueError(f"Adapter targets not found in the base model: {set(pairs) - merged}")

    result = {
        "base_snapshot": str(base),
        "adapter": str(adapter_dir),
        "targets_merged": len(merged),
        "merge_precision": "FP32 accumulation, cast back to base dtype",
    }
    (destination / "merge_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("adapter", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.adapter, args.destination)
