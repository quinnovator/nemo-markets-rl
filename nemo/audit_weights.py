"""Stop before any rollout if the policy did not load the published base weights.

`prepare.py` installs a call to `audit_base_weights` in NeMo Automodel's model
setup. Every tensor the model loaded is compared exactly with the original
BF16 checkpoint, and the report is written to base_weight_audit.json.

One known issue is repaired rather than failed: in the pinned container,
Nemotron-H's frozen Mamba `dt_bias` and `out_proj.weight` tensors (42 in all)
come back altered after initialization. Those are restored from the checkpoint
and checked again, on a single GPU only. Any other difference stops the run.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

KNOWN_ALTERED = (".mixer.dt_bias", ".mixer.out_proj.weight")


def audit_base_weights(model, model_name):
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    snapshot = Path(snapshot_download(model_name, local_files_only=True))
    keys_by_shard = defaultdict(list)
    for shard in snapshot.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as f:
            keys_by_shard[shard.name].extend(f.keys())
    if not keys_by_shard:
        raise RuntimeError("No base safetensors found")

    state = model.state_dict()
    parameters = dict(model.named_parameters())
    missing = sorted({k for keys in keys_by_shard.values() for k in keys} - state.keys())
    mismatches, restored, checked = [], [], 0
    for filename, keys in keys_by_shard.items():
        with safe_open(snapshot / filename, framework="pt", device="cpu") as f:
            for key in keys:
                if key not in state:
                    continue
                actual = state[key].detach()
                if hasattr(actual, "full_tensor"):
                    actual = actual.full_tensor()
                actual = actual.cpu()
                expected = f.get_tensor(key).to(actual.dtype)
                checked += 1
                if torch.equal(actual, expected):
                    continue
                parameter = parameters.get(key)
                repairable = (
                    key.endswith(KNOWN_ALTERED)
                    and parameter is not None
                    and not parameter.requires_grad
                    and torch.distributed.get_world_size() == 1
                )
                if not repairable:
                    mismatches.append(key)
                    continue
                target = state[key]
                if hasattr(target, "to_local"):
                    target = target.to_local()
                with torch.no_grad():
                    target.copy_(expected.to(target.device))
                if not torch.equal(target.detach().cpu(), expected):
                    raise RuntimeError(f"Could not restore {key}")
                restored.append(key)

    report = {
        "model_revision": snapshot.name,
        "checked_base_tensors": checked,
        "missing": missing,
        "mismatches": mismatches,
        "restored_frozen_tensors": restored,
    }
    (Path(os.environ["DEADLINE_RUN_DIR"]) / "base_weight_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("BASE_WEIGHT_AUDIT " + json.dumps(report), flush=True)
    if missing or mismatches:
        raise RuntimeError("Base weights differ from the checkpoint; see base_weight_audit.json")
