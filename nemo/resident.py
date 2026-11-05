"""Keep the policy and vLLM resident on one GPU between training and generation.

By default NeMo RL offloads the policy to CPU and puts vLLM to sleep at every
switch between rollouts and training. On one H100 there is room for both, and
skipping the round trip cut a training step from about 15 s to about 4 s.

This patches a few methods of the pinned NeMo RL source in place. Weight
refits, cache invalidation, forward/backward passes and optimizer updates stay
upstream code; only CPU offloading and a redundant allocator cleanup are
removed. Each patched file's original source and hashes are saved to the run
directory, and `restore` puts pristine source back first, so a reused
container never inherits a patch from an earlier run.
"""

import ast
import hashlib
import json
from pathlib import Path

POLICY_WORKER = "nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py"
VLLM_WORKER = "nemo_rl/models/generation/vllm/vllm_worker_async.py"


def restore(rl_root: Path) -> None:
    for relative in (POLICY_WORKER, VLLM_WORKER):
        path = rl_root / relative
        backup = path.with_suffix(".pristine")
        if backup.exists():
            path.write_bytes(backup.read_bytes())
        else:
            backup.write_bytes(path.read_bytes())


def install(rl_root: Path, run_dir: Path) -> None:
    guard = 'assert not self.cpu_offload, "Resident mode requires cpu_offload=False"\n'
    # move_to_cuda hides another gc.collect()/empty_cache pair; the lower-level
    # device mover keeps the placement guarantee without that sweep.
    to_gpu = 'self.model = self.move_to_device(self.model, "cuda")\n'
    policy_class = "DTensorPolicyWorkerV2Impl"
    patches = [
        _replace_methods(
            rl_root / POLICY_WORKER,
            {
                (policy_class, "prepare_for_lp_inference"): guard + to_gpu + "self.model.eval()",
                (policy_class, "prepare_for_training"): guard + to_gpu + "self.model.train()\n"
                "if self.optimizer is not None:\n"
                '    self.move_optimizer_to_device("cuda")',
                (policy_class, "offload_before_refit"): guard + "return None",
                (policy_class, "offload_after_refit"): guard + "self.model.eval()",
            },
        )
    ]

    vllm_path = rl_root / VLLM_WORKER
    owners = [
        node.name
        for node in ast.parse(vllm_path.read_text()).body
        if isinstance(node, ast.ClassDef)
        and any(isinstance(m, ast.AsyncFunctionDef) and m.name == "sleep_async" for m in node.body)
    ]
    if len(owners) != 1:
        raise RuntimeError("Unexpected async vLLM worker layout")
    patches.append(
        _replace_methods(
            vllm_path,
            {
                # Instead of sleeping, only drop the caches the refit invalidates.
                (owners[0], "sleep_async"): "assert self.llm is not None\n"
                "await self.llm.reset_prefix_cache()\n"
                'if hasattr(self.llm, "reset_mm_cache"):\n'
                "    await self.llm.reset_mm_cache()",
                (owners[0], "wake_up_async"): "assert self.llm is not None\nreturn None",
            },
        )
    )
    (run_dir / "resident_patches.json").write_text(json.dumps(patches, indent=2) + "\n")


def _replace_methods(path: Path, methods: dict[tuple[str, str], str]) -> dict:
    """Replace the bodies of the named methods, failing on any ambiguity."""
    source = path.read_text()
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    changes = []
    for (class_name, method_name), body in methods.items():
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
        if len(classes) != 1:
            raise RuntimeError(f"Unexpected runtime class: {class_name}")
        found = [
            n
            for n in classes[0].body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method_name
        ]
        if len(found) != 1:
            raise RuntimeError(f"Unexpected runtime method: {method_name}")
        node = found[0]
        indented = "".join("        " + line + "\n" for line in body.splitlines())
        changes.append((node.body[0].lineno - 1, node.end_lineno, indented))
    for start, end, replacement in sorted(changes, reverse=True):
        lines[start:end] = [replacement]
    patched = "".join(lines)
    compile(patched, str(path), "exec")
    path.write_text(patched)
    return {
        "path": str(path),
        "before_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "after_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "original_source": source,
        "patched_methods": [name for _, name in methods],
    }
