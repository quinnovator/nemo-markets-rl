"""Build a run directory and NeMo RL config for GRPO on the deadline market.

Runs inside the NeMo RL container (called by launch.py). It:
- installs the Gym resources server (gym/app.py + market.py) into NeMo Gym,
- copies the objective (deadline_grpo.py) and the base-weight audit into NeMo RL,
- applies the resident-model patch unless `lifecycle="stock"`,
- writes the train and validation splits, checked against their published hashes,
- writes config.yaml and manifest.json to the run directory.

The config starts from NVIDIA's GRPO and NeMo Gym example recipes; every
setting this experiment depends on is set explicitly below.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from deadline_offer import data  # noqa: E402

import resident  # noqa: E402

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
SEED = 3407
LORA_TARGETS = ["down_proj", "in_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"]
PROMPTS_PER_STEP = 8
REPLIES_PER_PROMPT = 8
MAX_COMPLETION_TOKENS = 128
LIFECYCLES = ("resident", "stock")


def prepare(rl_root: Path, run_dir: Path, steps: int, lifecycle: str = "resident") -> Path:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if lifecycle not in LIFECYCLES:
        raise ValueError(f"lifecycle must be one of {LIFECYCLES}")
    # Smoke runs of 10 steps or fewer would spend their whole schedule warming
    # up, so they borrow a 400-step horizon.
    horizon = steps if steps > 10 else 400
    run_dir.mkdir(parents=True, exist_ok=False)

    _install_code(rl_root)
    resident.restore(rl_root)
    if lifecycle == "resident":
        resident.install(rl_root, run_dir)
    data_dir = _write_training_data(run_dir / "data")

    cfg = OmegaConf.load(rl_root / "examples/configs/grpo_math_1B.yaml")
    gym_recipe = OmegaConf.load(rl_root / "examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml")
    cfg.env = gym_recipe.env
    cfg.data = gym_recipe.data
    cfg.data.train = {"data_path": str(data_dir / "train.jsonl")}
    cfg.data.validation = {"data_path": str(data_dir / "validation.jsonl")}
    gym = cfg.env.nemo_gym
    gym.config_paths = [
        "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
        "resources_servers/deadline_offer/config.yaml",
    ]
    del gym.code_gen

    cfg.grpo.update(
        {
            "num_prompts_per_step": PROMPTS_PER_STEP,
            "num_generations_per_prompt": REPLIES_PER_PROMPT,
            "max_num_steps": steps,
            "max_num_epochs": 10,
            # Advantage = reward minus the group mean: no std scaling, no leave-one-out.
            "normalize_rewards": False,
            "use_leave_one_out_baseline": False,
            "val_period": 0,
            "val_at_start": False,
            "val_at_end": False,
            "max_val_samples": None,
            "val_batch_size": None,
            "seed": SEED,
        }
    )
    cfg.loss_fn.reference_policy_kl_penalty = 0.0
    cfg.loss_fn.token_level_loss = True
    cfg.checkpointing.update(
        {
            "checkpoint_dir": str(run_dir / "checkpoints"),
            "save_period": min(50, steps),
            "keep_top_k": 30,
            "metric_name": "train:reward",
            "save_consolidated": True,
        }
    )

    policy = cfg.policy
    policy.model_name = MODEL
    policy.tokenizer.chat_template_kwargs = {"enable_thinking": False}
    policy.train_global_batch_size = PROMPTS_PER_STEP * REPLIES_PER_PROMPT
    policy.train_micro_batch_size = 8
    policy.logprob_batch_size = 8
    policy.max_total_sequence_length = 1024
    policy.max_grad_norm = 0.1
    policy.sequence_packing.enabled = False
    policy.dtensor_cfg.activation_checkpointing = True
    policy.dtensor_cfg.automodel_kwargs = {"force_hf": True}
    policy.dtensor_cfg.lora_cfg.update(
        {
            "enabled": True,
            "dim": 32,
            "alpha": 64,
            "dropout": 0.0,
            "target_modules": [f"*.{name}" for name in LORA_TARGETS],
            "match_all_linear": False,
            "use_triton": False,
            "lora_A_init": "uniform",
        }
    )
    policy.optimizer.kwargs.weight_decay = 0.1
    policy.scheduler = [
        {"name": "deadline_grpo.CosineWithWarmup", "kwargs": {"total_steps": horizon, "warmup_steps": 10}},
        {"milestones": []},
    ]
    policy.generation.max_new_tokens = MAX_COMPLETION_TOKENS
    policy.generation.vllm_cfg.update(
        {
            "async_engine": True,
            "expose_http_server": True,
            "skip_tokenizer_init": False,
            "gpu_memory_utilization": 0.35,
            "enforce_eager": True,
            "http_server_serving_chat_kwargs": {"enable_auto_tools": False},
        }
    )
    policy.generation.vllm_kwargs = {"mamba_ssm_cache_dtype": "float32"}
    if lifecycle == "resident":
        # Capture decode CUDA graphs up to the 64-reply workload. The eager
        # compiler backend keeps vLLM's custom kernels, as NVIDIA's recipe does.
        policy.generation.vllm_cfg.enforce_eager = False
        policy.generation.vllm_kwargs.update({"max_num_seqs": 64, "compilation_config": {"backend": "eager"}})
        # One optimizer update per rollout and zero dropout, so the old-policy
        # log-probs equal detached current ones (checked in check_objective.py).
        cfg.loss_fn.force_on_policy_ratio = True
        policy.dtensor_cfg.activation_checkpointing = False

    cfg.logger.update(
        {
            "log_dir": str(run_dir / "logs"),
            "wandb_enabled": False,
            "tensorboard_enabled": True,
            "monitor_gpus": True,
        }
    )
    cfg.cluster.gpus_per_node = 1
    cfg.cluster.num_nodes = 1
    config_path = run_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)

    manifest = {
        "model": MODEL,
        "steps": steps,
        "schedule_steps": horizon,
        "seed": SEED,
        "lifecycle": lifecycle,
        "data_sha256": {split: data.SHA256[split] for split in ("train", "validation")},
        "source_sha256": {
            str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted([*(REPO / "nemo").rglob("*.py"), REPO / "src/deadline_offer/market.py"])
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return config_path


def _install_code(rl_root: Path) -> None:
    resource = rl_root / "3rdparty/Gym-workspace/Gym/resources_servers/deadline_offer"
    resource.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "src/deadline_offer/market.py", resource / "market.py")
    shutil.copy2(REPO / "nemo/gym/app.py", resource / "app.py")
    shutil.copy2(REPO / "nemo/gym/config.yaml", resource / "config.yaml")
    (resource / "requirements.txt").write_text("-e nemo-gym[dev] @ ../../\n")
    (resource / "__init__.py").touch()
    shutil.copy2(REPO / "nemo/deadline_grpo.py", rl_root / "deadline_grpo.py")
    shutil.copy2(REPO / "nemo/audit_weights.py", rl_root / "audit_weights.py")

    # Call the base-weight audit right after Automodel builds the model.
    setup = rl_root / "nemo_rl/models/automodel/setup.py"
    code = setup.read_text()
    anchor = "    print(model)\n"
    hook = "    from audit_weights import audit_base_weights\n    audit_base_weights(model, model_name)\n"
    if hook not in code:
        if code.count(anchor) != 1:
            raise RuntimeError("Automodel setup changed; cannot install the base-weight audit")
        setup.write_text(code.replace(anchor, anchor + hook))


def _write_training_data(directory: Path) -> Path:
    """Write the train and validation splits in the form NeMo RL reads: each row
    names the Gym agent, and replies are capped at 128 tokens."""
    directory.mkdir()
    for split in ("train", "validation"):
        digest = hashlib.sha256()
        with (directory / f"{split}.jsonl").open("w") as f:
            for line in data.lines(split):
                digest.update(line.encode())
                row = json.loads(line)
                row["agent_ref"] = {"name": "deadline_offer_simple_agent"}
                row["responses_create_params"]["max_output_tokens"] = MAX_COMPLETION_TOKENS
                f.write(json.dumps(row) + "\n")
        if digest.hexdigest() != data.SHA256[split]:
            raise RuntimeError(f"Generated {split} split does not match the published data")
    return directory
