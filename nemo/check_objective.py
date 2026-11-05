"""Preflight check, run by launch.py before training.

Compares DrGRPOLoss's value and gradient with the Dr. GRPO formula written out
by hand, including microbatch accumulation and NeMo's on-policy shortcut
(`force_on_policy_ratio`), and checks the learning-rate schedule's endpoints.
Needs the NeMo RL container.
"""

import torch
from hydra.utils import get_class
from nemo_rl.algorithms.loss import ClippedPGLossConfig

from deadline_grpo import MAX_COMPLETION_TOKENS, DrGRPOLoss, cosine_multiplier


def check_loss():
    torch.manual_seed(123)
    mask = torch.tensor([[0, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0]])
    advantages = torch.tensor([1.0, -0.75, -0.25])[:, None].expand(3, 6)
    prev = torch.full((3, 6), -2.0)
    data = {
        "token_mask": mask,
        "sample_mask": torch.ones(3),
        "advantages": advantages,
        "prev_logprobs": prev,
        "generation_logprobs": prev.clone(),
    }
    current = (-2.0 + torch.randn(3, 5) * 0.3).requires_grad_()
    loss_fn = DrGRPOLoss(ClippedPGLossConfig(reference_policy_kl_penalty=0, token_level_loss=True))
    seqs, toks = torch.tensor(3.0), mask.sum()

    actual, _ = loss_fn(current, data, seqs, toks)
    ratio = (current - prev[:, 1:]).exp()
    adv = advantages[:, 1:]
    clipped = torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv)
    expected = -(clipped * mask[:, 1:]).sum() / (3 * MAX_COMPLETION_TOKENS)
    torch.testing.assert_close(actual, expected)
    expected_grad = torch.autograd.grad(expected, current, retain_graph=True)[0]
    torch.testing.assert_close(torch.autograd.grad(actual, current, retain_graph=True)[0], expected_grad)

    # Summing per-sample microbatches must give the same value and gradient.
    micro = sum(
        loss_fn(current[i : i + 1], {k: v[i : i + 1] for k, v in data.items()}, seqs, toks)[0] for i in range(3)
    )
    torch.testing.assert_close(micro, expected)
    torch.testing.assert_close(torch.autograd.grad(micro, current)[0], expected_grad)

    # With one optimizer update per rollout, the on-policy shortcut must match
    # an explicit frozen old-policy forward, including unequal reply lengths.
    on_policy = current.detach().clone().requires_grad_()
    old = torch.cat([torch.zeros(3, 1), on_policy.detach()], dim=1)
    on_data = {**data, "prev_logprobs": old, "generation_logprobs": old}
    explicit, _ = loss_fn(on_policy, on_data, seqs, toks)
    fast_fn = DrGRPOLoss(
        ClippedPGLossConfig(reference_policy_kl_penalty=0, token_level_loss=True, force_on_policy_ratio=True)
    )
    fast, _ = fast_fn(on_policy, on_data, seqs, toks)
    torch.testing.assert_close(fast, explicit)
    torch.testing.assert_close(
        torch.autograd.grad(fast, on_policy, retain_graph=True)[0], torch.autograd.grad(explicit, on_policy)[0]
    )


def check_schedule():
    assert cosine_multiplier(0, total_steps=400, warmup_steps=10) == 0
    assert cosine_multiplier(10, total_steps=400, warmup_steps=10) == 1
    assert cosine_multiplier(400, total_steps=400, warmup_steps=10) == 0
    # Resolve the class the way NeMo RL does, from its config name.
    scheduler_cls = get_class("deadline_grpo.CosineWithWarmup")
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=5e-6)
    scheduler = scheduler_cls(optimizer, total_steps=400, warmup_steps=10)
    assert optimizer.param_groups[0]["lr"] == 0
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 5e-6


if __name__ == "__main__":
    check_loss()
    check_schedule()
    print("PASS: Dr. GRPO value, gradient, microbatching, on-policy shortcut, and schedule endpoints")
