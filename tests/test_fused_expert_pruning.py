import torch
from torch import nn

from reap.model_util import slice_fused_expert_weights


class _PackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 4
        self.gate_up_proj = nn.Parameter(torch.arange(4 * 2 * 3).reshape(4, 2, 3).float())
        self.down_proj = nn.Parameter(torch.arange(4 * 3 * 2).reshape(4, 3, 2).float())
        self.register_buffer("gate_up_proj_scale_inv", torch.arange(4 * 2).reshape(4, 2).float())
        self.register_buffer("down_proj_scale_inv", torch.arange(4 * 3).reshape(4, 3).float())
        # Exercise loaders that leave split FP8 scale names after packing.
        self.register_buffer("gate_proj_scale_inv", torch.arange(4).float())
        self.register_buffer("up_proj_scale_inv", torch.arange(4).float() + 10)


def test_packed_expert_weights_and_fp8_scales_follow_expert_indices():
    experts = _PackedExperts()
    retained = [3, 1]
    original = {name: getattr(experts, name).detach().clone() for name in (
        "gate_up_proj", "down_proj", "gate_up_proj_scale_inv", "down_proj_scale_inv",
        "gate_proj_scale_inv", "up_proj_scale_inv",
    )}

    slice_fused_expert_weights(experts, retained)

    assert experts.num_experts == 2
    for name, value in original.items():
        assert torch.equal(getattr(experts, name), value[retained])


def test_qwen38_pruning_budgets_are_ratio_based():
    state = {
        layer: {"expert_frequency": torch.zeros(512)}
        for layer in range(48)
    }
    total_experts = next(iter(state.values()))["expert_frequency"].shape[0]
    for ratio, expected_pruned, expected_retained in (
        (0.25, 128, 384),
        (0.50, 256, 256),
    ):
        n_pruned = int(total_experts * ratio)
        assert n_pruned == expected_pruned
        assert total_experts - n_pruned == expected_retained

