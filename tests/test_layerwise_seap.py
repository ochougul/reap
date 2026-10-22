import torch

from reap.prune import allocate_layerwise_pruning_counts


def _state(num_experts, contribution):
    return {
        "expert_frequency": torch.ones(num_experts, dtype=torch.long),
        "moe_residual_contribution": torch.tensor(contribution),
    }


def test_layerwise_allocator_preserves_global_budget_and_favors_low_contribution():
    observer_data = {
        0: _state(8, 1.0),
        1: _state(8, 0.25),
    }

    counts = allocate_layerwise_pruning_counts(observer_data, 8)

    assert sum(counts.values()) == 8
    assert counts[1] > counts[0]
    assert all(count < 8 for count in counts.values())


def test_layerwise_allocator_handles_zero_contribution_with_caps():
    observer_data = {
        0: _state(4, 0.0),
        1: _state(4, 1.0),
    }

    counts = allocate_layerwise_pruning_counts(observer_data, 6)

    assert counts == {0: 3, 1: 3}


def test_layerwise_allocator_requires_contribution_statistics():
    observer_data = {0: {"expert_frequency": torch.ones(4, dtype=torch.long)}}

    try:
        allocate_layerwise_pruning_counts(observer_data, 1)
    except ValueError as exc:
        assert "moe_residual_contribution" in str(exc)
    else:
        raise AssertionError("Expected missing contribution statistics to fail")
