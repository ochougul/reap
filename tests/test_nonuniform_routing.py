import pytest
import torch

from reap.nonuniform_routing import masked_topk_routing, assert_routing_invariants


def test_mask_is_applied_before_topk_and_normalization():
    logits = torch.tensor([[100.0, 99.0, -1.0, -2.0]])
    valid = torch.tensor([True, False, True, False])
    probs, values, indices = masked_topk_routing(logits, valid, 2)
    assert set(indices[0].tolist()) == {0, 2}
    assert probs[0, 1] == 0 and probs[0, 3] == 0
    assert torch.allclose(values.sum(-1), torch.ones(1))
    assert_routing_invariants(logits, valid, 2)


def test_mask_requires_enough_retained_experts():
    with pytest.raises(ValueError, match="top_k"):
        masked_topk_routing(torch.randn(1, 3), torch.tensor([True, False, False]), 2)
