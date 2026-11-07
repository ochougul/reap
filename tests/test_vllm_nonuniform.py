import torch

from reap.vllm_nonuniform import mask_router_logits


def test_vllm_mask_prevents_padded_routes():
    logits = torch.tensor([[100.0, 99.0, -1.0, 98.0]])
    masked = mask_router_logits(logits, torch.tensor([True, False, True, False]))
    values, indices = torch.topk(torch.softmax(masked.float(), -1), 2, -1)
    assert set(indices[0].tolist()) == {0, 2}
    assert torch.allclose(values.sum(-1), torch.ones(1), atol=1e-5)
    assert masked[0, 1] < -1e30 and masked[0, 3] < -1e30
