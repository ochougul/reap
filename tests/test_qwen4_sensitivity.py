import torch

from experiments.qwen4_calibrate import _first_order_top11_removal


def test_top11_first_order_estimate_matches_direct_small_case():
    candidate_ids = torch.arange(11).reshape(1, 11)
    probabilities = torch.tensor(
        [[0.21, 0.16, 0.13, 0.11, 0.09, 0.08, 0.07, 0.06, 0.04, 0.03, 0.02]],
        dtype=torch.float64,
    )
    expert_outputs = torch.tensor(
        [[1.2, -0.7, 0.3, 2.0, -1.1, 0.4, 0.8, -0.2, 0.5, 1.7, -1.3]],
        dtype=torch.float64,
    )
    gradient = torch.tensor([0.6], dtype=torch.float64)
    projected = expert_outputs * gradient
    routed_scale = torch.tensor([1.4], dtype=torch.float64)

    ids, estimates = _first_order_top11_removal(
        candidate_ids, probabilities, projected, routed_scale
    )

    p11 = probabilities[0] / probabilities[0].sum()
    base_weights = p11[:10] / p11[:10].sum()
    base_output = (base_weights * expert_outputs[0, :10]).sum()
    direct = []
    for removed in range(10):
        keep = [index for index in range(11) if index != removed]
        weights = p11[keep] / p11[keep].sum()
        removed_output = (weights * expert_outputs[0, keep]).sum()
        direct.append((removed_output - base_output) * gradient.item() * routed_scale.item())

    assert torch.equal(ids, candidate_ids[:, :10])
    torch.testing.assert_close(estimates[0], torch.tensor(direct, dtype=torch.float32))


def test_top11_estimates_are_independent_per_token():
    ids = torch.arange(22).reshape(2, 11)
    probabilities = torch.softmax(torch.arange(22, dtype=torch.float32).reshape(2, 11), -1)
    projected = torch.tensor(
        [list(range(11)), list(range(10, -1, -1))], dtype=torch.float32
    )
    _, estimates = _first_order_top11_removal(ids, probabilities, projected)
    _, first = _first_order_top11_removal(ids[:1], probabilities[:1], projected[:1])
    _, second = _first_order_top11_removal(ids[1:], probabilities[1:], projected[1:])

    torch.testing.assert_close(estimates[:1], first)
    torch.testing.assert_close(estimates[1:], second)
    assert (estimates[0] > 0).any()
    assert (estimates[1] < 0).any()
