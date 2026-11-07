import torch

from experiments.qwen4_calibrate import (
    _local_output_rms,
    _top11_local_output_delta_squares,
)


def _direct_counterfactuals(ids, probs, outputs, scales):
    squared_sum = torch.zeros(int(ids.max()) + 1, dtype=torch.float64)
    counts = torch.zeros(int(ids.max()) + 1, dtype=torch.long)
    for token in range(ids.shape[0]):
        base_weights = probs[token, :10] / probs[token, :10].sum()
        baseline = scales[token] * (base_weights[:, None] * outputs[token, :10]).sum(0)
        for slot in range(10):
            replacement_weights = probs[token].clone()
            replacement_weights[slot] = 0
            replacement_weights /= replacement_weights.sum()
            counterfactual = scales[token] * (
                replacement_weights[:, None] * outputs[token]
            ).sum(0)
            delta_square = (counterfactual - baseline).square().sum()
            expert = int(ids[token, slot])
            squared_sum[expert] += delta_square.double()
            counts[expert] += 1
    return squared_sum, counts


def test_top11_local_output_deltas_match_direct_counterfactuals_and_slot_mapping():
    selected_ids = torch.tensor(
        [list(range(10)), list(range(2, 12))], dtype=torch.long
    )
    candidate_ids = torch.cat(
        [selected_ids, torch.tensor([[10], [12]], dtype=torch.long)], dim=-1
    )
    logits = torch.tensor(
        [
            [2.1, 1.9, 1.7, 1.5, 1.3, 1.1, 0.9, 0.7, 0.5, 0.3, 0.1],
            [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
        ],
        dtype=torch.float64,
    )
    probs = logits.softmax(dim=-1)
    scales = torch.tensor([1.3, 0.7], dtype=torch.float64)
    selected_weights = scales[:, None] * probs[:, :10] / probs[:, :10].sum(-1, keepdim=True)
    all_outputs = torch.tensor(
        [
            [[i * 0.2 - 0.5, i * -0.1 + 0.3, (i % 3) * 0.4] for i in range(11)],
            [[i * -0.15 + 0.7, i * 0.25 - 0.2, (i % 4) * -0.3] for i in range(11)],
        ],
        dtype=torch.float64,
    )

    # Simulate the expert dispatcher: rows are sorted by expert ID, and each
    # permutation value maps back to flattened (token, top-k slot) order.
    original_rows = torch.arange(selected_ids.numel())
    flat_ids = selected_ids.reshape(-1)
    sort_order = torch.argsort(flat_ids, stable=True)
    permutation = original_rows[sort_order]
    sorted_experts = flat_ids[sort_order]
    flattened_outputs = all_outputs[:, :10].reshape(-1, all_outputs.shape[-1])
    sorted_outputs = flattened_outputs[sort_order]

    actual_sum, actual_count = _top11_local_output_delta_squares(
        candidate_probs=probs.float(),
        selected_ids=selected_ids,
        selected_weights=selected_weights.float(),
        routed_scale=scales,
        eleventh_output=all_outputs[:, 10].float(),
        expert_ids=sorted_experts,
        expert_outputs=sorted_outputs.float(),
        route_permutation=permutation,
        num_experts=13,
        chunk_tokens=1,
    )
    expected_sum, expected_count = _direct_counterfactuals(
        selected_ids, probs, all_outputs, scales
    )

    torch.testing.assert_close(actual_sum[: expected_sum.numel()], expected_sum, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual_count[: expected_count.numel()], expected_count)
    assert actual_sum[12] == 0
    assert actual_count[12] == 0


def test_local_output_rms_is_aggregated_over_expert_occurrences():
    squared_sum = torch.tensor([20.0, 18.0, 0.0, 7.0], dtype=torch.float64)
    occurrences = torch.tensor([5, 2, 0, 1], dtype=torch.long)

    scores = _local_output_rms(squared_sum, occurrences)

    torch.testing.assert_close(
        scores,
        torch.tensor([2.0, 3.0, 0.0, 7.0**0.5], dtype=torch.float64),
    )
