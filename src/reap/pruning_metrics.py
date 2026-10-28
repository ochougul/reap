from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F

from reap.metrics import OnlineStatsTracker


@dataclass
class PreparedPruningBatch:
    activations: torch.Tensor
    selected_experts: torch.Tensor
    router_logits: torch.Tensor
    num_tokens: torch.Tensor
    expert_frequency: torch.Tensor
    pairwise_expert_frequency: torch.Tensor
    expert_indices: torch.Tensor


def initialize_pruning_state(
    num_experts: int,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Create the pruning-only per-layer state structure on the requested device.

    The returned mapping matches the pruning-related subset of the observer state used
    by both the standard and layerwise observers.
    """
    layer_state = {}
    layer_state["total_tokens"] = torch.tensor(0, device=device, dtype=torch.long)
    layer_state["expert_frequency"] = torch.zeros(
        num_experts, device=device, dtype=torch.long
    )
    layer_state["pairwise_expert_frequency"] = torch.zeros(
        num_experts, num_experts, dtype=torch.long, device=device
    )
    layer_state["ean_sum"] = torch.zeros(
        (num_experts,), device=device, dtype=torch.float64, requires_grad=False
    )
    layer_state["weighted_ean_sum"] = torch.zeros(
        (num_experts,), device=device, dtype=torch.float64, requires_grad=False
    )
    layer_state["ean_mean"] = OnlineStatsTracker(
        shape=(num_experts,),
        count_shape=(num_experts,),
        device=device,
        dtype=torch.float32,
    )
    layer_state["reap"] = OnlineStatsTracker(
        shape=(num_experts,),
        count_shape=(num_experts,),
        device=device,
        dtype=torch.float32,
    )
    layer_state["weighted_expert_frequency_sum"] = torch.zeros(
        (num_experts,), device=device, dtype=torch.float64, requires_grad=False
    )
    layer_state["max_activations"] = torch.zeros(
        (num_experts,), device=device, dtype=torch.float32, requires_grad=False
    )
    return layer_state


def _prepare_pruning_batch(
    *,
    activations: torch.Tensor,
    selected_experts: torch.Tensor,
    router_logits: torch.Tensor,
    num_experts: int,
    expert_indices: Optional[torch.Tensor] = None,
    valid_token_mask: Optional[torch.Tensor] = None,
) -> PreparedPruningBatch:
    """Normalize pruning inputs into a token-aligned batch representation.

    This flattens `selected_experts`, optionally filters all tensors with a valid-token
    mask, validates the resulting shapes, and precomputes token counts and routing
    frequencies needed by downstream pruning updates.
    """
    device = activations.device
    selected_experts = selected_experts.reshape(-1, selected_experts.shape[-1]).to(device)
    router_logits = router_logits.to(device)

    # Filter out padding tokens if attention mask is provided
    if valid_token_mask is not None:
        valid_token_mask = valid_token_mask.reshape(-1).bool().to(device)
        # Filter activations: (num_experts, total_tokens, hidden_dim) -> (num_experts, num_valid_tokens, hidden_dim)
        activations = activations[:, valid_token_mask, :]
        # Filter selected_experts: (total_tokens, top_k) -> (num_valid_tokens, top_k)
        selected_experts = selected_experts[valid_token_mask]
        # Filter router_logits: (total_tokens, num_experts) -> (num_valid_tokens, num_experts)
        router_logits = router_logits[valid_token_mask]

    if expert_indices is None:
        expert_indices = torch.arange(num_experts, device=device, dtype=torch.long)
    else:
        expert_indices = expert_indices.reshape(-1).to(device=device, dtype=torch.long)
    if activations.shape[0] != expert_indices.numel():
        raise ValueError(
            "Activation and expert-index counts do not match: "
            f"{activations.shape[0]} vs {expert_indices.numel()}"
        )
    if expert_indices.numel() and (
        expert_indices.min() < 0 or expert_indices.max() >= num_experts
    ):
        raise ValueError("expert_indices contains an invalid expert id")
    if router_logits.shape[1] != num_experts:
        raise ValueError(
            f"Expected router logits for {num_experts} experts, got {router_logits.shape[1]}"
        )
    if activations.shape[1] != selected_experts.shape[0]:
        raise ValueError(
            "Activations and selected expert token counts do not match: "
            f"{activations.shape[1]} vs {selected_experts.shape[0]}"
        )
    if router_logits.shape[0] != selected_experts.shape[0]:
        raise ValueError(
            "Router logits and selected expert token counts do not match: "
            f"{router_logits.shape[0]} vs {selected_experts.shape[0]}"
        )

    num_tokens = torch.tensor(selected_experts.shape[0], device="cpu", dtype=torch.long)
    if selected_experts.numel() == 0:
        expert_frequency = torch.zeros(num_experts, device=device, dtype=torch.long)
    else:
        expert_frequency = torch.bincount(
            selected_experts.reshape(-1), minlength=num_experts
        ).to(device)
    pairwise_expert_frequency = expert_frequency.unsqueeze(0) + expert_frequency.unsqueeze(1)

    return PreparedPruningBatch(
        activations=activations,
        selected_experts=selected_experts,
        router_logits=router_logits,
        num_tokens=num_tokens,
        expert_frequency=expert_frequency,
        pairwise_expert_frequency=pairwise_expert_frequency,
        expert_indices=expert_indices,
    )


def update_pruning_state(
    layer_state: dict[str, Any],
    *,
    activations: torch.Tensor,
    selected_experts: torch.Tensor,
    router_logits: torch.Tensor,
    num_experts: int,
    expert_indices: Optional[torch.Tensor] = None,
    accumulate_routing_stats: bool = True,
    valid_token_mask: Optional[torch.Tensor] = None,
    renormalize_router_weights: bool = False,
) -> PreparedPruningBatch:
    """Accumulate pruning saliency metrics for one routed batch into `layer_state`.

    The update computes expert/token counts, EAN aggregates, router-weighted variants,
    REAP scores, and maximum activation magnitudes. It returns the prepared batch so
    callers can reuse the filtered tensors and precomputed counts for additional metrics.
    """
    pruning_batch = _prepare_pruning_batch(
        activations=activations,
        selected_experts=selected_experts,
        router_logits=router_logits,
        num_experts=num_experts,
        expert_indices=expert_indices,
        valid_token_mask=valid_token_mask,
    )

    device = pruning_batch.activations.device
    if accumulate_routing_stats:
        layer_state["total_tokens"] += pruning_batch.num_tokens
        layer_state["expert_frequency"] += pruning_batch.expert_frequency.to("cpu", torch.long)
        layer_state["pairwise_expert_frequency"] += pruning_batch.pairwise_expert_frequency.to(
            "cpu", torch.long
        )

    metric_frequency = torch.zeros_like(pruning_batch.expert_frequency)
    metric_frequency[pruning_batch.expert_indices] = pruning_batch.expert_frequency[
        pruning_batch.expert_indices
    ]

    ean_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
    ean_mean = torch.zeros(num_experts, device=device, dtype=torch.float32)
    weighted_ean_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
    reap = torch.zeros(num_experts, device=device, dtype=torch.float32)
    weighted_expert_frequency_sum = torch.zeros(
        num_experts, device=device, dtype=torch.float64
    )

    routing_weights = F.softmax(pruning_batch.router_logits, dim=1, dtype=torch.float).to(
        device
    )
    if renormalize_router_weights and pruning_batch.selected_experts.numel() > 0:
        topk_weights = torch.gather(
            routing_weights,
            1,
            pruning_batch.selected_experts,
        )
        routing_weights = routing_weights / topk_weights.sum(dim=-1, keepdim=True)
        routing_weights = torch.clamp(
            routing_weights, min=torch.finfo(routing_weights.dtype).eps
        )

    for local_i, expert_id in enumerate(pruning_batch.expert_indices.tolist()):
        active_mask = (pruning_batch.selected_experts == expert_id).any(dim=-1).to(device)
        if not active_mask.any():
            continue

        selected_activations = pruning_batch.activations[local_i, active_mask, :]
        active_router_weights = routing_weights[active_mask, expert_id]
        ean_norm = torch.linalg.norm(selected_activations, dim=-1)
        ean_sum[expert_id] = ean_norm.sum().to(device)
        ean_mean[expert_id] = ean_norm.mean().to(device)
        weighted_expert_frequency_sum[expert_id] = active_router_weights.sum().to(device)
        weighted_ean_sum[expert_id] = (ean_norm * active_router_weights).sum().to(device)
        reap[expert_id] = (ean_norm * active_router_weights).mean().to(device)

        selected_activations_max = selected_activations.max().to(device="cpu")
        if selected_activations_max > layer_state["max_activations"][expert_id]:
            layer_state["max_activations"][expert_id] = selected_activations_max

    layer_state["ean_sum"] += ean_sum.to(device="cpu")
    layer_state["ean_mean"].update(
        ean_mean.to("cpu"), metric_frequency.to("cpu")
    )
    layer_state["weighted_ean_sum"] += weighted_ean_sum.to(device="cpu")
    layer_state["reap"].update(
        reap.to("cpu"), metric_frequency.to("cpu")
    )
    layer_state["weighted_expert_frequency_sum"] += weighted_expert_frequency_sum.to(
        device="cpu"
    )

    return pruning_batch


def update_pruning_state_from_routed_outputs(
    layer_state: dict[str, Any],
    *,
    expert_ids: torch.Tensor,
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
    selected_experts: torch.Tensor,
    router_logits: torch.Tensor,
    num_experts: int,
    renormalize_router_weights: bool = False,
    accumulate_routing_stats: bool = True,
) -> None:
    """Accumulate metrics from the model's already-computed routed outputs.

    ``expert_outputs`` contains one row per selected token/expert pair, in the
    same order as ``expert_ids``.  This avoids recomputing expert projections in
    the observer while retaining the statistics produced by ``update_pruning_state``.
    """
    device = expert_outputs.device
    selected_experts = selected_experts.reshape(-1, selected_experts.shape[-1]).to(device)
    router_logits = router_logits.to(device)
    expert_ids = expert_ids.reshape(-1).to(device=device, dtype=torch.long)
    token_indices = token_indices.reshape(-1).to(device=device, dtype=torch.long)
    if expert_outputs.shape[0] != expert_ids.numel() or token_indices.numel() != expert_ids.numel():
        raise ValueError("Routed expert outputs and routing indices must have matching rows")
    if router_logits.shape[0] != selected_experts.shape[0]:
        raise ValueError("Router logits and selected experts must have matching tokens")

    frequencies = torch.bincount(selected_experts.reshape(-1), minlength=num_experts)
    if accumulate_routing_stats:
        layer_state["total_tokens"] += torch.tensor(
            selected_experts.shape[0], device="cpu", dtype=torch.long
        )
        layer_state["expert_frequency"] += frequencies.cpu()
        layer_state["pairwise_expert_frequency"] += (
            frequencies.unsqueeze(0) + frequencies.unsqueeze(1)
        ).cpu()

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float).to(device)
    if renormalize_router_weights and selected_experts.numel() > 0:
        topk_weights = torch.gather(routing_weights, 1, selected_experts)
        routing_weights = routing_weights / topk_weights.sum(dim=-1, keepdim=True)
        routing_weights = torch.clamp(
            routing_weights, min=torch.finfo(routing_weights.dtype).eps
        )

    # The routed output is already flattened to one row per token/expert pair.
    # Aggregate all expert metrics in one vectorized pass.  The previous
    # implementation entered Python once per active expert, repeatedly masked
    # GPU tensors, and copied one maximum to CPU per expert.  With 512 experts
    # this made the callback a significant part of calibration time.
    metric_frequency = torch.bincount(expert_ids, minlength=num_experts)
    norms = torch.linalg.vector_norm(expert_outputs, dim=-1)
    weights = routing_weights[token_indices, expert_ids]
    weighted_norms = norms * weights

    ean_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
    ean_sum.scatter_add_(0, expert_ids, norms.to(torch.float64))
    weighted_ean_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
    weighted_ean_sum.scatter_add_(0, expert_ids, weighted_norms.to(torch.float64))
    weighted_frequency = torch.zeros(num_experts, device=device, dtype=torch.float64)
    weighted_frequency.scatter_add_(0, expert_ids, weights.to(torch.float64))

    counts = metric_frequency.to(torch.float32)
    ean_mean = torch.zeros(num_experts, device=device, dtype=torch.float32)
    reap = torch.zeros(num_experts, device=device, dtype=torch.float32)
    nonzero = counts > 0
    ean_mean[nonzero] = (ean_sum.float()[nonzero] / counts[nonzero])
    reap[nonzero] = (weighted_ean_sum.float()[nonzero] / counts[nonzero])

    row_max = expert_outputs.max(dim=-1).values
    max_activation = torch.full(
        (num_experts,), -torch.inf, device=device, dtype=row_max.dtype
    )
    max_activation.scatter_reduce_(0, expert_ids, row_max, reduce="amax", include_self=True)
    max_activation = max_activation.cpu()
    previous_max = layer_state["max_activations"]
    finite = torch.isfinite(max_activation)
    previous_max[finite] = torch.maximum(previous_max[finite], max_activation[finite])

    layer_state["ean_sum"] += ean_sum.cpu()
    layer_state["ean_mean"].update(ean_mean.cpu(), metric_frequency.cpu())
    layer_state["weighted_ean_sum"] += weighted_ean_sum.cpu()
    layer_state["reap"].update(reap.cpu(), metric_frequency.cpu())
    layer_state["weighted_expert_frequency_sum"] += weighted_frequency.cpu()
