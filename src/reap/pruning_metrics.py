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

    if activations.shape[0] != num_experts:
        raise ValueError(
            f"Expected activations for {num_experts} experts, got {activations.shape[0]}"
        )
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
    )


def update_pruning_state(
    layer_state: dict[str, Any],
    *,
    activations: torch.Tensor,
    selected_experts: torch.Tensor,
    router_logits: torch.Tensor,
    num_experts: int,
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
        valid_token_mask=valid_token_mask,
    )

    device = pruning_batch.activations.device
    layer_state["total_tokens"] += pruning_batch.num_tokens
    layer_state["expert_frequency"] += pruning_batch.expert_frequency.to("cpu", torch.long)
    layer_state["pairwise_expert_frequency"] += pruning_batch.pairwise_expert_frequency.to(
        "cpu", torch.long
    )

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

    for i in range(num_experts):
        active_mask = (pruning_batch.selected_experts == i).any(dim=-1).to(device)
        if not active_mask.any():
            continue

        selected_activations = pruning_batch.activations[i, active_mask, :]
        active_router_weights = routing_weights[active_mask, i]
        ean_norm = torch.linalg.norm(selected_activations, dim=-1)
        ean_sum[i] = ean_norm.sum().to(device)
        ean_mean[i] = ean_norm.mean().to(device)
        weighted_expert_frequency_sum[i] = active_router_weights.sum().to(device)
        weighted_ean_sum[i] = (ean_norm * active_router_weights).sum().to(device)
        reap[i] = (ean_norm * active_router_weights).mean().to(device)

        selected_activations_max = selected_activations.max().to(device="cpu")
        if selected_activations_max > layer_state["max_activations"][i]:
            layer_state["max_activations"][i] = selected_activations_max

    layer_state["ean_sum"] += ean_sum.to(device="cpu")
    layer_state["ean_mean"].update(
        ean_mean.to("cpu"), pruning_batch.expert_frequency.to("cpu")
    )
    layer_state["weighted_ean_sum"] += weighted_ean_sum.to(device="cpu")
    layer_state["reap"].update(
        reap.to("cpu"), pruning_batch.expert_frequency.to("cpu")
    )
    layer_state["weighted_expert_frequency_sum"] += weighted_expert_frequency_sum.to(
        device="cpu"
    )

    return pruning_batch


def update_pruning_state_from_selected_expert_outputs(
    layer_state: dict[str, Any],
    *,
    expert_outputs: dict[int, torch.Tensor],
    selected_experts: torch.Tensor,
    router_logits: torch.Tensor,
    num_experts: int,
    renormalize_router_weights: bool = False,
) -> None:
    """Accumulate REAP without materializing all expert/token activations.

    Packed-expert architectures such as Qwen3.5 store all expert weights in
    rank-three tensors. Creating the conventional ``[experts, tokens, hidden]``
    activation tensor for them is unnecessarily large: REAP only needs outputs
    for experts selected by a token. ``expert_outputs`` therefore maps each
    selected expert to its unweighted outputs for the tokens it served.
    """
    selected_experts = selected_experts.reshape(-1, selected_experts.shape[-1])
    router_logits = router_logits.reshape(-1, router_logits.shape[-1])
    if router_logits.shape[1] != num_experts:
        raise ValueError(
            f"Expected router logits for {num_experts} experts, got {router_logits.shape[1]}"
        )
    if router_logits.shape[0] != selected_experts.shape[0]:
        raise ValueError("Router logits and selected experts have different token counts")

    device = router_logits.device
    num_tokens = selected_experts.shape[0]
    expert_frequency = torch.bincount(
        selected_experts.reshape(-1), minlength=num_experts
    ).to(device)
    pairwise_expert_frequency = (
        expert_frequency.unsqueeze(0) + expert_frequency.unsqueeze(1)
    )
    layer_state["total_tokens"] += torch.tensor(num_tokens, dtype=torch.long)
    layer_state["expert_frequency"] += expert_frequency.to("cpu", torch.long)
    layer_state["pairwise_expert_frequency"] += pairwise_expert_frequency.to(
        "cpu", torch.long
    )

    routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float)
    if renormalize_router_weights and selected_experts.numel() > 0:
        topk_weights = routing_weights.gather(1, selected_experts)
        routing_weights = routing_weights / topk_weights.sum(dim=-1, keepdim=True)

    ean_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
    ean_mean = torch.zeros(num_experts, device=device, dtype=torch.float32)
    weighted_ean_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
    reap = torch.zeros(num_experts, device=device, dtype=torch.float32)
    weighted_frequency = torch.zeros(num_experts, device=device, dtype=torch.float64)

    for expert_index, output in expert_outputs.items():
        if not 0 <= expert_index < num_experts:
            raise ValueError(f"Invalid expert index {expert_index}")
        active_mask = (selected_experts == expert_index).any(dim=-1)
        if output.shape[0] != int(active_mask.sum()):
            raise ValueError(
                f"Expert {expert_index} has {output.shape[0]} outputs for "
                f"{int(active_mask.sum())} selected tokens"
            )
        norms = torch.linalg.vector_norm(output, dim=-1)
        weights = routing_weights[active_mask, expert_index]
        ean_sum[expert_index] = norms.sum().double()
        ean_mean[expert_index] = norms.mean()
        weighted_frequency[expert_index] = weights.sum().double()
        weighted_ean_sum[expert_index] = (norms * weights).sum().double()
        reap[expert_index] = (norms * weights).mean()
        layer_state["max_activations"][expert_index] = torch.maximum(
            layer_state["max_activations"][expert_index], output.max().detach().cpu()
        )

    layer_state["ean_sum"] += ean_sum.cpu()
    layer_state["ean_mean"].update(ean_mean.cpu(), expert_frequency.cpu())
    layer_state["weighted_ean_sum"] += weighted_ean_sum.cpu()
    layer_state["reap"].update(reap.cpu(), expert_frequency.cpu())
    layer_state["weighted_expert_frequency_sum"] += weighted_frequency.cpu()


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

    ean_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
    ean_mean = torch.zeros(num_experts, device=device, dtype=torch.float32)
    weighted_ean_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
    reap = torch.zeros(num_experts, device=device, dtype=torch.float32)
    weighted_frequency = torch.zeros(num_experts, device=device, dtype=torch.float64)
    metric_frequency = torch.bincount(expert_ids, minlength=num_experts)

    for expert_id in torch.unique(expert_ids).tolist():
        rows = expert_ids == expert_id
        activations = expert_outputs[rows]
        token_ids = token_indices[rows]
        norms = torch.linalg.vector_norm(activations, dim=-1)
        weights = routing_weights[token_ids, expert_id]
        ean_sum[expert_id] = norms.sum()
        ean_mean[expert_id] = norms.mean()
        weighted_frequency[expert_id] = weights.sum()
        weighted_ean_sum[expert_id] = (norms * weights).sum()
        reap[expert_id] = (norms * weights).mean()
        max_activation = activations.max().to(device="cpu")
        if max_activation > layer_state["max_activations"][expert_id]:
            layer_state["max_activations"][expert_id] = max_activation

    layer_state["ean_sum"] += ean_sum.cpu()
    layer_state["ean_mean"].update(ean_mean.cpu(), metric_frequency.cpu())
    layer_state["weighted_ean_sum"] += weighted_ean_sum.cpu()
    layer_state["reap"].update(reap.cpu(), metric_frequency.cpu())
    layer_state["weighted_expert_frequency_sum"] += weighted_frequency.cpu()
