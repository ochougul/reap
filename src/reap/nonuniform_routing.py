"""Routing primitives shared by non-uniform MoE checkpoint tools.

The mask is applied to logits before both softmax and top-k.  Applying it at
either later stage is incorrect: padded experts could otherwise displace a
valid expert or receive normalization mass.
"""

from __future__ import annotations

import torch


def masked_topk_routing(
    router_logits: torch.Tensor,
    valid_experts: torch.Tensor,
    top_k: int,
    *,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return probabilities, top-k probabilities, and indices for valid experts.

    ``valid_experts`` is broadcast over all leading dimensions of logits and
    its final dimension must equal the logits' expert dimension.
    """
    if router_logits.ndim < 2:
        raise ValueError("router_logits must have an expert dimension")
    if valid_experts.shape[-1] != router_logits.shape[-1]:
        raise ValueError("valid_experts and router_logits have different sizes")
    valid = valid_experts.to(device=router_logits.device, dtype=torch.bool)
    while valid.ndim < router_logits.ndim:
        valid = valid.unsqueeze(0)
    if (valid.sum(dim=-1) < top_k).any():
        raise ValueError("every layer must retain at least top_k experts")

    masked_logits = router_logits.masked_fill(~valid, torch.finfo(torch.float32).min)
    probs = torch.softmax(masked_logits.float(), dim=-1)
    probs = probs.masked_fill(~valid, 0.0)
    top_values, top_indices = torch.topk(probs, top_k, dim=-1)
    if normalize:
        top_values = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return probs.to(router_logits.dtype), top_values.to(router_logits.dtype), top_indices


def assert_routing_invariants(
    router_logits: torch.Tensor, valid_experts: torch.Tensor, top_k: int
) -> None:
    probs, top_values, indices = masked_topk_routing(
        router_logits, valid_experts, top_k
    )
    valid = valid_experts.to(indices.device).bool()
    while valid.ndim < router_logits.ndim:
        valid = valid.unsqueeze(0)
    valid = valid.expand_as(router_logits)
    selected_valid = valid.gather(-1, indices)
    if not bool(selected_valid.all()):
        raise AssertionError("padded expert selected by top-k routing")
    if not torch.allclose(top_values.float().sum(-1), torch.ones_like(top_values[..., 0]).float(), atol=1e-5):
        raise AssertionError("redistributed routing probabilities do not sum to one")
    if not torch.all(probs.masked_select(~valid) == 0):
        raise AssertionError("padded expert received routing probability")
