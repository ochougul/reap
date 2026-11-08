"""Faster, mathematically identical routed-expert forward for GLM-style MoE blocks.

The stock ``Glm4MoeMoE.moe`` builds a one-hot mask and calls ``torch.where`` for each
of the (128) experts, which forces a device sync per expert and many tiny kernels.
This version sorts the (token, slot) pairs by expert once, reads the per-expert
counts with a single sync, and runs each expert on a contiguous slice.
"""

from __future__ import annotations

import types

import torch


def sorted_moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
    num_tokens, top_k = topk_indices.shape
    flat_experts = topk_indices.reshape(-1)
    order = torch.argsort(flat_experts, stable=True)
    token_ids = order // top_k
    counts = torch.bincount(flat_experts, minlength=len(self.experts)).tolist()  # single sync
    weights = topk_weights.reshape(-1)[order]
    inputs = hidden_states[token_ids]

    # torch.split (not slicing): its backward is one cat instead of a full-size
    # zero-filled gradient per expert.
    active = [expert_id for expert_id, count in enumerate(counts) if count]
    chunks = inputs.split([counts[expert_id] for expert_id in active])
    outputs = [self.experts[expert_id](chunk) for expert_id, chunk in zip(active, chunks)]
    weighted = torch.cat(outputs, dim=0) * weights.unsqueeze(-1)
    final = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
    final.index_add_(0, token_ids, weighted)
    return final.type(hidden_states.dtype)


def patch_sorted_moe(model: torch.nn.Module) -> int:
    """Bind :func:`sorted_moe` as ``moe`` on every module with ``experts``, ``gate`` and ``moe``."""
    patched = 0
    for module in model.modules():
        if hasattr(module, "experts") and hasattr(module, "gate") and hasattr(module, "moe"):
            module.moe = types.MethodType(sorted_moe, module)
            patched += 1
    return patched
