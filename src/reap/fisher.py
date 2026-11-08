"""Gradient-aware (Fisher-style) expert importance scores.

For every MoE layer ``l`` and routed expert ``e`` this accumulates

    score(l, e) = E_tokens[ (G_l . y_e)^2 ],   y_e = g_e * f_e(x)

where ``G_l = dL/d(MoE layer output)`` is obtained from a single backward pass
and ``y_e`` is the router-weighted output of expert ``e`` on the tokens that
selected it.  ``G_l . y_e`` is the first-order estimate of the loss change when
expert ``e``'s contribution is removed (without re-routing); squaring avoids
sign cancellation across tokens.  Scores are in loss units, so they are
comparable across layers, unlike raw REAP scores.

Usage::

    collector = FisherCollector()
    collector.attach(model)            # hooks every module with .experts/.gate
    collector.set_token_mask(attention_mask)
    loss = model(**batch, labels=labels).loss
    loss.backward()                    # parameters may stay frozen
    state = collector.state()
"""

from __future__ import annotations

import re
from typing import Callable

import torch

_LAYER_RE = re.compile(r"layers\.(\d+)")


class FisherCollector:
    """Accumulate Fisher-style expert scores.

    ``mode="router_grad"`` (default, fast): since ``y_e = w * f_e(x)``,
    ``G . y_e == w * dL/dw`` where ``w`` is the token's routing weight for expert
    ``e``.  Autograd already computes ``dL/dw`` when backpropagating into the
    router, so a tensor hook on the gate's ``topk_weights`` gives the score with
    no extra expert computation.  ``mode="recompute"`` recomputes ``y_e`` in the
    backward hook (slower; kept as an independent reference).
    """

    def __init__(self, mode: str = "router_grad") -> None:
        if mode not in {"router_grad", "recompute"}:
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.fisher_sum: dict[int, torch.Tensor] = {}
        self.signed_sum: dict[int, torch.Tensor] = {}
        self.frequency: dict[int, torch.Tensor] = {}
        self.token_count: dict[int, int] = {}
        self._mask: torch.Tensor | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    # ------------------------------------------------------------------ setup
    def set_token_mask(self, mask: torch.Tensor | None) -> None:
        """Set the ``(batch, seq)`` mask of tokens that count toward the average."""
        self._mask = None if mask is None else mask.reshape(-1).bool()

    def attach(self, model: torch.nn.Module) -> list:
        for name, module in model.named_modules():
            if hasattr(module, "experts") and hasattr(module, "gate"):
                match = _LAYER_RE.search(name)
                if match:
                    layer = int(match.group(1))
                    if self.mode == "recompute":
                        self._handles.append(module.register_forward_hook(self._make_forward_hook(layer)))
                    else:
                        self._handles.append(
                            module.gate.register_forward_hook(self._make_gate_hook(layer, module))
                        )
        return self._handles

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    # ------------------------------------------------------------------ hooks
    def _make_forward_hook(self, layer: int) -> Callable:
        def hook(module, args, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            if not (torch.is_grad_enabled() and out.requires_grad):
                return
            hidden = args[0]
            flat = hidden.detach().reshape(-1, hidden.shape[-1])
            # Use the router's own forward so selection, bias, normalization and
            # scaling match the real forward pass exactly.
            with torch.no_grad():
                topk_indices, topk_weights = module.gate(flat)[:2]
            mask = self._mask
            if mask is None:
                mask = torch.ones(flat.shape[0], dtype=torch.bool, device=flat.device)
            else:
                mask = mask.to(flat.device)
            if mask.shape[0] != flat.shape[0]:
                raise ValueError(
                    f"token mask has {mask.shape[0]} entries but layer {layer} saw {flat.shape[0]} tokens"
                )
            out.register_hook(
                lambda grad: self._accumulate(layer, module, flat, topk_indices, topk_weights, mask, grad)
            )

        return hook

    def _ensure(self, layer: int, num_experts: int, device) -> None:
        if layer not in self.fisher_sum:
            self.fisher_sum[layer] = torch.zeros(num_experts, dtype=torch.float64, device=device)
            self.signed_sum[layer] = torch.zeros(num_experts, dtype=torch.float64, device=device)
            self.frequency[layer] = torch.zeros(num_experts, dtype=torch.long, device=device)
            self.token_count[layer] = 0

    def _make_gate_hook(self, layer: int, moe: torch.nn.Module) -> Callable:
        def hook(gate, args, output):
            topk_indices, topk_weights = output[0], output[1]
            if not (torch.is_grad_enabled() and topk_weights.requires_grad):
                return
            mask = self._mask
            if mask is None:
                mask = torch.ones(topk_weights.shape[0], dtype=torch.bool, device=topk_weights.device)
            else:
                mask = mask.to(topk_weights.device)
            if mask.shape[0] != topk_weights.shape[0]:
                raise ValueError(
                    f"token mask has {mask.shape[0]} entries but layer {layer} saw {topk_weights.shape[0]} tokens"
                )
            weights = topk_weights.detach()

            def accumulate(grad):
                with torch.no_grad():
                    num_experts = len(moe.experts)
                    self._ensure(layer, num_experts, weights.device)
                    dot = (weights.float() * grad.float()).double() * mask.unsqueeze(-1)
                    flat_idx = topk_indices.reshape(-1)
                    flat_dot = dot.reshape(-1)
                    self.fisher_sum[layer].index_add_(0, flat_idx, flat_dot * flat_dot)
                    self.signed_sum[layer].index_add_(0, flat_idx, flat_dot)
                    selected = mask.unsqueeze(-1).expand_as(topk_indices).reshape(-1)
                    self.frequency[layer].index_add_(0, flat_idx, selected.long())
                    self.token_count[layer] += int(mask.sum())

            topk_weights.register_hook(accumulate)

        return hook

    @torch.no_grad()
    def _accumulate(self, layer, module, flat, topk_indices, topk_weights, mask, grad) -> None:
        num_experts = len(module.experts)
        device = flat.device
        self._ensure(layer, num_experts, device)
        G = grad.reshape(-1, grad.shape[-1]).float()
        for expert_id in range(num_experts):
            tokens, slots = torch.where((topk_indices == expert_id) & mask.unsqueeze(-1))
            if tokens.numel() == 0:
                continue
            y = module.experts[expert_id](flat[tokens]).float() * topk_weights[tokens, slots].float().unsqueeze(-1)
            dot = (y * G[tokens]).sum(-1).double()
            self.fisher_sum[layer][expert_id] += (dot * dot).sum()
            self.signed_sum[layer][expert_id] += dot.sum()
            self.frequency[layer][expert_id] += tokens.numel()
        self.token_count[layer] += int(mask.sum())

    # ----------------------------------------------------------------- output
    def state(self) -> dict[int, dict]:
        """Return ``layer -> {...}`` compatible with ``reap.prune.prune``."""
        result = {}
        for layer in sorted(self.fisher_sum):
            total = max(self.token_count[layer], 1)
            result[layer] = {
                "fisher_score": (self.fisher_sum[layer] / total).float().cpu(),
                "fisher_signed": (self.signed_sum[layer] / total).float().cpu(),
                "expert_frequency": self.frequency[layer].cpu(),
                "total_tokens": self.token_count[layer],
            }
        return result


def assistant_label_mask(
    input_ids: torch.Tensor, attention_mask: torch.Tensor | None, assistant_token_id: int | None
) -> torch.Tensor:
    """Boolean ``(batch, seq)`` mask of tokens after the last assistant marker.

    Falls back to all non-pad tokens when the marker is unavailable or absent.
    """
    valid = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
    if assistant_token_id is None:
        return valid
    mask = valid.clone()
    for row in range(input_ids.shape[0]):
        positions = (input_ids[row] == assistant_token_id).nonzero()
        if positions.numel() == 0:
            continue
        mask[row, : int(positions[-1]) + 1] = False
    return mask
