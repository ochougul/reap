"""Single forward+backward calibration collecting REAP, stability and Fisher scores.

Per MoE layer, on the same batches and the same model forward:

* REAP        - mean over routed tokens of ||expert_out|| * routing_weight, taken from
                forward hooks on the experts the stock MoE already runs (no dense
                all-experts-on-all-tokens recompute).
* stability   - ``collect_seap_observation`` on the gate input (perturbed-router ranks).
* Fisher      - ``FisherCollector`` router-gradient trick during backward.
"""

from __future__ import annotations

import re

import torch

from reap.fisher import FisherCollector
from reap.router_stability import _route_once
from reap.seap import SeapCollector, collect_seap_observation

_LAYER_RE = re.compile(r"layers\.(\d+)")


class _DeviceSeapCollector(SeapCollector):
    """SeapCollector whose accumulators stay on the GPU (no per-batch host copies)."""

    def _ensure(self, layer, num_experts, device=None):
        if layer not in self.stability_sum:
            self.stability_sum[layer] = torch.zeros((len(self.deltas), num_experts), dtype=torch.float64, device=device)
            self.original_selected_count[layer] = torch.zeros(num_experts, dtype=torch.long, device=device)

    def observe(self, layer, observation, num_experts):
        selected = observation.original_selected.reshape(-1)
        self._ensure(layer, num_experts, selected.device)
        self.original_selected_count[layer] += torch.bincount(selected, minlength=num_experts)
        displacement = (observation.perturbed_ranks - observation.original_ranks).abs().double()
        for index, delta in enumerate(self.deltas):
            contribution = torch.exp(-torch.clamp(displacement - delta, min=0))
            self.stability_sum[layer][index].index_add_(0, selected, contribution.reshape(-1))

    def add_scores_to_state(self, state, **kwargs):
        """Non-destructive: accumulators stay on the GPU so collection can continue."""
        sums, counts = self.stability_sum, self.original_selected_count
        self.stability_sum = {k: v.cpu() for k, v in sums.items()}
        self.original_selected_count = {k: v.cpu() for k, v in counts.items()}
        try:
            return super().add_scores_to_state(state, **kwargs)
        finally:
            self.stability_sum, self.original_selected_count = sums, counts


class JointCollector:
    def __init__(self, output_dir, *, deltas=(2.0, 4.0), seed=42, variance_multiplier=2.0,
                 fisher_mode="router_grad", stability=True):
        self.fisher = FisherCollector(mode=fisher_mode)
        self.seap = _DeviceSeapCollector(output_dir, deltas=deltas, seed=seed)
        self.variance_multiplier = variance_multiplier
        self.stability = stability
        self.reap_sum: dict[int, torch.Tensor] = {}
        self.ean_sum: dict[int, torch.Tensor] = {}
        self.count: dict[int, torch.Tensor] = {}
        self.token_count: dict[int, int] = {}
        self._mask: torch.Tensor | None = None
        self._ctx: dict[int, dict] = {}
        self._handles: list = []

    def set_token_mask(self, mask) -> None:
        self._mask = None if mask is None else mask.reshape(-1).bool()
        self.fisher.set_token_mask(mask)

    def attach(self, model) -> None:
        self.fisher.attach(model)
        for name, module in model.named_modules():
            if hasattr(module, "experts") and hasattr(module, "gate"):
                match = _LAYER_RE.search(name)
                if not match:
                    continue
                layer = int(match.group(1))
                self._handles.append(module.gate.register_forward_hook(self._gate_hook(layer, module)))
                self._handles.append(module.register_forward_hook(self._moe_hook(layer)))
                for expert_id, expert in enumerate(module.experts):
                    self._handles.append(expert.register_forward_hook(self._expert_hook(layer, expert_id)))

    def detach(self) -> None:
        self.fisher.detach()
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _ensure(self, layer, num_experts, device):
        if layer not in self.reap_sum:
            self.reap_sum[layer] = torch.zeros(num_experts, dtype=torch.float64, device=device)
            self.ean_sum[layer] = torch.zeros(num_experts, dtype=torch.float64, device=device)
            self.count[layer] = torch.zeros(num_experts, dtype=torch.long, device=device)
            self.token_count[layer] = 0

    def _gate_hook(self, layer, moe):
        def hook(gate, args, output):
            with torch.no_grad():
                hidden = args[0].reshape(-1, args[0].shape[-1])
                mask = (torch.ones(hidden.shape[0], dtype=torch.bool, device=hidden.device)
                        if self._mask is None else self._mask.to(hidden.device))
                num_experts = int(gate.n_routed_experts)
                top_k = gate.top_k
                self._ensure(layer, num_experts, hidden.device)
                logits, _, _, _, selected, _ = _route_once(gate, hidden, top_k, boundary=0)
                # REAP weights: softmax over logits renormalised over the selected top-k
                # (identical to update_pruning_state(renormalize_router_weights=True)).
                softmax = torch.softmax(logits, dim=1, dtype=torch.float)
                denom = softmax.gather(1, selected).sum(-1, keepdim=True)
                weights = torch.clamp(softmax / denom, min=torch.finfo(torch.float).eps)
                flat = selected.reshape(-1)
                order = flat.argsort(stable=True)  # expert-major, token-ascending
                counts = torch.bincount(flat, minlength=num_experts).tolist()  # one sync/layer
                self._ctx[layer] = {
                    # (token, slot) flat positions of the rows each expert will output
                    "positions": order.split(counts),
                    "norms": torch.zeros(flat.numel(), dtype=torch.float32, device=hidden.device),
                    "selected": selected,
                    "weights": weights.gather(1, selected),
                    "mask": mask,
                }
                self.token_count[layer] += int(mask.sum())
                if self.stability:
                    valid = hidden[mask]
                    if valid.numel():
                        obs = collect_seap_observation(
                            gate, valid, self.seap.generator(hidden.device, layer),
                            variance_multiplier=self.variance_multiplier, top_k=top_k,
                        )
                        self.seap.observe(layer, obs, num_experts)
        return hook

    def _expert_hook(self, layer, expert_id):
        def hook(module, args, output):
            ctx = self._ctx.get(layer)
            if ctx is None:
                return
            with torch.no_grad():  # no host sync: scatter row norms into the slot buffer
                ctx["norms"][ctx["positions"][expert_id]] = output.detach().float().norm(dim=-1)
        return hook

    def _moe_hook(self, layer):
        def hook(module, args, output):
            ctx = self._ctx.pop(layer, None)
            if ctx is None:
                return
            with torch.no_grad():
                selected = ctx["selected"]
                norms = ctx["norms"].view_as(selected)
                valid = ctx["mask"].unsqueeze(-1).expand_as(selected).reshape(-1)
                flat = selected.reshape(-1)
                n = norms.reshape(-1)
                w = ctx["weights"].reshape(-1)
                self.count[layer].index_add_(0, flat, valid.long())
                self.ean_sum[layer].index_add_(0, flat, (n * valid).double())
                self.reap_sum[layer].index_add_(0, flat, (n * w * valid).double())
        return hook

    def state(self, *, delta=2.0, lambda_=0.5) -> dict[int, dict]:
        fisher = self.fisher.state()
        state = {}
        for layer in sorted(self.reap_sum):
            count = self.count[layer].cpu()
            reap = (self.reap_sum[layer].cpu() / count.clamp_min(1)).float()
            state[layer] = {
                "reap": reap,
                "ean_sum": self.ean_sum[layer].cpu(),
                "expert_frequency": count,
                "total_tokens": self.token_count[layer],
                **{k: v for k, v in fisher.get(layer, {}).items()
                   if k not in {"expert_frequency", "total_tokens"}},
            }
        if self.stability:
            state = self.seap.add_scores_to_state(state, delta=delta, lambda_=lambda_)
        return state
