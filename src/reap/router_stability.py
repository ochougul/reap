"""Streaming router-stability collection for MoE calibration."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class RouterObservation:
    top12_ids: torch.Tensor
    top12_ranking_values: torch.Tensor
    top8_applied_weights: torch.Tensor
    original_scores: torch.Tensor
    perturbed_scores: torch.Tensor
    original_selected: torch.Tensor
    perturbed_selected: torch.Tensor


def _glm_route(router: torch.nn.Module, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return GLM scores and the exact ranking mask used by its router."""
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = F.linear(
        hidden_states.float(), router.weight.float(),
    )
    scores = logits.sigmoid()
    correction = router.e_score_correction_bias.float().unsqueeze(0)
    corrected = scores + correction

    experts_per_group = router.n_routed_experts // router.n_group
    grouped = corrected.view(-1, router.n_group, experts_per_group)
    group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
    group_idx = group_scores.topk(router.topk_group, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    eligible = (
        group_mask.unsqueeze(-1)
        .expand(-1, router.n_group, experts_per_group)
        .reshape(-1, router.n_routed_experts)
        .bool()
    )
    ranking_values = corrected.masked_fill(~eligible, -torch.inf)
    return logits, scores, ranking_values


def _route_once(router: torch.nn.Module, hidden_states: torch.Tensor, top_k: int, boundary: int):
    if hasattr(router, "e_score_correction_bias") and hasattr(router, "n_group"):
        logits, scores, ranking_values = _glm_route(router, hidden_states)
    else:
        result = router(hidden_states)
        logits = result[-1] if isinstance(result, tuple) else result
        scores = logits.float()
        ranking_values = scores

    top12 = min(top_k + boundary, ranking_values.shape[-1])
    top12_values, top12_ids = ranking_values.topk(top12, dim=-1, sorted=True)
    selected_ids = top12_ids[:, :top_k]
    selected_scores = scores.gather(1, selected_ids)
    if getattr(router, "norm_topk_prob", False):
        selected_weights = selected_scores / (selected_scores.sum(-1, keepdim=True) + 1e-20)
    else:
        selected_weights = selected_scores
    selected_weights = selected_weights * getattr(router, "routed_scaling_factor", 1.0)
    return logits, scores, top12_ids, top12_values, selected_ids, selected_weights


def collect_router_observation(
    router: torch.nn.Module,
    hidden_states: torch.Tensor,
    variance_multipliers: Iterable[float],
    generator: torch.Generator,
    top_k: int,
    boundary: int,
) -> RouterObservation:
    """Collect one original and four perturbed router observations."""
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    original = _route_once(router, hidden_states, top_k, boundary)
    perturbed_outputs = []
    input_var = hidden_states.float().var(dim=-1, unbiased=False, keepdim=True)
    for multiplier in variance_multipliers:
        noise = torch.randn(
            hidden_states.shape,
            device=hidden_states.device,
            dtype=torch.float32,
            generator=generator,
        ) * torch.sqrt(input_var.clamp_min(torch.finfo(torch.float32).eps) * multiplier)
        perturbed_outputs.append(
            _route_once(router, hidden_states + noise.to(hidden_states.dtype), top_k, boundary)
        )

    return RouterObservation(
        top12_ids=torch.stack([original[2]] + [item[2] for item in perturbed_outputs]),
        top12_ranking_values=torch.stack(
            [original[3]] + [item[3] for item in perturbed_outputs]
        ),
        top8_applied_weights=torch.stack(
            [original[5]] + [item[5] for item in perturbed_outputs]
        ),
        original_scores=original[1],
        perturbed_scores=torch.stack([item[1] for item in perturbed_outputs]),
        original_selected=original[4],
        perturbed_selected=torch.stack([item[4] for item in perturbed_outputs]),
    )


class RouterStabilityCollector:
    """Streaming accumulators plus an incremental Parquet event writer."""

    def __init__(
        self,
        output_dir: str | pathlib.Path,
        variance_multipliers: Iterable[float],
        seed: int,
        boundary: int = 4,
        max_tokens: int | None = None,
        flush_rows: int = 1024,
    ) -> None:
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.variance_multipliers = tuple(float(x) for x in variance_multipliers)
        if len(self.variance_multipliers) != 4:
            raise ValueError("router stability requires exactly four variance multipliers")
        self.boundary = boundary
        self.max_tokens = max_tokens
        self.flush_rows = flush_rows
        self.generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        self.generator.manual_seed(seed)
        self.rows: list[dict[str, Any]] = []
        self.writer = None
        self.total_tokens = 0
        self._batch_token_counts: dict[int, int] = {}
        self.num_layers = 0
        self.num_experts = 0
        self.accumulators: dict[str, torch.Tensor] | None = None

    def _ensure_accumulators(self, num_layers: int, num_experts: int) -> None:
        if self.accumulators is not None:
            return
        self.num_layers = num_layers
        self.num_experts = num_experts
        # GLM has 46 transformer blocks; reserve a small amount for model variants.
        allocated_layers = max(num_layers, 64)
        shape = (4, allocated_layers, num_experts)
        self.accumulators = {
            "abs_change_sum": torch.zeros(shape, dtype=torch.float64),
            "squared_change_sum": torch.zeros(shape, dtype=torch.float64),
            "signed_change_sum": torch.zeros(shape, dtype=torch.float64),
            "original_score_sum": torch.zeros((allocated_layers, num_experts), dtype=torch.float64),
            "perturbed_score_sum": torch.zeros(shape, dtype=torch.float64),
            "original_selected_count": torch.zeros((allocated_layers, num_experts), dtype=torch.long),
            "perturbed_selected_count": torch.zeros(shape, dtype=torch.long),
            "selection_gained_count": torch.zeros(shape, dtype=torch.long),
            "selection_lost_count": torch.zeros(shape, dtype=torch.long),
        }

    def observe(self, layer: int, batch_id: int, observation: RouterObservation) -> None:
        tokens, experts = observation.original_scores.shape
        if self.max_tokens is not None:
            if batch_id not in self._batch_token_counts:
                remaining = self.max_tokens - self.total_tokens
                accepted = max(0, min(tokens, remaining))
                self._batch_token_counts[batch_id] = accepted
                self.total_tokens += accepted
            accepted = self._batch_token_counts[batch_id]
            if accepted <= 0:
                return
            if tokens > accepted:
                observation = RouterObservation(
                    top12_ids=observation.top12_ids[:, :accepted],
                    top12_ranking_values=observation.top12_ranking_values[:, :accepted],
                    top8_applied_weights=observation.top8_applied_weights[:, :accepted],
                    original_scores=observation.original_scores[:accepted],
                    perturbed_scores=observation.perturbed_scores[:, :accepted],
                    original_selected=observation.original_selected[:accepted],
                    perturbed_selected=observation.perturbed_selected[:, :accepted],
                )
                tokens = accepted
        elif batch_id not in self._batch_token_counts:
            self._batch_token_counts[batch_id] = tokens
            self.total_tokens += tokens
        self.num_layers = max(layer + 1, self.num_layers)
        self._ensure_accumulators(self.num_layers, experts)
        assert self.accumulators is not None
        original = observation.original_scores.detach().float().cpu()
        perturbed = observation.perturbed_scores.detach().float().cpu()
        delta = perturbed - original.unsqueeze(0)
        self.accumulators["abs_change_sum"][:, layer] += delta.abs().sum(1).double()
        self.accumulators["squared_change_sum"][:, layer] += delta.square().sum(1).double()
        self.accumulators["signed_change_sum"][:, layer] += delta.sum(1).double()
        self.accumulators["original_score_sum"][layer] += original.sum(0).double()
        self.accumulators["perturbed_score_sum"][:, layer] += perturbed.sum(1).double()

        original_mask = torch.zeros_like(original, dtype=torch.bool)
        original_mask.scatter_(1, observation.original_selected.detach().cpu(), True)
        perturbed_mask = torch.zeros((4, tokens, experts), dtype=torch.bool)
        perturbed_mask.scatter_(2, observation.perturbed_selected.detach().cpu(), True)
        self.accumulators["original_selected_count"][layer] += original_mask.sum(0)
        self.accumulators["perturbed_selected_count"][:, layer] += perturbed_mask.sum(1)
        self.accumulators["selection_gained_count"][:, layer] += (~original_mask.unsqueeze(0) & perturbed_mask).sum(1)
        self.accumulators["selection_lost_count"][:, layer] += (original_mask.unsqueeze(0) & ~perturbed_mask).sum(1)

        ids = observation.top12_ids.detach().cpu().numpy().astype(np.uint8).tolist()
        ranking = observation.top12_ranking_values.detach().cpu().numpy().astype(np.float16).tolist()
        weights = observation.top8_applied_weights.detach().cpu().numpy().astype(np.float16).tolist()
        for token in range(tokens):
            self.rows.append({
                "batch_id": int(batch_id),
                "token_position": token,
                "layer": int(layer),
                "top12_expert_ids": [condition[token] for condition in ids],
                "top12_ranking_values": [condition[token] for condition in ranking],
                "top8_applied_weights": [condition[token] for condition in weights],
            })
        if len(self.rows) >= self.flush_rows:
            self._flush()

    def _flush(self) -> None:
        if not self.rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(self.rows)
        path = self.output_dir / "routing_events.parquet"
        if self.writer is None:
            self.writer = pq.ParquetWriter(path, table.schema, compression="zstd")
        self.writer.write_table(table)
        self.rows.clear()

    def finalize(self) -> None:
        self._flush()
        if self.writer is not None:
            self.writer.close()
        if self.accumulators is not None:
            arrays = {
                key: value[:, : self.num_layers].numpy()
                if value.ndim == 3
                else value[: self.num_layers].numpy()
                for key, value in self.accumulators.items()
            }
            np.savez_compressed(self.output_dir / "router_stability_accumulators.npz", **arrays)
        metadata = {
            "variance_multipliers": list(self.variance_multipliers),
            "boundary_width": self.boundary,
            "top_k": 8,
            "total_tokens": self.total_tokens,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "noise": "zero_mean_gaussian_relative_to_input_variance",
        }
        (self.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
