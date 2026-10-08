"""Streaming SEAP (stability + normalized REAP) scoring."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from reap.router_stability import _glm_route, _route_once


@dataclass
class SeapObservation:
    original_selected: torch.Tensor
    original_ranks: torch.Tensor
    perturbed_ranks: torch.Tensor


def _rank_map(ranking_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one-based ranks and an eligibility mask for every expert."""
    _, order = torch.topk(
        ranking_values, ranking_values.shape[-1], dim=-1, sorted=True
    )
    ranks = torch.empty_like(order, dtype=torch.long)
    rank_values = torch.arange(1, order.shape[1] + 1, device=order.device).view(1, -1)
    ranks.scatter_(1, order, rank_values.expand_as(order))
    eligible = torch.isfinite(ranking_values)
    return ranks, eligible


def collect_seap_observation(
    router: torch.nn.Module,
    hidden_states: torch.Tensor,
    generator: torch.Generator,
    *,
    delta: float | tuple[float, ...] = (2.0, 4.0),
    variance_multiplier: float = 2.0,
    top_k: int,
) -> SeapObservation:
    """Collect exact full-expert ranks for the original and perturbed routes."""
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    original = _route_once(router, hidden_states, top_k, boundary=0)
    if hasattr(router, "e_score_correction_bias") and hasattr(router, "n_group"):
        _, _, original_ranking = _glm_route(router, hidden_states)
    else:
        original_ranking = original[0]
    original_ranks, _ = _rank_map(original_ranking)

    input_var = hidden_states.float().var(dim=-1, unbiased=False, keepdim=True)
    noise = torch.randn(
        hidden_states.shape,
        device=hidden_states.device,
        dtype=torch.float32,
        generator=generator,
    ) * torch.sqrt(input_var.clamp_min(torch.finfo(torch.float32).eps) * variance_multiplier)
    perturbed_hidden = hidden_states + noise.to(hidden_states.dtype)
    perturbed = _route_once(router, perturbed_hidden, top_k, boundary=0)
    if hasattr(router, "e_score_correction_bias") and hasattr(router, "n_group"):
        _, _, perturbed_ranking = _glm_route(router, perturbed_hidden)
    else:
        perturbed_ranking = perturbed[0]
    perturbed_ranks, perturbed_eligible = _rank_map(perturbed_ranking)

    selected = original[4]
    selected_perturbed_ranks = perturbed_ranks.gather(1, selected)
    selected_eligible = perturbed_eligible.gather(1, selected)
    selected_perturbed_ranks = selected_perturbed_ranks.masked_fill(
        ~selected_eligible, ranking_values_num_experts(router)
    )
    return SeapObservation(
        original_selected=selected,
        original_ranks=original_ranks.gather(1, selected),
        perturbed_ranks=selected_perturbed_ranks,
    )


def ranking_values_num_experts(router: torch.nn.Module) -> int:
    return int(getattr(router, "n_routed_experts", router.weight.shape[0])) + 1


class SeapCollector:
    """CPU-resident streaming stability accumulators."""

    def __init__(self, output_dir: str | pathlib.Path, *, deltas=(2.0, 4.0), seed=42):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.deltas = tuple(float(value) for value in deltas)
        self.seed = seed
        self.generator_by_device: dict[str, torch.Generator] = {}
        self.stability_sum: dict[int, torch.Tensor] = {}
        self.original_selected_count: dict[int, torch.Tensor] = {}

    def _ensure(self, layer: int, num_experts: int) -> None:
        if layer not in self.stability_sum:
            self.stability_sum[layer] = torch.zeros(
                (len(self.deltas), num_experts), dtype=torch.float64
            )
            self.original_selected_count[layer] = torch.zeros(
                num_experts, dtype=torch.long
            )

    def generator(self, device: torch.device, layer: int) -> torch.Generator:
        key = str(device)
        if key not in self.generator_by_device:
            self.generator_by_device[key] = torch.Generator(device=device).manual_seed(
                self.seed + layer
            )
        return self.generator_by_device[key]

    def observe(self, layer: int, observation: SeapObservation, num_experts: int) -> None:
        selected = observation.original_selected.detach().cpu()
        original = observation.original_ranks.detach().cpu()
        perturbed = observation.perturbed_ranks.detach().cpu()
        self._ensure(layer, num_experts)
        if selected.numel() and selected.max() >= num_experts:
            raise ValueError("SEAP observation contains an invalid expert id")
        self.original_selected_count[layer] += torch.bincount(
            selected.reshape(-1), minlength=num_experts
        )
        displacement = (perturbed - original).abs().double()
        for index, delta in enumerate(self.deltas):
            contribution = torch.exp(-torch.clamp(displacement - delta, min=0))
            self.stability_sum[layer][index].index_add_(
                0, selected.reshape(-1), contribution.reshape(-1)
            )

    def add_scores_to_state(
        self,
        state: dict[int, dict[str, Any]],
        *,
        delta: float = 2.0,
        lambda_: float = 0.5,
    ) -> dict[int, dict[str, Any]]:
        for layer, sums in self.stability_sum.items():
            counts = self.original_selected_count[layer].double()
            stability = torch.where(
                counts > 0,
                sums / counts.unsqueeze(0),
                torch.zeros_like(sums),
            ).float()
            reap = state[layer]["reap"]
            if not isinstance(reap, torch.Tensor):
                reap = reap.mean
            reap = reap.float().cpu()
            normalized = reap / reap.max() if reap.numel() and reap.max() > 0 else torch.zeros_like(reap)
            delta_index = min(range(len(self.deltas)), key=lambda i: abs(self.deltas[i] - delta))
            state[layer]["stability_score"] = stability[delta_index]
            for index, stored_delta in enumerate(self.deltas):
                state[layer][f"stability_score_delta_{stored_delta:g}"] = stability[index]
            state[layer]["stability_scores"] = stability
            state[layer]["reap_score"] = reap
            state[layer]["normalized_reap"] = normalized
            state[layer]["seap_score"] = (
                lambda_ * normalized + (1.0 - lambda_) * stability[delta_index]
            )
        return state

    def save(
        self,
        state: dict[int, dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        *,
        delta: float = 2.0,
        lambda_: float = 0.5,
    ) -> None:
        state = self.add_scores_to_state(state, delta=delta, lambda_=lambda_)
        torch.save(state, self.output_dir / "seap_state.pt")
        arrays = {
            f"layer_{layer}_stability": values.numpy()
            for layer, values in self.stability_sum.items()
        }
        np.savez_compressed(self.output_dir / "seap_accumulators.npz", **arrays)
        payload = {"deltas": list(self.deltas), "seed": self.seed, **(metadata or {})}
        (self.output_dir / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
