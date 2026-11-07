"""Load-time compacting of uniformly pruned Qwen4-Exp experts for vLLM.

This maps the original checkpoint's expert IDs onto a dense retained-ID range
and slices each router matrix to the same retained experts. vLLM then builds
ordinary expert-parallel layers using the reduced expert count; inference does
not need a router mask or a second materialized copy of the checkpoint.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch


_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.*\.layers\.(?P<layer>\d+)\.mlp\.experts\.)"
    r"(?P<expert>\d+)(?P<suffix>\..+)$"
)
_ROUTER_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.*\.layers\.(?P<layer>\d+)\.mlp\.gate\.weight)$"
)
_ORIGINAL_LOAD_WEIGHTS = None


def _read_compaction_plan(path: str | os.PathLike[str]) -> tuple[int, dict[int, list[int]]]:
    data = json.loads(Path(path).read_text())
    if data.get("routing") != "compact router and experts; no runtime routing mask":
        raise ValueError(f"Not a compact-expert metadata file: {path}")
    original_experts = int(data["original_experts"])
    raw = data.get("retained_original_ids_by_layer")
    if original_experts < 1 or not isinstance(raw, dict) or not raw:
        raise ValueError(f"Invalid expert compaction plan: {path}")

    retained: dict[int, list[int]] = {}
    for raw_layer, raw_ids in raw.items():
        layer = int(raw_layer)
        ids = [int(expert_id) for expert_id in raw_ids]
        if len(ids) != len(set(ids)) or any(
            expert_id < 0 or expert_id >= original_experts for expert_id in ids
        ):
            raise ValueError(f"Invalid retained expert IDs for layer {layer}")
        if ids != sorted(ids):
            raise ValueError(f"Retained IDs for layer {layer} must be sorted")
        retained[layer] = ids
    counts = {len(ids) for ids in retained.values()}
    if len(counts) != 1 or not counts or min(counts) < 1:
        raise ValueError("Native expert-parallel compaction requires equal nonzero layer counts")
    if int(data.get("retained_experts_per_layer", -1)) != next(iter(counts)):
        raise ValueError("Compaction metadata retained count is inconsistent")
    return original_experts, retained


def compact_checkpoint_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    original_experts: int,
    retained_ids_by_layer: dict[int, list[int]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Filter/remap expert tensors and select corresponding router rows."""
    id_maps = {
        layer: {original_id: compact_id for compact_id, original_id in enumerate(ids)}
        for layer, ids in retained_ids_by_layer.items()
    }
    seen_experts: dict[int, set[int]] = {layer: set() for layer in id_maps}
    seen_routers: set[int] = set()

    for name, weight in weights:
        expert_match = _EXPERT_WEIGHT_RE.match(name)
        if expert_match:
            layer = int(expert_match.group("layer"))
            original_id = int(expert_match.group("expert"))
            if layer not in id_maps:
                raise ValueError(f"Unexpected MoE layer {layer} in checkpoint tensor {name}")
            compact_id = id_maps[layer].get(original_id)
            if compact_id is None:
                continue
            seen_experts[layer].add(original_id)
            yield (
                f"{expert_match.group('prefix')}{compact_id}{expert_match.group('suffix')}",
                weight,
            )
            continue

        router_match = _ROUTER_WEIGHT_RE.match(name)
        if router_match:
            layer = int(router_match.group("layer"))
            if layer not in retained_ids_by_layer:
                raise ValueError(f"Unexpected router layer {layer} in checkpoint tensor {name}")
            retained = retained_ids_by_layer[layer]
            if weight.ndim != 2 or weight.shape[0] != original_experts:
                raise ValueError(
                    f"Expected {original_experts} router rows in {name}, got {tuple(weight.shape)}"
                )
            indices = torch.tensor(retained, dtype=torch.long, device=weight.device)
            seen_routers.add(layer)
            yield name, weight.index_select(0, indices)
            continue

        yield name, weight

    expected_layers = set(retained_ids_by_layer)
    missing_routers = expected_layers - seen_routers
    if missing_routers:
        raise RuntimeError(f"Checkpoint did not provide router weights for layers {sorted(missing_routers)}")
    missing_experts = {
        layer: sorted(set(ids) - seen_experts[layer])
        for layer, ids in retained_ids_by_layer.items()
        if set(ids) - seen_experts[layer]
    }
    if missing_experts:
        raise RuntimeError(f"Checkpoint did not provide retained expert tensors: {missing_experts}")


def install_vllm_qwen38_expert_compaction() -> None:
    """Patch the Qwen4-Exp vLLM loader for one explicitly selected run."""
    global _ORIGINAL_LOAD_WEIGHTS
    metadata_path = os.environ.get("REAP_QWEN38_COMPACTION_METADATA")
    if not metadata_path or _ORIGINAL_LOAD_WEIGHTS is not None:
        return

    original_experts, retained_ids = _read_compaction_plan(metadata_path)
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForConditionalGeneration

    _ORIGINAL_LOAD_WEIGHTS = Qwen4ExpForConditionalGeneration.load_weights

    def compacted_load_weights(self, weights):
        return _ORIGINAL_LOAD_WEIGHTS(
            self,
            compact_checkpoint_weights(weights, original_experts, retained_ids),
        )

    Qwen4ExpForConditionalGeneration.load_weights = compacted_load_weights
