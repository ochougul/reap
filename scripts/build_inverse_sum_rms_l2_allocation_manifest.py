#!/usr/bin/env python3
"""Build a LiveCodeBench manifest for inverse-layer-sum pruning counts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


ARTIFACT_ROOT = Path(
    "artifacts/Qwen3.8-Flash-Next-FP8/"
    "local-output-sensitivity-evol-codealpaca-12228-16384-unpruned"
)
DEFAULT_CALIBRATION_STATE = ARTIFACT_ROOT / "calibration_state.pt"
DEFAULT_METADATA = ARTIFACT_ROOT / "metadata.json"
DEFAULT_ALLOCATION_CSV = ARTIFACT_ROOT / "inverse_sum_rms_l2_pruning_allocation.csv"
DEFAULT_OUTPUT = (
    ARTIFACT_ROOT
    / "evaluations/lcb-p25-rms-l2-inverse-sum/allocation_manifest.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_CALIBRATION_STATE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--allocation-csv", type=Path, default=DEFAULT_ALLOCATION_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pruning-fraction", type=float, default=0.25)
    parser.add_argument("--max-pruned-per-layer", type=int, default=502)
    args = parser.parse_args()

    if not 0.0 < args.pruning_fraction < 1.0:
        raise ValueError("Pruning fraction must be between 0 and 1")
    if not 0 < args.max_pruned_per_layer <= 502:
        raise ValueError("Maximum removals per layer must be in 1..502")
    total_experts = 48 * 512
    target_pruned = round(total_experts * args.pruning_fraction)
    if target_pruned != total_experts * args.pruning_fraction:
        raise ValueError("Pruning fraction must produce an integer expert count")
    if target_pruned > 48 * args.max_pruned_per_layer:
        raise ValueError("Per-layer cap is too small for the requested total pruning")

    with args.allocation_csv.open(newline="") as f:
        allocation_rows = list(csv.DictReader(f))
    layers = [int(row["layer_idx"]) for row in allocation_rows]
    counts = {int(row["layer_idx"]): int(row["experts_pruned"]) for row in allocation_rows}
    if layers != list(range(48)):
        raise ValueError("Expected one allocation row for each layer 0..47")

    state = torch.load(args.state, map_location="cpu", weights_only=False)
    if sorted(int(layer) for layer in state) != list(range(48)):
        raise ValueError("Calibration state must contain layers 0..47")
    metadata: dict[str, Any] = json.loads(args.metadata.read_text())
    expert_count = 512
    if any(count < 0 or count > args.max_pruned_per_layer for count in counts.values()):
        raise ValueError("Removal counts violate the 10-expert minimum retention")
    if sum(counts.values()) != target_pruned:
        raise ValueError(
            f"Expected exactly {target_pruned:,} removals, got {sum(counts.values()):,}"
        )

    removed: dict[str, list[int]] = {}
    retained: dict[str, list[int]] = {}
    proxy_cost = 0.0
    uniform_proxy_cost = 0.0
    uniform_base, uniform_remainder = divmod(target_pruned, 48)
    for layer in range(48):
        layer_state = state[layer] if layer in state else state[str(layer)]
        order = torch.as_tensor(
            layer_state["local_output_sensitivity_removal_order_ascending_reap"]
        ).long().tolist()
        scores = torch.as_tensor(
            layer_state["local_output_sensitivity_rms_l2"]
        ).double().tolist()
        if len(order) != expert_count or len(set(order)) != expert_count:
            raise ValueError(f"Layer {layer} has an invalid REAP removal order")
        if len(scores) != expert_count or not np.isfinite(scores).all():
            raise ValueError(f"Layer {layer} has invalid per-expert sensitivity scores")

        removed_ids = [int(expert_id) for expert_id in order[: counts[layer]]]
        removed[str(layer)] = removed_ids
        removed_set = set(removed_ids)
        retained[str(layer)] = [
            expert_id for expert_id in range(expert_count) if expert_id not in removed_set
        ]
        proxy_cost += sum(scores[expert_id] for expert_id in removed_ids)
        uniform_count = uniform_base + int(layer < uniform_remainder)
        uniform_proxy_cost += sum(scores[expert_id] for expert_id in order[:uniform_count])

    total_removed = sum(counts.values())
    manifest = {
        "schema_version": 1,
        "variant": (
            f"nonuniform-rmsl2-inverse-layer-sum-p{round(args.pruning_fraction * 100)}"
            f"-r{target_pruned}-cap{args.max_pruned_per_layer}"
        ),
        "allocation": "prune counts inversely proportional to summed layer RMS-L2 score; ascending-REAP prefix within each layer",
        "objective_name": "inverse summed per-layer local-output RMS-L2 scores",
        "ranking": "ascending REAP within each layer; ties stable by original expert ID",
        "calibration_state": str(args.state.resolve()),
        "calibration_metadata": str(args.metadata.resolve()),
        "allocation_csv": str(args.allocation_csv.resolve()),
        "sample_set_sha256": metadata["sample_set_sha256"],
        "calibration_examples": int(metadata["retained_examples"]),
        "num_layers": 48,
        "num_experts": expert_count,
        "original_experts": expert_count,
        "top_k": 10,
        "total_removed": total_removed,
        "total_retained": 48 * expert_count - total_removed,
        "actual_pruning_fraction": total_removed / (48 * expert_count),
        "minimum_retained_experts": 10,
        "maximum_removals_per_layer": args.max_pruned_per_layer,
        "per_layer_removal_counts": {str(layer): counts[layer] for layer in range(48)},
        "retained_counts": {
            str(layer): expert_count - counts[layer] for layer in range(48)
        },
        "removed_expert_ids_original": removed,
        "retained_expert_ids_original": retained,
        "proxy_objective": proxy_cost,
        "uniform_proxy_objective": uniform_proxy_cost,
        "interpretation": (
            f"The layer-level counts use inverse proportional allocation with a {args.max_pruned_per_layer}-removal cap. "
            "Expert identities use the ascending-REAP prefix within each layer. "
            "The summed RMS-L2 scores are calibration proxies, not exact multi-expert damage."
        ),
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Saved {args.output}")
    print(f"Total removed: {total_removed}")
    print(f"Ascending-REAP-prefix proxy objective: {proxy_cost:.6f}")
    print(
        f"Uniform reference ({uniform_base} per layer plus remainder) proxy objective: "
        f"{uniform_proxy_cost:.6f}"
    )


if __name__ == "__main__":
    main()
