#!/usr/bin/env python3
"""Build a uniform-per-layer REAP pruning manifest from calibration state."""

from __future__ import annotations

import argparse
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
DEFAULT_STATE = ARTIFACT_ROOT / "calibration_state.pt"
DEFAULT_METADATA = ARTIFACT_ROOT / "metadata.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pruning-fraction", type=float, required=True)
    args = parser.parse_args()

    if not 0.0 < args.pruning_fraction < 1.0:
        raise ValueError("Pruning fraction must be between 0 and 1")
    num_layers, num_experts, top_k = 48, 512, 10
    target_pruned = round(num_layers * num_experts * args.pruning_fraction)
    if target_pruned != num_layers * num_experts * args.pruning_fraction:
        raise ValueError("Pruning fraction must produce an integer expert count")
    per_layer, remainder = divmod(target_pruned, num_layers)
    if per_layer + bool(remainder) > num_experts - top_k:
        raise ValueError("Requested pruning would leave fewer than top_k experts")

    state = torch.load(args.state, map_location="cpu", weights_only=False)
    metadata: dict[str, Any] = json.loads(args.metadata.read_text())
    if sorted(int(layer) for layer in state) != list(range(num_layers)):
        raise ValueError("Calibration state must contain layers 0..47")

    counts: dict[str, int] = {}
    removed: dict[str, list[int]] = {}
    retained: dict[str, list[int]] = {}
    proxy_cost = 0.0
    for layer in range(num_layers):
        layer_state = state[layer] if layer in state else state[str(layer)]
        reap = torch.as_tensor(layer_state["reap"]).double().numpy()
        if reap.shape != (num_experts,) or not np.isfinite(reap).all():
            raise ValueError(f"Layer {layer} has invalid REAP scores")
        order = np.lexsort((np.arange(num_experts), reap))
        count = per_layer + int(layer < remainder)
        removed_ids = order[:count].astype(int).tolist()
        counts[str(layer)] = count
        removed[str(layer)] = removed_ids
        removed_set = set(removed_ids)
        retained[str(layer)] = [
            expert for expert in range(num_experts) if expert not in removed_set
        ]
        sensitivity = torch.as_tensor(
            layer_state["local_output_sensitivity_rms_l2"]
        ).double().numpy()
        proxy_cost += float(sensitivity[removed_ids].sum())

    total_removed = sum(counts.values())
    variant = (
        f"reap-only-uniform-p{round(args.pruning_fraction * 100)}"
        f"-r{target_pruned}-cal{metadata['retained_examples']}"
    )
    manifest = {
        "schema_version": 1,
        "variant": variant,
        "allocation": f"uniform {per_layer} removals per layer"
        if remainder == 0
        else "uniform per-layer allocation with largest-remainder rounding",
        "objective_name": "raw per-expert REAP",
        "ranking": "ascending raw REAP within each layer; ties stable by original expert ID",
        "calibration_state": str(args.state.resolve()),
        "calibration_metadata": str(args.metadata.resolve()),
        "sample_set_sha256": metadata["sample_set_sha256"],
        "calibration_examples": int(metadata["retained_examples"]),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "original_experts": num_experts,
        "top_k": top_k,
        "total_removed": total_removed,
        "total_retained": num_layers * num_experts - total_removed,
        "actual_pruning_fraction": total_removed / (num_layers * num_experts),
        "minimum_retained_experts": min(num_experts - value for value in counts.values()),
        "per_layer_removal_counts": counts,
        "retained_counts": {
            str(layer): num_experts - counts[str(layer)] for layer in range(num_layers)
        },
        "removed_expert_ids_original": removed,
        "retained_expert_ids_original": retained,
        "local_output_rms_l2_proxy_cost": proxy_cost,
        "interpretation": (
            "Uniform per-layer pruning count; remove experts with the lowest raw REAP score "
            "within each layer. The RMS-L2 value is logged for comparison only."
        ),
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Saved {args.output}")
    print(f"Variant: {variant}")
    print(f"Total removed: {total_removed}")
    print(f"Per-layer removals: {min(counts.values())}–{max(counts.values())}")
    print(f"RMS-L2 proxy cost: {proxy_cost:.6f}")


if __name__ == "__main__":
    main()
