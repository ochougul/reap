#!/usr/bin/env python3
"""Build an exact-budget REAP tail-mass allocation with uniform edge layers.

This is an offline allocator: it uses the saved cumulative curves and ranking
orders, and never loads model weights or runs inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


LAYERS = 48
EXPERTS = 512
TOP_K = 10
DEFAULT_ROOT = Path("artifacts/Qwen3.8-Flash-Next-FP8/layerwise-reuse-v1")
TOTAL_REMOVED = 9180
RADIUS = 64
FIXED_EDGES = {**{layer: 192 for layer in range(4)}, **{layer: 191 for layer in range(44, 48)}}


def uniform_counts(total: int) -> dict[int, int]:
    base, extra = divmod(total, LAYERS)
    return {layer: base + int(layer < extra) for layer in range(LAYERS)}


def allocate_middle_dp(
    cumulative_mass: np.ndarray,
    total: int,
    uniform: dict[int, int],
    radius: int = RADIUS,
    fixed_edges: dict[int, int] = FIXED_EDGES,
) -> dict[int, int]:
    """Minimize summed tail mass over non-edge layers at the exact residual."""
    if cumulative_mass.shape != (LAYERS, EXPERTS + 1):
        raise ValueError(f"expected a {(LAYERS, EXPERTS + 1)} curve")
    if set(fixed_edges) != set(range(4)) | set(range(44, 48)):
        raise ValueError("this experiment requires exactly layers 0..3 and 44..47 fixed")
    if radius < 0 or total < 0:
        raise ValueError("radius and total must be non-negative")

    fixed_sum = sum(fixed_edges.values())
    remaining_total = total - fixed_sum
    middle_layers = [layer for layer in range(LAYERS) if layer not in fixed_edges]
    bounds: dict[int, tuple[int, int]] = {}
    for layer in middle_layers:
        low = max(0, uniform[layer] - radius)
        high = min(EXPERTS - TOP_K, uniform[layer] + radius)
        bounds[layer] = (low, high)
    if not sum(low for low, _ in bounds.values()) <= remaining_total <= sum(
        high for _, high in bounds.values()
    ):
        raise ValueError("residual budget is infeasible under middle-layer bounds")

    inf = float("inf")
    dp = np.full(remaining_total + 1, inf, dtype=np.float64)
    dp[0] = 0.0
    back: list[np.ndarray] = []
    for layer in middle_layers:
        low, high = bounds[layer]
        next_dp = np.full_like(dp, inf)
        previous = np.full(remaining_total + 1, -1, dtype=np.int16)
        for count in range(low, high + 1):
            if count > remaining_total:
                break
            destinations = np.arange(count, remaining_total + 1)
            candidates = dp[: remaining_total + 1 - count] + cumulative_mass[layer, count]
            improve = np.isfinite(candidates) & (candidates < next_dp[destinations] - 1e-12)
            next_dp[destinations[improve]] = candidates[improve]
            previous[destinations[improve]] = count
        dp = next_dp
        back.append(previous)

    if not math.isfinite(float(dp[remaining_total])):
        raise ValueError("no exact allocation exists for the requested residual budget")

    counts = dict(fixed_edges)
    remaining = remaining_total
    for index in reversed(range(len(middle_layers))):
        count = int(back[index][remaining])
        if count < 0:
            raise AssertionError("DP allocation backtrace failed")
        counts[middle_layers[index]] = count
        remaining -= count
    if remaining != 0 or sum(counts.values()) != total:
        raise AssertionError("allocation failed exact-total validation")
    for layer, count in counts.items():
        if layer in fixed_edges:
            if count != fixed_edges[layer]:
                raise AssertionError(f"fixed edge layer {layer} moved")
        else:
            low, high = bounds[layer]
            if not low <= count <= high:
                raise AssertionError(f"middle layer {layer} violates radius bound")
    if any(EXPERTS - count < TOP_K for count in counts.values()):
        raise AssertionError("allocation violates top-k feasibility")
    return dict(sorted(counts.items()))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    curves_path: Path,
    uniform_manifest_path: Path,
    cumulative_mass: np.ndarray,
    order: np.ndarray,
    total: int = TOTAL_REMOVED,
    radius: int = RADIUS,
) -> dict[str, Any]:
    if order.shape != (LAYERS, EXPERTS):
        raise ValueError(f"expected ranking order shape {(LAYERS, EXPERTS)}")
    if not np.array_equal(np.sort(order, axis=1), np.broadcast_to(np.arange(EXPERTS), order.shape)):
        raise ValueError("ranking order must be a permutation of original expert IDs per layer")
    uniform = uniform_counts(total)
    counts = allocate_middle_dp(cumulative_mass, total, uniform, radius)
    removed_ids = {str(layer): order[layer, :counts[layer]].astype(int).tolist() for layer in range(LAYERS)}
    proxy = float(sum(cumulative_mass[layer, count] for layer, count in counts.items()))
    uniform_proxy = float(sum(cumulative_mass[layer, count] for layer, count in uniform.items()))
    if proxy > uniform_proxy + 1e-10:
        raise AssertionError("tail-mass objective is worse than feasible uniform allocation")
    return {
        "variant": "P-reap-fixed-edges-R9180",
        "ranking": "reap",
        "allocation": "tail-mass-middle-fixed-uniform-edges",
        "objective_name": "tail-mass",
        "total_removed": total,
        "actual_pruning_fraction": total / (LAYERS * EXPERTS),
        "num_layers": LAYERS,
        "num_experts": EXPERTS,
        "original_experts": EXPERTS,
        "top_k": TOP_K,
        "radius": radius,
        "radius_applies_to": "middle layers 4..43 only; deviation from uniform removal count",
        "fixed_edge_layers": {str(layer): count for layer, count in FIXED_EDGES.items()},
        "per_layer_removal_counts": {str(layer): count for layer, count in counts.items()},
        "uniform_removal_counts": {str(layer): count for layer, count in uniform.items()},
        "middle_total_removed": sum(counts[layer] for layer in range(4, 44)),
        "retained_counts": {str(layer): EXPERTS - count for layer, count in counts.items()},
        "removed_expert_ids_original": removed_ids,
        "retained_expert_ids_original": {
            str(layer): [expert for expert in range(EXPERTS) if expert not in set(removed_ids[str(layer)])]
            for layer in range(LAYERS)
        },
        "proxy_objective": proxy,
        "uniform_proxy_objective": uniform_proxy,
        "proxy_improvement_vs_uniform": uniform_proxy - proxy,
        "solver": "dynamic programming over middle-layer integer counts at exact residual budget",
        "tie_rule": "ascending count transitions; strict improvement only; original expert ID breaks ranking ties",
        "inputs": {
            str(curves_path): {"sha256": sha256(curves_path)},
            str(uniform_manifest_path): {"sha256": sha256(uniform_manifest_path)},
        },
        "protocol_note": "Layers 0..3 and 44..47 retain exactly the uniform-REAP R=9180 removal counts. Tail-mass optimization and ±64 bounds apply only to layers 4..43.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--total", type=int, default=TOTAL_REMOVED)
    parser.add_argument("--radius", type=int, default=RADIUS)
    args = parser.parse_args()
    curves_path = args.root / "layer_removal_curves.npz"
    uniform_path = args.root / "allocations/U-reap-R9180/manifest.json"
    output_dir = args.root / "allocations/P-reap-fixed-edges-R9180"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "manifest.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing candidate manifest: {output_path}")
    with np.load(curves_path) as curves:
        cumulative_mass = curves["reap_cumulative_mass"]
        order = curves["reap_order"]
    manifest = build_manifest(curves_path, uniform_path, cumulative_mass, order, args.total, args.radius)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")
    counts = {int(layer): int(count) for layer, count in manifest["per_layer_removal_counts"].items()}
    print(json.dumps({
        "manifest": str(output_path),
        "total_removed": sum(counts.values()),
        "pruning_fraction": manifest["actual_pruning_fraction"],
        "fixed_edges": manifest["fixed_edge_layers"],
        "middle_removed": manifest["middle_total_removed"],
        "middle_min_max": [min(counts[layer] for layer in range(4, 44)), max(counts[layer] for layer in range(4, 44))],
        "tail_mass_proxy": manifest["proxy_objective"],
        "uniform_tail_mass_proxy": manifest["uniform_proxy_objective"],
    }, indent=2))


if __name__ == "__main__":
    main()
