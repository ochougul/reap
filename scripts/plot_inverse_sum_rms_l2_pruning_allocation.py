#!/usr/bin/env python3
"""Plot a pruning allocation inversely proportional to layer RMS-L2 sums."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SCORES = Path(
    "artifacts/Qwen3.8-Flash-Next-FP8/"
    "local-output-sensitivity-evol-codealpaca-12228-16384-unpruned/"
    "rms_l2_summed_score_by_layer.csv"
)
DEFAULT_PLOT = DEFAULT_SCORES.with_name("inverse_sum_rms_l2_pruned_experts_by_layer.png")
DEFAULT_ALLOCATION = DEFAULT_SCORES.with_name("inverse_sum_rms_l2_pruning_allocation.csv")
TOTAL_EXPERTS = 48 * 512


def calculate_allocation(
    scores: np.ndarray, target_pruned: int, max_pruned_per_layer: int
) -> np.ndarray:
    weights = 1.0 / scores
    active = set(range(len(scores)))
    quotas = np.zeros(len(scores), dtype=np.float64)
    remaining = target_pruned

    while active:
        weight_sum = sum(weights[layer] for layer in active)
        saturated = [
            layer
            for layer in active
            if remaining * weights[layer] / weight_sum >= max_pruned_per_layer
        ]
        if not saturated:
            for layer in active:
                quotas[layer] = remaining * weights[layer] / weight_sum
            break
        for layer in saturated:
            quotas[layer] = max_pruned_per_layer
            remaining -= max_pruned_per_layer
            active.remove(layer)

    allocation = np.minimum(np.floor(quotas).astype(np.int64), max_pruned_per_layer)
    remainder = target_pruned - int(allocation.sum())
    order = sorted(
        range(len(scores)),
        key=lambda layer: (quotas[layer] - allocation[layer], -scores[layer]),
        reverse=True,
    )
    for layer in order:
        if remainder == 0:
            break
        if allocation[layer] < max_pruned_per_layer:
            allocation[layer] += 1
            remainder -= 1
    if remainder:
        raise RuntimeError(f"Could not allocate {remainder} remaining removals")
    return allocation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--allocation-csv", type=Path, default=DEFAULT_ALLOCATION)
    parser.add_argument("--pruning-fraction", type=float, default=0.25)
    parser.add_argument("--max-pruned-per-layer", type=int, default=502)
    args = parser.parse_args()

    if not 0.0 < args.pruning_fraction < 1.0:
        raise ValueError("Pruning fraction must be between 0 and 1")
    if not 0 < args.max_pruned_per_layer <= 502:
        raise ValueError("Maximum removals per layer must be in 1..502")
    target_pruned = round(TOTAL_EXPERTS * args.pruning_fraction)
    if target_pruned != TOTAL_EXPERTS * args.pruning_fraction:
        raise ValueError("Pruning fraction must produce an integer expert count")
    if target_pruned > 48 * args.max_pruned_per_layer:
        raise ValueError("Per-layer cap is too small for the requested total pruning")

    with args.scores.open(newline="") as f:
        rows = list(csv.DictReader(f))
    layers = np.array([int(row["layer_idx"]) for row in rows], dtype=np.int64)
    scores = np.array([float(row["sum_rms_l2_score"]) for row in rows])
    if not np.array_equal(layers, np.arange(48)):
        raise ValueError("Expected 48 layers numbered 0..47")
    if not np.isfinite(scores).all() or np.any(scores <= 0):
        raise ValueError("Layer score sums must be finite and positive")

    pruned = calculate_allocation(scores, target_pruned, args.max_pruned_per_layer)
    if int(pruned.sum()) != target_pruned:
        raise ValueError(f"Expected {target_pruned} pruned experts, got {pruned.sum()}")

    args.allocation_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.allocation_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("layer_idx", "sum_rms_l2_score", "experts_pruned", "experts_kept"))
        writer.writerows(
        (int(layer), f"{scores[layer]:.10g}", int(pruned[layer]), 512 - int(pruned[layer]))
            for layer in layers
        )

    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    ax.bar(layers, pruned, color="#2673b8", width=0.82)
    ax.axhline(target_pruned / 48, color="#d1495b", linestyle="--", linewidth=1.4,
               label=f"Uniform reference = {target_pruned / 48:.0f} experts/layer")
    ax.set_xlim(-0.6, 47.6)
    ax.set_xticks(range(0, 48, 2))
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Number of experts pruned")
    ax.set_title(
        f"{args.pruning_fraction:.0%} pruning allocation inversely proportional to layer RMS-L2 sum\n"
        f"{target_pruned:,} experts total · at least 10 retained · cap {args.max_pruned_per_layer}/layer"
    )
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend()
    fig.savefig(args.plot, dpi=180)
    plt.close(fig)

    print(f"Saved plot: {args.plot}")
    print(f"Saved allocation: {args.allocation_csv}")
    print(f"Total pruned: {int(pruned.sum())}")
    print(f"Pruned per-layer range: {int(pruned.min())}–{int(pruned.max())}")


if __name__ == "__main__":
    main()
