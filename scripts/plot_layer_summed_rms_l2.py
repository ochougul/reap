#!/usr/bin/env python3
"""Plot the sum of per-expert local-output RMS-L2 scores for each layer."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_RESULTS = Path(
    "artifacts/Qwen3.8-Flash-Next-FP8/"
    "local-output-sensitivity-evol-codealpaca-12228-16384-unpruned/"
    "local_output_sensitivity_results.pt"
)
DEFAULT_PLOT = DEFAULT_RESULTS.with_name("rms_l2_summed_score_by_layer.png")
DEFAULT_CSV = DEFAULT_RESULTS.with_name("rms_l2_summed_score_by_layer.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    results = torch.load(args.results, map_location="cpu", weights_only=False)
    layer_results = results["layers"]
    layers = sorted(int(layer) for layer in layer_results)
    if layers != list(range(48)):
        raise ValueError(f"Expected layers numbered 0..47, got {layers}")

    scores = np.stack(
        [
            torch.as_tensor(layer_results[layer]["rms_l2_by_expert"])
            .float()
            .numpy()
            for layer in layers
        ]
    )
    if scores.shape != (48, 512) or not np.isfinite(scores).all():
        raise ValueError(f"Expected finite RMS-L2 values shaped (48, 512), got {scores.shape}")
    if np.any(scores < 0):
        raise ValueError("RMS-L2 sensitivity values must be nonnegative")

    totals = scores.sum(axis=1, dtype=np.float64)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("layer_idx", "sum_rms_l2_score"))
        writer.writerows((layer, f"{totals[layer]:.10g}") for layer in layers)

    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    ax.plot(layers, totals, color="#2673b8", marker="o", linewidth=1.8, markersize=4)
    ax.set_xlim(0, 47)
    ax.set_xticks(range(0, 48, 2))
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Sum of 512 per-expert RMS-L2 sensitivity scores")
    ax.set_title(
        "Qwen3.8: summed per-expert local-output RMS-L2 sensitivity by layer\n"
        "12,228 calibration examples"
    )
    ax.grid(True, alpha=0.28)
    fig.savefig(args.plot, dpi=180)
    plt.close(fig)

    print(f"Saved plot: {args.plot}")
    print(f"Saved layer totals: {args.csv}")
    print(f"Lowest-sum layer: {int(np.argmin(totals))} ({totals.min():.6f})")
    print(f"Highest-sum layer: {int(np.argmax(totals))} ({totals.max():.6f})")


if __name__ == "__main__":
    main()
