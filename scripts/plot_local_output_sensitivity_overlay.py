#!/usr/bin/env python3
"""Overlay per-expert local-output RMS-L2 sensitivity profiles by layer."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


DEFAULT_STATE = Path(
    "artifacts/Qwen3.8-Flash-Next-FP8/"
    "local-output-sensitivity-evol-codealpaca-12228-16384-unpruned/"
    "calibration_state.pt"
)
DEFAULT_OUTPUT = Path(
    "artifacts/Qwen3.8-Flash-Next-FP8/"
    "local-output-sensitivity-evol-codealpaca-12228-16384-unpruned/"
    "local_output_sensitivity_rms_l2_overlaid_layers.png"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    state = torch.load(args.state, map_location="cpu", weights_only=False)
    layers = sorted(int(layer) for layer in state)
    if len(layers) != 48 or layers != list(range(48)):
        raise ValueError(f"Expected 48 layers numbered 0..47, got {layers}")

    scores = np.stack(
        [
            torch.as_tensor(state[layer]["local_output_sensitivity_rms_l2"])
            .float()
            .numpy()
            for layer in layers
        ]
    )
    if scores.shape != (48, 512) or not np.isfinite(scores).all():
        raise ValueError(f"Expected finite sensitivity values shaped (48, 512), got {scores.shape}")
    if np.any(scores < 0):
        raise ValueError("RMS-L2 sensitivity values must be nonnegative")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    norm = Normalize(vmin=layers[0], vmax=layers[-1])
    cmap = plt.get_cmap("turbo")
    expert_ids = np.arange(scores.shape[1])

    fig, ax = plt.subplots(figsize=(15, 8), constrained_layout=True)
    for layer, values in zip(layers, scores):
        ax.plot(
            expert_ids,
            values,
            color=cmap(norm(layer)),
            linewidth=0.9,
            alpha=0.76,
        )

    global_median = float(np.median(scores))
    ax.axhline(
        global_median,
        color="#202020",
        linestyle="--",
        linewidth=1.1,
        alpha=0.75,
        label=f"Overall median = {global_median:.3f}",
    )
    ax.set_yscale("symlog", linthresh=0.03, linscale=1.0)
    ax.set_ylim(bottom=0)
    ax.set_yticks([0, 0.03, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    ax.set_xlim(0, 511)
    ax.set_xlabel("Expert ID (original checkpoint order)")
    ax.set_ylabel("Per-expert local-output sensitivity (RMS-L2, symlog scale)")
    ax.set_title(
        "Qwen3.8: per-expert local-output RMS-L2 sensitivity — all layers overlaid\n"
        "48 layers × 512 experts · 12,228 calibration examples"
    )
    ax.grid(True, which="major", alpha=0.22)
    ax.legend(loc="upper right", framealpha=0.9)

    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        ticks=[0, 8, 16, 24, 32, 40, 47],
        pad=0.015,
    )
    colorbar.set_label("Decoder layer")
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
