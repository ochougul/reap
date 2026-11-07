"""Plot layer MLP importance defined as the sum of per-expert REAP scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    state = torch.load(args.state, map_location="cpu", weights_only=False)
    layers = sorted(k for k, value in state.items() if isinstance(value, dict))
    scores = []
    expert_counts = []
    for layer in layers:
        reap = state[layer]["reap"].float().reshape(-1)
        scores.append(float(reap.sum()))
        expert_counts.append(int(reap.numel()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(14, 6.5))
    bars = axis.bar(layers, scores, color="tab:purple", alpha=0.86)
    axis.set_xlabel("MoE layer")
    axis.set_ylabel("Layer REAP importance (sum of expert REAP scores)")
    axis.set_title("Qwen3.8-Flash-Next-FP8: layer MLP importance from REAP")
    axis.set_xticks(layers)
    axis.grid(axis="y", alpha=0.25)
    maximum = max(range(len(scores)), key=scores.__getitem__)
    minimum = min(range(len(scores)), key=scores.__getitem__)
    bars[maximum].set_color("tab:orange")
    bars[minimum].set_color("tab:green")
    axis.legend(
        [bars[maximum], bars[minimum]],
        [f"max: layer {layers[maximum]}", f"min: layer {layers[minimum]}"],
        loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=220)
    plt.close(fig)

    report = {
        "definition": "sum of state[layer]['reap'] over experts",
        "source": str(args.state),
        "num_layers": len(layers),
        "expert_counts": expert_counts,
        "layer_importance": {str(layer): score for layer, score in zip(layers, scores)},
        "min_layer": layers[minimum],
        "max_layer": layers[maximum],
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
