"""Plot layerwise overlap between REAP and SEAP pruning sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    parser.add_argument("--prune-fraction", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    state = torch.load(args.state, weights_only=False)
    rows = []
    for layer in sorted(state):
        reap = state[layer]["reap_score"].float().cpu().numpy()
        seap = state[layer]["seap_score"].float().cpu().numpy()
        num_pruned = int(len(reap) * args.prune_fraction)
        reap_dropped = set(np.argsort(reap, kind="stable")[:num_pruned].tolist())
        seap_dropped = set(np.argsort(seap, kind="stable")[:num_pruned].tolist())
        overlap = len(reap_dropped & seap_dropped)
        rows.append({
            "layer": int(layer),
            "num_experts": len(reap),
            "num_pruned": num_pruned,
            "overlap_count": overlap,
            "overlap_percentage": 100.0 * overlap / num_pruned,
            "reap_dropped": sorted(reap_dropped),
            "seap_dropped": sorted(seap_dropped),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(12, 6))
    layers = [row["layer"] for row in rows]
    percentages = [row["overlap_percentage"] for row in rows]
    counts = [row["overlap_count"] for row in rows]
    axis.plot(layers, percentages, marker="o", linewidth=1.8, color="tab:blue")
    axis.set_xlabel("MoE layer index")
    axis.set_ylabel("REAP/SEAP pruning-set overlap (%)")
    axis.set_ylim(0, 105)
    axis.set_xticks(layers)
    axis.grid(alpha=0.25)
    axis.set_title(
        f"Layerwise REAP vs SEAP pruning overlap "
        f"({rows[0]['num_pruned']}/{rows[0]['num_experts']} experts pruned per layer)"
    )
    for layer, percentage, count in zip(layers, percentages, counts):
        axis.annotate(
            f"{count}",
            (layer, percentage),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(args.output, dpi=220)
    plt.close(fig)

    report = {
        "prune_fraction": args.prune_fraction,
        "layers": rows,
        "mean_overlap_count": float(np.mean(counts)),
        "mean_overlap_percentage": float(np.mean(percentages)),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "mean_overlap_count": report["mean_overlap_count"],
        "mean_overlap_percentage": report["mean_overlap_percentage"],
        "min_overlap_percentage": float(min(percentages)),
        "max_overlap_percentage": float(max(percentages)),
    }, indent=2))


if __name__ == "__main__":
    main()
