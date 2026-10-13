"""Plot layerwise REAP/SEAP pruning overlap for multiple prune fractions."""

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
    parser.add_argument("--prune-fractions", nargs="+", type=float, default=[0.25, 0.5])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    state = torch.load(args.state, weights_only=False)
    results = {}
    layers = sorted(state)
    for fraction in args.prune_fractions:
        fraction_key = f"{fraction:g}"
        rows = []
        for layer in layers:
            reap = state[layer]["reap_score"].float().cpu().numpy()
            seap = state[layer]["seap_score"].float().cpu().numpy()
            num_pruned = int(len(reap) * fraction)
            reap_dropped = set(np.argsort(reap, kind="stable")[:num_pruned].tolist())
            seap_dropped = set(np.argsort(seap, kind="stable")[:num_pruned].tolist())
            overlap = len(reap_dropped & seap_dropped)
            rows.append({
                "layer": int(layer),
                "num_experts": len(reap),
                "num_pruned": num_pruned,
                "overlap_count": overlap,
                "overlap_percentage": 100.0 * overlap / num_pruned,
            })
        results[fraction_key] = rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(12, 6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for index, (fraction_key, rows) in enumerate(results.items()):
        x = [row["layer"] for row in rows]
        y = [row["overlap_percentage"] for row in rows]
        counts = [row["overlap_count"] for row in rows]
        line = axis.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            color=colors[index % len(colors)],
            label=f"{float(fraction_key):g} pruning ({rows[0]['num_pruned']}/128)",
        )[0]
        for layer, percentage, count in zip(x, y, counts):
            axis.annotate(
                str(count),
                (layer, percentage),
                textcoords="offset points",
                xytext=(0, 5 if index % 2 == 0 else -11),
                ha="center",
                fontsize=6,
                color=line.get_color(),
            )

    axis.set_xlabel("MoE layer index")
    axis.set_ylabel("REAP/SEAP pruning-set overlap (%)")
    axis.set_ylim(0, 105)
    axis.set_xticks(layers)
    axis.grid(alpha=0.25)
    axis.set_title("Layerwise REAP vs SEAP pruning overlap")
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=220)
    plt.close(fig)

    report = {
        "prune_fractions": args.prune_fractions,
        "results": results,
        "summary": {
            fraction: {
                "mean_overlap_count": float(np.mean([r["overlap_count"] for r in rows])),
                "mean_overlap_percentage": float(np.mean([r["overlap_percentage"] for r in rows])),
                "min_overlap_percentage": float(min(r["overlap_percentage"] for r in rows)),
                "max_overlap_percentage": float(max(r["overlap_percentage"] for r in rows)),
            }
            for fraction, rows in results.items()
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
