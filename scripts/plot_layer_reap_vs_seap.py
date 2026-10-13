"""Plot REAP and SEAP scores with their respective dropped experts."""

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
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--prune-fraction", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    layer_state = torch.load(args.state, weights_only=False)[args.layer]
    reap = layer_state["reap_score"].float().cpu().numpy()
    seap = layer_state["seap_score"].float().cpu().numpy()
    experts = np.arange(len(reap))
    num_pruned = int(len(experts) * args.prune_fraction)
    reap_dropped = np.argsort(reap, kind="stable")[:num_pruned]
    seap_dropped = np.argsort(seap, kind="stable")[:num_pruned]
    overlap = np.intersect1d(reap_dropped, seap_dropped)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, reap_axis = plt.subplots(figsize=(13, 6))
    seap_axis = reap_axis.twinx()

    reap_line = reap_axis.plot(
        experts, reap, color="tab:red", linewidth=1.4, label="REAP score"
    )[0]
    seap_line = seap_axis.plot(
        experts, seap, color="tab:blue", linewidth=1.4, label="SEAP score"
    )[0]
    reap_markers = reap_axis.scatter(
        reap_dropped,
        reap[reap_dropped],
        marker="x",
        s=58,
        linewidths=1.8,
        color="darkred",
        label="REAP dropped",
        zorder=5,
    )
    seap_markers = seap_axis.scatter(
        seap_dropped,
        seap[seap_dropped],
        marker="o",
        facecolors="none",
        edgecolors="navy",
        s=58,
        linewidths=1.5,
        label="SEAP dropped",
        zorder=5,
    )

    reap_axis.set_xlabel("Expert index")
    reap_axis.set_ylabel("REAP score", color="tab:red")
    seap_axis.set_ylabel("SEAP score", color="tab:blue")
    reap_axis.tick_params(axis="y", labelcolor="tab:red")
    seap_axis.tick_params(axis="y", labelcolor="tab:blue")
    reap_axis.set_xlim(0, len(experts) - 1)
    reap_axis.grid(alpha=0.22)
    reap_axis.set_title(
        f"Layer {args.layer}: REAP vs SEAP "
        f"({num_pruned}/{len(experts)} experts dropped; "
        f"overlap {len(overlap)}/{num_pruned})"
    )
    reap_axis.legend(
        [reap_line, seap_line, reap_markers, seap_markers],
        ["REAP score", "SEAP score", "REAP dropped", "SEAP dropped"],
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=220)
    plt.close(fig)

    report = {
        "layer": args.layer,
        "num_experts": len(experts),
        "prune_fraction": args.prune_fraction,
        "num_pruned": num_pruned,
        "reap_dropped": reap_dropped.tolist(),
        "seap_dropped": seap_dropped.tolist(),
        "overlap": overlap.tolist(),
        "overlap_fraction": float(len(overlap) / num_pruned),
        "reap_score_min": float(reap.min()),
        "reap_score_max": float(reap.max()),
        "seap_score_min": float(seap.min()),
        "seap_score_max": float(seap.max()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
