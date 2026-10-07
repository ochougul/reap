"""Plot REAP vs stability from a SEAP state and emit comparison statistics."""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch


def rank(values):
    order = np.argsort(values, kind="stable")
    result = np.empty_like(order, dtype=float)
    result[order] = np.arange(len(values), dtype=float)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("artifacts/seap/smoke"))
    parser.add_argument("--delta-index", type=int, default=0)
    args = parser.parse_args()
    state = torch.load(args.state, weights_only=False)
    reap = torch.cat([state[layer]["reap_score"].reshape(-1).cpu() for layer in sorted(state)]).numpy()
    seap = torch.cat([state[layer]["seap_score"].reshape(-1).cpu() for layer in sorted(state)]).numpy()
    stability = torch.cat([
        (state[layer].get("stability_scores", state[layer]["stability_score"])[args.delta_index]
         if state[layer].get("stability_scores", state[layer]["stability_score"]).ndim == 2
         else state[layer]["stability_score"]).reshape(-1).cpu()
        for layer in sorted(state)
    ]).numpy()
    valid = np.isfinite(reap) & np.isfinite(stability) & np.isfinite(seap)
    reap, stability, seap = reap[valid], stability[valid], seap[valid]
    pearson = float(np.corrcoef(reap, stability)[0, 1]) if len(reap) > 1 else float("nan")
    spearman = float(np.corrcoef(rank(reap), rank(stability))[0, 1]) if len(reap) > 1 else float("nan")
    k = max(1, len(reap) // 10)
    top_reap = set(np.argsort(reap)[-k:])
    top_stability = set(np.argsort(stability)[-k:])
    stats = {
        "num_experts": int(len(reap)),
        "pearson": pearson,
        "spearman": spearman,
        "top_10_percent_overlap": len(top_reap & top_stability) / k,
        "reap_mean": float(reap.mean()),
        "stability_mean": float(stability.mean()),
        "seap_mean": float(seap.mean()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    plt.figure(figsize=(7, 6))
    plt.scatter(reap, stability, s=8, alpha=0.35)
    plt.xlabel("REAP score")
    plt.ylabel("Stability score")
    plt.title("REAP vs router stability")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(args.output / "reap_vs_stability.png", dpi=180)
    plt.figure(figsize=(12, 5))
    x = np.arange(len(reap))
    reap_normalized = reap / reap.max() if reap.max() > 0 else reap
    plt.plot(x, reap_normalized, label="normalized REAP", linewidth=0.7, alpha=0.8)
    plt.plot(x, stability, label="stability", linewidth=0.7, alpha=0.8)
    plt.plot(x, seap, label="SEAP", linewidth=0.7, alpha=0.8)
    plt.xlabel("Flattened layer/expert index")
    plt.ylabel("Score (normalized where needed)")
    plt.title("REAP, router stability, and SEAP")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(args.output / "reap_stability_seap.png", dpi=180)
    per_layer = args.output / "per_layer"
    per_layer.mkdir(parents=True, exist_ok=True)
    for layer in sorted(state):
        layer_reap = state[layer]["reap_score"].float().cpu().numpy()
        layer_stability = state[layer].get("stability_scores", state[layer]["stability_score"])
        if layer_stability.ndim == 2:
            layer_stability = layer_stability[args.delta_index]
        layer_stability = layer_stability.float().cpu().numpy()
        layer_seap = state[layer]["seap_score"].float().cpu().numpy()
        layer_reap = layer_reap / layer_reap.max() if layer_reap.max() > 0 else layer_reap
        plt.figure(figsize=(8, 5))
        experts = np.arange(len(layer_reap))
        plt.plot(experts, layer_reap, marker=".", linewidth=1, label="normalized REAP")
        plt.plot(experts, layer_stability, marker=".", linewidth=1, label="stability")
        plt.plot(experts, layer_seap, marker=".", linewidth=1, label="SEAP")
        plt.xlabel("Expert index")
        plt.ylabel("Score")
        plt.title(f"Layer {layer}: REAP, stability, and SEAP")
        plt.xlim(0, len(layer_reap) - 1)
        plt.grid(alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(per_layer / f"layer_{int(layer):03d}.png", dpi=180)
        plt.close()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
