"""Plot layer scores for a SEAP lambda sweep and compare pruning sets."""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument("state", type=pathlib.Path)
parser.add_argument("--layer", type=int, default=20)
parser.add_argument("--lambdas", nargs="+", type=float, default=[0.25, 0.5, 0.75])
parser.add_argument("--prune-fraction", type=float, default=0.25)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()

state = torch.load(args.state, weights_only=False)[args.layer]
reap = state["reap_score"].float().cpu().numpy()
normalized_reap = state.get("normalized_reap", reap / reap.max()).float().cpu().numpy()
stability = state["stability_score"].float().cpu().numpy()
experts = np.arange(len(reap))
num_pruned = int(len(experts) * args.prune_fraction)
reap_drop = set(np.argsort(reap)[:num_pruned].tolist())

args.output.mkdir(parents=True, exist_ok=True)
report = {
    "layer": args.layer,
    "prune_fraction": args.prune_fraction,
    "num_experts": len(experts),
    "num_pruned": num_pruned,
    "reap_only_dropped": sorted(reap_drop),
    "lambdas": {},
}

for lam in args.lambdas:
    seap = lam * normalized_reap + (1.0 - lam) * stability
    seap_drop = set(np.argsort(seap)[:num_pruned].tolist())
    report["lambdas"][str(lam)] = {
        "seap_dropped": sorted(seap_drop),
        "newly_dropped_vs_reap": sorted(seap_drop - reap_drop),
        "reap_dropped_instead": sorted(reap_drop - seap_drop),
        "drop_set_overlap": len(seap_drop & reap_drop) / num_pruned,
    }
    plt.figure(figsize=(12, 5))
    plt.plot(experts, normalized_reap, marker=".", linewidth=1, label="normalized REAP")
    plt.plot(experts, stability, marker=".", linewidth=1, label="stability")
    plt.plot(experts, seap, marker=".", linewidth=1.5, label=f"SEAP λ={lam:g}")
    plt.scatter(sorted(seap_drop), seap[sorted(seap_drop)], s=22, zorder=4, label="SEAP dropped")
    plt.xlabel("Expert index")
    plt.ylabel("Score")
    plt.title(f"Layer {args.layer}: REAP, stability, and SEAP λ={lam:g}")
    plt.xlim(0, len(experts) - 1)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output / f"layer_{args.layer:03d}_lambda_{lam:.2f}.png", dpi=200)
    plt.close()

(args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
