"""Plot normalized REAP and stability for one layer."""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch


def normalize_max(values: np.ndarray) -> np.ndarray:
    maximum = values.max()
    return values / maximum if maximum > 0 else np.zeros_like(values)


def normalize_sum(values: np.ndarray) -> np.ndarray:
    total = values.sum()
    return values / total if total > 0 else np.zeros_like(values)


parser = argparse.ArgumentParser()
parser.add_argument("state", type=pathlib.Path)
parser.add_argument("--layer", type=int, default=27)
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--raw-stability", action="store_true")
parser.add_argument("--sum-normalize", action="store_true")
args = parser.parse_args()

state = torch.load(args.state, weights_only=False)
layer = state[args.layer]
reap = layer["reap_score"].float().cpu().numpy()
reap = normalize_sum(reap) if args.sum_normalize else normalize_max(reap)
stability = layer.get("stability_scores", layer["stability_score"])
if stability.ndim == 2:
    stability = stability[0]
stability = stability.float().cpu().numpy()
if not args.raw_stability:
    stability = normalize_sum(stability) if args.sum_normalize else normalize_max(stability)

experts = np.arange(len(reap))
args.output.parent.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(12, 5))
plt.plot(experts, reap, marker=".", linewidth=1, label="normalized REAP")
plt.plot(experts, stability, marker=".", linewidth=1, label="normalized stability")
plt.xlabel("Expert index")
plt.ylabel("Score")
normalization_label = "sum-normalized" if args.sum_normalize else "max-normalized"
stability_label = "stability" if args.raw_stability else f"{normalization_label} stability"
plt.title(f"Layer {args.layer}: normalized REAP vs {stability_label}")
plt.xlim(0, len(experts) - 1)
plt.ylim(0, 1.05)
plt.grid(alpha=0.2)
reap_label = f"{normalization_label} REAP"
plt.legend([reap_label, stability_label])
plt.tight_layout()
plt.savefig(args.output, dpi=200)
plt.close()
print(args.output)
