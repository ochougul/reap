#!/usr/bin/env python3
"""Allocate an exact Qwen3.8 expert-pruning budget from measured damage curves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import sys

import numpy as np
import torch


DEFAULT_REAP_STATE = pathlib.Path(
    "artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-"
    "12228-16384-unpruned/calibration_state.pt"
)
PROXIES = (
    "mean_absolute_l2",
    "mean_relative_l2",
    "mean_residual_normalized_l2",
    "p95_relative_l2",
    "p99_relative_l2",
    "top1pct_relative_l2",
)


def stable_reap_orders(state_path: pathlib.Path) -> tuple[dict[int, list[int]], str]:
    """Read REAP vectors only and create ascending orders, stable by expert ID."""
    raw = state_path.read_bytes()
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    orders: dict[int, list[int]] = {}
    for layer, layer_state in sorted(state.items(), key=lambda item: int(item[0])):
        scores = torch.as_tensor(layer_state["reap"], dtype=torch.float64).reshape(-1)
        if scores.numel() != 512 or not torch.isfinite(scores).all():
            raise ValueError(f"Layer {layer} has malformed or non-finite REAP scores")
        # argsort(stable=True) leaves equal scores in input order (expert ID).
        orders[int(layer)] = torch.argsort(scores, stable=True).tolist()
    if len(orders) != 48 or sorted(orders) != list(range(48)):
        raise ValueError(f"Expected REAP scores for layers 0..47, got {sorted(orders)}")
    return orders, hashlib.sha256(raw).hexdigest()


def _pava_non_decreasing(values: np.ndarray) -> np.ndarray:
    """Least-squares isotonic fit using pooled adjacent violators."""
    means: list[float] = []
    weights: list[int] = []
    for value in values.astype(np.float64):
        means.append(float(value))
        weights.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            weight = weights[-2] + weights[-1]
            mean = (means[-2] * weights[-2] + means[-1] * weights[-1]) / weight
            means[-2:] = [mean]
            weights[-2:] = [weight]
    return np.concatenate([np.full(weight, mean) for mean, weight in zip(means, weights)])


def convexify_costs(costs: np.ndarray) -> np.ndarray:
    """Make C(0)=0 and fit nonnegative, nondecreasing discrete marginal costs."""
    values = np.asarray(costs, dtype=np.float64).reshape(-1).copy()
    if values.shape != (385,) or not np.isfinite(values).all():
        raise ValueError("Each layer curve must contain finite costs for k=0..384")
    values -= values[0]
    raw_marginals = np.diff(values)
    marginals = _pava_non_decreasing(np.maximum(raw_marginals, 0.0))
    result = np.concatenate(([0.0], np.cumsum(marginals)))
    return result


def read_curves(path: pathlib.Path, proxy: str,
                winsorize_percentile: float | None = None) -> dict[int, np.ndarray]:
    if proxy not in PROXIES:
        raise ValueError(f"Unknown proxy {proxy!r}; choose one of {PROXIES}")
    rows: dict[int, dict[int, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            layer, k = int(row["layer"]), int(row["k"])
            value = float(row[proxy])
            if layer in rows and k in rows[layer]:
                raise ValueError(f"Duplicate curve point layer={layer}, k={k}")
            rows.setdefault(layer, {})[k] = value
    if sorted(rows) != list(range(48)):
        raise ValueError("Curve summary must contain exactly layers 0..47")
    expected_ks = sorted(rows[0])
    if len(expected_ks) < 2 or expected_ks[0] != 0 or expected_ks[-1] != 384:
        raise ValueError("Curve grid must include both k=0 and k=384")
    if winsorize_percentile is not None:
        # Cap values above the cross-layer percentile at each measured k (no rows dropped).
        for k in expected_ks:
            threshold = float(np.percentile([rows[l][k] for l in rows], winsorize_percentile))
            for l in rows:
                rows[l][k] = min(rows[l][k], threshold)
    curves: dict[int, np.ndarray] = {}
    for layer, points in rows.items():
        if sorted(points) != expected_ks:
            raise ValueError(f"Layer {layer} must have k points {expected_ks}")
        values = np.interp(np.arange(385), expected_ks, [points[k] for k in expected_ks])
        curves[layer] = convexify_costs(values)
    return curves


def apply_power(curves: dict[int, np.ndarray], power: float) -> dict[int, np.ndarray]:
    """Raise convexified costs to a power >= 1 (still convex, nondecreasing).

    Large powers approach a min-max (equal-damage) allocation and penalize
    concentrating removals in one layer.
    """
    if power < 1:
        raise ValueError("--power must be >= 1")
    return {layer: np.power(cost, power) for layer, cost in curves.items()}


def allocate(curves: dict[int, np.ndarray], total_pruned: int,
             cap: int = 384) -> tuple[dict[int, int], float]:
    layers = sorted(curves)
    if layers != list(range(48)):
        raise ValueError("Expected cost curves for layers 0..47")
    if not 1 <= cap <= 384:
        raise ValueError("cap must be in [1, 384]")
    if not 0 <= total_pruned <= 48 * cap:
        raise ValueError(f"--total-pruned must be in [0, {48 * cap}]")

    # dp[r] is the least cost for exactly r removals after prior layers.
    dp = np.full(total_pruned + 1, np.inf, dtype=np.float64)
    dp[0] = 0.0
    back = np.full((48, total_pruned + 1), -1, dtype=np.int16)
    for layer in layers:
        next_dp = np.full_like(dp, np.inf)
        cost = curves[layer]
        for k in range(min(cap, total_pruned) + 1):
            limit = total_pruned - k
            candidate = dp[: limit + 1] + cost[k]
            target = next_dp[k : k + limit + 1]
            improve = candidate < target - 1e-14
            if improve.any():
                target[improve] = candidate[improve]
                back[layer, k : k + limit + 1][improve] = k
        dp = next_dp
    if not math.isfinite(float(dp[total_pruned])):
        raise RuntimeError(f"No allocation found for exact budget {total_pruned}")
    allocation: dict[int, int] = {}
    remaining = total_pruned
    for layer in range(47, -1, -1):
        k = int(back[layer, remaining])
        if k < 0:
            raise RuntimeError(f"Broken DP backpointer at layer={layer}, budget={remaining}")
        allocation[layer] = k
        remaining -= k
    if remaining != 0 or sum(allocation.values()) != total_pruned:
        raise RuntimeError("DP backtracking did not preserve the exact requested budget")
    return allocation, float(dp[total_pruned])


def blend_with_uniform(counts: dict[int, int], total_pruned: int, lam: float,
                       cap: int) -> dict[int, int]:
    """k = (1-lam)*uniform + lam*dp, rounded by largest remainder to the exact total."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError("--blend-lambda must be in [0, 1]")
    uniform = total_pruned / 48
    target = {l: (1 - lam) * uniform + lam * counts[l] for l in range(48)}
    result = {l: min(cap, int(math.floor(v))) for l, v in target.items()}
    short = total_pruned - sum(result.values())
    order = sorted(range(48), key=lambda l: (-(target[l] - math.floor(target[l])), l))
    while short > 0:
        progressed = False
        for l in order:
            if short and result[l] < cap:
                result[l] += 1
                short -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("Cannot place blended removals under the cap")
    return result


def plot_allocation(counts: dict[int, int], path: pathlib.Path, title: str, cap: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(20, 7))
    layers = list(range(48))
    values = [counts[l] for l in layers]
    colors = ["#e55656" if v >= cap else "#4c78a8" for v in values]
    ax.bar(layers, values, color=colors)
    for l, v in zip(layers, values):
        ax.annotate(str(v), (l, v), ha="center", va="bottom", fontsize=8,
                    xytext=(0, 3), textcoords="offset points")
    ax.axhline(cap, color="#a02020", linestyle="--", linewidth=1, label=f"Per-layer cap: {cap}")
    ax.axhline(sum(values) / 48, color="gray", linestyle=":", linewidth=1,
               label=f"Uniform: {sum(values) / 48:.0f}")
    ax.set_xticks(layers)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Experts pruned")
    ax.set_ylim(0, max(cap, max(values)) * 1.12)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=pathlib.Path, required=True)
    parser.add_argument("--reap-state", type=pathlib.Path, default=DEFAULT_REAP_STATE)
    parser.add_argument("--proxy", choices=PROXIES, default="mean_relative_l2")
    parser.add_argument("--total-pruned", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--power", type=float, default=1.0,
                        help="Raise convexified layer costs to this power (>=1).")
    parser.add_argument("--max-pruned-per-layer", type=int, default=384)
    parser.add_argument("--blend-lambda", type=float, default=1.0,
                        help="1 = pure DP; 0 = uniform; in between blends counts.")
    parser.add_argument("--plot", type=pathlib.Path, default=None)
    parser.add_argument("--winsorize-percentile", type=float, default=None,
                        help="Cap each layer's curve value at this cross-layer percentile, per k.")
    parser.add_argument("--reap-layer-scale", type=float, default=0.0,
                        help="Multiply each layer's curve by (sum of its REAP scores)^this "
                             "before the power; 0 disables, 1 is the multiplicative form.")
    args = parser.parse_args()

    orders, reap_sha256 = stable_reap_orders(args.reap_state)
    cap = args.max_pruned_per_layer
    curves = read_curves(args.curves, args.proxy, args.winsorize_percentile)
    if args.reap_layer_scale:
        state = torch.load(args.reap_state, map_location="cpu", weights_only=False)
        sums = {int(l): float(torch.as_tensor(v["reap"]).double().sum()) for l, v in state.items()}
        curves = {l: c * sums[l] ** args.reap_layer_scale for l, c in curves.items()}
    curves = apply_power(curves, args.power)
    counts, objective = allocate(curves, args.total_pruned, cap)
    dp_counts = dict(counts)
    if args.blend_lambda != 1.0:
        counts = blend_with_uniform(counts, args.total_pruned, args.blend_lambda, cap)
        objective = float(sum(curves[l][counts[l]] for l in range(48)))
    layers = {}
    valid_masks = {}
    for layer in range(48):
        removed = orders[layer][: counts[layer]]
        kept = sorted(set(range(512)) - set(removed))
        layers[str(layer)] = {
            "pruned_expert_indices_ascending_reap_prefix": removed,
            "retained_expert_indices": kept,
            "num_pruned": counts[layer],
            "num_retained": len(kept),
        }
        mask = [True] * 512
        for expert in removed:
            mask[expert] = False
        valid_masks[str(layer)] = mask

    manifest = {
        "variant": f"qwen4-damage-{args.proxy}-R{args.total_pruned}",
        "method": "measured damage curve with convex integer dynamic programming",
        "proxy": args.proxy,
        "curve_summary": str(args.curves),
        "reap_state": str(args.reap_state),
        "reap_state_sha256": reap_sha256,
        "original_experts": 512,
        "top_k": 10,
        "minimum_retained_experts": 512 - cap,
        "total_removed": args.total_pruned,
        "total_retained": 48 * 512 - args.total_pruned,
        "objective_value": objective,
        "allocation": "exact integer DP over linearly interpolated, convexified layer costs",
        "power": args.power,
        "reap_layer_scale": args.reap_layer_scale,
        "winsorize_percentile": args.winsorize_percentile,
        "max_pruned_per_layer": cap,
        "blend_lambda": args.blend_lambda,
        "dp_counts_before_blend": {str(l): dp_counts[l] for l in range(48)},
        "routing": {"mask_before_softmax_and_topk": True},
        "layers": layers,
        "valid_expert_mask": valid_masks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.plot:
        plot_allocation(counts, args.plot,
                        f"Allocation ({args.proxy}, power={args.power}, cap={cap}, "
                        f"lambda={args.blend_lambda}): {args.total_pruned} experts", cap)
    print(json.dumps({"output": str(args.output), "total_pruned": args.total_pruned,
                      "counts": counts, "objective_value": objective}, indent=2))


if __name__ == "__main__":
    main()
