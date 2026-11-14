#!/usr/bin/env python3
"""Inverse-sum layer allocation from 0.5*stability + 0.5*REAP layer scores.

Per layer: stability = fraction of originally selected top-k expert slots that stay
selected under the router-input perturbation (1 - lost/selected); reap = sum of the 512
per-expert REAP scores. Each component is divided by its cross-layer mean so both have
mean 1, then score = w_stab*stab_n + w_reap*reap_n. Removals are proportional to 1/score
(capped, largest-remainder rounding, same solver as the RMS-L2 inverse-sum allocation).
Within a layer, experts are removed in ascending-REAP order (stable by expert ID).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments"))
from plot_inverse_sum_rms_l2_pruning_allocation import calculate_allocation  # noqa: E402
from qwen4_allocate_damage import stable_reap_orders  # noqa: E402

CAL = ROOT / ("artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-"
              "evol-codealpaca-12228-16384-unpruned")


def layer_scores(cal: pathlib.Path):
    z = np.load(cal / "router_stability/router_stability_accumulators.npz")
    selected = z["original_selected_count"].astype(np.float64)
    lost = z["selection_lost_count"][0].astype(np.float64)
    stability = 1.0 - lost.sum(1) / selected.sum(1)
    state = torch.load(cal / "calibration_state.pt", map_location="cpu", weights_only=False)
    reap = np.array([float(torch.as_tensor(state[l]["reap"]).double().sum()) for l in range(48)])
    return stability, reap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration-dir", type=pathlib.Path, default=CAL)
    ap.add_argument("--total-pruned", type=int, default=6144)
    ap.add_argument("--max-pruned-per-layer", type=int, default=502)
    ap.add_argument("--stability-weight", type=float, default=0.5)
    ap.add_argument("--reap-weight", type=float, default=0.5)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    args = ap.parse_args()

    stability, reap = layer_scores(args.calibration_dir)
    score = args.stability_weight * stability / stability.mean() \
        + args.reap_weight * reap / reap.mean()
    counts = calculate_allocation(score, args.total_pruned, args.max_pruned_per_layer)
    assert int(counts.sum()) == args.total_pruned

    state_path = args.calibration_dir / "calibration_state.pt"
    orders, sha = stable_reap_orders(state_path)
    manifest = {
        "variant": f"inverse-sum-stability{args.stability_weight}-reap{args.reap_weight}"
                   f"-r{args.total_pruned}-cap{args.max_pruned_per_layer}",
        "method": "layer quota proportional to 1/(w_s*stability/mean + w_r*REAPsum/mean); "
                  "experts removed by ascending REAP within layer",
        "stability_definition": "1 - selection_lost / original_selected (router perturbation)",
        "stability_weight": args.stability_weight, "reap_weight": args.reap_weight,
        "max_pruned_per_layer": args.max_pruned_per_layer,
        "total_removed": args.total_pruned, "total_retained": 48 * 512 - args.total_pruned,
        "num_layers": 48, "num_experts": 512, "original_experts": 512, "top_k": 10,
        "reap_state": str(state_path), "reap_state_sha256": sha,
        "layer_stability": [float(x) for x in stability],
        "layer_reap_sum": [float(x) for x in reap],
        "layer_score": [float(x) for x in score],
        "per_layer_removal_counts": {str(l): int(counts[l]) for l in range(48)},
        "removed_expert_ids_original": {str(l): orders[l][: int(counts[l])] for l in range(48)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "counts": counts.tolist()}))


if __name__ == "__main__":
    main()
