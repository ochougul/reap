"""Materialize REAP and handoff router-selection-stability pruning states."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_state", type=pathlib.Path)
    parser.add_argument("router_accumulators", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    args = parser.parse_args()

    state = torch.load(args.calibration_state, map_location="cpu", weights_only=False)
    arrays = np.load(args.router_accumulators)
    variance_index = 0  # handoff contains one perturbation: variance multiplier 1.0
    lost = torch.from_numpy(arrays["selection_lost_count"][variance_index]).long()
    selected = torch.from_numpy(arrays["original_selected_count"]).long()

    reap_state = {}
    stability_state = {}
    for layer, values in state.items():
        reap = values["reap"].float().cpu()
        reap_state[layer] = dict(values)
        reap_state[layer]["reap_score"] = reap
        retained = selected[layer].float() - lost[layer].float()
        stability = torch.where(
            selected[layer] > 0,
            retained / selected[layer].float(),
            torch.zeros_like(retained),
        ).clamp(0.0, 1.0)
        stability_state[layer] = dict(values)
        stability_state[layer]["stability_score"] = stability
        stability_state[layer]["seap_score"] = stability

    reap_dir = args.output_dir / "reap"
    stability_dir = args.output_dir / "selection-stability"
    reap_dir.mkdir(parents=True, exist_ok=True)
    stability_dir.mkdir(parents=True, exist_ok=True)
    torch.save(reap_state, reap_dir / "state.pt")
    torch.save(stability_state, stability_dir / "state.pt")
    metadata = {
        "reap_formula": "stored calibration_state[layer]['reap']",
        "stability_formula": "1 - selection_lost_count / original_selected_count",
        "variance_index": variance_index,
        "variance_multiplier": 1.0,
        "note": "Selection-retention stability from the Qwen3.8 handoff; not paper rank stability.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
