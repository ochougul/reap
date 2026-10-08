"""Build offline SEAP states for lambda/delta sweeps."""

from __future__ import annotations

import argparse
import copy
import json
import pathlib

import torch


parser = argparse.ArgumentParser()
parser.add_argument("state", type=pathlib.Path)
parser.add_argument("--output-dir", type=pathlib.Path, required=True)
parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
parser.add_argument("--deltas", nargs="+", type=float, default=[2.0, 4.0])
args = parser.parse_args()

base = torch.load(args.state, weights_only=False)
args.output_dir.mkdir(parents=True, exist_ok=True)

for delta in args.deltas:
    delta_key = f"stability_score_delta_{delta:g}"
    for lambda_ in args.lambdas:
        state = copy.deepcopy(base)
        for layer, values in state.items():
            stability = values.get(delta_key)
            if stability is None:
                all_stability = values.get("stability_scores")
                if all_stability is None:
                    raise KeyError(f"Missing {delta_key} in layer {layer}")
                stability = all_stability[0 if delta == 2.0 else 1]
            normalized_reap = values["normalized_reap"].float()
            values["stability_score"] = stability.float()
            values["seap_score"] = (
                lambda_ * normalized_reap + (1.0 - lambda_) * stability.float()
            )
        output = args.output_dir / f"delta-{delta:g}" / f"lambda-{lambda_:.2f}"
        output.mkdir(parents=True, exist_ok=True)
        torch.save(state, output / "state.pt")
        (output / "metadata.json").write_text(json.dumps({
            "source_state": str(args.state),
            "delta": delta,
            "lambda": lambda_,
            "formula": "lambda * normalized_reap + (1 - lambda) * stability",
        }, indent=2) + "\n")
