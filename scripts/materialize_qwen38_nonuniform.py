"""Build compact stability-only and padded serving Qwen3.8 checkpoints.

The compact directory is the canonical artifact.  The padded directory is a
derived serving artifact and includes the per-layer validity mask needed by a
mask-aware Qwen4-Exp router.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reap.model_util import get_moe, slice_fused_expert_weights
from reap.nonuniform_routing import assert_routing_invariants
from reap.prune import allocate_layerwise_pruning_counts


EXPECTED_DATASET = "theblackcat102/evol-codealpaca-v1"
EXPECTED_LAYERS = 48
EXPECTED_EXPERTS = 512
EXPECTED_TOP_K = 10


def _load_and_verify_state(state_path: pathlib.Path, calibration_metadata: pathlib.Path, calibration_state_path: pathlib.Path) -> dict[int, dict[str, Any]]:
    metadata = json.loads(calibration_metadata.read_text())
    if metadata.get("dataset_name") != EXPECTED_DATASET:
        raise ValueError(
            f"Refusing state: calibration dataset is {metadata.get('dataset_name')!r}, "
            f"expected {EXPECTED_DATASET!r}"
        )
    expected_calibration_state = calibration_metadata.parent / "calibration_state.pt"
    if calibration_state_path.resolve() != expected_calibration_state.resolve():
        raise ValueError(f"Expected final calibration artifact {expected_calibration_state}, got {calibration_state_path}")
    if not calibration_state_path.exists():
        raise FileNotFoundError(calibration_state_path)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    calibration_state = torch.load(calibration_state_path, map_location="cpu", weights_only=False)
    layers = sorted(int(layer) for layer in state)
    if layers != list(range(EXPECTED_LAYERS)):
        raise ValueError(f"Expected 48 layers numbered 0..47, got {layers}")
    for layer in layers:
        values = state[layer]
        for key in ("stability_score", "seap_score", "moe_residual_contribution"):
            if key not in values:
                raise ValueError(f"Layer {layer} is missing {key}")
        if values["stability_score"].numel() != EXPECTED_EXPERTS:
            raise ValueError(f"Layer {layer} does not contain 512 stability scores")
        if not torch.equal(torch.as_tensor(values["moe_residual_contribution"]), torch.as_tensor(calibration_state[layer]["moe_residual_contribution"])):
            raise ValueError(f"Layer {layer} importance is not from the final calibration state")
    return state


def _choose_indices(state, removals: dict[int, int], score_state=None) -> dict[int, list[int]]:
    retained = {}
    for layer in sorted(state):
        source = score_state if score_state is not None else state
        scores = source[layer]["score"].float() if score_state is not None else state[layer]["stability_score"].float()
        remove = removals[layer]
        # Stable tie-breaking by expert index makes materialization reproducible.
        order = sorted(range(scores.numel()), key=lambda i: (float(scores[i]), i))
        removed = set(order[:remove])
        retained[layer] = [i for i in range(scores.numel()) if i not in removed]
    return retained


def _bounded_reap_removals(
    layer_importance: dict[int, float],
    *,
    experts_per_layer: int,
    min_fraction: float = 0.25,
    max_fraction: float = 0.50,
) -> dict[int, int]:
    """Map inverse summed-REAP priorities into independent layer bounds."""
    if not 0 <= min_fraction <= max_fraction <= 1:
        raise ValueError("invalid bounded pruning fractions")
    priorities = {layer: 1.0 / max(value, 1e-12) for layer, value in layer_importance.items()}
    low, high = min(priorities.values()), max(priorities.values())
    span = high - low
    removals = {}
    for layer, priority in priorities.items():
        normalized = 0.0 if span == 0 else (priority - low) / span
        fraction = min_fraction + (max_fraction - min_fraction) * normalized
        removals[layer] = min(
            int(experts_per_layer * max_fraction),
            max(int(experts_per_layer * min_fraction), int(round(experts_per_layer * fraction))),
        )
    return removals


def _metadata(state, retained, removals, ratio, source_state, calibration_metadata, calibration_state, layer_importance, allocation_name, variant_name, min_retained_experts, pruning_bounds=None):
    return {
        "variant": variant_name,
        "method": "stability-only",
        "expert_selection": "stability_score",
        "source_state": str(source_state),
        "calibration_metadata": str(calibration_metadata),
        "calibration_state": str(calibration_state),
        "calibration_dataset": EXPECTED_DATASET,
        "stability_score_key": "stability_score (same values as seap_score)",
        "layer_importance_key": allocation_name,
        "allocation": f"inverse {allocation_name}, largest-remainder water-fill",
        "original_experts": EXPECTED_EXPERTS,
        "top_k": EXPECTED_TOP_K,
        "minimum_retained_experts": min_retained_experts,
        "pruning_bounds": pruning_bounds,
        "total_removed": sum(removals.values()),
        "total_retained": sum(len(x) for x in retained.values()),
        "layers": {
            str(layer): {
                "retained_expert_indices": indices,
                "removed_count": removals[layer],
                "retained_count": len(indices),
                "layer_importance": float(layer_importance[layer]),
            }
            for layer, indices in retained.items()
        },
        "routing": {
            "mask_before_softmax_and_topk": True,
            "padded_experts_receive_zero_probability": True,
            "redistribute_over_retained_experts_only": True,
        },
    }


def _pad_first_dim(value: torch.Tensor, count: int) -> torch.Tensor:
    if value.shape[0] == count:
        return value
    if value.shape[0] > count:
        raise ValueError("cannot pad tensor to a smaller expert count")
    return torch.cat((value, torch.zeros((count - value.shape[0], *value.shape[1:]), dtype=value.dtype, device=value.device)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_model", type=pathlib.Path)
    parser.add_argument("state", type=pathlib.Path)
    parser.add_argument("output_root", type=pathlib.Path)
    parser.add_argument("--ratio", type=float, choices=(0.25, 0.5), action="append", default=None)
    parser.add_argument("--calibration-metadata", type=pathlib.Path, default=None)
    parser.add_argument("--calibration-state", type=pathlib.Path, default=pathlib.Path("artifacts/Qwen3.8-Flash-Next-FP8/full-packed-2048/calibration_state.pt"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--layer-importance", choices=("moe_residual_contribution", "reap_sum"), default="moe_residual_contribution")
    parser.add_argument("--min-retained-experts", type=int, default=10)
    parser.add_argument("--bounded-reap", action="store_true", help="Use independent 25%%..50%% inverse-REAP pruning bounds per layer.")
    parser.add_argument("--uniform", action="store_true", help="Use an exact global removal count distributed uniformly across layers.")
    parser.add_argument("--expert-score", choices=("stability", "reap"), default="stability")
    parser.add_argument("--total-remove", type=int, default=None)
    args = parser.parse_args()

    calibration_metadata = args.calibration_metadata or args.state.parents[2] / "full-packed-2048" / "metadata.json"
    state = _load_and_verify_state(args.state, calibration_metadata, args.calibration_state)
    calibration_state = torch.load(args.calibration_state, map_location="cpu", weights_only=False)
    if args.layer_importance == "reap_sum" or args.bounded_reap or (args.uniform and args.expert_score == "reap"):
        layer_importance = {
            layer: float(torch.as_tensor(calibration_state[layer]["reap"]).float().sum())
            for layer in state
        }
        allocation_name = "sum of per-expert reap scores"
        variant_prefix = "stability-reapsum-nonuniform"
    else:
        layer_importance = {
            layer: float(torch.as_tensor(state[layer]["moe_residual_contribution"]).item())
            for layer in state
        }
        allocation_name = "moe_residual_contribution"
        variant_prefix = "stability-nonuniform"
    if args.uniform:
        if args.bounded_reap:
            raise ValueError("--uniform and --bounded-reap are mutually exclusive")
        total_remove = args.total_remove
        if total_remove is None:
            raise ValueError("--total-remove is required with --uniform")
        ratios = [total_remove / (EXPECTED_LAYERS * EXPECTED_EXPERTS)]
    else:
        ratios = [0.5] if args.bounded_reap else (args.ratio or [0.25, 0.5])
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), trust_remote_code=True, local_files_only=True)

    for ratio in ratios:
        if args.uniform:
            total_remove = args.total_remove
            base_remove, extra = divmod(total_remove, EXPECTED_LAYERS)
            removals = {layer: base_remove + int(index < extra) for index, layer in enumerate(sorted(state))}
            allocation_name = "uniform exact global removal count"
            variant_name = f"{args.expert_score}-uniform-{int(round(ratio * 10000))}bp"
            pruning_bounds = {"total_remove": total_remove, "per_layer_removal_counts": removals}
            if args.expert_score == "reap":
                score_state = {layer: {"score": calibration_state[layer]["reap"]} for layer in state}
                expert_selection = "raw per-expert reap score"
            else:
                score_state = None
                expert_selection = "stability_score"
            total_remove = sum(removals.values())
        elif args.bounded_reap:
            removals = _bounded_reap_removals(
                layer_importance,
                experts_per_layer=EXPECTED_EXPERTS,
                min_fraction=0.25,
                max_fraction=0.50,
            )
            total_remove = sum(removals.values())
            allocation_name = "bounded inverse sum of per-expert reap scores (25%-50% per layer)"
            variant_name = f"{args.expert_score}-reapsum-bounded-25to50"
            pruning_bounds = {"minimum_pruning_fraction": 0.25, "maximum_pruning_fraction": 0.50}
            if args.expert_score == "reap":
                score_state = {layer: {"score": calibration_state[layer]["reap"]} for layer in state}
                expert_selection = "raw per-expert reap score"
            else:
                score_state = None
                expert_selection = "stability_score"
        else:
            total_remove = int(EXPECTED_LAYERS * EXPECTED_EXPERTS * ratio)
            removals = allocate_layerwise_pruning_counts(
                state,
                total_remove,
                layer_importance=layer_importance,
                min_experts_per_layer=args.min_retained_experts,
            )
            allocation_name = allocation_name
            variant_name = f"{variant_prefix}{f'-min{args.min_retained_experts}' if args.min_retained_experts != 10 else ''}-{int(ratio * 100)}"
            pruning_bounds = None
        retained = _choose_indices(
            state,
            removals,
            score_state if (args.uniform or args.bounded_reap) else None,
        )
        if sum(removals.values()) != total_remove or min(map(len, retained.values())) < max(EXPECTED_TOP_K, args.min_retained_experts):
            raise AssertionError("invalid global allocation or fewer than top-k retained experts")
        meta = _metadata(state, retained, removals, ratio, args.state, calibration_metadata, args.calibration_state, layer_importance, allocation_name, variant_name, args.min_retained_experts, pruning_bounds)
        if args.uniform:
            meta["method"] = "uniform-global-budget"
            meta["expert_selection"] = expert_selection
        elif args.bounded_reap:
            meta["expert_selection"] = expert_selection
        variant = args.output_root / variant_name
        compact = variant / "compact"
        padded = variant / "padded"
        for path in (compact, padded):
            path.mkdir(parents=True, exist_ok=True)

        # Re-load the original weights for each ratio so no prior ratio can
        # contaminate the next checkpoint.
        model = AutoModelForCausalLM.from_pretrained(
            str(args.base_model), device_map={"": "cpu"}, torch_dtype="auto", trust_remote_code=True, local_files_only=True
        )
        for layer, indices in retained.items():
            moe = get_moe(model, layer)
            slice_fused_expert_weights(moe.experts, indices)
            moe.num_experts = len(indices)
            moe.gate.weight.data = moe.gate.weight.data[indices].clone()
            moe.gate.num_experts = len(indices)
            moe.gate.out_features = len(indices)
        # Compact is deliberately a self-describing state-dict artifact: its
        # layers have different expert dimensions and therefore must not claim
        # to be a uniform Transformers checkpoint.
        torch.save(model.state_dict(), compact / "model_state.pt")
        tokenizer.save_pretrained(compact)
        (compact / "nonuniform_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

        max_retained = max(map(len, retained.values()))
        for layer, indices in retained.items():
            moe = get_moe(model, layer)
            moe.experts.gate_up_proj.data = _pad_first_dim(moe.experts.gate_up_proj.data, max_retained)
            moe.experts.down_proj.data = _pad_first_dim(moe.experts.down_proj.data, max_retained)
            moe.experts.num_experts = max_retained
            moe.gate.weight.data = _pad_first_dim(moe.gate.weight.data, max_retained)
            moe.gate.weight.data[len(indices):] = 0
            moe.gate.num_experts = max_retained
            moe.gate.out_features = max_retained
        model.config.num_experts = max_retained
        if hasattr(model.config, "text_config"):
            model.config.text_config.num_experts = max_retained
        meta["padded_experts"] = max_retained
        meta["valid_expert_mask"] = {str(layer): [1] * len(indices) + [0] * (max_retained - len(indices)) for layer, indices in retained.items()}
        model.save_pretrained(padded)
        tokenizer.save_pretrained(padded)
        (padded / "nonuniform_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        if args.smoke_test:
            for layer, indices in retained.items():
                mask = torch.tensor(meta["valid_expert_mask"][str(layer)], dtype=torch.bool)
                logits = torch.randn(32, max_retained)
                assert_routing_invariants(logits, mask, EXPECTED_TOP_K)
        print(json.dumps({"variant": meta["variant"], "compact": str(compact), "padded": str(padded), "removed": meta["total_removed"], "min_retained": min(map(len, retained.values()))}))


if __name__ == "__main__":
    main()
