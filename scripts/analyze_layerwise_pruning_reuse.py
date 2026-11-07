"""Offline audit and allocation builder for layerwise pruning reuse.

This script deliberately loads calibration/evaluation states only.  It never
loads model weights or runs inference.  The checkpoint merger mirrors the
EP-state merge in ``experiments/qwen4_calibrate.py`` so intermediate states
are not accidentally interpreted as rank-0 complete states.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


LAYERS = 48
EXPERTS = 512
TOP_K = 10
DEFAULT_ROOT = Path("artifacts/Qwen3.8-Flash-Next-FP8/layerwise-reuse-v1")
DEFAULT_CALIBRATION = Path("artifacts/Qwen3.8-Flash-Next-FP8/full-packed-2048")
DEFAULT_REAP = Path("artifacts/Qwen3.8-Flash-Next-FP8/eval_states/reap/state.pt")
DEFAULT_MIXTURE = Path(
    "artifacts/Qwen3.8-Flash-Next-FP8/eval_states/lambda-sweep/delta-2/lambda-0.75/state.pt"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor(value: Any) -> torch.Tensor:
    return value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)


def _tracker_parts(value: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
    if hasattr(value, "mean") and hasattr(value, "count"):
        return tensor(value.mean), tensor(value.count)
    return None


def merge_states(states: list[dict[int, dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    """Merge rank-local calibration states using sums, weighted means, maxima."""
    if not states:
        raise ValueError("at least one state is required")
    layers = sorted(states[0])
    merged: dict[int, dict[str, Any]] = {}
    for layer in layers:
        keys = set().union(*(state[layer] for state in states))
        output: dict[str, Any] = {}
        for key in keys:
            values = [state[layer][key] for state in states if key in state[layer]]
            tracker_values = [_tracker_parts(value) for value in values]
            if values and all(parts is not None for parts in tracker_values):
                means = [parts[0].double() for parts in tracker_values if parts is not None]
                counts = [parts[1].double() for parts in tracker_values if parts is not None]
                count = torch.stack(counts).sum(0)
                weighted = torch.stack([mean * count_i for mean, count_i in zip(means, counts)]).sum(0)
                # Preserve the lightweight tracker interface used by saved states.
                first = values[0]
                first.mean = (weighted / count.clamp_min(1)).to(means[0].dtype)
                first.count = count.to(counts[0].dtype)
                if hasattr(first, "mean_compensation"):
                    first.mean_compensation = torch.zeros_like(first.mean)
                output[key] = first
                continue
            if all(isinstance(value, torch.Tensor) for value in values):
                values_t = [tensor(value) for value in values]
                if key == "max_activations":
                    output[key] = torch.stack(values_t).amax(0)
                else:
                    output[key] = torch.stack(values_t).sum(0)
            else:
                output[key] = values[0]
        merged[layer] = output
    return merged


def load_state(path: Path) -> dict[int, dict[str, Any]]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state" in state and isinstance(state["state"], dict):
        state = state["state"]
    return {int(layer): values for layer, values in state.items() if str(layer).lstrip("-").isdigit()}


def load_checkpoint_set(root: Path, packed_sequences: int) -> dict[int, dict[str, Any]]:
    # The calibration writer uses a nine-digit sequence field (e.g. 000010016).
    checkpoint_name = f"calibration_state_checkpoint_{packed_sequences:09d}.pt"
    paths = [root / checkpoint_name]
    paths += [root / f"rank{rank}" / paths[0].name for rank in range(1, 4)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing checkpoint ranks: {missing}")
    return merge_states([load_state(path) for path in paths])


def validate_shape(state: dict[int, dict[str, Any]], name: str) -> None:
    if sorted(state) != list(range(LAYERS)):
        raise ValueError(f"{name}: expected layers 0..47, got {sorted(state)}")
    for layer, values in state.items():
        for key in ("reap", "expert_frequency", "weighted_ean_sum"):
            if key not in values:
                raise KeyError(f"{name}: layer {layer} missing {key}")
            if tensor(values[key]).numel() != EXPERTS:
                raise ValueError(f"{name}: layer {layer} {key} is not length {EXPERTS}")


def uniform_counts(total: int) -> dict[int, int]:
    base, extra = divmod(total, LAYERS)
    return {layer: base + int(layer < extra) for layer in range(LAYERS)}


def score_and_mass(state: dict[int, dict[str, Any]], score_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, list[int]]]:
    scores = np.zeros((LAYERS, EXPERTS), dtype=np.float64)
    mass = np.zeros_like(scores)
    frequency = np.zeros_like(scores)
    orders: dict[int, list[int]] = {}
    for layer in range(LAYERS):
        values = state[layer]
        score = tensor(values[score_key]).double().numpy().reshape(-1)
        frequencies = tensor(values["expert_frequency"]).double().numpy().reshape(-1)
        total = float(tensor(values["total_tokens"]).item())
        contribution = tensor(values["weighted_ean_sum"]).double().numpy().reshape(-1) / total
        if not np.isfinite(score).all() or (score < 0).any():
            raise ValueError(f"layer {layer} has non-finite or negative {score_key}")
        if not np.isfinite(contribution).all() or (contribution < 0).any():
            raise ValueError(f"layer {layer} has non-finite or negative contribution proxy")
        if score.sum() <= 0 or contribution.sum() <= 0:
            raise ValueError(f"layer {layer} has a zero total")
        scores[layer] = score
        mass[layer] = contribution
        frequency[layer] = frequencies
        orders[layer] = sorted(range(EXPERTS), key=lambda expert: (float(score[expert]), expert))
    return scores, mass, frequency, orders


def curves(scores: np.ndarray, mass: np.ndarray, frequency: np.ndarray, orders: dict[int, list[int]]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name, values in (("score", scores), ("mass", mass), ("frequency", frequency)):
        sorted_values = np.stack([values[layer, orders[layer]] for layer in range(LAYERS)])
        output[f"sorted_{name}"] = sorted_values
        if name == "frequency":
            output["removed_frequency"] = np.concatenate(
                [np.zeros((LAYERS, 1)), np.cumsum(sorted_values, axis=1)], axis=1
            )
        else:
            total = values.sum(axis=1, keepdims=True)
            cumulative = np.concatenate([np.zeros((LAYERS, 1)), np.cumsum(sorted_values, axis=1)], axis=1)
            output[f"cumulative_{name}"] = cumulative / total
            output[f"increments_{name}"] = sorted_values / total
    return output


def allocate_dp(curve: np.ndarray, total: int, radius: int) -> dict[int, int]:
    uniform = uniform_counts(total)
    bounds = [(max(0, uniform[layer] - radius), min(EXPERTS - TOP_K, uniform[layer] + radius)) for layer in range(LAYERS)]
    inf = float("inf")
    dp = np.full(total + 1, inf)
    dp[0] = 0.0
    back: list[np.ndarray] = []
    for layer, (low, high) in enumerate(bounds):
        next_dp = np.full(total + 1, inf)
        previous = np.full(total + 1, -1, dtype=np.int16)
        # Vectorize each count transition.  Counts are still considered in
        # ascending order and strict replacement preserves the same stable tie
        # rule as the scalar recurrence.
        for count in range(low, high + 1):
            if count > total:
                break
            destinations = np.arange(count, total + 1)
            candidates = dp[: total + 1 - count] + curve[layer, count]
            valid = np.isfinite(candidates) & (candidates < next_dp[destinations] - 1e-12)
            next_dp[destinations[valid]] = candidates[valid]
            previous[destinations[valid]] = count
        dp, back = next_dp, back + [previous]
    if not math.isfinite(float(dp[total])):
        raise ValueError(f"no allocation for total={total} radius={radius}")
    counts = {}
    remaining = total
    for layer in reversed(range(LAYERS)):
        count = int(back[layer][remaining])
        if count < 0:
            raise AssertionError("allocation backtrace failed")
        counts[layer] = count
        remaining -= count
    if remaining or sum(counts.values()) != total:
        raise AssertionError("allocation total mismatch")
    return dict(sorted(counts.items()))


def allocate_heap_score(curve: np.ndarray, total: int, radius: int) -> dict[int, int]:
    """Greedy equivalent for sorted score tails; used as a solver check."""
    import heapq

    uniform = uniform_counts(total)
    lower = {layer: max(0, uniform[layer] - radius) for layer in range(LAYERS)}
    upper = {layer: min(EXPERTS - TOP_K, uniform[layer] + radius) for layer in range(LAYERS)}
    counts = dict(lower)
    heap: list[tuple[float, int]] = []
    for layer in range(LAYERS):
        if counts[layer] < upper[layer]:
            heapq.heappush(heap, (float(curve[layer, counts[layer] + 1] - curve[layer, counts[layer]]), layer))
    remaining = total - sum(counts.values())
    while remaining:
        if not heap:
            raise ValueError("heap allocator exhausted its feasible range")
        _, layer = heapq.heappop(heap)
        counts[layer] += 1
        remaining -= 1
        if counts[layer] < upper[layer]:
            heapq.heappush(heap, (float(curve[layer, counts[layer] + 1] - curve[layer, counts[layer]]), layer))
    return counts


def run_solver_self_checks() -> None:
    toy = np.zeros((LAYERS, EXPERTS + 1), dtype=np.float64)
    for layer in range(LAYERS):
        increments = np.full(EXPERTS, 1.0 + layer / 1000.0)
        if layer == 0:
            # The second expert is intentionally expensive.  A feasible
            # allocation should spend the budget in cheaper other layers.
            increments[1] = 100.0
        toy[layer, 1:] = np.cumsum(increments)
    dp = allocate_dp(toy, total=48, radius=64)
    heap = allocate_heap_score(toy, total=48, radius=64)
    if dp != heap or dp[0] > 1:
        raise AssertionError("score DP/heap or expensive-expert preservation check failed")


def validate_allocation(counts: dict[int, int], orders: dict[int, list[int]], total: int, radius: int) -> None:
    uniform = uniform_counts(total)
    if sum(counts.values()) != total:
        raise AssertionError("allocation does not meet exact global budget")
    for layer, count in counts.items():
        if not max(0, uniform[layer] - radius) <= count <= min(EXPERTS - TOP_K, uniform[layer] + radius):
            raise AssertionError(f"layer {layer} is outside its declared radius")
        if len(orders[layer]) != EXPERTS or len(set(orders[layer])) != EXPERTS:
            raise AssertionError(f"layer {layer} does not preserve original expert identity")
        if EXPERTS - count < TOP_K:
            raise AssertionError(f"layer {layer} violates top-k feasibility")


def allocation_payload(name: str, ranking: str, counts: dict[int, int], orders: dict[int, list[int]], total: int, radius: int, curve: np.ndarray, uniform: dict[int, int]) -> dict[str, Any]:
    retained = {str(layer): EXPERTS - count for layer, count in counts.items()}
    removed = {str(layer): orders[layer][:count] for layer, count in counts.items()}
    return {
        "variant": name,
        "ranking": ranking,
        "allocation": "uniform" if name.startswith("U-") else name,
        "total_removed": total,
        "actual_pruning_fraction": total / (LAYERS * EXPERTS),
        "radius": radius,
        "per_layer_removal_counts": {str(k): v for k, v in counts.items()},
        "uniform_removal_counts": {str(k): v for k, v in uniform.items()},
        "retained_counts": retained,
        "removed_expert_ids_original": removed,
        "retained_expert_ids_original": {str(layer): [i for i in range(EXPERTS) if i not in set(removed[str(layer)])] for layer in counts},
        "proxy_objective": float(sum(curve[layer, count] for layer, count in counts.items())),
        "tie_rule": "original expert ID for ranking; ascending count transitions and strict DP improvement for allocation",
        "top_k": TOP_K,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def segment_report(
    final_reference: dict[int, dict[str, Any]],
    first: dict[int, dict[str, Any]],
    second: dict[int, dict[str, Any]],
    final_segment: dict[int, dict[str, Any]],
    budgets: list[int],
) -> dict[str, Any]:
    segments = {
        "first_snapshot": first,
        "second_minus_first": second,
        "final_minus_second": final_segment,
    }
    report: dict[str, Any] = {"segments": {}, "note": "Segments are consistency diagnostics, not independent test sets."}
    _, _, _, final_order = score_and_mass(final_reference, "reap")
    for name, state in segments.items():
        _, _, _, order = score_and_mass(state, "reap")
        report["segments"][name] = {"total_tokens": [int(tensor(state[layer]["total_tokens"]).item()) for layer in range(LAYERS)], "boundary_agreement": {}}
        for budget in budgets:
            agreements = []
            for layer in range(LAYERS):
                count = uniform_counts(budget)[layer]
                segment_set = set(order[layer][:count])
                reference_set = set(final_order[layer][:count])
                agreements.append(len(segment_set & reference_set) / max(1, count))
            report["segments"][name]["boundary_agreement"][str(budget)] = {
                "removed_count": budget,
                "retained_set_agreement_vs_final_mean": float(np.mean(agreements)),
            }
        report["segments"][name]["layer_order_checksum"] = hashlib.sha256(
            json.dumps(order, separators=(",", ":")).encode()
        ).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--reap-state", type=Path, default=DEFAULT_REAP)
    parser.add_argument("--mixture-state", type=Path, default=DEFAULT_MIXTURE)
    parser.add_argument("--radius", type=int, default=64)
    parser.add_argument("--budgets", nargs="+", type=int, default=[9180, 9830, 11059, 12288])
    args = parser.parse_args()
    if args.radius < 0 or any(b < 0 or b > LAYERS * (EXPERTS - TOP_K) for b in args.budgets):
        raise ValueError("invalid radius or budget")
    run_solver_self_checks()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = args.calibration_root / "metadata.json"
    calibration_path = args.calibration_root / "calibration_state.pt"
    metadata = json.loads(metadata_path.read_text())
    final = load_state(calibration_path)
    reap_state = load_state(args.reap_state)
    mixture_state = load_state(args.mixture_state)
    validate_shape(final, "final calibration")
    validate_shape(reap_state, "REAP state")
    validate_shape(mixture_state, "mixture state")
    if any(not torch.equal(tensor(final[layer]["total_tokens"]), tensor(reap_state[layer]["total_tokens"])) for layer in range(LAYERS)):
        raise ValueError("REAP state is not aligned with final calibration")

    source_paths = [metadata_path, calibration_path, args.reap_state, args.mixture_state]
    source_paths += [args.calibration_root / f"calibration_state_checkpoint_{n:09d}.pt" for n in (10016, 20000)]
    source_paths += [args.calibration_root / f"rank{rank}" / f"calibration_state_checkpoint_{n:09d}.pt" for rank in range(1, 4) for n in (10016, 20000)]
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        diff = subprocess.run(["git", "diff", "--no-ext-diff"], capture_output=True, check=True).stdout
        git_diff_sha = hashlib.sha256(diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        git, git_diff_sha = "unavailable", "unavailable"
    manifest = {
        "schema_version": 1,
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "objective": "offline layerwise pruning allocation using reused full calibration",
        "model": metadata.get("model_path"),
        "calibration_metadata": metadata,
        "inputs": {str(path): {"sha256": sha256(path), "size": path.stat().st_size} for path in source_paths},
        "git": {"head": git, "working_tree_diff_sha256": git_diff_sha},
        "runtime": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__},
        "protocol_freeze": {"seed": 42, "top_k": TOP_K, "layers": LAYERS, "experts_per_layer": EXPERTS, "radius": args.radius, "budgets": args.budgets, "score_tie_break": "original expert ID"},
        "evaluation_status": "offline audit only; no model inference or accuracy result",
    }
    (root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Identity check and expert-level table use the authoritative final state.
    stats_rows = []
    identity_errors = []
    for layer in range(LAYERS):
        values = final[layer]
        total = float(tensor(values["total_tokens"]).item())
        reap = tensor(values["reap"]).double().numpy()
        freq = tensor(values["expert_frequency"]).double().numpy()
        weighted = tensor(values["weighted_ean_sum"]).double().numpy()
        expected = reap * freq / total
        relative = np.abs(weighted / total - expected) / np.maximum(np.abs(weighted / total), 1e-30)
        identity_errors.append(float(np.max(relative)))
        for expert in range(EXPERTS):
            stats_rows.append({"layer": layer, "expert": expert, "reap": float(reap[expert]), "expert_frequency": int(freq[expert]), "total_tokens": int(total), "contribution_proxy_c": float(weighted[expert] / total), "reap_frequency_over_total": float(expected[expert]), "identity_relative_error": float(relative[expert])})
    write_csv(root / "expert_statistics.csv", stats_rows, list(stats_rows[0]))
    (root / "input_audit.json").write_text(json.dumps({"layers": LAYERS, "experts_per_layer": EXPERTS, "total_tokens": [int(tensor(final[l]["total_tokens"]).item()) for l in range(LAYERS)], "positive_score_and_mass_totals": True, "reap_c_identity_max_relative_error": max(identity_errors), "pairwise_semantics": "f[i] + f[j]; not co-selection counts", "max_activations_semantics": "maximum activation element; not expert contribution norm"}, indent=2) + "\n")

    all_curves: dict[str, np.ndarray] = {}
    allocation_index = []
    for ranking, state, score_key in (("reap", reap_state, "reap"), ("mixture-lambda-0.75", mixture_state, "seap_score")):
        scores, mass, frequency, orders = score_and_mass(state, score_key)
        curve = curves(scores, mass, frequency, orders)
        all_curves[ranking.replace("-", "_") + "_order"] = np.asarray([orders[l] for l in range(LAYERS)], dtype=np.int16)
        for key, value in curve.items():
            all_curves[f"{ranking.replace('-', '_')}_{key}"] = value
        for budget in args.budgets:
            uniform = uniform_counts(budget)
            uniform_score = {layer: uniform[layer] for layer in range(LAYERS)}
            for objective, objective_curve, prefix in (("tail-score", curve["cumulative_score"], "T"), ("tail-mass", curve["cumulative_mass"], "P")):
                counts = allocate_dp(objective_curve, budget, args.radius)
                validate_allocation(counts, orders, budget, args.radius)
                if objective == "tail-score" and counts != allocate_heap_score(objective_curve, budget, args.radius):
                    raise AssertionError(f"tail-score DP and heap disagree for {ranking} R={budget}")
                payload = allocation_payload(f"{prefix}-{ranking}-R{budget}", ranking, counts, orders, budget, args.radius, objective_curve, uniform)
                payload["objective_name"] = objective
                variant_dir = root / "allocations" / payload["variant"]
                variant_dir.mkdir(parents=True, exist_ok=True)
                (variant_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
                allocation_index.append({"variant": payload["variant"], "ranking": ranking, "budget": budget, "objective": objective, "proxy_sum": payload["proxy_objective"], "uniform_proxy_sum": float(sum(objective_curve[layer, uniform[layer]] for layer in range(LAYERS))), "transferred_experts": int(sum(abs(counts[l] - uniform[l]) for l in range(LAYERS)) // 2), "retained_min": min(EXPERTS - count for count in counts.values()), "retained_max": max(EXPERTS - count for count in counts.values())})
            # Uniform itself is a first-class feasible control.
            payload = allocation_payload(f"U-{ranking}-R{budget}", ranking, uniform, orders, budget, args.radius, curve["cumulative_score"], uniform)
            payload["objective_name"] = "uniform-control"
            variant_dir = root / "allocations" / payload["variant"]
            variant_dir.mkdir(parents=True, exist_ok=True)
            (variant_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(root / "layer_removal_curves.npz", **all_curves)
    write_csv(root / "allocation_index.csv", allocation_index, list(allocation_index[0]))

    # Curves plot: the full one-expert resolution is kept in NPZ; the plot
    # highlights the requested operating points without hiding the tails.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    x = np.arange(EXPERTS + 1)
    for column, ranking in enumerate(("reap", "mixture_lambda_0.75")):
        for row, metric in enumerate(("score", "mass")):
            axis = axes[row, column]
            values = all_curves[f"{ranking}_cumulative_{metric}"]
            for layer in range(LAYERS):
                axis.plot(x, values[layer], alpha=0.18, linewidth=0.7)
            axis.axvline(512 * 0.37353515625, color="black", linestyle="--", linewidth=0.9)
            axis.axvline(512 * 0.45, color="tab:orange", linestyle=":", linewidth=0.9)
            axis.axvline(512 * 0.50, color="tab:red", linestyle=":", linewidth=0.9)
            axis.set_title(f"{ranking.replace('_', ' ')} — cumulative removed {metric}")
            axis.set_xlabel("experts removed in layer")
            axis.set_ylabel("fraction of original layer proxy")
            axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "layer_removal_curves.png", dpi=180)
    plt.close(fig)

    first = load_checkpoint_set(args.calibration_root, 10016)
    second_cumulative = load_checkpoint_set(args.calibration_root, 20000)
    # Difference only additive sufficient statistics; never difference means or maxima.
    second = {}
    for layer in range(LAYERS):
        second[layer] = {key: tensor(second_cumulative[layer][key]) - tensor(first[layer][key]) for key in ("total_tokens", "expert_frequency", "weighted_ean_sum")}
        second[layer]["reap"] = second[layer]["weighted_ean_sum"] / second[layer]["expert_frequency"].clamp_min(1)
    first_segment = {layer: {key: tensor(first[layer][key]) for key in ("total_tokens", "expert_frequency", "weighted_ean_sum")} for layer in range(LAYERS)}
    for layer in range(LAYERS):
        first_segment[layer]["reap"] = first_segment[layer]["weighted_ean_sum"] / first_segment[layer]["expert_frequency"].clamp_min(1)
    final_segment = {layer: {key: tensor(final[layer][key]) - tensor(second_cumulative[layer][key]) for key in ("total_tokens", "expert_frequency", "weighted_ean_sum")} for layer in range(LAYERS)}
    for layer in range(LAYERS):
        final_segment[layer]["reap"] = final_segment[layer]["weighted_ean_sum"] / final_segment[layer]["expert_frequency"].clamp_min(1)
    cumulative_counts = [int(tensor(first[l]["total_tokens"]).item()) for l in range(LAYERS)] + [int(tensor(second_cumulative[l]["total_tokens"]).item()) for l in range(LAYERS)] + [int(tensor(final[l]["total_tokens"]).item()) for l in range(LAYERS)]
    consistency = segment_report(final, first_segment, second, final_segment, args.budgets)
    consistency["checkpoint_token_counts_monotonic"] = all(a <= b for a, b in zip(cumulative_counts, cumulative_counts[1:]))
    consistency["checkpoint_counts"] = {"10016": cumulative_counts[:LAYERS], "20000": cumulative_counts[LAYERS:2 * LAYERS], "final": cumulative_counts[2 * LAYERS:]}
    (root / "segment_consistency.json").write_text(json.dumps(consistency, indent=2) + "\n")
    print(json.dumps({"output_root": str(root), "identity_max_relative_error": max(identity_errors), "allocations": len(allocation_index), "checkpoint_counts": consistency["checkpoint_counts"]["10016"][0]}, indent=2))


if __name__ == "__main__":
    main()
