#!/usr/bin/env python3
"""Write calibration statistics and plots for a SEAP experiment root."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def stable_ranks(values: np.ndarray) -> np.ndarray:
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[np.argsort(values, kind="stable")] = np.arange(values.size)
    return ranks


def fraction_overlap(left: np.ndarray, right: np.ndarray, fraction: float) -> tuple[int, int, float]:
    count = int(len(left) * fraction)
    left_ids = set(np.argsort(left, kind="stable")[:count].tolist())
    right_ids = set(np.argsort(right, kind="stable")[:count].tolist())
    overlap = len(left_ids & right_ids)
    return count, overlap, overlap / count if count else 1.0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def layer_statistics(state: dict, fractions: list[float], plots_dir: Path) -> list[dict]:
    rows = []
    for layer in sorted(state):
        values = state[layer]
        reap = values["reap_score"].float().cpu().numpy()
        normalized = values["normalized_reap"].float().cpu().numpy()
        stability_2 = values["stability_score_delta_2"].float().cpu().numpy()
        stability_4 = values["stability_score_delta_4"].float().cpu().numpy()
        seap = values["seap_score"].float().cpu().numpy()
        selected = values.get("expert_frequency")
        row = {
            "layer": int(layer),
            "num_experts": int(reap.size),
            "reap_min": float(reap.min()),
            "reap_max": float(reap.max()),
            "reap_mean": float(reap.mean()),
            "stability_delta_2_mean": float(stability_2.mean()),
            "stability_delta_4_mean": float(stability_4.mean()),
            "stability_delta_2_min": float(stability_2.min()),
            "stability_delta_2_max": float(stability_2.max()),
            "pearson_reap_stability_delta_2": correlation(normalized, stability_2),
            "spearman_reap_stability_delta_2": correlation(
                stable_ranks(normalized), stable_ranks(stability_2)
            ),
            "pearson_reap_stability_delta_4": correlation(normalized, stability_4),
            "spearman_reap_stability_delta_4": correlation(
                stable_ranks(normalized), stable_ranks(stability_4)
            ),
            "selected_expert_min": int(selected.min()) if selected is not None else None,
            "selected_expert_max": int(selected.max()) if selected is not None else None,
            "zero_selection_experts": int((selected == 0).sum()) if selected is not None else None,
        }
        for fraction in fractions:
            count, overlap, percentage = fraction_overlap(reap, seap, fraction)
            key = f"prune_{fraction:g}".replace(".", "_")
            row[f"{key}_count"] = count
            row[f"{key}_overlap_count"] = overlap
            row[f"{key}_overlap_fraction"] = percentage
        rows.append(row)

        figure, axis = plt.subplots(figsize=(13, 5), constrained_layout=True)
        experts = np.arange(reap.size)
        axis.plot(experts, normalized, label="normalized REAP", linewidth=1.2)
        axis.plot(experts, stability_2, label="stability δ=2", linewidth=1.2)
        axis.plot(experts, stability_4, label="stability δ=4", linewidth=1.2)
        axis.plot(experts, seap, label="default SEAP", linewidth=1.4, color="black")
        axis.set(
            title=f"Layer {layer}: REAP, stability, and default SEAP",
            xlabel="Routed expert index",
            ylabel="Normalized score",
            xlim=(0, reap.size - 1),
        )
        axis.grid(alpha=0.25)
        axis.legend(ncol=4)
        figure.savefig(plots_dir / f"layer_{int(layer):03d}_scores.png", dpi=200)
        plt.close(figure)
    return rows


def plot_calibration_summary(rows: list[dict], output: Path) -> None:
    layers = [row["layer"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    axes[0].plot(layers, [row["pearson_reap_stability_delta_2"] for row in rows], marker="o", label="Pearson")
    axes[0].plot(layers, [row["spearman_reap_stability_delta_2"] for row in rows], marker="o", label="Spearman")
    axes[0].set(title="REAP vs stability δ=2", xlabel="MoE layer", ylabel="Correlation", ylim=(-1.05, 1.05))
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(layers, [row["stability_delta_2_mean"] for row in rows], marker="o", label="δ=2")
    axes[1].plot(layers, [row["stability_delta_4_mean"] for row in rows], marker="o", label="δ=4")
    axes[1].set(title="Mean stability by layer", xlabel="MoE layer", ylabel="Mean stability", ylim=(0, 1.05))
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def sweep_statistics(root: Path, fractions: list[float]) -> list[dict]:
    rows = []
    for path in sorted((root / "sweeps").glob("delta-*/lambda-*/state.pt")):
        metadata = json.loads(path.with_name("metadata.json").read_text())
        state = torch.load(path, weights_only=False)
        for fraction in fractions:
            overlaps = []
            for layer in sorted(state):
                reap = state[layer]["reap_score"].float().cpu().numpy()
                seap = state[layer]["seap_score"].float().cpu().numpy()
                _, _, overlap = fraction_overlap(reap, seap, fraction)
                overlaps.append(overlap)
            rows.append(
                {
                    "delta": metadata["delta"],
                    "lambda": metadata["lambda"],
                    "prune_fraction": fraction,
                    "mean_overlap_fraction": float(np.mean(overlaps)),
                    "min_overlap_fraction": float(np.min(overlaps)),
                    "max_overlap_fraction": float(np.max(overlaps)),
                }
            )
    return rows


def plot_sweep_overlap(rows: list[dict], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for fraction in sorted({row["prune_fraction"] for row in rows}):
        for delta in sorted({row["delta"] for row in rows}):
            selected = sorted(
                (row for row in rows if row["prune_fraction"] == fraction and row["delta"] == delta),
                key=lambda row: row["lambda"],
            )
            if selected:
                axis.plot(
                    [row["lambda"] for row in selected],
                    [100 * row["mean_overlap_fraction"] for row in selected],
                    marker="o",
                    label=f"δ={delta:g}, prune={fraction:.0%}",
                )
    axis.set(
        title="Mean REAP/SEAP pruning-set overlap",
        xlabel="SEAP lambda (1 = REAP-only)",
        ylabel="Overlap with REAP (%)",
        ylim=(0, 105),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--prune-fractions", nargs="+", type=float, default=[0.25, 0.5])
    args = parser.parse_args()

    root = args.experiment_root
    state_path = args.state or root / "calibration" / "seap_state.pt"
    reports = root / "reports"
    calibration_plots = root / "plots" / "calibration"
    overlap_plots = root / "plots" / "overlap"
    reports.mkdir(parents=True, exist_ok=True)
    calibration_plots.mkdir(parents=True, exist_ok=True)
    overlap_plots.mkdir(parents=True, exist_ok=True)

    state = torch.load(state_path, weights_only=False)
    calibration_rows = layer_statistics(state, args.prune_fractions, calibration_plots)
    write_csv(reports / "calibration_statistics.csv", calibration_rows)
    (reports / "calibration_statistics.json").write_text(
        json.dumps({"state": str(state_path), "layers": calibration_rows}, indent=2) + "\n"
    )
    plot_calibration_summary(calibration_rows, calibration_plots / "summary.png")

    sweep_rows = sweep_statistics(root, args.prune_fractions)
    write_csv(reports / "pruning_overlap.csv", sweep_rows)
    (reports / "pruning_overlap.json").write_text(json.dumps(sweep_rows, indent=2) + "\n")
    if sweep_rows:
        plot_sweep_overlap(sweep_rows, overlap_plots / "lambda_delta_overlap.png")


if __name__ == "__main__":
    main()
