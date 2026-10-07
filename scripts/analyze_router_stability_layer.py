#!/usr/bin/env python3
"""Analyze per-expert router selection probability from router-stability events.

This script only reads the existing routing_events.parquet file.  The first
eight expert IDs in each stored condition are the selected top-k experts; the
remaining four are boundary experts and are intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args()


def load_metadata(path: Path | None, parquet: Path) -> dict:
    if path is None:
        candidate = parquet.with_name("metadata.json")
        path = candidate if candidate.exists() else None
    return json.loads(path.read_text()) if path else {}


def collect_counts(parquet: Path, layer: int, num_experts: int, top_k: int, batch_size: int):
    counts = np.zeros((5, num_experts), dtype=np.int64)
    tokens = 0
    table = pq.read_table(
        parquet,
        columns=["top12_expert_ids"],
        filters=[("layer", "=", layer)],
        use_threads=True,
    )
    for batch in table.to_batches(max_chunksize=batch_size):
        # Each batch is shaped [tokens, conditions, top12].  Converting one
        # batch at a time avoids materializing the multi-gigabyte parquet file.
        ids = np.asarray(batch.column(0).to_pylist(), dtype=np.int16)
        if ids.size == 0:
            continue
        selected = ids[:, :, :top_k]
        for condition in range(selected.shape[1]):
            counts[condition] += np.bincount(
                selected[:, condition, :].ravel(), minlength=num_experts
            )
        tokens += ids.shape[0]
    if tokens == 0:
        raise ValueError(f"No rows found for layer {layer}")
    return counts, tokens


def save_csv(output_dir: Path, probabilities: np.ndarray, deltas: np.ndarray, labels: list[str]):
    header = ["expert_idx"] + [f"prob_{label}" for label in labels] + [
        f"delta_vs_original_{label}" for label in labels[1:]
    ]
    values = np.column_stack([np.arange(probabilities.shape[1]), probabilities.T, deltas[1:].T])
    np.savetxt(
        output_dir / "expert_selection_probability.csv",
        values,
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt=["%d"] + ["%.9f"] * (values.shape[1] - 1),
    )


def plot_probability_heatmap(probabilities: np.ndarray, labels: list[str], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 3.8), constrained_layout=True)
    image = ax.imshow(probabilities, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set(yticks=np.arange(len(labels)), yticklabels=labels, xlabel="Expert index", ylabel="Router input")
    ax.set_xticks(np.arange(0, probabilities.shape[1], 8))
    ax.set_title("Layer 20: probability of being selected in top-8")
    fig.colorbar(image, ax=ax, label="Selection probability")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_probability_lines(probabilities: np.ndarray, labels: list[str], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    x = np.arange(probabilities.shape[1])
    for row, label in zip(probabilities, labels):
        ax.plot(x, row, linewidth=1.25, marker=".", markersize=2.5, label=label)
    ax.set(xlabel="Expert index", ylabel="Selection probability", xlim=(0, len(x) - 1))
    ax.set_title("Layer 20: expert selection probability by perturbation strength")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_deltas(deltas: np.ndarray, labels: list[str], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    x = np.arange(deltas.shape[1])
    max_abs = np.max(np.abs(deltas[1:]))
    for row, label in zip(deltas[1:], labels[1:]):
        ax.plot(x, row * 100, linewidth=1.1, marker=".", markersize=2.5, label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Expert index", ylabel="Change vs original (percentage points)", xlim=(0, len(x) - 1))
    ax.set_title("Layer 20: selection-probability change caused by router-input noise")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata, args.parquet)
    num_experts = int(metadata.get("num_experts", 128))
    top_k = int(metadata.get("top_k", 8))
    variances = metadata.get("variance_multipliers", [0.25, 0.5, 1.0, 2.0])
    labels = ["original"] + [f"variance {value:g}" for value in variances]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts, tokens = collect_counts(args.parquet, args.layer, num_experts, top_k, args.batch_size)
    # An expert can appear at most once per token.  This is therefore the
    # fraction of tokens on which each expert is selected, and probabilities
    # across experts sum to top_k (eight selected experts per token).
    probabilities = counts / tokens
    deltas = probabilities - probabilities[0]
    save_csv(args.output_dir, probabilities, deltas, labels)
    plot_probability_heatmap(probabilities, labels, args.output_dir / "expert_selection_probability_heatmap.png")
    plot_probability_lines(probabilities, labels, args.output_dir / "expert_selection_probability_lines.png")
    plot_deltas(deltas, labels, args.output_dir / "expert_selection_probability_delta.png")

    summary = {
        "layer": args.layer,
        "tokens": tokens,
        "top_k": top_k,
        "num_experts": num_experts,
        "conditions": labels,
        "mean_absolute_delta_vs_original": {
            label: float(np.mean(np.abs(deltas[index])))
            for index, label in enumerate(labels[1:], start=1)
        },
        "selection_probability_range_original": [float(probabilities[0].min()), float(probabilities[0].max())],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
