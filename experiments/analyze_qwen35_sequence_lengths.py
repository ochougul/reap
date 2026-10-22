#!/usr/bin/env python3
"""Measure exact Qwen3.5 calibration-sequence lengths for a chat dataset.

The calibration runner tokenizes each complete ``messages`` conversation after
Qwen's chat template.  This utility uses the same representation and writes
row and token retention figures for possible maximum lengths.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


DEFAULT_LIMITS = (512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096)


def describe(lengths: np.ndarray) -> dict[str, float | int]:
    percentiles = (1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9)
    result: dict[str, float | int] = {
        "rows": int(lengths.size),
        "tokens": int(lengths.sum()),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
    }
    for percentile, value in zip(percentiles, np.percentile(lengths, percentiles)):
        result[f"p{str(percentile).replace('.', '_')}"] = float(value)
    return result


def write_plot(rows: list[dict[str, float | int]], output_dir: Path) -> None:
    """Write a compact visual summary without making plotting a dependency of analysis."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    limits = [int(row["max_length"]) for row in rows]
    retained_rows = [float(row["retained_rows_pct"]) for row in rows]
    retained_tokens = [float(row["retained_tokens_pct"]) for row in rows]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(limits, retained_rows, marker="o", label="Retained rows")
    axis.plot(limits, retained_tokens, marker="o", label="Retained tokens")
    axis.set(xlabel="Maximum calibration-sequence length (tokens)", ylabel="Retained (%)")
    axis.set_ylim(0, 102)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "retention_by_max_length.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limits", type=int, nargs="+", default=DEFAULT_LIMITS)
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    dataset = load_dataset(args.dataset, split=args.split, revision=args.revision)

    calibration_lengths: list[int] = []
    user_prompt_lengths: list[int] = []
    total_rows = len(dataset)
    for start in range(0, total_rows, args.batch_size):
        rows = dataset[start : min(start + args.batch_size, total_rows)]
        # This matches ChatDatasetProcessor._encode_sample: template first,
        # then tokenization, including both user and assistant turns.
        encoded = tokenizer.apply_chat_template(
            rows["messages"], add_generation_prompt=False, tokenize=True,
            return_dict=True, padding=False,
        )
        calibration_lengths.extend(len(ids) for ids in encoded["input_ids"])
        user_prompt_lengths.extend(
            len(ids)
            for ids in tokenizer(rows["prompt"], add_special_tokens=False)["input_ids"]
        )
        if (start // args.batch_size) % 100 == 0 or start + args.batch_size >= total_rows:
            print(f"tokenized rows={min(start + args.batch_size, total_rows)}/{total_rows}", flush=True)

    calibration = np.asarray(calibration_lengths, dtype=np.int32)
    prompts = np.asarray(user_prompt_lengths, dtype=np.int32)
    np.savez_compressed(
        output_dir / "lengths.npz",
        calibration_sequence_tokens=calibration,
        user_prompt_tokens=prompts,
    )

    rows: list[dict[str, float | int]] = []
    total_tokens = int(calibration.sum())
    for limit in sorted(set(args.limits)):
        retained = calibration <= limit
        retained_rows = int(retained.sum())
        retained_tokens = int(calibration[retained].sum())
        dropped_rows = int((~retained).sum())
        dropped_tokens = total_tokens - retained_tokens
        rows.append(
            {
                "max_length": limit,
                "retained_rows": retained_rows,
                "retained_rows_pct": 100 * retained_rows / total_rows,
                "dropped_rows": dropped_rows,
                "dropped_rows_pct": 100 * dropped_rows / total_rows,
                "retained_tokens": retained_tokens,
                "retained_tokens_pct": 100 * retained_tokens / total_tokens,
                "dropped_tokens": dropped_tokens,
                "dropped_tokens_pct": 100 * dropped_tokens / total_tokens,
                "mean_retained_length": float(calibration[retained].mean()) if retained_rows else 0.0,
            }
        )

    with (output_dir / "retention_by_max_length.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_plot(rows, output_dir)
    summary = {
        "dataset": args.dataset,
        "revision": args.revision,
        "split": args.split,
        "model_path": str(args.model_path),
        "tokenizer": tokenizer.name_or_path,
        "definition": {
            "calibration_sequence_tokens": "Qwen chat template over all messages, then tokenized; this is the runner input.",
            "user_prompt_tokens": "Dataset prompt field tokenized without special tokens; reported for reference only.",
            "retention": "A row is retained when calibration_sequence_tokens <= max_length, matching SEAP_TRUNCATE=false.",
        },
        "calibration_sequence": describe(calibration),
        "user_prompt": describe(prompts),
        "retention_by_max_length": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
