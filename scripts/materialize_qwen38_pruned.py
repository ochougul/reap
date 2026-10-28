"""Materialize a Qwen3.8 25% pruned checkpoint from a saved observer state."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reap.prune import prune


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("base_model", type=Path)
    p.add_argument("state", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--method", choices=["reap", "seap"], required=True)
    p.add_argument("--compression-ratio", type=float, default=0.25)
    args = p.parse_args()

    observer_state = torch.load(args.state, map_location="cpu", weights_only=False)
    if not observer_state:
        raise ValueError("Observer state is empty")
    expert_counts = {
        int(layer): int(values["expert_frequency"].shape[0])
        for layer, values in observer_state.items()
    }
    if len(set(expert_counts.values())) != 1:
        raise ValueError(f"Expected one expert count across layers, got {expert_counts}")
    if not 0 <= args.compression_ratio < 1:
        raise ValueError("compression ratio must be in [0, 1)")
    total_experts = next(iter(expert_counts.values()))
    n_experts_to_prune = int(total_experts * args.compression_ratio)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        # Keep Transformers' FP8 expert-conversion temporaries off the GPUs;
        # the source checkpoint is ~173 GiB and host RAM is available.  vLLM
        # will place the resulting checkpoint on the H100s for evaluation.
        device_map={"": "cpu"},
        torch_dtype="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    prune_args = SimpleNamespace(
        prune_method=args.method,
        perserve_super_experts=False,
        perserve_outliers=False,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    prune(observer_state, model, prune_args, n_experts_to_prune, args.output)
    AutoTokenizer.from_pretrained(
        str(args.base_model), trust_remote_code=True, local_files_only=True
    ).save_pretrained(args.output)


if __name__ == "__main__":
    main()
