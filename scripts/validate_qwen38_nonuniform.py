"""Validate non-uniform allocation and padded routing metadata without loading weights."""

from __future__ import annotations

import argparse
import json
import pathlib

import torch

from reap.nonuniform_routing import assert_routing_invariants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", type=pathlib.Path)
    args = parser.parse_args()
    metadata = json.loads((args.variant / "padded" / "nonuniform_metadata.json").read_text())
    layers = metadata["layers"]
    assert len(layers) == 48
    assert metadata["total_removed"] in (6144, 12288)
    assert sum(x["removed_count"] for x in layers.values()) == metadata["total_removed"]
    assert min(x["retained_count"] for x in layers.values()) >= metadata["top_k"]
    padded = metadata["padded_experts"]
    for layer, values in layers.items():
        indices = values["retained_expert_indices"]
        assert len(indices) == values["retained_count"]
        assert len(metadata["valid_expert_mask"][layer]) == padded
        assert_routing_invariants(torch.randn(64, padded), torch.tensor(metadata["valid_expert_mask"][layer]), metadata["top_k"])
    print(json.dumps({"variant": metadata["variant"], "layers": 48, "removed": metadata["total_removed"], "padded_experts": padded, "routing": "ok"}, indent=2))


if __name__ == "__main__":
    main()
