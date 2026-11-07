"""Runtime support for mask-aware routing in padded Qwen4-Exp vLLM models.

The patch is intentionally narrow: it masks the already-computed router
logits in ``MoERunner._apply_quant_method``. This is before both vLLM's
modular router and its monolithic fused kernels, and therefore before softmax,
top-k, and probability renormalization.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch


_LAYER_RE = re.compile(r"layers\.(\d+)\.mlp(?:\.experts)?")
_ORIGINAL_APPLY = None
_ORIGINAL_INIT = None


def _load_masks(path: str | os.PathLike[str]) -> dict[int, torch.Tensor]:
    metadata = json.loads(Path(path).read_text())
    if not metadata.get("routing", {}).get("mask_before_softmax_and_topk", False):
        raise ValueError(f"Metadata does not enable mask-aware routing: {path}")
    masks = metadata.get("valid_expert_mask")
    if not masks:
        raise ValueError(f"Missing valid_expert_mask in {path}")
    return {int(layer): torch.tensor(mask, dtype=torch.bool) for layer, mask in masks.items()}


def mask_router_logits(router_logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Mask padded experts without changing the logits' dtype or shape."""
    if router_logits.shape[-1] != valid_mask.numel():
        raise ValueError(
            f"Router has {router_logits.shape[-1]} experts but mask has "
            f"{valid_mask.numel()} entries"
        )
    mask = valid_mask if valid_mask.device == router_logits.device else valid_mask.to(device=router_logits.device)
    return router_logits.masked_fill(~mask, -torch.finfo(router_logits.dtype).max)


def install_vllm_qwen38_mask() -> None:
    """Install the patch when ``REAP_QWEN38_MASK_METADATA`` is set."""
    global _ORIGINAL_APPLY, _ORIGINAL_INIT
    if _ORIGINAL_APPLY is not None:
        return
    metadata_path = os.environ.get("REAP_QWEN38_MASK_METADATA")
    if not metadata_path:
        return

    masks = _load_masks(metadata_path)
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    _ORIGINAL_APPLY = MoERunner._apply_quant_method
    _ORIGINAL_INIT = MoERunner.__init__

    def masked_init(self, *args, **kwargs):
        _ORIGINAL_INIT(self, *args, **kwargs)
        match = _LAYER_RE.search(getattr(self, "layer_name", ""))
        if match is None:
            raise RuntimeError(
                "Mask-aware Qwen routing could not identify the decoder layer "
                f"from vLLM layer name {getattr(self, 'layer_name', '')!r}"
            )
        layer = int(match.group(1))
        if layer not in masks:
            raise RuntimeError(f"No valid-expert mask for Qwen layer {layer}")
        # Materialize before CUDA graph capture. Copying a CPU tensor during
        # capture is illegal, even for this small static mask.
        device = torch.device("cuda", torch.cuda.current_device())
        self._reap_valid_expert_mask = masks[layer].to(device=device)

    MoERunner.__init__ = masked_init

    def masked_apply(
        self,
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids=None,
        shared_experts_overlapping=False,
    ):
        match = _LAYER_RE.search(getattr(self, "layer_name", ""))
        if match is None:
            raise RuntimeError(
                "Mask-aware Qwen routing could not identify the decoder layer "
                f"from vLLM layer name {getattr(self, 'layer_name', '')!r}"
            )
        layer = int(match.group(1))
        valid_mask = getattr(self, "_reap_valid_expert_mask", masks.get(layer))
        if valid_mask is None:
            raise RuntimeError(f"No valid-expert mask for Qwen layer {layer}")
        router_logits = mask_router_logits(router_logits, valid_mask)
        return _ORIGINAL_APPLY(
            self,
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            shared_experts_overlapping,
        )

    MoERunner._apply_quant_method = masked_apply
