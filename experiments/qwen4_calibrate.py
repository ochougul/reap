"""Single-pass Qwen4-Exp FP8 calibration on four GPUs.

The runner collects REAP, router stability, and layerwise MoE residual
contribution in one model forward pass.  Qwen4 stores experts in packed
projection tensors; its expert dispatcher calls ``_reap_capture_callback``
with the already-computed routed outputs, so calibration does not re-run
experts or materialize a ``[experts, tokens, hidden]`` activation tensor.
"""

from __future__ import annotations

import json
import hashlib
import os
import pathlib
import re
import time
from dataclasses import replace
from types import SimpleNamespace
from types import MethodType

import torch
import torch.nn.functional as F
from accelerate.hooks import remove_hook_from_module
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

from reap.data import load_category_batches
from reap.data import CodeAlpacaChatDataset, _load_raw_dataset
from reap.pruning_metrics import (
    initialize_pruning_state,
    update_pruning_state_from_routed_outputs,
)
from reap.router_stability import (
    RouterStabilityCollector,
    _route_once,
    collect_router_observation,
)


def _patch_triton_kernel_compat() -> None:
    """Bridge the Triton 3.4 symbol move used by the hub FP8 kernel."""
    import triton
    import triton.runtime.autotuner as autotuner
    import triton.runtime.cache as cache
    from triton.runtime.jit import JITFunction

    compatibility = {
        "JITFunction": JITFunction,
        "get_cache_manager": cache.get_cache_manager,
        "get_cache_invalidating_env_vars": lambda: {},
        "triton_key": lambda: triton.__version__,
    }
    for name, value in compatibility.items():
        if not hasattr(autotuner, name):
            setattr(autotuner, name, value)


def _patch_ep_tp_compat() -> None:
    """Skip TP validation/sharding for replicated scalar FP8 scale tensors."""
    from transformers.distributed.tensor_parallel import ColwiseParallel

    if getattr(ColwiseParallel, "_qwen4_ep_compat", False):
        return
    old_validate = ColwiseParallel.validate_param
    old_shard = ColwiseParallel.shard_param

    def validate(self, module, param, mesh, parameter_name=None):
        meta = module._parameters.get(param)
        if meta is not None and meta.ndim < 2:
            return
        return old_validate(self, module, param, mesh, parameter_name)

    def shard(self, module, param, mesh):
        meta = module._parameters.get(param)
        if meta is not None and meta.ndim < 2:
            return
        return old_shard(self, module, param, mesh)

    ColwiseParallel.validate_param = validate
    ColwiseParallel.shard_param = shard
    ColwiseParallel._qwen4_ep_compat = True


_QWEN4_CAPTURE_CONTEXT = None
_QWEN4_GROUPED_MM_ORIGINAL = None
_QWEN4_EP_CAPTURE_PATCHED = False
_QWEN4_DEEPGEMM_CAPTURE_PATCHED = False
_QWEN4_SENSITIVITY_ENABLED = False
_QWEN4_SENSITIVITY_CURRENT = {"routes": {}, "sensitivity": {}, "capture_mode": "ordinary"}


def _install_qwen4_capture_dispatch() -> None:
    """Expose packed FP8 expert outputs to the callback observer.

    Qwen4's Transformers dispatcher does not expose a callback.  Its grouped
    FP8 implementation performs two grouped matmuls; the result of the second
    one is one row per routed token/expert pair, before routing-weight combine.
    Intercepting that operation preserves the fused kernel path and avoids a
    second expert evaluation.
    """
    global _QWEN4_GROUPED_MM_ORIGINAL
    if _QWEN4_GROUPED_MM_ORIGINAL is not None:
        return
    from transformers.integrations import finegrained_fp8

    # Transformers 5.x stores the loaded kernels in a FineGrainedFP8 bundle;
    # `grouped_matmul` is not a module-level symbol.  Load the bundle once,
    # then replace only its grouped entry with a callback-preserving wrapper.
    bundle = finegrained_fp8.load_finegrained_fp8_kernel()
    if bundle is None:
        bundle = finegrained_fp8._FINEGRAINED_FP8
    _QWEN4_GROUPED_MM_ORIGINAL = bundle.grouped_matmul

    def capture_grouped_matmul(*args, **kwargs):
        output = _QWEN4_GROUPED_MM_ORIGINAL(*args, **kwargs)
        context = _QWEN4_CAPTURE_CONTEXT
        if context is not None:
            if context.calls == 1:
                expert_ids = context.expert_ids
                perm = context.perm
                if hasattr(context, "ep_expert_ids"):
                    expert_ids, local_perm = torch.sort(context.ep_expert_ids)
                    token_ids = context.ep_token_ids[local_perm]
                    expert_ids = expert_ids + torch.distributed.get_rank() * context.ep_local_experts
                    perm = token_ids * context.num_top_k
                context.callback(
                    expert_ids,
                    output,
                    perm,
                    context.num_top_k,
                )
            context.calls += 1
        return output

    finegrained_fp8._FINEGRAINED_FP8 = replace(
        bundle, grouped_matmul=capture_grouped_matmul
    )


def _install_qwen4_ep_capture_dispatch() -> None:
    """Expose the EP all-to-all token mapping to the grouped-MM observer."""
    global _QWEN4_EP_CAPTURE_PATCHED
    if _QWEN4_EP_CAPTURE_PATCHED:
        return
    from transformers.distributed.tensor_parallel import EpDispatchExpertsParallel

    original_dispatch = EpDispatchExpertsParallel._dispatch_tokens

    def dispatch(self, hidden_states, top_k_index, num_local_experts, ep_group, ep_size):
        result = original_dispatch(
            self, hidden_states, top_k_index, num_local_experts, ep_group, ep_size
        )
        context = _QWEN4_CAPTURE_CONTEXT
        if context is not None:
            _, _, order, send_sizes, recv_sizes = result
            local_tokens = hidden_states.shape[0]
            token_ids = order.reshape(-1) // top_k_index.shape[-1]
            # The TP transform gives each rank a contiguous token shard.
            token_ids = token_ids + torch.distributed.get_rank() * local_tokens
            received = torch.empty(
                sum(recv_sizes), device=token_ids.device, dtype=token_ids.dtype
            )
            torch.distributed.all_to_all_single(
                received, token_ids, recv_sizes, send_sizes, group=ep_group
            )
            context.ep_expert_ids = result[1].reshape(-1)
            context.ep_token_ids = received
            context.ep_local_experts = num_local_experts
        return result

    EpDispatchExpertsParallel._dispatch_tokens = dispatch
    _QWEN4_EP_CAPTURE_PATCHED = True


def _install_qwen4_deepgemm_capture_dispatch() -> None:
    """Capture DeepGEMM rows immediately before its EP combine step."""
    global _QWEN4_DEEPGEMM_CAPTURE_PATCHED
    if _QWEN4_DEEPGEMM_CAPTURE_PATCHED:
        return
    from transformers.integrations import deepgemm

    original_dispatch = deepgemm._dispatch_routed_input
    original_combine = deepgemm._combine_routed_output

    def dispatch(*args, **kwargs):
        result = original_dispatch(*args, **kwargs)
        context = _QWEN4_CAPTURE_CONTEXT
        if context is not None:
            context.deep_expert_ids = result[2]
            context.deep_perm = result[4]
        return result

    def combine(out_padded, sorted_weights, sentinel_mask, perm, sorted_to_padded,
                num_tokens, num_top_k, hidden_dim, out_dtype):
        context = _QWEN4_CAPTURE_CONTEXT
        if context is not None and hasattr(context, "deep_expert_ids"):
            out = deepgemm._unpad_from_deepgemm_contiguous_layout(
                out_padded, sorted_to_padded
            )
            if out.shape[0] == context.expert_ids.numel():
                # DeepGEMM may receive the original top-k route directly;
                # its unpadded rows then align with the already-sorted route
                # captured by the expert wrapper.
                expert_ids = context.expert_ids
                perm_for_callback = context.perm
            else:
                token_ids = context.ep_token_ids[context.deep_perm]
                expert_ids = context.deep_expert_ids
                if expert_ids.numel() != out.shape[0] or token_ids.numel() != out.shape[0]:
                    raise RuntimeError(
                        "DeepGEMM EP observer layout changed: cannot safely align "
                        f"expert rows ({expert_ids.numel()}), token rows "
                        f"({token_ids.numel()}), and outputs ({out.shape[0]}). "
                        "Use grouped_mm for callback-based calibration."
                    )
                expert_ids = expert_ids + torch.distributed.get_rank() * context.ep_local_experts
                perm_for_callback = token_ids * context.num_top_k
            context.callback(
                expert_ids,
                out,
                perm_for_callback,
                context.num_top_k,
            )
        return original_combine(
            out_padded, sorted_weights, sentinel_mask, perm, sorted_to_padded,
            num_tokens, num_top_k, hidden_dim, out_dtype,
        )

    deepgemm._dispatch_routed_input = dispatch
    deepgemm._combine_routed_output = combine
    _QWEN4_DEEPGEMM_CAPTURE_PATCHED = True


def _wrap_qwen4_expert_forward(
    experts: torch.nn.Module,
    layer: int | None = None,
    router: torch.nn.Module | None = None,
) -> None:
    original_forward = experts.forward

    def wrapped_forward(*args, **kwargs):
        global _QWEN4_CAPTURE_CONTEXT
        callback = getattr(experts, "_reap_capture_callback", None)
        if callback is None or len(args) < 3:
            return original_forward(*args, **kwargs)
        top_k_index = args[1]
        expert_ids, perm = torch.sort(top_k_index.reshape(-1))
        previous = _QWEN4_CAPTURE_CONTEXT
        try:
            if not _QWEN4_SENSITIVITY_ENABLED:
                _QWEN4_CAPTURE_CONTEXT = SimpleNamespace(
                    callback=callback,
                    expert_ids=expert_ids,
                    perm=perm,
                    num_top_k=top_k_index.shape[-1],
                    calls=0,
                )
                return original_forward(*args, **kwargs)

            if layer is None:
                raise RuntimeError("Output sensitivity capture requires a layer-indexed expert call")
            route = _QWEN4_SENSITIVITY_CURRENT["routes"].get(layer)
            if route is None:
                raise RuntimeError(f"Missing router result for output-sensitivity layer {layer}")
            logits = route[0].reshape(-1, route[0].shape[-1]).float()
            selected = route[4].reshape(-1, route[4].shape[-1])
            if selected.shape != (logits.shape[0], 10):
                raise RuntimeError(
                    f"Expected the actual router to select ten experts, got {tuple(selected.shape)}"
                )
            # Match the gate's exact score domain (softmax probabilities), then
            # append the highest-probability expert excluded from its actual
            # top-10. This also remains well-defined if softmax rounding creates
            # ties that differ from a top-k directly on lower-precision logits.
            all_probs = logits.softmax(dim=-1)
            excluded_probs = all_probs.clone()
            excluded_probs.scatter_(1, selected, -torch.inf)
            eleventh_id = excluded_probs.argmax(dim=-1, keepdim=True)
            if (all_probs.gather(1, eleventh_id) > all_probs.gather(1, selected).min(-1, keepdim=True).values).any():
                raise RuntimeError("The appended candidate outranks a selected top-10 expert")
            top11 = torch.cat((selected, eleventh_id), dim=-1)
            probs11 = all_probs.gather(1, top11)
            probs11 = probs11 / probs11.sum(dim=-1, keepdim=True).clamp_min(1e-20)
            selected_weights = route[5].reshape_as(selected).float()
            if router is None:
                raise RuntimeError("Output sensitivity capture requires the layer router")
            selected_probs = all_probs.gather(1, selected)
            if getattr(router, "norm_topk_prob", False):
                routed_scale = selected_weights.sum(-1, keepdim=True)
                expected_weights = routed_scale * selected_probs / selected_probs.sum(
                    -1, keepdim=True
                ).clamp_min(1e-20)
            else:
                routed_scale = (
                    (selected_weights * selected_probs).sum(-1, keepdim=True)
                    / selected_probs.square().sum(-1, keepdim=True).clamp_min(1e-20)
                )
                expected_weights = routed_scale * selected_probs
            if not torch.allclose(
                selected_weights, expected_weights, rtol=1e-2, atol=5e-4
            ):
                max_error = float((selected_weights - expected_weights).abs().max())
                raise RuntimeError(
                    "Captured top-10 router weights disagree with softmax and the "
                    "router's norm_topk_prob/scaling settings "
                    f"(max_abs_error={max_error:.6g})"
                )
            extra_ids = top11[:, -1:].to(device=top_k_index.device)
            extra_weights = torch.ones(
                extra_ids.shape,
                dtype=args[2].dtype,
                device=args[2].device,
            )
            current = _QWEN4_SENSITIVITY_CURRENT
            current["sensitivity"][layer] = {
                "candidate_ids": top11.detach(),
                "candidate_probs": probs11.detach(),
                "selected_ids": selected.detach(),
                "selected_weights": selected_weights.detach(),
                "routed_scale": routed_scale.reshape(-1).detach(),
                "eleventh_output": torch.empty(
                    (top11.shape[0], args[0].shape[-1]),
                    device=extra_ids.device,
                    dtype=args[0].dtype,
                ),
                "eleventh_seen": torch.zeros(
                    top11.shape[0], device=top11.device, dtype=torch.bool
                ),
            }

            # Evaluate the replacement expert separately, but do not add its
            # result to the model output. The baseline remains exactly top-10.
            sorted_extra_ids, extra_perm = torch.sort(extra_ids.reshape(-1))
            _QWEN4_CAPTURE_CONTEXT = SimpleNamespace(
                callback=callback,
                expert_ids=sorted_extra_ids,
                perm=extra_perm,
                num_top_k=1,
                calls=0,
            )
            current["capture_mode"] = "eleventh"
            original_forward(
                args[0], extra_ids, extra_weights, *args[3:], **kwargs
            )
            payload = current["sensitivity"].get(layer)
            if payload is None or not payload["eleventh_seen"].all():
                raise RuntimeError(f"Layer {layer} did not capture all candidate-11 outputs")

            _QWEN4_CAPTURE_CONTEXT = SimpleNamespace(
                callback=callback,
                expert_ids=expert_ids,
                perm=perm,
                num_top_k=top_k_index.shape[-1],
                calls=0,
            )
            current["capture_mode"] = "ordinary"
            return original_forward(*args, **kwargs)
        finally:
            if _QWEN4_SENSITIVITY_ENABLED:
                _QWEN4_SENSITIVITY_CURRENT["capture_mode"] = "ordinary"
            _QWEN4_CAPTURE_CONTEXT = previous

    experts.forward = wrapped_forward


def _wrap_qwen4_router_forward(router: torch.nn.Module, layer: int, current) -> None:
    """Record the router result that the model already computed.

    The previous calibration pre-hook called the router a second time just to
    reconstruct the original route.  Qwen4 returns logits, weights, and ids
    from the gate, so retain that exact result for the observer and remove one
    512-way router GEMM per MoE layer.
    """
    original_forward = router.forward

    def wrapped_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        if isinstance(result, tuple) and len(result) >= 3:
            logits, weights, selected = result[:3]
            if torch.is_tensor(logits) and torch.is_tensor(selected):
                selected = selected.reshape(-1, selected.shape[-1])
                current["routes"][layer] = (
                    logits.reshape(-1, logits.shape[-1]),
                    logits.reshape(-1, logits.shape[-1]).float(),
                    selected,
                    logits.reshape(-1, logits.shape[-1]).float().gather(1, selected),
                    selected,
                    weights.reshape(-1, weights.shape[-1]).float()
                    * getattr(router, "routed_scaling_factor", 1.0),
                )
        return result

    router.forward = wrapped_forward


MODEL_PATH = os.environ.get(
    "QWEN4_MODEL_PATH",
    "/home/ubuntu/hf_cache/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/"
    "236dfdf285828023ca3bcd3f37366c58a3469b13",
)
SENSITIVITY_SMOKE_SAMPLES = int(os.environ.get("QWEN4_SENSITIVITY_SMOKE_SAMPLES", "0"))
SENSITIVITY_MODE = SENSITIVITY_SMOKE_SAMPLES > 0 or os.environ.get(
    "QWEN4_SENSITIVITY_MODE", "0"
).lower() in {
    "1", "true", "yes"
}
if SENSITIVITY_SMOKE_SAMPLES < 0 or SENSITIVITY_SMOKE_SAMPLES > 12228:
    raise ValueError("QWEN4_SENSITIVITY_SMOKE_SAMPLES must be in [0, 12228]")
OUTPUT_DIR = pathlib.Path(
    os.environ.get(
        "QWEN4_OUTPUT_DIR",
        (
            "artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-smoke-evol-codealpaca-"
            f"{SENSITIVITY_SMOKE_SAMPLES}-16384-unpruned"
            if SENSITIVITY_SMOKE_SAMPLES
            else "artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-12228-16384-unpruned"
            if SENSITIVITY_MODE
            else "artifacts/Qwen3.8-Flash-Next-FP8/evol-codealpaca"
        ),
    )
)
DATASET_NAME = os.environ.get(
    "QWEN4_DATASET_NAME", "theblackcat102/evol-codealpaca-v1"
)
DATASET_SPLIT = os.environ.get("QWEN4_DATASET_SPLIT", "train")
BATCH_SIZE = 1 if SENSITIVITY_MODE else int(os.environ.get("QWEN4_BATCH_SIZE", "32"))
FORWARD_BATCH_SIZE = int(
    "1" if SENSITIVITY_MODE else os.environ.get("QWEN4_FORWARD_BATCH_SIZE", str(min(BATCH_SIZE, 4)))
)
MODEL_MAX_LENGTH = 16384 if SENSITIVITY_MODE else int(os.environ.get("QWEN4_MODEL_MAX_LENGTH", "2048"))
# GLM-4.5-Air's full-dataset calibration uses unpacked rows with
# truncate=False, which filters overlong rows. Keep packed mode as the
# historical Qwen default, but allow an apples-to-apples comparison.
PACK_SAMPLES = False if SENSITIVITY_MODE else os.environ.get("QWEN4_PACK_SAMPLES", "1").lower() in {
    "1", "true", "yes"
}
# The data loader's default packed mode fills each calibration sequence to
# model_max_length from multiple source rows. Therefore this value counts
# packed calibration sequences, not raw dataset rows. Use
# QWEN4_MAX_SAMPLES=100 for the smoke test or set it explicitly for a larger
# study; the source dataset contains 111k rows.
MAX_SAMPLES = (
    SENSITIVITY_SMOKE_SAMPLES
    if SENSITIVITY_SMOKE_SAMPLES
    else 12228 if SENSITIVITY_MODE
    else int(os.environ.get("QWEN4_MAX_SAMPLES", "1000"))
)
SEED = 42 if SENSITIVITY_MODE else int(os.environ.get("QWEN4_SEED", "42"))
ROUTER_MAX_TOKENS = int(os.environ.get("QWEN4_ROUTER_MAX_TOKENS", "0")) or None
SAVE_ROUTING_EVENTS = os.environ.get("QWEN4_SAVE_ROUTING_EVENTS", "0").lower() in {
    "1", "true", "yes"
}
CHECKPOINT_EVERY = 1024 if SENSITIVITY_MODE else int(os.environ.get("QWEN4_CHECKPOINT_EVERY", "10000"))
_EP_REQUESTED = os.environ.get("QWEN4_EXPERT_PARALLEL", "0").lower() in {
    "1", "true", "yes"
}
EXPERTS_IMPLEMENTATION = os.environ.get(
    "QWEN4_EXPERTS_IMPLEMENTATION",
    "deepgemm" if _EP_REQUESTED else "grouped_mm",
)
ATTN_IMPLEMENTATION = os.environ.get("QWEN4_ATTN_IMPLEMENTATION", "") or None
if os.environ.get("QWEN4_DISABLE_CUDNN_SDP", "0").lower() in {"1", "true", "yes"}:
    # Torch 2.10 may select an invalid cuDNN SDPA plan with the host cuDNN
    # runtime.  Leave flash/memory-efficient SDPA available as alternatives.
    torch.backends.cuda.enable_cudnn_sdp(False)
VARIANCE_MULTIPLIERS = tuple(
    float(value.strip())
    for value in os.environ.get("QWEN4_VARIANCE_MULTIPLIERS", "1.0").split(",")
    if value.strip()
)
if SENSITIVITY_MODE:
    VARIANCE_MULTIPLIERS = (1.0,)
if not VARIANCE_MULTIPLIERS:
    raise ValueError("QWEN4_VARIANCE_MULTIPLIERS must contain at least one value")
if FORWARD_BATCH_SIZE < 1:
    raise ValueError("QWEN4_FORWARD_BATCH_SIZE must be positive")


def _layer_index(name: str) -> int | None:
    match = re.search(r"(?:model\.)?(?:language_model\.)?layers\.(\d+)(?:\.mlp)?$", name)
    return int(match.group(1)) if match else None


def _num_experts(router: torch.nn.Module) -> int:
    for name in ("num_experts", "n_routed_experts"):
        value = getattr(router, name, None)
        if value is not None:
            return int(value)
    return int(router.weight.shape[0])


def _prepare_sensitivity_samples(tokenizer) -> tuple[list[dict], dict]:
    """Select deterministic, unpadded eligible source rows for the sensitivity run."""
    import numpy as np

    if DATASET_NAME != "theblackcat102/evol-codealpaca-v1" or DATASET_SPLIT != "train":
        raise RuntimeError("Sensitivity mode is fixed to Evol-CodeAlpaca's train split")
    raw = _load_raw_dataset(DATASET_NAME, DATASET_SPLIT, subset=None)
    processor = CodeAlpacaChatDataset(
        dataset=raw,
        tokenizer=tokenizer,
        max_input_len=MODEL_MAX_LENGTH,
        pack_samples=False,
        split=DATASET_SPLIT,
        split_by_category=False,
        return_vllm_tokens_prompt=False,
        truncate=False,
        batch_size=1,
    )
    permutation = np.random.default_rng(SEED).permutation(len(raw))
    samples: list[dict] = []
    selected_indices: list[int] = []
    token_lengths: list[int] = []
    token_hashes: list[str] = []
    for row_index in permutation:
        encoded = processor._encode_sample(processor._map_fn(raw[int(row_index)]))
        length = int(encoded.shape[-1])
        if length > MODEL_MAX_LENGTH:
            continue
        input_ids = encoded.reshape(1, -1).to(dtype=torch.long)
        id_bytes = input_ids[0].cpu().numpy().astype("<i8", copy=False).tobytes()
        token_hash = hashlib.sha256(id_bytes).hexdigest()
        samples.append({
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        })
        selected_indices.append(int(row_index))
        token_lengths.append(length)
        token_hashes.append(token_hash)
        if len(samples) == MAX_SAMPLES:
            break
    if len(samples) != MAX_SAMPLES:
        raise RuntimeError(
            f"Only {len(samples)} eligible examples found; need exactly {MAX_SAMPLES}"
        )
    if len(set(selected_indices)) != MAX_SAMPLES:
        raise RuntimeError("Sensitivity sample selection contains duplicate source rows")
    if any(length < 2 or length > MODEL_MAX_LENGTH for length in token_lengths):
        raise RuntimeError("Sensitivity selection contains a sequence outside [2, 16384]")

    manifest_payload = {
        "dataset_name": DATASET_NAME,
        "split": DATASET_SPLIT,
        "seed": SEED,
        "max_length": MODEL_MAX_LENGTH,
        "selected_row_indices": selected_indices,
        "token_lengths": token_lengths,
        "token_id_sha256": token_hashes,
    }
    sample_hash = hashlib.sha256(
        json.dumps(manifest_payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        **manifest_payload,
        "selected_examples": len(samples),
        "sample_set_sha256": sample_hash,
        "selection": "numpy.default_rng(seed).permutation; accept rows in order, skip rows longer than max_length",
        "padding": False,
        "truncation": False,
        "packing": False,
    }
    return samples, manifest


def _top11_local_output_delta_squares(
    *,
    candidate_probs: torch.Tensor,
    selected_ids: torch.Tensor,
    selected_weights: torch.Tensor,
    routed_scale: float | torch.Tensor,
    eleventh_output: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_outputs: torch.Tensor,
    route_permutation: torch.Tensor,
    num_experts: int,
    chunk_tokens: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-expert sum(||delta MoE output||^2) and occurrence counts.

    The callback outputs are sorted by expert. `route_permutation` maps each
    output row back to the flattened original [token, top-k slot] ordering.
    Only one top-k slot is materialized at a time to bound peak memory.
    """
    if candidate_probs.ndim != 2 or candidate_probs.shape[-1] != 11:
        raise ValueError("Expected normalized probabilities for 11 router candidates")
    tokens = candidate_probs.shape[0]
    if selected_ids.shape != (tokens, 10) or selected_weights.shape != (tokens, 10):
        raise ValueError("Expected ten selected expert IDs and weights per token")
    if eleventh_output.ndim != 2 or eleventh_output.shape[0] != tokens:
        raise ValueError("Candidate-11 outputs must have shape [tokens, hidden]")
    if expert_outputs.ndim != 2 or expert_outputs.shape[1] != eleventh_output.shape[1]:
        raise ValueError("Routed outputs and candidate-11 outputs must share hidden size")
    if expert_ids.numel() != tokens * 10 or route_permutation.numel() != tokens * 10:
        raise ValueError("Expected one routed output row per top-10 token/expert pair")

    device = expert_outputs.device
    token_rows = route_permutation.reshape(-1).long() // 10
    slot_rows = route_permutation.reshape(-1).long() % 10
    expert_ids = expert_ids.reshape(-1).long()
    if token_rows.min() < 0 or token_rows.max() >= tokens:
        raise ValueError("Routed-output permutation refers to an invalid token")
    if not torch.equal(selected_ids[token_rows, slot_rows], expert_ids):
        raise RuntimeError("Routed output rows do not align with the captured top-10 route")
    if (expert_ids >= num_experts).any():
        raise RuntimeError("Sensitivity mode does not support EP sentinel expert rows")

    probabilities = candidate_probs.float()
    probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-20)
    weights = selected_weights.to(device=device, dtype=torch.float32)
    scale = torch.as_tensor(routed_scale, device=device, dtype=torch.float32).reshape(-1)
    if scale.numel() == 1:
        scale = scale.expand(tokens)
    if scale.shape != (tokens,) or (scale <= 0).any():
        raise RuntimeError("Router produced an invalid routed scaling factor")
    if (probabilities[:, :10].sum(-1) <= 0).any():
        raise RuntimeError("Router produced a non-positive selected-route mass")

    # Reconstruct the unchanged top-10 mixture once from the actual routed
    # outputs and weights. Then use the top-11 probabilities to replace each
    # selected expert, preserving the original total router scale.
    baseline = torch.zeros(
        (tokens, expert_outputs.shape[-1]), device=device, dtype=torch.float32
    )
    baseline.index_add_(
        0,
        token_rows,
        expert_outputs.float() * weights[token_rows, slot_rows, None],
    )
    probability_weighted_sum = torch.zeros_like(baseline)
    probability_weighted_sum.index_add_(
        0,
        token_rows,
        expert_outputs.float() * probabilities[token_rows, slot_rows, None],
    )
    rows_by_slot = torch.full((tokens, 10), -1, device=device, dtype=torch.long)
    rows_by_slot[token_rows, slot_rows] = torch.arange(tokens * 10, device=device)
    if (rows_by_slot < 0).any():
        raise RuntimeError("Missing routed expert output for a token/top-k slot")

    replacement = eleventh_output.float()
    replacement_prob = probabilities[:, 10]
    squared_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
    occurrence_count = torch.zeros(num_experts, device=device, dtype=torch.long)
    for slot in range(10):
        for start in range(0, tokens, chunk_tokens):
            stop = min(start + chunk_tokens, tokens)
            rows = rows_by_slot[start:stop, slot]
            current_ids = selected_ids[start:stop, slot].to(device=device, dtype=torch.long)
            removed_prob = probabilities[start:stop, slot]
            denominator = (1.0 - removed_prob).clamp_min(1e-12)
            current_baseline = baseline[start:stop]
            current_expert_output = expert_outputs[rows].float()
            counterfactual = (
                probability_weighted_sum[start:stop]
                + replacement_prob[start:stop, None] * replacement[start:stop]
                - removed_prob[:, None] * current_expert_output
            ) / denominator[:, None]
            counterfactual = counterfactual * scale[start:stop, None]
            delta = counterfactual - current_baseline
            delta_squared_norm = delta.square().sum(-1).double()
            squared_sum.scatter_add_(0, current_ids, delta_squared_norm)
            occurrence_count.scatter_add_(
                0, current_ids, torch.ones_like(current_ids, dtype=torch.long)
            )
    return squared_sum, occurrence_count


def _local_output_rms(squared_sum: torch.Tensor, occurrence_count: torch.Tensor) -> torch.Tensor:
    """Aggregate independent local MoE-output deltas into per-expert RMS-L2."""
    if squared_sum.ndim != 1 or occurrence_count.shape != squared_sum.shape:
        raise ValueError("Expected matching one-dimensional sums and counts")
    if (occurrence_count < 0).any() or (squared_sum < 0).any():
        raise ValueError("Squared sums and occurrence counts must be non-negative")
    return torch.sqrt(
        squared_sum / occurrence_count.clamp_min(1).to(dtype=squared_sum.dtype)
    )


def _save_sensitivity_checkpoint(
    output_dir: pathlib.Path,
    state: dict,
    stability: RouterStabilityCollector,
    manifest: dict,
    processed_examples: int,
    processed_tokens: int,
    elapsed_seconds: float,
) -> None:
    checkpoint = output_dir / f"local_output_sensitivity_checkpoint_{processed_examples:09d}.pt"
    torch.save(
        {
            "state": state,
            "processed_examples": processed_examples,
            "processed_tokens": processed_tokens,
            "elapsed_seconds": elapsed_seconds,
            "sample_set_sha256": manifest["sample_set_sha256"],
            "stability_accumulators": stability.accumulators,
            "stability_num_layers": stability.num_layers,
            "stability_num_experts": stability.num_experts,
            "stability_total_tokens": stability.total_tokens,
            "stability_batch_token_counts": stability._batch_token_counts,
            "stability_generator_state": stability.generator.get_state(),
        },
        checkpoint,
    )
    stability.save_accumulators(
        output_dir / f"local_output_router_stability_checkpoint_{processed_examples:09d}.npz"
    )
    print(
        f"Saved sensitivity checkpoint: {checkpoint} "
        f"(examples={processed_examples}, tokens={processed_tokens})",
        flush=True,
    )


def _restore_sensitivity_checkpoint(
    output_dir: pathlib.Path,
    state: dict,
    stability: RouterStabilityCollector,
    manifest: dict,
) -> tuple[int, int, float]:
    checkpoints = sorted(output_dir.glob("local_output_sensitivity_checkpoint_*.pt"))
    if not checkpoints:
        return 0, 0, 0.0
    checkpoint = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
    if checkpoint.get("sample_set_sha256") != manifest["sample_set_sha256"]:
        raise RuntimeError("Existing sensitivity checkpoint belongs to a different sample set")
    restored = checkpoint["state"]
    if set(restored) != set(state):
        raise RuntimeError("Existing sensitivity checkpoint has a different layer set")
    state.clear()
    state.update(restored)
    stability.accumulators = checkpoint["stability_accumulators"]
    stability.num_layers = checkpoint["stability_num_layers"]
    stability.num_experts = checkpoint["stability_num_experts"]
    stability.total_tokens = checkpoint["stability_total_tokens"]
    stability._batch_token_counts = checkpoint["stability_batch_token_counts"]
    stability.generator.set_state(checkpoint["stability_generator_state"])
    return (
        int(checkpoint["processed_examples"]),
        int(checkpoint["processed_tokens"]),
        float(checkpoint["elapsed_seconds"]),
    )


def _patch_sensitivity_rotary_device(model) -> None:
    """Keep Qwen position IDs beside rotary frequencies under layer sharding."""
    for module in model.modules():
        if not module.__class__.__name__.lower().endswith("rotaryembedding"):
            continue
        if not hasattr(module, "inv_freq") or getattr(
            module, "_reap_position_device_patch", False
        ):
            continue
        original_forward = module.forward

        def forward(self, hidden_states, position_ids, *args, _original=original_forward, **kwargs):
            position_ids = position_ids.to(device=self.inv_freq.device)
            return _original(hidden_states, position_ids, *args, **kwargs)

        module.forward = MethodType(forward, module)
        module._reap_position_device_patch = True


def _forward_local_output_sensitivity_sample(
    model,
    sample: dict[str, torch.Tensor],
    input_device: torch.device,
    current: dict,
    state: dict,
    batch_id: int,
) -> int:
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]
    if input_ids.shape[0] != 1 or not torch.all(attention_mask == 1):
        raise RuntimeError("Sensitivity inputs must be one unpadded all-real-token sequence")
    current["batch"] = batch_id
    current["mask"] = attention_mask
    current["routes"].clear()
    current["sensitivity"].clear()
    inputs = {
        "input_ids": input_ids.to(input_device),
        "attention_mask": attention_mask.to(input_device),
        "use_cache": False,
        "return_dict": True,
    }
    with torch.no_grad():
        model.model(**inputs)
    if current["sensitivity"] and any(
        not torch.isfinite(payload["eleventh_output"]).all()
        or not payload["eleventh_seen"].all()
        for payload in current["sensitivity"].values()
    ):
        raise RuntimeError("Candidate-11 expert outputs were incomplete or non-finite")
    for layer, layer_state in state.items():
        if not torch.isfinite(layer_state["local_output_delta_squared_sum"]).all():
            raise RuntimeError(f"Layer {layer} produced non-finite local output deltas")
    return int(input_ids.shape[-1])


def _top_k(router: torch.nn.Module) -> int:
    for name in ("top_k", "num_experts_per_tok"):
        value = getattr(router, name, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"Could not determine top-k routing for {type(router).__name__}")


def _qwen4_device_map(config) -> dict[str, str]:
    """Place Qwen4 language layers across all four H100s without disk offload."""
    text = getattr(config, "text_config", config)
    layers = int(text.num_hidden_layers)
    device_map: dict[str, str] = {
        "model.visual": "cpu",
        "model.language_model.embed_tokens": "cuda:0",
        "model.language_model.hyper_connection_mixer": "cuda:2",
        "model.language_model.norm": "cuda:2",
        "lm_head": "cuda:3",
    }
    # Keep consecutive decoder layers on the same GPU.  The previous
    # round-robin placement made the hidden state cross PCIe/NVLink between
    # almost every layer, which is especially expensive for this 48-layer
    # model.  Contiguous blocks reduce the steady-state boundaries to three.
    devices = 4
    block = (layers + devices - 1) // devices
    for index in range(layers):
        device_map[f"model.language_model.layers.{index}"] = f"cuda:{min(index // block, devices - 1)}"
    # The PLE n-gram table is a non-FP8 auxiliary table and is too large for a GPU.
    # Split layer 1 explicitly; otherwise Accelerate initializes the complete
    # parent on its assigned GPU before seeing the child override.
    layer_one = "model.language_model.layers.1"
    device_map.pop(layer_one)
    for child in (
        "linear_attn", "mlp", "attn_hyper_connection", "mlp_hyper_connection",
        "ple.key_proj", "ple.value_proj", "ple.norm_key", "ple.norm_query",
        "ple.norm_conv", "ple.conv1d",
    ):
        device_map[f"{layer_one}.{child}"] = "cuda:0"
    device_map[f"{layer_one}.ple.ple_embedding"] = "cpu"
    # Keep a little headroom on the last device for the packed checkpoint tail.
    device_map["model.language_model.layers.47"] = "cuda:2"
    return device_map


def _keep_ngram_embedding_on_cpu(model) -> None:
    for name, module in model.named_modules():
        if not name.endswith("ple.ple_embedding") or not hasattr(module, "ngram_embedding"):
            continue
        embedding = module.ngram_embedding
        remove_hook_from_module(embedding, recurse=False)
        # In EP mode Transformers otherwise leaves this large auxiliary table
        # replicated on every rank.  Its lookup is inexpensive on CPU and the
        # result is copied back to the layer's CUDA device below.
        embedding.to("cpu")

        def cpu_forward(self, input_ids):
            values = F.embedding(input_ids.to(self.weight.device), self.weight)
            return values.to(device=input_ids.device, dtype=torch.bfloat16)

        embedding.forward = MethodType(cpu_forward, embedding)
        return
    raise RuntimeError("Could not find Qwen4 PLE n-gram embedding")


def _bridge_qwen4_ple_layer(model) -> None:
    """Keep all layer-1 PLE inputs on the device hosting its submodules."""
    layer = model.model.language_model.layers[1]
    target = torch.device("cuda:0")

    def move(value):
        if torch.is_tensor(value):
            return value.to(target)
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if isinstance(value, list):
            return [move(item) for item in value]
        return value

    def pre_hook(_module, args, kwargs):
        args = tuple(move(item) for item in args)
        kwargs = {
            key: value if key == "past_key_values" else move(value)
            for key, value in kwargs.items()
        }
        return args, kwargs

    layer.register_forward_pre_hook(pre_hook, with_kwargs=True)


def _assert_fp8_loaded(model) -> None:
    config = getattr(model, "config", None)
    quant = getattr(config, "quantization_config", None)
    quant_method = quant.get("quant_method") if isinstance(quant, dict) else getattr(quant, "quant_method", None)
    if quant_method != "fp8":
        raise RuntimeError(f"Expected FP8 checkpoint metadata, got {quant_method!r}")
    fp8_parameters = [parameter for name, parameter in model.named_parameters()
                      if "expert" in name or "proj" in name]
    fp8_dtypes = {torch.float8_e4m3fn}
    if hasattr(torch, "float8_e4m3fnuz"):
        fp8_dtypes.add(torch.float8_e4m3fnuz)
    if not any(parameter.dtype in fp8_dtypes for parameter in fp8_parameters):
        raise RuntimeError("No FP8 expert/projection parameters found; weights may have been dequantized")


def _shard_ep_expert_scales(model, rank: int) -> None:
    """Apply the missing local shards for Transformers' EP FP8 scales."""
    from torch.distributed.tensor import DTensor

    for module in model.modules():
        if not hasattr(module, "gate_up_proj_scale_inv"):
            continue
        weight = module.gate_up_proj
        scale = module.gate_up_proj_scale_inv
        if isinstance(weight, DTensor) and not isinstance(scale, DTensor):
            local_experts = weight.to_local().shape[0]
            start = rank * local_experts
            def shard_scale(value):
                local = value.detach()[start : start + local_experts].to(weight.device)
                return torch.nn.Parameter(
                    DTensor.from_local(
                        local,
                        weight.device_mesh,
                        [weight.placements[0]],
                        shape=value.shape,
                        stride=value.stride(),
                    ),
                    requires_grad=False,
                )
            module._parameters["gate_up_proj_scale_inv"] = shard_scale(scale)
            module._parameters["down_proj_scale_inv"] = shard_scale(module.down_proj_scale_inv)
        for name in ("gate_up_proj_scale_inv", "down_proj_scale_inv"):
            value = getattr(module, name, None)
            if isinstance(value, DTensor) and value.dtype != torch.float32:
                module._parameters[name] = torch.nn.Parameter(
                    DTensor.from_local(
                        value.to_local().float(), value.device_mesh,
                        value.placements, shape=value.shape, stride=value.stride(),
                    ),
                    requires_grad=False,
                )


def _all_reduce_ep_state(state: dict[int, dict], dist, rank: int) -> None:
    """Merge callback metrics collected by one EP rank per GPU."""
    device = torch.device(f"cuda:{rank}")
    for layer_state in state.values():
        for key, value in layer_state.items():
            if key in {"ean_mean", "reap"} or not torch.is_tensor(value):
                continue
            reduced = value.to(device)
            op = dist.ReduceOp.MAX if key == "max_activations" else dist.ReduceOp.SUM
            dist.all_reduce(reduced, op=op)
            value.copy_(reduced.cpu())

        # OnlineStatsTracker stores per-expert means and counts. Merge the
        # sufficient statistics rather than summing normalized means.
        for key in ("ean_mean", "reap"):
            tracker = layer_state[key]
            count = tracker.count.to(device)
            weighted = (tracker.mean * tracker.count.to(tracker.mean.dtype)).to(device)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            dist.all_reduce(weighted, op=dist.ReduceOp.SUM)
            # State tensors may have been created under inference_mode; replace
            # them instead of mutating an inference tensor outside its context.
            tracker.count = count.cpu().detach().clone()
            tracker.mean = (
                weighted / count.clamp_min(1).to(weighted.dtype)
            ).cpu().detach().clone()
            tracker.mean_compensation = torch.zeros_like(tracker.mean)


def _load_batches(tokenizer) -> list[dict[str, torch.Tensor]]:
    data = load_category_batches(
        dataset_name=DATASET_NAME,
        split=DATASET_SPLIT,
        subset=None,
        tokenizer=tokenizer,
        model_max_length=MODEL_MAX_LENGTH,
        batch_size=BATCH_SIZE,
        split_by_category=False,
        return_vllm_tokens_prompt=False,
        truncate=False,
        batches_per_category=MAX_SAMPLES,
        pack_samples=PACK_SAMPLES,
    )
    batches = next(iter(data.values()))
    # The legacy loader's ``batches_per_category`` counts processed batches.
    # In packed mode each row is a packed calibration sequence; in unpacked
    # mode it is a batch of filtered source rows. Trim the final tensor batch
    # so MAX_SAMPLES remains exact in either mode.
    selected = batches[: (MAX_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE]
    remaining = MAX_SAMPLES
    trimmed = []
    for batch in selected:
        count = int(batch["input_ids"].shape[0])
        keep = min(count, remaining)
        trimmed.append({
            key: value[:keep] if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == count else value
            for key, value in batch.items()
        })
        remaining -= keep
        if remaining <= 0:
            break
    return trimmed


def _save_calibration_checkpoint(
    state: dict,
    stability: RouterStabilityCollector,
    output_dir: pathlib.Path,
    processed_sequences: int,
    total_tokens: int,
    calibration_seconds: float,
) -> None:
    """Write a resumable CPU metric snapshot during long calibration runs."""
    checkpoint = output_dir / f"calibration_state_checkpoint_{processed_sequences:09d}.pt"
    torch.save(
        {
            "state": state,
            "processed_packed_sequences": processed_sequences,
            "total_tokens": total_tokens,
            "calibration_seconds": calibration_seconds,
        },
        checkpoint,
    )
    stability.save_accumulators(
        output_dir / f"router_stability_checkpoint_{processed_sequences:09d}.npz"
    )
    print(
        f"Saved calibration checkpoint: {checkpoint} "
        f"(packed_sequences={processed_sequences}, tokens={total_tokens})",
        flush=True,
    )


def main() -> None:
    global _QWEN4_SENSITIVITY_ENABLED, _QWEN4_SENSITIVITY_CURRENT
    _patch_triton_kernel_compat()
    set_seed(SEED)
    started = time.perf_counter()
    ep_enabled = os.environ.get("QWEN4_EXPERT_PARALLEL", "0").lower() in {
        "1", "true", "yes"
    }
    if SENSITIVITY_MODE and ep_enabled:
        raise RuntimeError(
            "Sensitivity mode requires the non-EP grouped_mm path so activation "
            "gradients remain local and directly capturable"
        )
    if SENSITIVITY_MODE and EXPERTS_IMPLEMENTATION != "grouped_mm":
        raise RuntimeError("Sensitivity mode requires QWEN4_EXPERTS_IMPLEMENTATION=grouped_mm")
    _QWEN4_SENSITIVITY_ENABLED = SENSITIVITY_MODE
    rank = 0
    dist = None
    if ep_enabled:
        import torch.distributed as torch_dist

        torch_dist.init_process_group("nccl")
        rank = torch_dist.get_rank()
        torch.cuda.set_device(rank)
        dist = torch_dist
        _patch_ep_tp_compat()
        _install_qwen4_ep_capture_dispatch()
    config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    if getattr(config, "model_type", None) != "qwen4_exp":
        raise RuntimeError(f"Expected qwen4_exp checkpoint, got {getattr(config, 'model_type', None)!r}")

    load_kwargs = dict(
        pretrained_model_name_or_path=MODEL_PATH,
        device_map=_qwen4_device_map(config),
        dtype=None,
        experts_implementation=EXPERTS_IMPLEMENTATION,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        offload_state_dict=True,
        offload_buffers=True,
    )
    if ep_enabled:
        from transformers.distributed import DistributedConfig

        load_kwargs.pop("device_map")
        load_kwargs["distributed_config"] = DistributedConfig(
            tp_size=int(os.environ.get("QWEN4_TP_SIZE", "4")),
            fsdp_size=1,
            ep_size=int(os.environ.get("QWEN4_EP_SIZE", "4")),
        )
    if ATTN_IMPLEMENTATION is not None:
        load_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION
    model = AutoModelForImageTextToText.from_pretrained(**load_kwargs).eval()
    if SENSITIVITY_MODE:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        _patch_sensitivity_rotary_device(model)
    actual_experts_implementation = model.get_experts_implementation()
    actual_impl = actual_experts_implementation.get("", EXPERTS_IMPLEMENTATION)
    if actual_impl != EXPERTS_IMPLEMENTATION:
        raise RuntimeError(
            "Requested experts implementation was not selected: "
            f"requested={EXPERTS_IMPLEMENTATION!r}, actual={actual_experts_implementation!r}"
        )
    if actual_impl == "deepgemm":
        if not ep_enabled:
            raise RuntimeError("DeepGEMM calibration requires QWEN4_EXPERT_PARALLEL=1")
        _install_qwen4_deepgemm_capture_dispatch()
    else:
        _install_qwen4_capture_dispatch()
    _assert_fp8_loaded(model)
    if ep_enabled:
        _shard_ep_expert_scales(model, rank)
    if not ep_enabled:
        _keep_ngram_embedding_on_cpu(model)
        _bridge_qwen4_ple_layer(model)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    sample_manifest = None
    if SENSITIVITY_MODE:
        batches, sample_manifest = _prepare_sensitivity_samples(tokenizer)
    else:
        batches = _load_batches(tokenizer)
    if not batches:
        raise RuntimeError("No calibration examples were loaded")

    output_dir = OUTPUT_DIR if rank == 0 else OUTPUT_DIR / f"rank{rank}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if sample_manifest is not None:
        (output_dir / "sample_manifest.json").write_text(
            json.dumps(sample_manifest, indent=2) + "\n"
        )
    stability_dir = output_dir / "router_stability"
    stability = RouterStabilityCollector(
        stability_dir,
        variance_multipliers=VARIANCE_MULTIPLIERS,
        seed=SEED,
        boundary=4,
        max_tokens=ROUTER_MAX_TOKENS,
        top_k=int(text_config.num_experts_per_tok),
        save_events=SAVE_ROUTING_EVENTS,
    )
    state: dict[int, dict] = {}
    current = {
        "mask": None,
        "routes": {},
        "sensitivity": {},
        "capture_mode": "ordinary",
    }
    _QWEN4_SENSITIVITY_CURRENT = current
    handles = []
    layer_names: dict[int, str] = {}

    def make_moe_hook(layer: int, moe: torch.nn.Module):
        router = moe.gate
        num_experts = _num_experts(router)
        top_k = _top_k(router)

        def hook(_module, args, output):
            hidden_all = args[0].reshape(-1, args[0].shape[-1])
            mask = current["mask"]
            valid = torch.ones(hidden_all.shape[0], device=hidden_all.device, dtype=torch.bool)
            if mask is not None:
                valid = mask.reshape(-1).to(hidden_all.device).bool()
            hidden = hidden_all[valid]
            if hidden.numel() == 0:
                return

            route = current["routes"].get(layer)
            if route is None:
                raise RuntimeError(f"Missing Qwen4 route capture for layer {layer}")
            selected = route[4][valid]
            logits = route[0][valid]
            original_route = tuple(value[valid] for value in route)
            with torch.no_grad():
                observation = collect_router_observation(
                    router, hidden, stability.variance_multipliers, stability.generator,
                    top_k=top_k, boundary=stability.boundary,
                    original_route=original_route,
                )
            stability.observe(layer, current["batch"], observation)

            if layer not in state:
                state[layer] = initialize_pruning_state(num_experts, device="cpu")
                state[layer]["moe_residual_contribution_sum"] = torch.zeros((), dtype=torch.float64)
                state[layer]["moe_residual_contribution_count"] = torch.zeros((), dtype=torch.long)

            moe_output = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(moe_output) and moe_output.shape == args[0].shape:
                contribution = torch.linalg.vector_norm(moe_output, dim=-1).reshape(-1)[valid]
                residual = torch.linalg.vector_norm(args[0], dim=-1).reshape(-1)[valid]
                state[layer]["moe_residual_contribution_sum"] += (
                    (contribution / (residual + torch.finfo(args[0].dtype).eps))
                    .detach().sum().double().cpu()
                )
                state[layer]["moe_residual_contribution_count"] += torch.tensor(
                    contribution.numel(), dtype=torch.long
                )

        return hook

    def capture_route(layer: int, moe: torch.nn.Module):
        router = moe.gate
        top_k = _top_k(router)

        def pre_hook(_module, args):
            hidden = args[0].reshape(-1, args[0].shape[-1])
            current["routes"][layer] = _route_once(router, hidden, top_k, boundary=0)

        return pre_hook

    def make_callback(layer: int, num_experts: int, top_k: int):
        def callback(expert_ids, expert_outputs, perm, num_top_k):
            route = current["routes"].get(layer)
            if route is None:
                raise RuntimeError(f"Missing Qwen4 route capture for layer {layer}")
            capture_mode = current.get("capture_mode", "ordinary")
            if SENSITIVITY_MODE:
                payload = current["sensitivity"].get(layer)
                if payload is None:
                    raise RuntimeError(f"Missing sensitivity payload for layer {layer}")
                expert_rows = expert_ids.reshape(-1).to(torch.long)
                if capture_mode == "eleventh":
                    if num_top_k != 1 or perm.numel() != payload["candidate_ids"].shape[0]:
                        raise RuntimeError("Candidate-11 callback has an unexpected token layout")
                    token_rows = perm.reshape(-1).to(torch.long)
                    candidate_ids = payload["candidate_ids"]
                    if not torch.equal(
                        candidate_ids[token_rows, 10].to(expert_rows.device), expert_rows
                    ):
                        raise RuntimeError("Candidate-11 callback rows do not align with tokens")
                    if (expert_rows >= num_experts).any():
                        raise RuntimeError("Candidate-11 route contains an invalid expert id")
                    if payload["eleventh_seen"][token_rows].any():
                        raise RuntimeError("Candidate-11 output was captured twice for a token")
                    payload["eleventh_output"].index_copy_(
                        0, token_rows, expert_outputs.detach()
                    )
                    payload["eleventh_seen"][token_rows] = True
                    return

                if num_top_k != top_k or top_k != 10:
                    raise RuntimeError(
                        f"Expected ordinary top-{top_k} output callback, got top-{num_top_k}"
                    )
                sum_squares, occurrence_count = _top11_local_output_delta_squares(
                    candidate_probs=payload["candidate_probs"],
                    selected_ids=payload["selected_ids"],
                    selected_weights=payload["selected_weights"],
                    routed_scale=payload["routed_scale"],
                    eleventh_output=payload["eleventh_output"],
                    expert_ids=expert_rows,
                    expert_outputs=expert_outputs.detach(),
                    route_permutation=perm,
                    num_experts=num_experts,
                )
                state[layer]["local_output_delta_squared_sum"] += sum_squares.cpu()
                state[layer]["local_output_occurrence_count"] += occurrence_count.cpu()

            selected = route[4].reshape(-1, num_top_k)
            logits = route[0]
            mask = current["mask"]
            valid = torch.ones(selected.shape[0], device=selected.device, dtype=torch.bool)
            if mask is not None:
                valid = mask.reshape(-1).to(selected.device).bool()
            if valid.numel() != selected.shape[0]:
                raise RuntimeError(
                    "Observer route/token layout mismatch; this EP dispatcher "
                    "does not expose an alignable callback route"
                )
            original_token = perm.reshape(-1) // num_top_k
            keep = valid[original_token] & (expert_ids.reshape(-1) < num_experts)
            if not keep.any():
                return
            valid_token_ids = torch.nonzero(valid, as_tuple=False).reshape(-1)
            compact = torch.full((valid.numel(),), -1, device=selected.device, dtype=torch.long)
            compact[valid_token_ids] = torch.arange(valid_token_ids.numel(), device=selected.device)
            if layer not in state:
                state[layer] = initialize_pruning_state(num_experts, device="cpu")
                state[layer]["moe_residual_contribution_sum"] = torch.zeros((), dtype=torch.float64)
                state[layer]["moe_residual_contribution_count"] = torch.zeros((), dtype=torch.long)
            with torch.no_grad():
                update_pruning_state_from_routed_outputs(
                    state[layer],
                    expert_ids=expert_ids.reshape(-1)[keep],
                    expert_outputs=expert_outputs[keep],
                    token_indices=compact[original_token[keep]],
                    selected_experts=selected[valid],
                    router_logits=logits[valid],
                    num_experts=num_experts,
                    renormalize_router_weights=True,
                    accumulate_routing_stats=not ep_enabled or rank == 0,
                )

        return callback

    for name, module in model.named_modules():
        if not (hasattr(module, "experts") and hasattr(module, "gate")):
            continue
        layer = _layer_index(name)
        if layer is None:
            continue
        layer_names[layer] = name
        router = module.gate
        num_experts, top_k = _num_experts(router), _top_k(router)
        state[layer] = initialize_pruning_state(num_experts, device="cpu")
        state[layer]["moe_residual_contribution_sum"] = torch.zeros((), dtype=torch.float64)
        state[layer]["moe_residual_contribution_count"] = torch.zeros((), dtype=torch.long)
        if SENSITIVITY_MODE:
            state[layer]["local_output_delta_squared_sum"] = torch.zeros(
                num_experts, dtype=torch.float64
            )
            state[layer]["local_output_occurrence_count"] = torch.zeros(
                num_experts, dtype=torch.long
            )
        handles.append(module.register_forward_hook(make_moe_hook(layer, module)))
        _wrap_qwen4_router_forward(router, layer, current)
        module.experts._reap_capture_callback = make_callback(layer, num_experts, top_k)
        _wrap_qwen4_expert_forward(module.experts, layer, router)

    expected_layers = int(getattr(text_config, "num_hidden_layers", 0))
    if len(layer_names) != expected_layers:
        raise RuntimeError(f"Expected {expected_layers} Qwen4 MoE layers, found {layer_names}")

    input_device = model.get_input_embeddings().weight.device
    calibration_started = time.perf_counter()
    total_examples = total_tokens = 0
    next_checkpoint = CHECKPOINT_EVERY if CHECKPOINT_EVERY > 0 else None
    progress = tqdm(
        batches if not SENSITIVITY_MODE else range(len(batches)),
        desc="Qwen3.8 sensitivity calibration" if SENSITIVITY_MODE else "Qwen4 FP8 calibration",
        unit="example" if SENSITIVITY_MODE else "batch",
        disable=ep_enabled and rank != 0,
    )
    try:
        if SENSITIVITY_MODE:
            assert sample_manifest is not None
            existing = sorted(output_dir.glob("local_output_sensitivity_checkpoint_*.pt"))
            if not existing:
                longest = max(
                    range(len(batches)), key=lambda index: batches[index]["input_ids"].shape[-1]
                )
                longest_length = int(batches[longest]["input_ids"].shape[-1])
                longest_row = sample_manifest["selected_row_indices"][longest]
                print(
                    "Running required local-output sensitivity feasibility check on longest selected row: "
                    f"index={longest_row}, tokens={longest_length}",
                    flush=True,
                )
                try:
                    _forward_local_output_sensitivity_sample(
                        model, batches[longest], input_device, current, state, -1
                    )
                except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                    (output_dir / "local_output_sensitivity_run_status.json").write_text(
                        json.dumps(
                            {
                                "status": "stopped_at_longest_sequence_feasibility_check",
                                "complete": False,
                                "sample_set_sha256": sample_manifest["sample_set_sha256"],
                                "row_index": longest_row,
                                "sequence_tokens": longest_length,
                                "feasibility_passed": False,
                                "gradient_capture_status": "not_required",
                                "calibration_scores_written": False,
                                "error": str(exc),
                                "partial_preflight_artifacts": (
                                    "router_stability contains observations from the failed "
                                    "longest-row feasibility forward only; it is not a complete calibration"
                                ),
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                    message = (
                        "Longest-sequence local-output sensitivity feasibility check failed; "
                        "the requested calibration was not weakened or substituted. "
                        f"row_index={longest_row}, tokens={longest_length}, error={exc}"
                    )
                    if "out of memory" in str(exc).lower() or isinstance(
                        exc, torch.cuda.OutOfMemoryError
                    ):
                        torch.cuda.empty_cache()
                    raise RuntimeError(message) from exc
                if any(
                    not torch.isfinite(layer_state["local_output_delta_squared_sum"]).all()
                    for layer_state in state.values()
                ):
                    raise RuntimeError("Feasibility check produced non-finite local output deltas")

                # The preflight validates the exact longest-row path but is
                # excluded from the calibration aggregates.
                state.clear()
                for layer in range(expected_layers):
                    num_experts = _num_experts(model.get_submodule(layer_names[layer]).gate)
                    state[layer] = initialize_pruning_state(num_experts, device="cpu")
                    state[layer]["moe_residual_contribution_sum"] = torch.zeros(
                        (), dtype=torch.float64
                    )
                    state[layer]["moe_residual_contribution_count"] = torch.zeros(
                        (), dtype=torch.long
                    )
                    state[layer]["local_output_delta_squared_sum"] = torch.zeros(
                        num_experts, dtype=torch.float64
                    )
                    state[layer]["local_output_occurrence_count"] = torch.zeros(
                        num_experts, dtype=torch.long
                    )
                stability = RouterStabilityCollector(
                    stability_dir,
                    variance_multipliers=(1.0,),
                    seed=42,
                    boundary=4,
                    max_tokens=None,
                    top_k=int(text_config.num_experts_per_tok),
                    save_events=False,
                )
                current["routes"].clear()
                current["sensitivity"].clear()

            start_index, processed_tokens, prior_seconds = _restore_sensitivity_checkpoint(
                output_dir, state, stability, sample_manifest
            )
            total_examples = start_index
            total_tokens = sum(sample_manifest["token_lengths"][:start_index])
            calibration_started = time.perf_counter()
            for sample_index in range(start_index, len(batches)):
                processed_tokens += _forward_local_output_sensitivity_sample(
                    model,
                    batches[sample_index],
                    input_device,
                    current,
                    state,
                    sample_index,
                )
                total_examples = sample_index + 1
                total_tokens += int(batches[sample_index]["input_ids"].shape[-1])
                progress.update(1)
                progress.set_postfix(examples=total_examples, tokens=total_tokens)
                if total_examples % CHECKPOINT_EVERY == 0:
                    _save_sensitivity_checkpoint(
                        output_dir,
                        state,
                        stability,
                        sample_manifest,
                        total_examples,
                        processed_tokens,
                        prior_seconds + time.perf_counter() - calibration_started,
                    )
            calibration_seconds = prior_seconds + time.perf_counter() - calibration_started
            _save_sensitivity_checkpoint(
                output_dir,
                state,
                stability,
                sample_manifest,
                total_examples,
                processed_tokens,
                calibration_seconds,
            )
        else:
            with torch.inference_mode():
                for batch_id, batch in enumerate(progress):
                    batch_size = int(batch["input_ids"].shape[0])
                    for offset in range(0, batch_size, FORWARD_BATCH_SIZE):
                        microbatch = {
                            key: value[offset : offset + FORWARD_BATCH_SIZE]
                            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size
                            else value
                            for key, value in batch.items()
                        }
                        current["batch"] = batch_id * 100000 + offset
                        current["mask"] = microbatch.get("attention_mask")
                        current["routes"].clear()
                        inputs = {
                            key: value.to(input_device) if torch.is_tensor(value) else value
                            for key, value in microbatch.items()
                        }
                        # Calibration only consumes routed expert outputs and
                        # hidden states. Avoid the unused vocabulary projection.
                        model.model(**inputs, use_cache=False, return_dict=True)
                    total_examples += int(batch["input_ids"].shape[0])
                    batch_mask = batch.get("attention_mask")
                    total_tokens += int(batch_mask.sum()) if batch_mask is not None else int(batch["input_ids"].numel())
                    progress.set_postfix(examples=total_examples, tokens=total_tokens)
                    if next_checkpoint is not None and total_examples >= next_checkpoint:
                        _save_calibration_checkpoint(
                            state,
                            stability,
                            output_dir,
                            total_examples,
                            total_tokens,
                            time.perf_counter() - calibration_started,
                        )
                        next_checkpoint += CHECKPOINT_EVERY
    finally:
        progress.close()
        for handle in handles:
            handle.remove()
        for module in model.modules():
            experts = getattr(module, "experts", None)
            if experts is not None and hasattr(experts, "_reap_capture_callback"):
                delattr(experts, "_reap_capture_callback")
        stability.finalize()

    if len(state) != expected_layers:
        raise RuntimeError(f"Missing calibration state for layers: {set(range(expected_layers)) - set(state)}")
    if SENSITIVITY_MODE:
        assert sample_manifest is not None
        expected_experts = int(text_config.num_experts)
        top_k = int(text_config.num_experts_per_tok)
        if (expected_layers, expected_experts, top_k) != (48, 512, 10):
            raise RuntimeError(
                "Sensitivity protocol expects exactly 48 layers, 512 experts, and top-k 10; "
                f"got {(expected_layers, expected_experts, top_k)}"
            )
        if (
            total_examples != MAX_SAMPLES
            or len(set(sample_manifest["selected_row_indices"])) != MAX_SAMPLES
        ):
            raise RuntimeError("Sensitivity sample count/uniqueness validation failed")
        if total_tokens != sum(sample_manifest["token_lengths"]):
            raise RuntimeError("Processed token count does not match the selected manifest")
        if stability.total_tokens != total_tokens:
            raise RuntimeError("Router stability did not observe every real input token")
        for layer, layer_state in state.items():
            frequencies = layer_state["expert_frequency"]
            if int(layer_state["total_tokens"].item()) != total_tokens:
                raise RuntimeError(f"Layer {layer} route-token count is inconsistent")
            if int(frequencies.sum().item()) != total_tokens * top_k:
                raise RuntimeError(f"Layer {layer} routing frequencies are inconsistent")
            expected_pairwise = frequencies[:, None] + frequencies[None, :]
            if not torch.equal(layer_state["pairwise_expert_frequency"], expected_pairwise):
                raise RuntimeError(f"Layer {layer} pairwise route counts are inconsistent")
            if int(layer_state["reap"].count.sum().item()) != total_tokens * top_k:
                raise RuntimeError(f"Layer {layer} REAP observation count is inconsistent")
            if int(layer_state["moe_residual_contribution_count"].item()) != total_tokens:
                raise RuntimeError(f"Layer {layer} MoE contribution count is inconsistent")
            if not torch.equal(
                layer_state["local_output_occurrence_count"], frequencies
            ):
                raise RuntimeError(
                    f"Layer {layer} local-output sensitivity counts disagree with routing frequencies"
                )
    if ep_enabled:
        assert dist is not None
        _all_reduce_ep_state(state, dist, rank)
    sensitivity_results = {}
    for layer, layer_state in state.items():
        count = int(layer_state["moe_residual_contribution_count"].item())
        layer_state["moe_residual_contribution"] = (
            layer_state["moe_residual_contribution_sum"] / count if count else torch.tensor(0.0)
        )
        layer_state.pop("moe_residual_contribution_sum")
        layer_state.pop("moe_residual_contribution_count")
        for key, value in list(layer_state.items()):
            if hasattr(value, "mean") and not torch.is_tensor(value):
                layer_state[key] = value.mean
        if SENSITIVITY_MODE:
            squared_sum = layer_state.pop("local_output_delta_squared_sum")
            occurrence_count = layer_state.pop("local_output_occurrence_count")
            rms_scores = _local_output_rms(squared_sum, occurrence_count)
            reap_scores = torch.as_tensor(layer_state["reap"])
            removal_order = torch.argsort(reap_scores, stable=True)
            cumulative = torch.cat((
                torch.zeros(1, dtype=torch.float64),
                rms_scores[removal_order].cumsum(dim=0),
            ))
            if rms_scores.shape != (int(text_config.num_experts),):
                raise RuntimeError(f"Layer {layer} has a malformed RMS-L2 score vector")
            if (
                not torch.isfinite(rms_scores).all()
                or not torch.isfinite(reap_scores).all()
            ):
                raise RuntimeError(f"Layer {layer} contains non-finite REAP or RMS-L2 values")
            layer_state["local_output_sensitivity_rms_l2"] = rms_scores
            layer_state["local_output_sensitivity_occurrence_count"] = occurrence_count
            layer_state["local_output_sensitivity_removal_order_ascending_reap"] = removal_order
            layer_state["local_output_sensitivity_cumulative_rms_l2_ascending_reap"] = cumulative
            sensitivity_results[layer] = {
                "rms_l2_by_expert": rms_scores,
                "occurrence_count_by_expert": occurrence_count,
                "ascending_reap_expert_ids": removal_order,
                "cumulative_rms_l2_heuristic": cumulative,
            }

    if not SENSITIVITY_MODE:
        calibration_seconds = time.perf_counter() - calibration_started
    if rank == 0:
        torch.save(state, output_dir / "calibration_state.pt")
        if SENSITIVITY_MODE:
            torch.save(
                {
                    "interpretation": (
                        "per-expert one-at-a-time local MoE-output RMS-L2 deltas; "
                        "the cumulative curve is a ranking heuristic, not a multi-expert damage estimate"
                    ),
                    "processed_token_count": processed_tokens,
                    "layers": sensitivity_results,
                },
                output_dir / "local_output_sensitivity_results.pt",
            )
    metadata = {
        "model_path": MODEL_PATH, "model_type": config.model_type,
        "dataset_name": DATASET_NAME, "split": DATASET_SPLIT,
        "max_samples": MAX_SAMPLES,
        # Keep the old fields for compatibility, but make the packed-unit
        # semantics explicit for timing comparisons and projections.
        "retained_examples": total_examples,
        "max_packed_sequences": MAX_SAMPLES,
        "retained_packed_sequences": total_examples,
        "pack_samples": PACK_SAMPLES,
        "checkpoint_every_packed_sequences": CHECKPOINT_EVERY,
        "total_tokens": total_tokens, "batch_size": BATCH_SIZE,
        "forward_batch_size": FORWARD_BATCH_SIZE,
        "model_max_length": MODEL_MAX_LENGTH, "seed": SEED,
        "experts_implementation": actual_experts_implementation,
        "variance_multipliers": list(VARIANCE_MULTIPLIERS),
        "routing_events_saved": SAVE_ROUTING_EVENTS,
        "num_layers": expected_layers, "num_experts": int(text_config.num_experts),
        "calibration_seconds": calibration_seconds,
        "total_seconds": time.perf_counter() - started,
        "examples_per_second": total_examples / max(calibration_seconds, 1e-9),
        "tokens_per_second": total_tokens / max(calibration_seconds, 1e-9),
        "full_dataset_estimate_seconds_per_100_examples": calibration_seconds,
        "estimate_unit": "packed_sequences",
        "full_dataset_estimate_seconds_per_100_packed_sequences": calibration_seconds,
    }
    if SENSITIVITY_MODE:
        metadata.update({
            "mode": "single_run_local_output_sensitivity",
            "smoke_test": SENSITIVITY_SMOKE_SAMPLES > 0,
            "requested_sample_count": MAX_SAMPLES,
            "retained_examples": total_examples,
            "selected_row_indices_file": "sample_manifest.json",
            "sample_set_sha256": sample_manifest["sample_set_sha256"],
            "processed_tokens": processed_tokens,
            "max_sequence_length": MODEL_MAX_LENGTH,
            "packing": False,
            "truncation": False,
            "padding": False,
            "selection_without_replacement": True,
            "sensitivity": {
                "kind": "per_expert_local_output_rms_l2",
                "counterfactual": (
                    "for each token, replace each selected top-10 expert in turn with "
                    "candidate 11 and renormalize the top-10 router weights"
                ),
                "aggregation": "sqrt(sum squared MoE-output delta L2 / expert occurrence count)",
                "curves": (
                    "cumulative sum of independent per-expert RMS-L2 scores in "
                    "ascending-REAP removal order; ranking heuristic only"
                ),
                "gradient_or_loss_sensitivity": False,
                "exact_multi_expert_damage": False,
            },
            "router_stability": {
                "top_k": int(text_config.num_experts_per_tok),
                "variance_multipliers": [1.0],
                "boundary_width": 4,
                "seed": 42,
                "events_saved": False,
            },
            "checkpoint_every_examples": CHECKPOINT_EVERY,
            "feasibility_checked_longest_selected_example": True,
        })
    if rank == 0:
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if SENSITIVITY_MODE:
            (output_dir / "local_output_sensitivity_run_status.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "complete": True,
                        "sample_set_sha256": sample_manifest["sample_set_sha256"],
                        "processed_examples": total_examples,
                        "processed_tokens": total_tokens,
                        "calibration_scores_written": True,
                    },
                    indent=2,
                )
                + "\n"
            )
        print(json.dumps(metadata, indent=2))
    if ep_enabled:
        assert dist is not None
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
