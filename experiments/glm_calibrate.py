"""Single-pass GLM-4.5-Air calibration on four GPUs.

Collects REAP, router stability, and (in SENSITIVITY_MODE) layerwise per-expert
local-output RMS-L2 sensitivity scores in one model forward pass.

GLM 4.5 Air uses unfused BF16 experts; the expert loop in Glm4MoeMoE.moe() is
patched per-instance to capture individual expert outputs without re-running them.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import time
from types import MethodType

import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from reap.data import load_category_batches
from reap.data import CodeAlpacaChatDataset, _load_raw_dataset
from reap.pruning_metrics import (
    initialize_pruning_state,
    update_pruning_state_from_routed_outputs,
)
from reap.router_stability import RouterStabilityCollector, collect_router_observation



MODEL_PATH = os.environ.get(
    "GLM_MODEL_PATH",
    "artifacts/models/GLM-4.5-Air",
)
SENSITIVITY_SMOKE_SAMPLES = int(os.environ.get("GLM_SENSITIVITY_SMOKE_SAMPLES", "0"))
SENSITIVITY_MODE = SENSITIVITY_SMOKE_SAMPLES > 0 or os.environ.get(
    "GLM_SENSITIVITY_MODE", "0"
).lower() in {"1", "true", "yes"}
if SENSITIVITY_SMOKE_SAMPLES < 0 or SENSITIVITY_SMOKE_SAMPLES > 12228:
    raise ValueError("GLM_SENSITIVITY_SMOKE_SAMPLES must be in [0, 12228]")
OUTPUT_DIR = pathlib.Path(
    os.environ.get(
        "GLM_OUTPUT_DIR",
        (
            f"artifacts/GLM-4.5-Air/local-output-sensitivity-smoke-evol-codealpaca-"
            f"{SENSITIVITY_SMOKE_SAMPLES}-16384-unpruned"
            if SENSITIVITY_SMOKE_SAMPLES
            else "artifacts/GLM-4.5-Air/local-output-sensitivity-evol-codealpaca-12228-16384-unpruned"
            if SENSITIVITY_MODE
            else "artifacts/GLM-4.5-Air/evol-codealpaca"
        ),
    )
)
DATASET_NAME = os.environ.get("GLM_DATASET_NAME", "theblackcat102/evol-codealpaca-v1")
DATASET_SPLIT = os.environ.get("GLM_DATASET_SPLIT", "train")
BATCH_SIZE = 1 if SENSITIVITY_MODE else int(os.environ.get("GLM_BATCH_SIZE", "32"))
FORWARD_BATCH_SIZE = int(
    "1" if SENSITIVITY_MODE else os.environ.get("GLM_FORWARD_BATCH_SIZE", str(min(BATCH_SIZE, 4)))
)
MODEL_MAX_LENGTH = 16384 if SENSITIVITY_MODE else int(os.environ.get("GLM_MODEL_MAX_LENGTH", "2048"))
PACK_SAMPLES = False if SENSITIVITY_MODE else os.environ.get("GLM_PACK_SAMPLES", "1").lower() in {
    "1", "true", "yes"
}
MAX_SAMPLES = (
    SENSITIVITY_SMOKE_SAMPLES
    if SENSITIVITY_SMOKE_SAMPLES
    else 12228 if SENSITIVITY_MODE
    else int(os.environ.get("GLM_MAX_SAMPLES", "1000"))
)
SEED = 42 if SENSITIVITY_MODE else int(os.environ.get("GLM_SEED", "42"))
ROUTER_MAX_TOKENS = int(os.environ.get("GLM_ROUTER_MAX_TOKENS", "0")) or None
SAVE_ROUTING_EVENTS = os.environ.get("GLM_SAVE_ROUTING_EVENTS", "0").lower() in {
    "1", "true", "yes"
}
CHECKPOINT_EVERY = 1024 if SENSITIVITY_MODE else int(os.environ.get("GLM_CHECKPOINT_EVERY", "10000"))
VARIANCE_MULTIPLIERS = tuple(
    float(v.strip())
    for v in os.environ.get("GLM_VARIANCE_MULTIPLIERS", "1.0").split(",")
    if v.strip()
)
if SENSITIVITY_MODE:
    VARIANCE_MULTIPLIERS = (1.0,)
SENSITIVITY_BATCH_SIZE = int(os.environ.get("GLM_SENSITIVITY_BATCH_SIZE", "1")) if SENSITIVITY_MODE else 1
if not VARIANCE_MULTIPLIERS:
    raise ValueError("GLM_VARIANCE_MULTIPLIERS must contain at least one value")
if FORWARD_BATCH_SIZE < 1:
    raise ValueError("GLM_FORWARD_BATCH_SIZE must be positive")



def _layer_index(name: str) -> int | None:
    match = re.search(r"model\.layers\.(\d+)\.mlp$", name)
    return int(match.group(1)) if match else None


def _num_experts(router) -> int:
    for attr in ("n_routed_experts", "num_experts"):
        v = getattr(router, attr, None)
        if v is not None:
            return int(v)
    return int(router.weight.shape[0])


def _top_k(router) -> int:
    for attr in ("top_k", "num_experts_per_tok"):
        v = getattr(router, attr, None)
        if v is not None:
            return int(v)
    raise AttributeError(f"Cannot determine top-k for {type(router).__name__}")


def _glm_device_map(config) -> dict[str, str]:
    """Distribute GLM-4.5-Air's 46 layers across four GPUs."""
    layers = int(config.num_hidden_layers)
    devices = 4
    block = (layers + devices - 1) // devices
    device_map = {
        "model.embed_tokens": "cuda:0",
        "model.norm": "cuda:3",
        "lm_head": "cuda:3",
    }
    for i in range(layers):
        device_map[f"model.layers.{i}"] = f"cuda:{min(i // block, devices - 1)}"
    return device_map



def _wrap_glm_router_forward(router, layer: int, current: dict) -> None:
    """Record the router result (logits, topk_indices, topk_weights) once per forward."""
    original = router.forward

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        topk_indices, topk_weights, router_logits = result
        current["routes"][layer] = (
            router_logits.detach(),
            topk_indices.detach(),
            topk_weights.detach(),
        )
        return result

    router.forward = wrapped



def _top_k_plus_1_local_output_delta_squares(
    *,
    candidate_probs: torch.Tensor,
    selected_ids: torch.Tensor,
    selected_weights: torch.Tensor,
    routed_scale,
    ninth_output: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_outputs: torch.Tensor,
    route_permutation: torch.Tensor,
    num_experts: int,
    top_k: int,
    chunk_tokens: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-expert sum(||delta MoE output||^2) and occurrence counts.

    Identical in structure to qwen4_calibrate._top11_local_output_delta_squares
    but parameterized for arbitrary top_k (GLM uses top_k=8, candidate count=9).
    """
    tokens = candidate_probs.shape[0]
    if candidate_probs.ndim != 2 or candidate_probs.shape[-1] != top_k + 1:
        raise ValueError(f"Expected normalized probabilities for {top_k + 1} candidates")
    if selected_ids.shape != (tokens, top_k) or selected_weights.shape != (tokens, top_k):
        raise ValueError(f"Expected {top_k} selected expert IDs and weights per token")
    if ninth_output.ndim != 2 or ninth_output.shape[0] != tokens:
        raise ValueError("Candidate-(k+1) outputs must have shape [tokens, hidden]")
    if expert_outputs.ndim != 2 or expert_outputs.shape[1] != ninth_output.shape[1]:
        raise ValueError("Routed outputs and candidate outputs must share hidden size")
    if expert_ids.numel() != tokens * top_k or route_permutation.numel() != tokens * top_k:
        raise ValueError(f"Expected one routed output row per top-{top_k} token/expert pair")

    device = expert_outputs.device
    token_rows = route_permutation.reshape(-1).long() // top_k
    slot_rows = route_permutation.reshape(-1).long() % top_k
    expert_ids = expert_ids.reshape(-1).long()
    if token_rows.min() < 0 or token_rows.max() >= tokens:
        raise ValueError("Routed-output permutation refers to an invalid token")
    if not torch.equal(selected_ids[token_rows, slot_rows], expert_ids):
        raise RuntimeError("Routed output rows do not align with the captured route")
    if (expert_ids >= num_experts).any():
        raise RuntimeError("Sensitivity mode: invalid expert id encountered")

    probabilities = candidate_probs.float()
    probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-20)
    weights = selected_weights.to(device=device, dtype=torch.float32)
    scale = torch.as_tensor(routed_scale, device=device, dtype=torch.float32).reshape(-1)
    if scale.numel() == 1:
        scale = scale.expand(tokens)
    if scale.shape != (tokens,) or (scale <= 0).any():
        raise RuntimeError("Router produced an invalid routed scaling factor")
    if (probabilities[:, :top_k].sum(-1) <= 0).any():
        raise RuntimeError("Router produced a non-positive selected-route mass")

    baseline = torch.zeros(
        (tokens, expert_outputs.shape[-1]), device=device, dtype=torch.float32
    )
    baseline.index_add_(
        0, token_rows,
        expert_outputs.float() * weights[token_rows, slot_rows, None],
    )
    probability_weighted_sum = torch.zeros_like(baseline)
    probability_weighted_sum.index_add_(
        0, token_rows,
        expert_outputs.float() * probabilities[token_rows, slot_rows, None],
    )
    rows_by_slot = torch.full((tokens, top_k), -1, device=device, dtype=torch.long)
    rows_by_slot[token_rows, slot_rows] = torch.arange(tokens * top_k, device=device)
    if (rows_by_slot < 0).any():
        raise RuntimeError("Missing routed expert output for a token/top-k slot")

    replacement = ninth_output.float()
    replacement_prob = probabilities[:, top_k]
    squared_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
    occurrence_count = torch.zeros(num_experts, device=device, dtype=torch.long)
    for slot in range(top_k):
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
    if squared_sum.ndim != 1 or occurrence_count.shape != squared_sum.shape:
        raise ValueError("Expected matching one-dimensional sums and counts")
    return torch.sqrt(
        squared_sum / occurrence_count.clamp_min(1).to(dtype=squared_sum.dtype)
    )



def _patch_glm_moe_method(
    glm_moe,
    layer: int,
    router,
    num_experts: int,
    top_k: int,
    state: dict,
    current: dict,
    sensitivity_enabled: bool,
) -> None:
    """Replace Glm4MoeMoE.moe() to capture expert outputs for REAP/sensitivity.

    Runs each expert exactly once (no extra forward passes in standard mode).
    In SENSITIVITY_MODE the 9th expert is additionally evaluated per token.
    """

    def patched_moe(hidden_states: torch.Tensor,
                    topk_indices: torch.Tensor,
                    topk_weights: torch.Tensor) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = F.one_hot(topk_indices, num_classes=num_experts).permute(2, 0, 1)

        exp_id_parts: list[torch.Tensor] = []
        tok_id_parts: list[torch.Tensor] = []
        slot_id_parts: list[torch.Tensor] = []
        output_parts: list[torch.Tensor] = []

        for expert_idx in range(num_experts):
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)
            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = glm_moe.experts[expert_idx](expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)
                exp_id_parts.append(torch.full_like(token_indices, expert_idx))
                tok_id_parts.append(token_indices)
                slot_id_parts.append(weight_indices)
                output_parts.append(expert_output.detach())

        if exp_id_parts:
            exp_ids = torch.cat(exp_id_parts)
            tok_ids = torch.cat(tok_id_parts)
            slot_ids = torch.cat(slot_id_parts)
            outputs = torch.cat(output_parts)
            sort_order = torch.argsort(exp_ids, stable=True)
            perm = (tok_ids * top_k + slot_ids)[sort_order]

            callback = getattr(glm_moe, "_reap_capture_callback", None)
            if callback is not None:
                callback(
                    exp_ids[sort_order], outputs[sort_order], perm, top_k,
                    topk_indices=topk_indices, topk_weights=topk_weights, layer=layer,
                )

            if sensitivity_enabled:
                route = current["routes"].get(layer)
                if route is None:
                    raise RuntimeError(f"Missing router capture for GLM MoE layer {layer}")
                router_logits_all, _, _ = route

                # Filter to valid (non-padding) tokens — supports batch_size > 1
                mask = current.get("mask")
                if mask is not None:
                    valid_flat = mask.reshape(-1).bool().to(hidden_states.device)
                else:
                    valid_flat = torch.ones(hidden_states.shape[0], dtype=torch.bool, device=hidden_states.device)

                hs_v = hidden_states[valid_flat]
                topk_idx_v = topk_indices[valid_flat]
                topk_wts_v = topk_weights[valid_flat]
                scores_v = router_logits_all[valid_flat].sigmoid()

                # Find 9th expert for each valid token
                scores_for_choice = scores_v + router.e_score_correction_bias.unsqueeze(0)
                excluded = scores_for_choice.clone()
                excluded.scatter_(1, topk_idx_v, -torch.inf)
                ninth_id_v = excluded.argmax(dim=-1)

                # Run 9th expert for valid tokens only
                num_valid = hs_v.shape[0]
                ninth_out_v = torch.zeros(num_valid, hs_v.shape[-1], device=hs_v.device, dtype=hs_v.dtype)
                for ninth_exp_id in ninth_id_v.unique().tolist():
                    tok_mask = ninth_id_v == ninth_exp_id
                    ninth_out_v[tok_mask] = glm_moe.experts[ninth_exp_id](hs_v[tok_mask]).detach()

                top9_ids_v = torch.cat([topk_idx_v, ninth_id_v.unsqueeze(-1)], dim=-1)
                top9_scores_v = scores_v.gather(1, top9_ids_v)
                candidate_probs_v = top9_scores_v / top9_scores_v.sum(-1, keepdim=True).clamp_min(1e-20)

                tok_ids_sorted = tok_ids[sort_order]
                valid_routing = valid_flat[tok_ids_sorted]
                exp_ids_v = exp_ids[sort_order][valid_routing]
                outputs_v = outputs[sort_order][valid_routing]
                slot_ids_v = slot_ids[sort_order][valid_routing]
                remap = torch.full((hidden_states.shape[0],), -1, dtype=torch.long, device=hidden_states.device)
                remap[valid_flat] = torch.arange(num_valid, device=hidden_states.device)
                tok_ids_v = remap[tok_ids_sorted[valid_routing]]
                perm_v = tok_ids_v * top_k + slot_ids_v

                if layer not in state:
                    state[layer] = initialize_pruning_state(num_experts, device="cpu")
                    state[layer]["local_output_delta_squared_sum"] = torch.zeros(num_experts, dtype=torch.float64)
                    state[layer]["local_output_occurrence_count"] = torch.zeros(num_experts, dtype=torch.long)

                sum_sq, occ_count = _top_k_plus_1_local_output_delta_squares(
                    candidate_probs=candidate_probs_v.float(),
                    selected_ids=topk_idx_v,
                    selected_weights=topk_wts_v.float(),
                    routed_scale=1.0,
                    ninth_output=ninth_out_v,
                    expert_ids=exp_ids_v,
                    expert_outputs=outputs_v,
                    route_permutation=perm_v,
                    num_experts=num_experts,
                    top_k=top_k,
                )
                state[layer]["local_output_delta_squared_sum"] += sum_sq.cpu()
                state[layer]["local_output_occurrence_count"] += occ_count.cpu()

        return final_hidden_states.type(hidden_states.dtype)

    glm_moe.moe = patched_moe



def _prepare_sensitivity_samples(tokenizer) -> tuple[list[dict], dict]:
    """Select 12,228 deterministic unpadded rows from evol-codealpaca."""
    import numpy as np

    if DATASET_NAME != "theblackcat102/evol-codealpaca-v1" or DATASET_SPLIT != "train":
        raise RuntimeError("Sensitivity mode is fixed to Evol-CodeAlpaca train split")
    raw = _load_raw_dataset(DATASET_NAME, DATASET_SPLIT, subset=None)
    processor = CodeAlpacaChatDataset(
        dataset=raw, tokenizer=tokenizer, max_input_len=MODEL_MAX_LENGTH,
        pack_samples=False, split=DATASET_SPLIT, split_by_category=False,
        return_vllm_tokens_prompt=False, truncate=False, batch_size=1,
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
        samples.append({
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        })
        selected_indices.append(int(row_index))
        token_lengths.append(length)
        token_hashes.append(hashlib.sha256(id_bytes).hexdigest())
        if len(samples) == MAX_SAMPLES:
            break
    if len(samples) != MAX_SAMPLES:
        raise RuntimeError(f"Only {len(samples)} eligible examples found; need {MAX_SAMPLES}")
    if len(set(selected_indices)) != MAX_SAMPLES:
        raise RuntimeError("Sensitivity sample selection contains duplicate source rows")

    manifest_payload = {
        "dataset_name": DATASET_NAME, "split": DATASET_SPLIT, "seed": SEED,
        "max_length": MODEL_MAX_LENGTH, "selected_row_indices": selected_indices,
        "token_lengths": token_lengths, "token_id_sha256": token_hashes,
    }
    sample_hash = hashlib.sha256(
        json.dumps(manifest_payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        **manifest_payload,
        "selected_examples": len(samples),
        "sample_set_sha256": sample_hash,
        "padding": False, "truncation": False, "packing": False,
    }
    return samples, manifest


def _load_batches(tokenizer) -> list[dict[str, torch.Tensor]]:
    data = load_category_batches(
        dataset_name=DATASET_NAME, split=DATASET_SPLIT, subset=None,
        tokenizer=tokenizer, model_max_length=MODEL_MAX_LENGTH, batch_size=BATCH_SIZE,
        split_by_category=False, return_vllm_tokens_prompt=False, truncate=False,
        batches_per_category=MAX_SAMPLES, pack_samples=PACK_SAMPLES,
    )
    batches = next(iter(data.values()))
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



def _save_sensitivity_checkpoint(output_dir, state, stability, manifest, processed, tokens, elapsed):
    path = output_dir / f"local_output_sensitivity_checkpoint_{processed:09d}.pt"
    torch.save({
        "state": state, "processed_examples": processed,
        "processed_tokens": tokens, "elapsed_seconds": elapsed,
        "sample_set_sha256": manifest["sample_set_sha256"],
        "stability_accumulators": stability.accumulators,
        "stability_num_layers": stability.num_layers,
        "stability_num_experts": stability.num_experts,
        "stability_total_tokens": stability.total_tokens,
        "stability_batch_token_counts": stability._batch_token_counts,
        "stability_generator_state": stability.generator.get_state(),
    }, path)
    stability.save_accumulators(
        output_dir / f"local_output_router_stability_checkpoint_{processed:09d}.npz"
    )
    print(f"Saved sensitivity checkpoint: {path} (examples={processed}, tokens={tokens})", flush=True)


def _restore_sensitivity_checkpoint(output_dir, state, stability, manifest):
    checkpoints = sorted(output_dir.glob("local_output_sensitivity_checkpoint_*.pt"))
    if not checkpoints:
        return 0, 0, 0.0
    ckpt = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
    if ckpt.get("sample_set_sha256") != manifest["sample_set_sha256"]:
        raise RuntimeError("Checkpoint belongs to a different sample set")
    restored = ckpt["state"]
    if set(restored) != set(state):
        raise RuntimeError("Checkpoint has a different layer set")
    state.clear()
    state.update(restored)
    stability.accumulators = ckpt["stability_accumulators"]
    stability.num_layers = ckpt["stability_num_layers"]
    stability.num_experts = ckpt["stability_num_experts"]
    stability.total_tokens = ckpt["stability_total_tokens"]
    stability._batch_token_counts = ckpt["stability_batch_token_counts"]
    stability.generator.set_state(ckpt["stability_generator_state"])
    return int(ckpt["processed_examples"]), int(ckpt["processed_tokens"]), float(ckpt["elapsed_seconds"])


def _save_calibration_checkpoint(state, stability, output_dir, sequences, tokens, seconds):
    path = output_dir / f"calibration_state_checkpoint_{sequences:09d}.pt"
    torch.save({
        "state": state, "processed_packed_sequences": sequences,
        "total_tokens": tokens, "calibration_seconds": seconds,
    }, path)
    stability.save_accumulators(
        output_dir / f"router_stability_checkpoint_{sequences:09d}.npz"
    )
    print(f"Saved calibration checkpoint: {path} (sequences={sequences}, tokens={tokens})", flush=True)



def main() -> None:
    set_seed(SEED)
    started = time.perf_counter()

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    if getattr(config, "model_type", None) != "glm4_moe":
        raise RuntimeError(
            f"Expected glm4_moe checkpoint, got {getattr(config, 'model_type', None)!r}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=_glm_device_map(config),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    if SENSITIVITY_MODE:
        for param in model.parameters():
            param.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    sample_manifest = None
    if SENSITIVITY_MODE:
        batches, sample_manifest = _prepare_sensitivity_samples(tokenizer)
    else:
        batches = _load_batches(tokenizer)
    if not batches:
        raise RuntimeError("No calibration examples were loaded")

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if sample_manifest is not None:
        (output_dir / "sample_manifest.json").write_text(json.dumps(sample_manifest, indent=2) + "\n")

    stability_dir = output_dir / "router_stability"
    stability = RouterStabilityCollector(
        stability_dir,
        variance_multipliers=VARIANCE_MULTIPLIERS,
        seed=SEED,
        boundary=4,
        max_tokens=ROUTER_MAX_TOKENS,
        top_k=int(config.num_experts_per_tok),
        save_events=SAVE_ROUTING_EVENTS,
    )

    state: dict[int, dict] = {}
    current: dict = {"mask": None, "batch": 0, "routes": {}}
    handles = []
    layer_names: dict[int, str] = {}

    for name, module in model.named_modules():
        if not (hasattr(module, "experts") and hasattr(module, "gate")):
            continue
        layer = _layer_index(name)
        if layer is None:
            continue
        layer_names[layer] = name
        router = module.gate
        num_exp = _num_experts(router)
        top_k_val = _top_k(router)

        state[layer] = initialize_pruning_state(num_exp, device="cpu")
        state[layer]["moe_residual_contribution_sum"] = torch.zeros((), dtype=torch.float64)
        state[layer]["moe_residual_contribution_count"] = torch.zeros((), dtype=torch.long)
        if SENSITIVITY_MODE:
            state[layer]["local_output_delta_squared_sum"] = torch.zeros(num_exp, dtype=torch.float64)
            state[layer]["local_output_occurrence_count"] = torch.zeros(num_exp, dtype=torch.long)

        _wrap_glm_router_forward(router, layer, current)

        # MoE residual contribution (post-hook)
        def make_moe_hook(l, moe):
            rtr = moe.gate

            def hook(_module, args, output):
                moe_output = output[0] if isinstance(output, (tuple, list)) else output
                if torch.is_tensor(moe_output) and moe_output.shape == args[0].shape:
                    hidden_all = args[0].reshape(-1, args[0].shape[-1])
                    mask = current["mask"]
                    valid = torch.ones(hidden_all.shape[0], device=hidden_all.device, dtype=torch.bool)
                    if mask is not None:
                        valid = mask.reshape(-1).to(hidden_all.device).bool()
                    contribution = torch.linalg.vector_norm(moe_output.reshape(-1, moe_output.shape[-1]), dim=-1)[valid]
                    residual = torch.linalg.vector_norm(hidden_all, dim=-1)[valid]
                    state[l]["moe_residual_contribution_sum"] += (
                        (contribution / (residual + torch.finfo(args[0].dtype).eps))
                        .detach().sum().double().cpu()
                    )
                    state[l]["moe_residual_contribution_count"] += torch.tensor(
                        contribution.numel(), dtype=torch.long
                    )
                # Router stability
                route = current["routes"].get(l)
                if route is None:
                    return
                router_logits, topk_indices, topk_weights = route
                hidden_all = args[0].reshape(-1, args[0].shape[-1])
                mask = current["mask"]
                valid = torch.ones(hidden_all.shape[0], device=hidden_all.device, dtype=torch.bool)
                if mask is not None:
                    valid = mask.reshape(-1).to(hidden_all.device).bool()
                hidden = hidden_all[valid]
                if hidden.numel() == 0:
                    return
                top_k_local = _top_k(rtr)
                # Build original_route tuple expected by collect_router_observation
                logits_all = router_logits.reshape(-1, router_logits.shape[-1])
                selected_all = topk_indices.reshape(-1, top_k_local)
                # top12_ids must be top_k + boundary wide to match _route_once output
                top12_count = min(top_k_local + stability.boundary, logits_all.shape[-1])
                top12_vals, top12_ids = logits_all[valid].float().topk(top12_count, dim=-1, sorted=True)
                original_route = (
                    logits_all[valid],
                    logits_all[valid].float(),
                    top12_ids,
                    top12_vals,
                    selected_all[valid],
                    topk_weights.reshape(-1, top_k_local)[valid].float(),
                )
                with torch.no_grad():
                    observation = collect_router_observation(
                        rtr, hidden, stability.variance_multipliers, stability.generator,
                        top_k=top_k_local, boundary=stability.boundary,
                        original_route=original_route,
                    )
                stability.observe(l, current["batch"], observation)

            return hook

        handles.append(module.register_forward_hook(make_moe_hook(layer, module)))

        # Patch the moe() method and wire REAP callback
        _patch_glm_moe_method(
            module, layer, router, num_exp, top_k_val, state, current,
            sensitivity_enabled=SENSITIVITY_MODE,
        )

        def make_reap_callback(l, n_experts, tk):
            def callback(expert_ids, expert_outputs, perm, num_top_k, *,
                         topk_indices=None, topk_weights=None, layer=l):
                route = current["routes"].get(l)
                if route is None:
                    return
                router_logits, route_indices, route_weights = route
                selected_all = route_indices.reshape(-1, tk)
                logits_all = router_logits.reshape(-1, n_experts)
                mask = current["mask"]
                if mask is None:
                    valid = torch.ones(selected_all.shape[0], device=selected_all.device, dtype=torch.bool)
                else:
                    valid = mask.reshape(-1).to(selected_all.device).bool()
                original_token = perm.reshape(-1) // tk
                row_valid = valid[original_token] & (expert_ids.reshape(-1) < n_experts)
                if not row_valid.any():
                    return
                valid_token_ids = torch.nonzero(valid, as_tuple=False).reshape(-1)
                compact = torch.full((valid.numel(),), -1, device=valid.device, dtype=torch.long)
                compact[valid_token_ids] = torch.arange(valid_token_ids.numel(), device=valid.device)
                with torch.no_grad():
                    update_pruning_state_from_routed_outputs(
                        state[l],
                        expert_ids=expert_ids.reshape(-1)[row_valid],
                        expert_outputs=expert_outputs[row_valid],
                        token_indices=compact[original_token[row_valid]],
                        selected_experts=selected_all[valid],
                        router_logits=logits_all[valid],
                        num_experts=n_experts,
                        renormalize_router_weights=True,
                    )
            return callback

        module._reap_capture_callback = make_reap_callback(layer, num_exp, top_k_val)

    expected_total_layers = int(config.num_hidden_layers)
    expected_moe_layers = expected_total_layers - int(getattr(config, "first_k_dense_replace", 1))
    if len(layer_names) != expected_moe_layers:
        raise RuntimeError(
            f"Expected {expected_moe_layers} GLM MoE layers, found {len(layer_names)}: {layer_names}"
        )

    input_device = model.get_input_embeddings().weight.device
    calibration_started = time.perf_counter()
    total_examples = total_tokens = 0
    next_checkpoint = CHECKPOINT_EVERY if CHECKPOINT_EVERY > 0 else None
    progress = tqdm(
        batches if not SENSITIVITY_MODE else range(len(batches)),
        desc="GLM sensitivity calibration" if SENSITIVITY_MODE else "GLM calibration",
        unit="example" if SENSITIVITY_MODE else "batch",
    )

    try:
        if SENSITIVITY_MODE:
            assert sample_manifest is not None
            existing = sorted(output_dir.glob("local_output_sensitivity_checkpoint_*.pt"))
            if not existing:
                # Feasibility check on the longest sequence
                longest = max(range(len(batches)), key=lambda i: batches[i]["input_ids"].shape[-1])
                longest_len = int(batches[longest]["input_ids"].shape[-1])
                print(
                    f"Feasibility check on longest selected row: tokens={longest_len}", flush=True
                )
                try:
                    sample = batches[longest]
                    current["batch"] = -1
                    current["mask"] = sample["attention_mask"]
                    current["routes"].clear()
                    with torch.no_grad():
                        model(**{
                            k: v.to(input_device) if torch.is_tensor(v) else v
                            for k, v in sample.items()
                        }, use_cache=False, return_dict=True)
                except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                    raise RuntimeError(
                        f"Longest-sequence feasibility check failed (tokens={longest_len}): {exc}"
                    ) from exc

                # Reset state after preflight
                state.clear()
                for l in layer_names:
                    num_exp = _num_experts(model.get_submodule(layer_names[l]).gate)
                    state[l] = initialize_pruning_state(num_exp, device="cpu")
                    state[l]["moe_residual_contribution_sum"] = torch.zeros((), dtype=torch.float64)
                    state[l]["moe_residual_contribution_count"] = torch.zeros((), dtype=torch.long)
                    state[l]["local_output_delta_squared_sum"] = torch.zeros(num_exp, dtype=torch.float64)
                    state[l]["local_output_occurrence_count"] = torch.zeros(num_exp, dtype=torch.long)
                stability = RouterStabilityCollector(
                    stability_dir, variance_multipliers=(1.0,), seed=42,
                    boundary=4, max_tokens=None,
                    top_k=int(config.num_experts_per_tok), save_events=False,
                )
                current["routes"].clear()

            start_index, processed_tokens, prior_seconds = _restore_sensitivity_checkpoint(
                output_dir, state, stability, sample_manifest
            )
            total_examples = start_index
            total_tokens = sum(sample_manifest["token_lengths"][:start_index])
            calibration_started = time.perf_counter()

            last_checkpoint_examples = start_index
            for batch_start in range(start_index, len(batches), SENSITIVITY_BATCH_SIZE):
                group = batches[batch_start: batch_start + SENSITIVITY_BATCH_SIZE]
                actual_bs = len(group)
                if actual_bs == 1:
                    input_ids = group[0]["input_ids"]
                    attn_mask = group[0]["attention_mask"]
                else:
                    max_len = max(s["input_ids"].shape[-1] for s in group)
                    input_ids = torch.zeros(actual_bs, max_len, dtype=torch.long)
                    attn_mask = torch.zeros(actual_bs, max_len, dtype=torch.long)
                    for j, s in enumerate(group):
                        L = s["input_ids"].shape[-1]
                        input_ids[j, :L] = s["input_ids"][0]
                        attn_mask[j, :L] = 1
                current["batch"] = batch_start
                current["mask"] = attn_mask
                current["routes"].clear()
                with torch.no_grad():
                    model(
                        input_ids=input_ids.to(input_device),
                        attention_mask=attn_mask.to(input_device),
                        use_cache=False, return_dict=True,
                    )
                batch_tokens = int(attn_mask.sum())
                total_examples = batch_start + actual_bs
                total_tokens += batch_tokens
                processed_tokens += batch_tokens
                progress.update(actual_bs)
                progress.set_postfix(examples=total_examples, tokens=total_tokens)
                if total_examples - last_checkpoint_examples >= CHECKPOINT_EVERY:
                    last_checkpoint_examples = total_examples
                    _save_sensitivity_checkpoint(
                        output_dir, state, stability, sample_manifest,
                        total_examples, processed_tokens,
                        prior_seconds + time.perf_counter() - calibration_started,
                    )
            calibration_seconds = prior_seconds + time.perf_counter() - calibration_started
            _save_sensitivity_checkpoint(
                output_dir, state, stability, sample_manifest,
                total_examples, processed_tokens, calibration_seconds,
            )
        else:
            with torch.inference_mode():
                for batch_id, batch in enumerate(progress):
                    batch_size = int(batch["input_ids"].shape[0])
                    for offset in range(0, batch_size, FORWARD_BATCH_SIZE):
                        microbatch = {
                            k: v[offset: offset + FORWARD_BATCH_SIZE]
                            if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == batch_size else v
                            for k, v in batch.items()
                        }
                        current["batch"] = batch_id * 100000 + offset
                        current["mask"] = microbatch.get("attention_mask")
                        current["routes"].clear()
                        model(**{
                            k: v.to(input_device) if torch.is_tensor(v) else v
                            for k, v in microbatch.items()
                        }, use_cache=False, return_dict=True)
                    total_examples += batch_size
                    batch_mask = batch.get("attention_mask")
                    total_tokens += int(batch_mask.sum()) if batch_mask is not None else int(batch["input_ids"].numel())
                    progress.set_postfix(examples=total_examples, tokens=total_tokens)
                    if next_checkpoint is not None and total_examples >= next_checkpoint:
                        _save_calibration_checkpoint(
                            state, stability, output_dir, total_examples, total_tokens,
                            time.perf_counter() - calibration_started,
                        )
                        next_checkpoint += CHECKPOINT_EVERY
    finally:
        progress.close()
        for handle in handles:
            handle.remove()
        for module in model.modules():
            if hasattr(module, "_reap_capture_callback"):
                delattr(module, "_reap_capture_callback")
        stability.finalize()

    if not SENSITIVITY_MODE:
        calibration_seconds = time.perf_counter() - calibration_started

    if len(state) != expected_moe_layers:
        raise RuntimeError(
            f"Missing calibration state for layers: {set(layer_names) - set(state)}"
        )

    top_k_val = int(config.num_experts_per_tok)
    num_experts_val = int(config.n_routed_experts)

    if SENSITIVITY_MODE:
        assert sample_manifest is not None
        if (expected_moe_layers, num_experts_val, top_k_val) != (45, 128, 8):
            raise RuntimeError(
                "Sensitivity protocol expects exactly 45 MoE layers, 128 experts, top-k 8; "
                f"got {(expected_moe_layers, num_experts_val, top_k_val)}"
            )

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

    torch.save(state, output_dir / "calibration_state.pt")
    if SENSITIVITY_MODE:
        torch.save(
            {
                "interpretation": "per-expert one-at-a-time local MoE-output RMS-L2 deltas",
                "processed_token_count": total_tokens,
                "layers": sensitivity_results,
            },
            output_dir / "local_output_sensitivity_results.pt",
        )

    metadata = {
        "model_path": MODEL_PATH, "model_type": "glm4_moe",
        "dataset_name": DATASET_NAME, "split": DATASET_SPLIT,
        "max_samples": MAX_SAMPLES,
        "retained_examples": total_examples,
        "max_packed_sequences": MAX_SAMPLES,
        "retained_packed_sequences": total_examples,
        "pack_samples": PACK_SAMPLES,
        "checkpoint_every_packed_sequences": CHECKPOINT_EVERY,
        "total_tokens": total_tokens, "batch_size": BATCH_SIZE,
        "forward_batch_size": FORWARD_BATCH_SIZE,
        "model_max_length": MODEL_MAX_LENGTH, "seed": SEED,
        "variance_multipliers": list(VARIANCE_MULTIPLIERS),
        "routing_events_saved": SAVE_ROUTING_EVENTS,
        "num_total_layers": expected_total_layers,
        "num_moe_layers": expected_moe_layers,
        "num_experts": num_experts_val,
        "top_k": top_k_val,
        "calibration_seconds": calibration_seconds,
        "total_seconds": time.perf_counter() - started,
        "examples_per_second": total_examples / max(calibration_seconds, 1e-9),
        "tokens_per_second": total_tokens / max(calibration_seconds, 1e-9),
    }
    if SENSITIVITY_MODE:
        metadata.update({
            "mode": "single_run_local_output_sensitivity",
            "smoke_test": SENSITIVITY_SMOKE_SAMPLES > 0,
            "sample_set_sha256": sample_manifest["sample_set_sha256"],
            "processed_tokens": total_tokens,
            "max_sequence_length": MODEL_MAX_LENGTH,
        })
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
