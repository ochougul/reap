"""Measure exact local MoE damage curves for Qwen3.8-Flash-Next."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import qwen4_calibrate as calibration


EXPECTED_SAMPLE_SHA256 = "042789c5a55407f9205f6d8dbdfdbd446ea72b452442d2835f081e7d0901466a"
DEFAULT_REAP_STATE = pathlib.Path(
    "artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-"
    "12228-16384-unpruned/calibration_state.pt"
)
DEFAULT_OUTPUT_DIR = pathlib.Path(
    "artifacts/Qwen3.8-Flash-Next-FP8/reap-damage-curves-evol-codealpaca-"
    "12228-16384-unpruned"
)
K_GRID = tuple(range(0, 385, 32))


def _config() -> tuple[pathlib.Path, pathlib.Path, float, int, int, int, int, tuple[int, ...]]:
    reap_state = pathlib.Path(os.environ.get("QWEN4_CURVE_REAP_STATE", DEFAULT_REAP_STATE))
    output_dir = pathlib.Path(os.environ.get("QWEN4_CURVE_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    fraction = float(os.environ.get("QWEN4_CURVE_TOKEN_FRACTION", "0.01"))
    max_samples = int(os.environ.get("QWEN4_CURVE_MAX_SAMPLES", "12228"))
    max_length = int(os.environ.get("QWEN4_CURVE_MAX_LENGTH", "16384"))
    seed = int(os.environ.get("QWEN4_CURVE_SEED", "42"))
    chunk_pairs = int(os.environ.get("QWEN4_CURVE_EXPERT_PAIR_CHUNK", "1024"))
    k_grid = tuple(
        int(value.strip())
        for value in os.environ.get("QWEN4_CURVE_K_GRID", ",".join(map(str, K_GRID))).split(",")
        if value.strip()
    )
    if not 0 < fraction <= 1:
        raise ValueError("QWEN4_CURVE_TOKEN_FRACTION must be in (0, 1]")
    if not 1 <= max_samples <= 12228:
        raise ValueError("QWEN4_CURVE_MAX_SAMPLES must be in [1, 12228]")
    if max_length < 2:
        raise ValueError("QWEN4_CURVE_MAX_LENGTH must be at least 2")
    if chunk_pairs < 1:
        raise ValueError("QWEN4_CURVE_EXPERT_PAIR_CHUNK must be positive")
    if len(k_grid) < 2 or k_grid[0] != 0 or k_grid[-1] != 384:
        raise ValueError("QWEN4_CURVE_K_GRID must be increasing, start at 0, and end at 384")
    if any(left >= right for left, right in zip(k_grid, k_grid[1:])):
        raise ValueError("QWEN4_CURVE_K_GRID must be strictly increasing")
    return reap_state, output_dir, fraction, max_samples, max_length, seed, chunk_pairs, k_grid


def _load_reap_orders(path: pathlib.Path) -> tuple[dict[int, torch.Tensor], str]:
    data = path.read_bytes()
    state = torch.load(path, map_location="cpu", weights_only=False)
    if len(state) != 48:
        raise ValueError(f"REAP state must have 48 layers, found {len(state)}")
    orders = {}
    for key, layer_state in state.items():
        layer = int(key)
        scores = torch.as_tensor(layer_state["reap"]).reshape(-1).double()
        if scores.numel() != 512 or not torch.isfinite(scores).all():
            raise ValueError(f"Layer {layer} has malformed or non-finite REAP scores")
        # Stable sorting preserves the original expert ID order when scores tie.
        orders[layer] = torch.argsort(scores, stable=True)
    if sorted(orders) != list(range(48)):
        raise ValueError("REAP state layer IDs must be exactly 0..47")
    return orders, hashlib.sha256(data).hexdigest()


def _empty_state() -> dict[int, dict[str, torch.Tensor]]:
    state = {}
    for layer in range(48):
        state[layer] = {
            key: torch.zeros(len(K_GRID), dtype=torch.float64)
            for key in (
                "absolute_l2_sum",
                "relative_l2_sum",
                "residual_normalized_l2_sum",
                "p95_relative_l2_sum",
                "p99_relative_l2_sum",
                "top1pct_relative_l2_sum",
                "routing_changed_token_sum",
                "routing_jaccard_sum",
            )
        }
        state[layer]["token_count"] = torch.zeros((), dtype=torch.long)
        state[layer]["tail_sample_count"] = torch.zeros((), dtype=torch.long)
    return state


def _sample_token_positions(length: int, fraction: float, seed: int, sample_index: int) -> torch.Tensor:
    count = min(length, max(1, int(math.ceil(length * fraction))))
    generator = torch.Generator(device="cpu")
    generator.manual_seed((seed + 0x9E3779B1 * (sample_index + 1)) % (2**63 - 1))
    return torch.randperm(length, generator=generator)[:count].sort().values


def _new_state_like(state: dict[int, dict[str, torch.Tensor]]) -> dict[int, dict[str, torch.Tensor]]:
    return {layer: {key: value.clone() for key, value in values.items()} for layer, values in state.items()}


def _checkpoint_path(output_dir: pathlib.Path, examples: int) -> pathlib.Path:
    return output_dir / f"curve_checkpoint_{examples:09d}.pt"


def _save_checkpoint(
    output_dir, state, examples, seconds, sample_sha, rank=0,
    sequence_work_seconds=0.0, sequence_work_examples=0,
):
    torch.save({"state": state, "processed_examples": examples,
                "elapsed_seconds": seconds, "sample_set_sha256": sample_sha,
                "sequence_work_seconds": sequence_work_seconds,
                "sequence_work_examples": sequence_work_examples},
               _checkpoint_path(output_dir, examples))
    print(f"Saved damage-curve checkpoint at {examples} examples (rank {rank})", flush=True)


def _restore_checkpoint(output_dir, state, sample_sha):
    files = sorted(output_dir.glob("curve_checkpoint_*.pt"))
    if not files:
        return state, 0, 0.0, 0.0, 0
    checkpoint = torch.load(files[-1], map_location="cpu", weights_only=False)
    if checkpoint.get("sample_set_sha256") != sample_sha:
        raise RuntimeError("Curve checkpoint belongs to a different calibration sample set")
    restored = checkpoint["state"]
    if set(restored) != set(state) or any(set(restored[x]) != set(state[x]) for x in state):
        raise RuntimeError("Curve checkpoint has a different state schema")
    return (
        restored,
        int(checkpoint["processed_examples"]),
        float(checkpoint["elapsed_seconds"]),
        float(checkpoint.get("sequence_work_seconds", 0.0)),
        int(checkpoint.get("sequence_work_examples", 0)),
    )


def _capture_callback_factory(current: dict[str, Any], layer: int, top_k: int):
    def callback(expert_ids, expert_outputs, perm, num_top_k):
        mode = current.get("capture_mode")
        if mode == "curve_baseline":
            if num_top_k != top_k or top_k != 10:
                raise RuntimeError(f"Baseline capture expected top-10; received {num_top_k}")
            if current.get("ep_enabled", False):
                # EP returns the combined result to the source rank. Curve mode
                # re-evaluates each required expert through that return path.
                return
            payload = current.get("curve_payloads", {}).get(layer)
            if payload is None or not payload.get("active", False):
                return
            route = current["routes"][layer]
            token_rows = perm.reshape(-1).long() // num_top_k
            slot_rows = perm.reshape(-1).long() % num_top_k
            sampled_token_to_row = payload["sampled_token_to_row"].to(token_rows.device)
            sampled_rows = sampled_token_to_row[token_rows]
            keep = sampled_rows >= 0
            if not keep.any():
                return
            if not torch.equal(
                route[4][token_rows[keep], slot_rows[keep]].to(expert_ids.device),
                expert_ids.reshape(-1)[keep],
            ):
                raise RuntimeError(f"Layer {layer}: baseline callback route alignment failed")
            rows = sampled_rows[keep]
            slots = slot_rows[keep].to(rows.device)
            linear = rows * top_k + slots
            payload["baseline_outputs"].index_copy_(0, linear, expert_outputs[keep].detach())
            payload["baseline_seen"][linear] = True
            return
        if mode == "curve_candidates":
            payload = current.get("curve_candidate_capture")
            if payload is None:
                raise RuntimeError("Candidate expert callback has no active output buffer")
            output_rows = perm.reshape(-1).long()
            if output_rows.numel() != expert_outputs.shape[0] or output_rows.numel() != expert_ids.numel():
                raise RuntimeError("Candidate output callback layout changed unexpectedly")
            if output_rows.numel() and (output_rows.min() < 0 or output_rows.max() >= payload["outputs"].shape[0]):
                raise RuntimeError("Candidate callback permutation is outside its chunk")
            if not torch.equal(expert_ids.reshape(-1).to(payload["expert_ids"].device),
                               payload["expert_ids"][output_rows]):
                raise RuntimeError("Candidate expert output IDs do not align with requested pairs")
            payload["outputs"].index_copy_(0, output_rows.to(payload["outputs"].device), expert_outputs.detach())
            payload["seen"][output_rows.to(payload["seen"].device)] = True
            return
        if mode == "curve_ep_candidates":
            # The EP dispatcher combines each one-expert request back to its
            # source rank; the caller reads experts.forward's returned tensor.
            return

        # The normal REAP callback is not needed in this mode.

    return callback


def _candidate_outputs(moe, layer, hidden, pairs, current, chunk_pairs):
    result = torch.empty((len(pairs), hidden.shape[-1]), device=hidden.device, dtype=hidden.dtype)
    seen = torch.zeros(len(pairs), device=hidden.device, dtype=torch.bool)
    for start in range(0, len(pairs), chunk_pairs):
        stop = min(start + chunk_pairs, len(pairs))
        block = pairs[start:stop]
        token_rows = torch.tensor([item[0] for item in block], device=hidden.device, dtype=torch.long)
        expert_ids = torch.tensor([item[1] for item in block], device=hidden.device, dtype=torch.long)
        pair_hidden = hidden.index_select(0, token_rows)
        indices = expert_ids[:, None]
        weights = torch.ones((len(block), 1), device=hidden.device, dtype=hidden.dtype)
        outputs = torch.empty((len(block), hidden.shape[-1]), device=hidden.device, dtype=hidden.dtype)
        block_seen = torch.zeros(len(block), device=hidden.device, dtype=torch.bool)
        ep_enabled = current.get("ep_enabled", False)
        if ep_enabled:
            current["capture_mode"] = "curve_ep_candidates"
        else:
            current["capture_mode"] = "curve_candidates"
            current["curve_candidate_capture"] = {
                "outputs": outputs,
                "seen": block_seen,
                "expert_ids": expert_ids,
            }
        with torch.no_grad():
            expert_result = moe.experts(pair_hidden, indices, weights)
        if ep_enabled:
            outputs.copy_(expert_result.reshape_as(outputs))
            block_seen.fill_(True)
        if not block_seen.all():
            raise RuntimeError(f"Layer {layer}: candidate expert output buffer overflow or missing rows")
        result[start:stop].copy_(outputs)
        seen[start:stop] = block_seen
    current["curve_candidate_capture"] = None
    return result, seen


def _record_layer_curve(
    layer: int,
    moe: torch.nn.Module,
    module_args,
    module_output,
    current: dict[str, Any],
    state: dict[int, dict[str, torch.Tensor]],
    orders: dict[int, torch.Tensor],
    k_grid: tuple[int, ...],
    chunk_pairs: int,
) -> None:
    payload = current.get("curve_payloads", {}).get(layer)
    if payload is None or not payload.get("active", False):
        return
    if not payload["baseline_seen"].all():
        if current.get("ep_enabled", False):
            pass
        else:
            missing = int((~payload["baseline_seen"]).sum().item())
            raise RuntimeError(f"Layer {layer}: original routed output capture missed {missing} slots")

    hidden_all = module_args[0].reshape(-1, module_args[0].shape[-1])
    positions = payload["positions"].to(hidden_all.device)
    hidden = hidden_all.index_select(0, positions)
    route = current["routes"][layer]
    logits = route[0].index_select(0, positions).float()
    original_ids = route[4].index_select(0, positions).long()
    original_weights = route[5].index_select(0, positions).float()
    routed_outputs = payload["baseline_outputs"].reshape(-1, 10, hidden.shape[-1]).float()

    all_probs = logits.softmax(-1)
    selected_ids_by_k = []
    selected_weights_by_k = []
    for k in k_grid:
        if k == 0:
            ids = original_ids
            weights = original_weights
        else:
            valid_logits = logits.clone()
            removed = orders[layer][:k].to(valid_logits.device)
            valid_logits[:, removed] = -torch.inf
            probs = valid_logits.softmax(-1)
            ids = probs.topk(10, dim=-1, sorted=True).indices
            weights = probs.gather(1, ids)
            if getattr(moe.gate, "norm_topk_prob", False):
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
            weights = weights * getattr(moe.gate, "routed_scaling_factor", 1.0)
        selected_ids_by_k.append(ids)
        selected_weights_by_k.append(weights)

    computed_zero = selected_ids_by_k[0]
    if not torch.equal(computed_zero, original_ids):
        raise RuntimeError(f"Layer {layer}: unmasked top-10 routing does not match model route")
    missing_pairs: list[tuple[int, int]] = []
    baseline_id_set = (
        [set() for _ in range(len(original_ids))]
        if current.get("ep_enabled", False)
        else [set(map(int, row)) for row in original_ids.detach().cpu().tolist()]
    )
    pair_set = set()
    for ids in selected_ids_by_k:
        for token_row, row in enumerate(ids.detach().cpu().tolist()):
            for expert in row:
                if expert not in baseline_id_set[token_row]:
                    pair_set.add((token_row, int(expert)))
    missing_pairs = sorted(pair_set)
    candidate_by_pair: dict[tuple[int, int], torch.Tensor] = {}
    if missing_pairs:
        output, _ = _candidate_outputs(moe, layer, hidden, missing_pairs, current, chunk_pairs)
        candidate_by_pair = {pair: output[index] for index, pair in enumerate(missing_pairs)}
    current["capture_mode"] = "curve_baseline"

    if current.get("ep_enabled", False):
        routed_baseline = torch.zeros(
            (len(original_ids), hidden.shape[-1]), device=hidden.device, dtype=torch.float32
        )
        for token_row, expert_row in enumerate(original_ids.detach().cpu().tolist()):
            for slot, expert in enumerate(expert_row):
                routed_baseline[token_row] += (
                    candidate_by_pair[(token_row, int(expert))].float()
                    * original_weights[token_row, slot].to(hidden.device)
                )
    else:
        routed_baseline = (routed_outputs * original_weights.to(hidden.device)[:, :, None]).sum(dim=1)

    # Preserve unchanged non-routed contributions (the shared expert, when present).
    output = module_output[0] if isinstance(module_output, (tuple, list)) else module_output
    original_full = output.reshape(-1, output.shape[-1]).index_select(0, positions).float()
    shared_contribution = original_full - routed_baseline
    baseline_norm = torch.linalg.vector_norm(original_full, dim=-1).clamp_min(1e-12)
    residual = torch.linalg.vector_norm(hidden.float(), dim=-1).clamp_min(1e-12)
    original_sets = [set(map(int, row)) for row in original_ids.detach().cpu().tolist()]

    damages_abs = []
    damages_rel = []
    damages_resid = []
    changed_rates = []
    jaccards = []
    for curve_index, (ids, weights) in enumerate(zip(selected_ids_by_k, selected_weights_by_k)):
        expert_values = torch.empty((ids.shape[0], 10, hidden.shape[-1]), device=hidden.device, dtype=torch.float32)
        for token_row, expert_row in enumerate(ids.detach().cpu().tolist()):
            for slot, expert in enumerate(expert_row):
                if expert in baseline_id_set[token_row]:
                    original_slot = int((original_ids[token_row] == expert).nonzero()[0].item())
                    expert_values[token_row, slot] = routed_outputs[token_row, original_slot]
                else:
                    expert_values[token_row, slot] = candidate_by_pair[(token_row, int(expert))].float()
        routed = (expert_values * weights.float().to(hidden.device)[:, :, None]).sum(dim=1)
        counterfactual_full = shared_contribution + routed
        delta_norm = torch.linalg.vector_norm(counterfactual_full - original_full, dim=-1)
        rel = delta_norm / baseline_norm
        resid_rel = delta_norm / residual
        damages_abs.append(delta_norm)
        damages_rel.append(rel)
        damages_resid.append(resid_rel)
        changed = []
        overlap = []
        for token_row, expert_row in enumerate(ids.detach().cpu().tolist()):
            candidate_set = set(map(int, expert_row))
            changed.append(candidate_set != original_sets[token_row])
            overlap.append(len(candidate_set & original_sets[token_row]) / len(candidate_set | original_sets[token_row]))
        changed_rates.append(torch.tensor(changed, device=hidden.device, dtype=torch.float64))
        jaccards.append(torch.tensor(overlap, device=hidden.device, dtype=torch.float64))

    # A zero-prune result must reproduce the unmodified layer output within FP32
    # accumulation tolerance; this catches route weights and callback alignment bugs.
    if damages_abs[0].max().item() > 2e-3 * max(1.0, float(baseline_norm.max().item())):
        raise RuntimeError(f"Layer {layer}: k=0 curve differs from captured baseline")

    layer_state = state[layer]
    for index in range(len(k_grid)):
        abs_values = damages_abs[index].double()
        rel_values = damages_rel[index].double()
        residual_values = damages_resid[index].double()
        n = rel_values.numel()
        tail_count = max(1, int(math.ceil(n * 0.01)))
        layer_state["absolute_l2_sum"][index] += abs_values.sum().cpu()
        layer_state["relative_l2_sum"][index] += rel_values.sum().cpu()
        layer_state["residual_normalized_l2_sum"][index] += residual_values.sum().cpu()
        layer_state["p95_relative_l2_sum"][index] += torch.quantile(rel_values, 0.95).cpu()
        layer_state["p99_relative_l2_sum"][index] += torch.quantile(rel_values, 0.99).cpu()
        layer_state["top1pct_relative_l2_sum"][index] += rel_values.topk(tail_count).values.mean().cpu()
        layer_state["routing_changed_token_sum"][index] += changed_rates[index].sum().cpu()
        layer_state["routing_jaccard_sum"][index] += jaccards[index].sum().cpu()
    layer_state["token_count"] += len(positions)
    layer_state["tail_sample_count"] += 1


def _summary_rows(state: dict[int, dict[str, torch.Tensor]]) -> list[dict[str, Any]]:
    rows = []
    for layer in range(48):
        values = state[layer]
        tokens = int(values["token_count"])
        samples = int(values["tail_sample_count"])
        if tokens <= 0 or samples <= 0:
            raise RuntimeError(f"Layer {layer} has no sampled token observations")
        for index, k in enumerate(K_GRID):
            rows.append({
                "layer": layer,
                "k": k,
                "sampled_tokens": tokens,
                "mean_absolute_l2": float(values["absolute_l2_sum"][index] / tokens),
                "mean_relative_l2": float(values["relative_l2_sum"][index] / tokens),
                "mean_residual_normalized_l2": float(values["residual_normalized_l2_sum"][index] / tokens),
                "p95_relative_l2": float(values["p95_relative_l2_sum"][index] / samples),
                "p99_relative_l2": float(values["p99_relative_l2_sum"][index] / samples),
                "top1pct_relative_l2": float(values["top1pct_relative_l2_sum"][index] / samples),
                "routing_changed_token_fraction": float(values["routing_changed_token_sum"][index] / tokens),
                "routing_jaccard": float(values["routing_jaccard_sum"][index] / tokens),
            })
    return rows


def _write_outputs(output_dir, state, metadata):
    import csv

    rows = _summary_rows(state)
    fields = list(rows[0])
    with (output_dir / "curve_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    torch.save({"k_grid": K_GRID, "layers": state}, output_dir / "curve_state.pt")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _all_reduce_curve_state(state, torch_dist, rank: int) -> None:
    """Sum EP-rank statistics and counts before means/tails are finalized."""
    device = torch.device(f"cuda:{rank}")
    for layer_state in state.values():
        for value in layer_state.values():
            reduced = value.to(device)
            torch_dist.all_reduce(reduced, op=torch_dist.ReduceOp.SUM)
            value.copy_(reduced.cpu())


def main() -> None:
    reap_path, output_dir, token_fraction, max_samples, max_length, seed, chunk_pairs, k_grid = _config()
    global K_GRID
    K_GRID = k_grid
    calibration._patch_triton_kernel_compat()
    set_seed(seed)
    ep_enabled = os.environ.get("QWEN4_EXPERT_PARALLEL", "0").lower() in {"1", "true", "yes"}
    rank = 0
    torch_dist = None
    if ep_enabled:
        import torch.distributed as torch_dist_module

        torch_dist_module.init_process_group("nccl")
        rank = torch_dist_module.get_rank()
        torch.cuda.set_device(rank)
        torch_dist = torch_dist_module
        calibration._patch_ep_tp_compat()
        calibration._install_qwen4_ep_capture_dispatch()
    metadata_source = reap_path.parent / "metadata.json"
    sample_manifest_path = reap_path.parent / "sample_manifest.json"
    if not reap_path.exists() or not metadata_source.exists() or not sample_manifest_path.exists():
        raise FileNotFoundError("REAP state, calibration metadata, and sample_manifest.json are required")
    source_metadata = json.loads(metadata_source.read_text())
    sample_manifest = json.loads(sample_manifest_path.read_text())
    if source_metadata.get("sample_set_sha256") != EXPECTED_SAMPLE_SHA256:
        raise RuntimeError("REAP calibration metadata does not match the required sample-set SHA-256")
    if sample_manifest.get("sample_set_sha256") != EXPECTED_SAMPLE_SHA256:
        raise RuntimeError("REAP sample manifest does not match the required sample-set SHA-256")
    if max_samples == 12228 and int(source_metadata.get("retained_examples", -1)) != 12228:
        raise RuntimeError("Full curve runs require the completed 12,228-example calibration")
    orders, reap_sha256 = _load_reap_orders(reap_path)

    # Make the existing strict, unpadded sample loader use the completed run's protocol.
    calibration.DATASET_NAME = "theblackcat102/evol-codealpaca-v1"
    calibration.DATASET_SPLIT = "train"
    calibration.MODEL_MAX_LENGTH = max_length
    calibration.MAX_SAMPLES = max_samples
    calibration.SEED = 42
    config = AutoConfig.from_pretrained(calibration.MODEL_PATH, local_files_only=True, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    experts_implementation = (
        os.environ.get("QWEN4_EXPERTS_IMPLEMENTATION", "deepgemm")
        if ep_enabled else "grouped_mm"
    )
    load_kwargs = {
        "pretrained_model_name_or_path": calibration.MODEL_PATH,
        "dtype": None,
        "experts_implementation": experts_implementation,
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "offload_state_dict": True,
    }
    if ep_enabled:
        if experts_implementation != "deepgemm":
            raise RuntimeError("Qwen4 EP damage curves currently require the DeepGEMM experts implementation")
        from transformers.distributed import DistributedConfig

        load_kwargs["distributed_config"] = DistributedConfig(
            tp_size=int(os.environ.get("QWEN4_TP_SIZE", "4")),
            fsdp_size=1,
            ep_size=int(os.environ.get("QWEN4_EP_SIZE", "4")),
        )
    else:
        load_kwargs["device_map"] = calibration._qwen4_device_map(config)
        load_kwargs["offload_buffers"] = True
    model = AutoModelForImageTextToText.from_pretrained(**load_kwargs).eval()
    calibration._assert_fp8_loaded(model)
    if ep_enabled:
        calibration._shard_ep_expert_scales(model, rank)
    else:
        calibration._keep_ngram_embedding_on_cpu(model)
        calibration._bridge_qwen4_ple_layer(model)
    calibration._patch_sensitivity_rotary_device(model)
    if experts_implementation == "deepgemm":
        calibration._install_qwen4_deepgemm_capture_dispatch()
    else:
        calibration._install_qwen4_capture_dispatch()
    tokenizer = AutoTokenizer.from_pretrained(calibration.MODEL_PATH, local_files_only=True, trust_remote_code=True)
    samples, manifest = calibration._prepare_sensitivity_samples(tokenizer)
    if max_length == int(sample_manifest.get("max_length", 16384)):
        for key in ("selected_row_indices", "token_lengths", "token_id_sha256"):
            if manifest[key] != sample_manifest[key][:max_samples]:
                raise RuntimeError(
                    f"Reconstructed calibration sample prefix differs from source manifest ({key})"
                )
    run_sample_sha256 = manifest["sample_set_sha256"]

    local_output_dir = output_dir if rank == 0 else output_dir / f"rank{rank}"
    local_output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (output_dir / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if ep_enabled:
        torch_dist.barrier()
    input_device = model.get_input_embeddings().weight.device
    current: dict[str, Any] = {
        "routes": {},
        "capture_mode": "curve_baseline",
        "curve_payloads": {},
        "ep_enabled": ep_enabled,
        "rank": rank,
    }
    layer_modules: dict[int, torch.nn.Module] = {}
    handles = []
    state = _empty_state()
    expected_layers = int(getattr(text_config, "num_hidden_layers", 0))
    for name, module in model.named_modules():
        if not (hasattr(module, "experts") and hasattr(module, "gate")):
            continue
        layer = calibration._layer_index(name)
        if layer is None:
            continue
        layer_modules[layer] = module
        calibration._wrap_qwen4_router_forward(module.gate, layer, current)
        module.experts._reap_capture_callback = _capture_callback_factory(
            current, layer, calibration._top_k(module.gate)
        )
        calibration._wrap_qwen4_expert_forward(module.experts, layer, module.gate)

        def pre_hook(_module, args, _layer=layer):
            if not current.get("enabled", False):
                current["curve_payloads"].pop(_layer, None)
                return
            hidden = args[0].reshape(-1, args[0].shape[-1])
            sample_index = current["sample_index"]
            positions = _sample_token_positions(hidden.shape[0], token_fraction, seed, sample_index)
            inverse = torch.full((hidden.shape[0],), -1, dtype=torch.long)
            inverse[positions] = torch.arange(len(positions))
            current["curve_payloads"][_layer] = {
                "active": True,
                "positions": positions,
                "sampled_token_to_row": inverse,
                "baseline_outputs": torch.empty(
                    (len(positions) * 10, hidden.shape[-1]), device=hidden.device, dtype=hidden.dtype
                ),
                "baseline_seen": torch.zeros(len(positions) * 10, device=hidden.device, dtype=torch.bool),
            }

        def post_hook(_module, args, output, _layer=layer):
            if current.get("enabled", False):
                _record_layer_curve(
                    _layer, _module, args, output, current, state, orders, K_GRID, chunk_pairs
                )

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))
    if len(layer_modules) != expected_layers or expected_layers != 48:
        raise RuntimeError(f"Expected 48 Qwen4 MoE layers, found {sorted(layer_modules)}")

    # Reject incompatible or stale checkpoint state before the longest-row preflight.
    (
        state,
        start_index,
        prior_seconds,
        sequence_work_seconds,
        sequence_work_examples,
    ) = _restore_checkpoint(local_output_dir, state, run_sample_sha256)
    started = time.perf_counter()
    if start_index == 0:
        longest = max(range(len(samples)), key=lambda index: samples[index]["input_ids"].shape[-1])
        current["enabled"] = False
        with torch.no_grad():
            model.model(
                input_ids=samples[longest]["input_ids"].to(input_device),
                attention_mask=samples[longest]["attention_mask"].to(input_device),
                use_cache=False,
                return_dict=True,
            )
        current["routes"].clear()
        current["curve_payloads"].clear()

    next_checkpoint = ((start_index // 1024) + 1) * 1024
    progress = tqdm(
        range(start_index, len(samples)), desc="Qwen4 damage curves", unit="example",
        disable=ep_enabled and rank != 0,
    )
    current["enabled"] = True
    try:
        for sample_index in progress:
            sample_started = time.perf_counter()
            current["sample_index"] = sample_index
            current["routes"].clear()
            current["curve_payloads"].clear()
            current["capture_mode"] = "curve_baseline"
            sample = samples[sample_index]
            with torch.no_grad():
                model.model(
                    input_ids=sample["input_ids"].to(input_device),
                    attention_mask=sample["attention_mask"].to(input_device),
                    use_cache=False,
                    return_dict=True,
                )
            sequence_work_seconds += time.perf_counter() - sample_started
            sequence_work_examples += 1
            processed = sample_index + 1
            progress.set_postfix(sampled_fraction=token_fraction, examples=processed)
            if processed >= next_checkpoint and processed % 1024 == 0:
                _save_checkpoint(
                    local_output_dir, state, processed,
                    prior_seconds + time.perf_counter() - started,
                    run_sample_sha256,
                    rank,
                    sequence_work_seconds,
                    sequence_work_examples,
                )
                next_checkpoint += 1024
    finally:
        progress.close()
        current["enabled"] = False
        for handle in handles:
            handle.remove()
        for module in model.modules():
            experts = getattr(module, "experts", None)
            if experts is not None and hasattr(experts, "_reap_capture_callback"):
                delattr(experts, "_reap_capture_callback")

    calibration_seconds = prior_seconds + time.perf_counter() - started
    _save_checkpoint(
        local_output_dir, state, len(samples), calibration_seconds, run_sample_sha256,
        rank, sequence_work_seconds, sequence_work_examples,
    )
    if ep_enabled:
        _all_reduce_curve_state(state, torch_dist, rank)
    metadata = {
        "mode": "reap_ordered_layerwise_damage_curves",
        "model_path": calibration.MODEL_PATH,
        "dataset_name": "theblackcat102/evol-codealpaca-v1",
        "dataset_split": "train",
        "source_reap_sample_set_sha256": EXPECTED_SAMPLE_SHA256,
        "sample_set_sha256": run_sample_sha256,
        "curve_run_sample_set_sha256": run_sample_sha256,
        "reap_state_path": str(reap_path),
        "reap_state_sha256": reap_sha256,
        "sample_manifest_path": str(sample_manifest_path),
        "processed_examples": len(samples),
        "complete": len(samples) == 12228,
        "max_sequence_length": max_length,
        "truncation": False,
        "packing": False,
        "token_fraction": token_fraction,
        "token_selection": "CPU torch.randperm with seed=(run seed + 0x9E3779B1*(sample index+1)); choose ceil(length*fraction), sorted positions",
        "token_selection_seed": seed,
        "k_grid": list(K_GRID),
        "num_layers": 48,
        "num_experts": 512,
        "router_top_k": 10,
        "experts_implementation": experts_implementation,
        "locality": "each layer's own input; held-out KL harness measures full-model effects",
        "full_moe_norm_includes_shared_expert": True,
        "tail_estimates": "per-sequence token p95, p99, and top-one-percent mean, averaged across calibration sequences",
        "curve_seconds": calibration_seconds,
        "measured_sequence_work_seconds": sequence_work_seconds,
        "measured_sequence_work_examples": sequence_work_examples,
        "measured_seconds_per_sequence": (
            sequence_work_seconds / sequence_work_examples if sequence_work_examples else None
        ),
        "checkpoint_every_examples": 1024,
        "ep_enabled": ep_enabled,
        "ep_world_size": int(os.environ.get("QWEN4_EP_SIZE", "1")) if ep_enabled else 1,
        "outputs": ["curve_state.pt", "curve_summary.csv", "metadata.json"],
    }
    if rank == 0:
        _write_outputs(output_dir, state, metadata)
        print(json.dumps(metadata, indent=2), flush=True)
    if ep_enabled:
        torch_dist.barrier()
        torch_dist.destroy_process_group()

