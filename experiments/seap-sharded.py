"""Resident four-GPU SEAP calibration runner for GLM-4.5-Air.

The REAP state is supplied from a matching calibration run; this process collects
the expensive full-rank stability signal while the model is resident on all GPUs.
"""

from __future__ import annotations

import pathlib
import re
import os

import torch
from tqdm.auto import tqdm
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils import set_seed
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from reap.args import DatasetArgs, ObserverArgs
from reap.layerwise_prune import prepare_calibration_batches
from reap.seap import SeapCollector, collect_seap_observation
from reap.router_stability import _route_once
from reap.pruning_metrics import (
    initialize_pruning_state,
    update_pruning_state_from_routed_outputs,
)


MODEL_PATH = os.environ.get(
    "SEAP_MODEL_PATH", "/local/mnt2/workspace/ochougul/artifacts/models/GLM-4.5-Air"
)
OUTPUT_DIR = pathlib.Path(os.environ.get("SEAP_OUTPUT_DIR", "artifacts/seap/glm4.5-air-500k"))
DATASET_NAME = os.environ.get("SEAP_DATASET_NAME", "theblackcat102/evol-codealpaca-v1")
DATASET_SPLIT = os.environ.get("SEAP_SPLIT", "train")
BATCH_SIZE = int(os.environ.get("SEAP_BATCH_SIZE", "1"))
MAX_SAMPLES = int(os.environ.get("SEAP_MAX_SAMPLES", "0")) or None
CHECKPOINT_INTERVAL = int(os.environ.get("SEAP_CHECKPOINT_INTERVAL", "5000"))
FULL_DATASET = os.environ.get("SEAP_FULL_DATASET", "false").lower() in {"1", "true", "yes"}
# Match experiments/pruning-cli.sh unless full-dataset mode is requested.
BATCHES_PER_CATEGORY = int(os.environ.get("SEAP_BATCHES_PER_CATEGORY", "1024"))
if FULL_DATASET:
    # load_category_batches stops after exhausting the dataset when this
    # exceeds the available sample count.
    BATCHES_PER_CATEGORY = 10**12
MODEL_MAX_LENGTH = int(os.environ.get("SEAP_MODEL_MAX_LENGTH", "2048"))
# Match REAP's ObserverArgs default: overlong examples are skipped, not
# truncated. This keeps the sampled calibration examples identical.
TRUNCATE = os.environ.get("SEAP_TRUNCATE", "false").lower() in {"1", "true", "yes"}
MAX_TOKENS = int(os.environ.get("SEAP_MAX_TOKENS", "0" if FULL_DATASET else "500000")) or None
SEED = int(os.environ.get("SEAP_SEED", "42"))


def build_device_map(config):
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    device_map = infer_auto_device_map(
        empty_model,
        max_memory={
            0: os.environ.get("SEAP_GPU0_MAX_MEMORY", "54GiB"),
            1: os.environ.get("SEAP_GPU1_MAX_MEMORY", "54GiB"),
            2: os.environ.get("SEAP_GPU2_MAX_MEMORY", "54GiB"),
            3: os.environ.get("SEAP_GPU3_MAX_MEMORY", "54GiB"),
            "cpu": os.environ.get("SEAP_CPU_MAX_MEMORY", "0GiB"),
        },
        no_split_module_classes=["Glm4MoeDecoderLayer"],
        dtype=torch.bfloat16,
    )
    placements = {str(device) for device in device_map.values()}
    if not {"0", "1", "2", "3"}.issubset(placements):
        raise RuntimeError(f"SEAP requires all four GPUs, got {placements}")
    return device_map


def save_checkpoint(collector, state, *, samples, total_tokens):
    """Save the latest state after crossing a checkpoint boundary.

    Batch sizes do not necessarily divide ``CHECKPOINT_INTERVAL``. The caller
    therefore invokes this once the processed sample count has reached the
    next boundary; the state file is intentionally overwritten in place.
    """
    collector.save(
        state,
        {
            "total_samples": samples,
            "total_tokens": total_tokens,
            "checkpoint": True,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
        },
        delta=2.0,
        lambda_=0.5,
    )


def main() -> None:
    set_seed(SEED)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=build_device_map(config),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    batches = prepare_calibration_batches(
        tokenizer,
        DatasetArgs(dataset_name=DATASET_NAME, split=DATASET_SPLIT),
        ObserverArgs(
            batch_size=BATCH_SIZE,
            model_max_length=MODEL_MAX_LENGTH,
            truncate=TRUNCATE,
            batches_per_category=BATCHES_PER_CATEGORY,
        ),
    )

    collector = SeapCollector(OUTPUT_DIR, deltas=(2.0, 4.0), seed=SEED)
    total_tokens = 0
    current = {"mask": None, "batch": 0, "residuals": {}, "routes": {}}
    reap_state = {}
    handles = []

    def make_hook(layer, moe):
        router = moe.gate
        num_experts = int(
            getattr(router, "n_routed_experts", moe.config.n_routed_experts)
        )
        def hook(module, args, output):
            residual = current["residuals"].get(layer)
            hidden = args[0].reshape(-1, args[0].shape[-1])
            combined_output = output[0] if isinstance(output, (tuple, list)) else output
            if residual is None or combined_output.shape != args[0].shape:
                raise RuntimeError(
                    f"Missing or incompatible residual capture for GLM MoE layer {layer}"
                )
            contribution = torch.linalg.vector_norm(combined_output, dim=-1) / (
                torch.linalg.vector_norm(residual, dim=-1)
                + torch.finfo(args[0].dtype).eps
            )
            mask = current["mask"]
            if mask is not None:
                valid = mask.reshape(-1).to(hidden.device).bool()
                contribution = contribution.reshape(-1)[valid]
                hidden = hidden[valid]
            else:
                contribution = contribution.reshape(-1)
            if hidden.numel() == 0:
                return
            route = _route_once(router, hidden, router.top_k, boundary=0)
            obs = collect_seap_observation(
                router, hidden, collector.generator(hidden.device, layer),
                variance_multiplier=2.0, top_k=router.top_k,
            )
            collector.observe(layer, obs, num_experts)
            if layer not in reap_state:
                reap_state[layer] = initialize_pruning_state(
                    num_experts, device="cpu"
                )
                reap_state[layer]["moe_residual_contribution_sum"] = torch.zeros(
                    (), dtype=torch.float64
                )
                reap_state[layer]["moe_residual_contribution_count"] = torch.zeros(
                    (), dtype=torch.long
                )
            reap_state[layer]["moe_residual_contribution_sum"] += contribution.sum(
                dtype=torch.float64
            ).cpu()
            reap_state[layer]["moe_residual_contribution_count"] += torch.tensor(
                contribution.numel(), dtype=torch.long
            )
        return hook

    for name, module in model.named_modules():
        if hasattr(module, "experts") and hasattr(module, "gate"):
            match = re.search(r"model\.layers\.(\d+)", name)
            if match:
                layer = int(match.group(1))
                num_experts = int(
                    getattr(module.gate, "n_routed_experts", module.config.n_routed_experts)
                )
                reap_state[layer] = initialize_pruning_state(num_experts, device="cpu")
                reap_state[layer]["moe_residual_contribution_sum"] = torch.zeros((), dtype=torch.float64)
                reap_state[layer]["moe_residual_contribution_count"] = torch.zeros((), dtype=torch.long)

                def capture_route(m, args, layer=layer, module=module):
                    hidden = args[0].reshape(-1, args[0].shape[-1])
                    current["routes"][layer] = _route_once(
                        module.gate, hidden, module.gate.top_k, boundary=0
                    )

                def capture_expert_outputs(
                    expert_ids_g,
                    expert_outputs,
                    perm,
                    num_top_k,
                    layer=layer,
                    num_experts=num_experts,
                ):
                    route = current["routes"].get(layer)
                    if route is None:
                        raise RuntimeError(f"Missing captured route for MoE layer {layer}")
                    selected_all = route[4].reshape(-1, num_top_k)
                    logits_all = route[1]
                    valid = current["mask"]
                    if valid is None:
                        valid = torch.ones(
                            selected_all.shape[0], device=selected_all.device, dtype=torch.bool
                        )
                    else:
                        valid = valid.reshape(-1).to(selected_all.device).bool()
                    original_token = perm // num_top_k
                    row_valid = valid[original_token] & (expert_ids_g < num_experts)
                    if not row_valid.any():
                        return
                    valid_token_ids = torch.nonzero(valid, as_tuple=False).reshape(-1)
                    compact_ids = torch.full(
                        (valid.numel(),), -1, device=valid.device, dtype=torch.long
                    )
                    compact_ids[valid_token_ids] = torch.arange(
                        valid_token_ids.numel(), device=valid.device
                    )
                    update_pruning_state_from_routed_outputs(
                        reap_state[layer],
                        expert_ids=expert_ids_g[row_valid],
                        expert_outputs=expert_outputs[row_valid],
                        token_indices=compact_ids[original_token[row_valid]],
                        selected_experts=selected_all[valid],
                        router_logits=logits_all[valid],
                        num_experts=num_experts,
                        renormalize_router_weights=True,
                    )

                module.register_forward_pre_hook(capture_route)
                module.experts._reap_capture_callback = capture_expert_outputs
                handles.append(module.register_forward_hook(make_hook(layer, module)))

    # GLM's residual added to the MoE output is the input to the
    # post-attention normalization, not the normalized MoE input.
    for layer_idx, layer_module in enumerate(model.model.layers):
        moe = getattr(layer_module, "mlp", None)
        norm = getattr(layer_module, "post_attention_layernorm", None)
        if hasattr(moe, "experts") and hasattr(moe, "gate") and norm is not None:
            def capture_residual(module, args, layer_idx=layer_idx):
                if args and torch.is_tensor(args[0]):
                    current["residuals"][layer_idx] = args[0].detach()

            handles.append(norm.register_forward_pre_hook(capture_residual))

    input_device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        total_samples = 0
        next_checkpoint = CHECKPOINT_INTERVAL if CHECKPOINT_INTERVAL > 0 else None
        progress_total = len(batches)
        if MAX_SAMPLES is not None:
            progress_total = min(progress_total, (MAX_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE)
        progress = tqdm(
            enumerate(batches),
            total=progress_total,
            desc="SEAP calibration",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch_id, batch in progress:
            if MAX_SAMPLES is not None:
                remaining_samples = MAX_SAMPLES - total_samples
                if remaining_samples <= 0:
                    break
                batch_size = int(batch["input_ids"].shape[0])
                if batch_size > remaining_samples:
                    batch = {
                        key: value[:remaining_samples]
                        if torch.is_tensor(value) and value.dim() > 0
                        else value
                        for key, value in batch.items()
                    }
            total_samples += int(batch["input_ids"].shape[0])
            current["batch"] = batch_id
            current["mask"] = batch.get("attention_mask")
            current["residuals"].clear()
            model(**{key: value.to(input_device) if torch.is_tensor(value) else value
                     for key, value in batch.items()}, use_cache=False, return_dict=True)
            total_tokens += int(current["mask"].sum()) if current["mask"] is not None else 0
            progress.set_postfix(samples=total_samples, tokens=total_tokens)
            if next_checkpoint is not None and total_samples >= next_checkpoint:
                save_checkpoint(
                    collector,
                    reap_state,
                    samples=total_samples,
                    total_tokens=total_tokens,
                )
                while next_checkpoint <= total_samples:
                    next_checkpoint += CHECKPOINT_INTERVAL
            if MAX_TOKENS is not None and total_tokens >= MAX_TOKENS:
                break
        progress.close()
    for handle in handles:
        handle.remove()

    state = collector.add_scores_to_state(reap_state, delta=2.0, lambda_=0.5)
    for layer_state in state.values():
        count = layer_state.get("moe_residual_contribution_count")
        total = layer_state.get("moe_residual_contribution_sum")
        if count is not None and int(count.item()) > 0:
            layer_state["moe_residual_contribution"] = total / count
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(state, OUTPUT_DIR / "seap_state.pt")
    collector.save(
        state,
        {
            "total_tokens": total_tokens,
            "total_samples": total_samples,
            "variance_multiplier": 2.0,
            "dataset_name": DATASET_NAME,
            "split": DATASET_SPLIT,
            "batch_size": BATCH_SIZE,
            "batches_per_category": BATCHES_PER_CATEGORY,
            "full_dataset": FULL_DATASET,
            "model_max_length": MODEL_MAX_LENGTH,
            "truncate": TRUNCATE,
            "split_by_category": False,
            "seed": SEED,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
        },
        delta=2.0,
        lambda_=0.5,
    )


if __name__ == "__main__":
    main()
