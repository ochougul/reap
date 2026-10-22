"""Four-GPU SEAP calibration for Qwen3.5 packed-expert MoE checkpoints.

The runner is text-only even though Qwen3.5 is a conditional-generation
model. It measures all 256 routed experts and leaves the shared expert out of
both score construction and pruning.
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
from types import MethodType
from importlib.metadata import PackageNotFoundError, version

import torch
import torch.nn.functional as F
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils import set_seed
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Qwen3_5MoeForConditionalGeneration,
)

from reap.data import load_category_batches
from reap.pruning_metrics import (
    initialize_pruning_state,
    update_pruning_state_from_routed_outputs,
)
from reap.router_stability import _route_once
from reap.seap import SeapCollector, collect_seap_observation


EXPERIMENT_ROOT = pathlib.Path(
    os.environ.get(
        "SEAP_EXPERIMENT_ROOT",
        "artifacts/seap/qwen3.5-122b-a10b-tulu-3-sft-personas-math",
    )
)
MODEL_ID = os.environ.get("SEAP_MODEL_ID", "Qwen/Qwen3.5-122B-A10B")
MODEL_REVISION = os.environ.get(
    "SEAP_MODEL_REVISION", "dc4d348443bc740c68e2d77492492c11606384d5"
)
MODEL_PATH = os.environ.get("SEAP_MODEL_PATH")
DATASET_NAME = os.environ.get("SEAP_DATASET_NAME", "allenai/tulu-3-sft-personas-math")
DATASET_REVISION = os.environ.get(
    "SEAP_DATASET_REVISION", "46add032f5203fd96efbd54d074198981ed57d25"
)
DATASET_SPLIT = os.environ.get("SEAP_SPLIT", "train")
FULL_DATASET = os.environ.get("SEAP_FULL_DATASET", "true").lower() in {"1", "true", "yes"}
BATCH_SIZE = int(os.environ.get("SEAP_BATCH_SIZE", "1"))
BATCHES_PER_CATEGORY = int(os.environ.get("SEAP_BATCHES_PER_CATEGORY", "1024"))
if FULL_DATASET:
    BATCHES_PER_CATEGORY = 10**12
MODEL_MAX_LENGTH = int(os.environ.get("SEAP_MODEL_MAX_LENGTH", "2048"))
TRUNCATE = os.environ.get("SEAP_TRUNCATE", "false").lower() in {"1", "true", "yes"}
MAX_TOKENS = int(os.environ.get("SEAP_MAX_TOKENS", "0")) or None
MAX_SAMPLES = int(os.environ.get("SEAP_MAX_SAMPLES", "0")) or None
SEED = int(os.environ.get("SEAP_SEED", "42"))
VARIANCE_MULTIPLIER = float(os.environ.get("SEAP_VARIANCE_MULTIPLIER", "2.0"))
MAX_MEMORY_PER_GPU = os.environ.get("SEAP_MAX_MEMORY_PER_GPU", "70GiB")
PROGRESS_INTERVAL = int(os.environ.get("SEAP_PROGRESS_INTERVAL", "10"))
CHECKPOINT_INTERVAL_BATCHES = int(
    os.environ.get("SEAP_CHECKPOINT_INTERVAL_BATCHES", "100")
)
RESUME_CHECKPOINT = os.environ.get("SEAP_RESUME_CHECKPOINT")


def _package_versions() -> dict[str, str]:
    names = ("torch", "transformers", "accelerate", "datasets", "vllm")
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _build_device_map(config):
    with init_empty_weights():
        empty_model = Qwen3_5MoeForConditionalGeneration(config)
    max_memory = {
        index: os.environ.get(f"SEAP_MAX_MEMORY_GPU_{index}", MAX_MEMORY_PER_GPU)
        for index in range(4)
    }
    max_memory["cpu"] = "0GiB"
    device_map = infer_auto_device_map(
        empty_model,
        max_memory=max_memory,
        no_split_module_classes=["Qwen3_5MoeDecoderLayer", "Qwen3_5MoeVisionBlock"],
        dtype=torch.bfloat16,
    )
    placements = {str(device) for device in device_map.values()}
    if not {"0", "1", "2", "3"}.issubset(placements):
        raise RuntimeError(f"Qwen3.5 calibration requires all four GPUs, got {placements}")
    return device_map


def _qwen35_capture_forward(self, hidden_states, top_k_index, top_k_weights):
    """Native Qwen3.5 routed-expert execution with optional unweighted-output capture."""
    final_hidden_states = torch.zeros_like(hidden_states)
    capture = getattr(self, "_reap_capture_callback", None)
    captured_ids, captured_outputs, captured_tokens = [], [], []
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(
            top_k_index, num_classes=self.num_experts + 1
        ).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx_tensor in expert_hit:
        expert_idx = int(expert_idx_tensor.item())
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        expert_output = F.linear(
            self.act_fn(gate) * up, self.down_proj[expert_idx]
        )
        if capture is not None:
            captured_ids.append(torch.full_like(token_idx, expert_idx))
            captured_outputs.append(expert_output)
            captured_tokens.append(token_idx)
        weighted_output = expert_output * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(
            0, token_idx, weighted_output.to(final_hidden_states.dtype)
        )
    if capture is not None and captured_outputs:
        capture(
            torch.cat(captured_ids), torch.cat(captured_outputs), torch.cat(captured_tokens)
        )
    return final_hidden_states

def _processed_examples(batches) -> int:
    return sum(int(batch["input_ids"].shape[0]) for batch in batches)


def _run_identity() -> dict[str, object]:
    """Fields that must agree before a raw calibration checkpoint is resumed."""
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "split": DATASET_SPLIT,
        "batch_size": BATCH_SIZE,
        "model_max_length": MODEL_MAX_LENGTH,
        "truncate": TRUNCATE,
        "full_dataset": FULL_DATASET,
        "max_samples": MAX_SAMPLES,
        "max_tokens": MAX_TOKENS,
        "variance_multiplier": VARIANCE_MULTIPLIER,
        "seed": SEED,
        "deltas": [2.0, 4.0],
    }


def _save_checkpoint(
    path: pathlib.Path,
    *,
    reap_state: dict,
    collector: SeapCollector,
    processed_batches: int,
    processed_rows: int,
    total_tokens: int,
) -> None:
    """Atomically save raw state so a later process can continue calibration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 3,
        "identity": _run_identity(),
        "progress": {
            "processed_batches": processed_batches,
            "processed_rows": processed_rows,
            "total_tokens": total_tokens,
        },
        "reap_state": reap_state,
        "collector": collector.checkpoint_state(),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    print(
        "checkpoint "
        f"batches={processed_batches} rows={processed_rows} tokens={total_tokens} "
        f"path={path}",
        flush=True,
    )


def _load_checkpoint(path: pathlib.Path, collector: SeapCollector) -> tuple[dict, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 3:
        raise ValueError(f"Unsupported Qwen calibration checkpoint: {path}")
    identity = payload.get("identity", {})
    expected = _run_identity()
    mismatches = {
        key: (identity.get(key), value)
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint identity does not match this run: {mismatches}")
    collector.restore_checkpoint_state(payload["collector"])
    progress = payload["progress"]
    for key in ("processed_batches", "processed_rows", "total_tokens"):
        if not isinstance(progress.get(key), int) or progress[key] < 0:
            raise ValueError(f"Invalid checkpoint progress field {key!r}")
    return payload["reap_state"], progress


def _validate_callback_parity(
    reap_state: dict[int, dict], collector: SeapCollector, *, total_tokens: int, top_k: int
) -> None:
    """Verify native Qwen routed-output capture against independent router routes.

    ``SeapCollector`` observes router selections directly.  The routed-output
    callback independently records selected expert outputs.  Their per-expert
    counts must match exactly in every layer, and each valid token must produce
    ``top_k`` captured routed outputs.
    """
    expected_per_layer = total_tokens * top_k
    for layer, state in reap_state.items():
        callback_frequency = state["expert_frequency"].cpu()
        router_frequency = collector.original_selected_count.get(layer)
        if router_frequency is None:
            raise RuntimeError(f"Missing router observations for Qwen3.5 layer {layer}")
        if int(callback_frequency.sum()) != expected_per_layer:
            raise RuntimeError(
                f"Qwen3.5 callback captured {int(callback_frequency.sum())} routed outputs "
                f"at layer {layer}; expected {expected_per_layer}"
            )
        if not torch.equal(callback_frequency, router_frequency):
            raise RuntimeError(
                f"Qwen3.5 callback routing counts differ from router observations at "
                f"layer {layer}"
            )


def main() -> None:
    if MODEL_PATH is None:
        raise ValueError("SEAP_MODEL_PATH must point to a local Qwen3.5 checkpoint")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        raise RuntimeError("Qwen3.5 calibration requires four visible CUDA devices")

    set_seed(SEED)
    root = EXPERIMENT_ROOT
    calibration_dir = root / "calibration"
    log_dir = calibration_dir / "logs"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(
        MODEL_PATH, revision=MODEL_REVISION, trust_remote_code=True, local_files_only=True
    )
    if config.model_type != "qwen3_5_moe":
        raise ValueError(f"Expected qwen3_5_moe, got {config.model_type}")
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        revision=MODEL_REVISION,
        device_map=_build_device_map(config),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, revision=MODEL_REVISION, trust_remote_code=True, local_files_only=True
    )

    category_batches, packing_stats = load_category_batches(
        dataset_name=DATASET_NAME,
        split=DATASET_SPLIT,
        subset=None,
        tokenizer=tokenizer,
        model_max_length=MODEL_MAX_LENGTH,
        batch_size=BATCH_SIZE,
        split_by_category=False,
        return_vllm_tokens_prompt=False,
        truncate=TRUNCATE,
        batches_per_category=BATCHES_PER_CATEGORY,
        revision=DATASET_REVISION,
        return_packing_stats=True,
    )
    batches = next(iter(category_batches.values()))
    packing_stats = packing_stats["all"]
    source_rows = len(
        load_dataset(
            DATASET_NAME, split=DATASET_SPLIT, revision=DATASET_REVISION
        )
    )
    prepared_packed_sequences = _processed_examples(batches)
    if prepared_packed_sequences != packing_stats["packed_sequences"]:
        raise RuntimeError("Packed sequence accounting does not match prepared batches")

    collector = SeapCollector(calibration_dir, deltas=(2.0, 4.0), seed=SEED)
    reap_state = {}
    current = {"attention_mask": None, "routes": {}}
    handles = []

    def capture_route(layer_index, moe):
        def hook(module, args):
            hidden = args[0].reshape(-1, args[0].shape[-1])
            mask = current["attention_mask"]
            if mask is not None:
                hidden = hidden[mask.reshape(-1).to(hidden.device).bool()]
            current["routes"][layer_index] = (
                hidden,
                _route_once(moe.gate, hidden, moe.gate.top_k, boundary=0),
            )
        return hook

    def capture_routed_outputs(layer_index, num_experts):
        def callback(expert_ids, expert_outputs, token_indices):
            route_entry = current["routes"].get(layer_index)
            if route_entry is None:
                raise RuntimeError(f"Missing route capture for Qwen3.5 layer {layer_index}")
            _, route = route_entry
            mask = current["attention_mask"]
            if mask is None:
                valid = torch.ones(route[4].shape[0], device=route[4].device, dtype=torch.bool)
                compact_indices = token_indices
                row_valid = torch.ones_like(token_indices, dtype=torch.bool)
            else:
                valid = mask.reshape(-1).to(token_indices.device).bool()
                row_valid = valid[token_indices]
                compact = torch.cumsum(valid.to(torch.long), dim=0) - 1
                compact_indices = compact[token_indices[row_valid]]
            if not row_valid.any():
                return
            update_pruning_state_from_routed_outputs(
                reap_state[layer_index],
                expert_ids=expert_ids[row_valid],
                expert_outputs=expert_outputs[row_valid],
                token_indices=compact_indices,
                selected_experts=route[4],
                router_logits=route[0],
                num_experts=num_experts,
                renormalize_router_weights=True,
            )
        return callback

    def make_hook(layer_index, moe):
        router = moe.gate
        def hook(module, args, output):
            hidden, _ = current["routes"].get(layer_index, (None, None))
            if hidden is None or hidden.numel() == 0:
                return
            observation = collect_seap_observation(
                router, hidden, collector.generator(hidden.device, layer_index),
                variance_multiplier=VARIANCE_MULTIPLIER, top_k=router.top_k,
            )
            collector.observe(layer_index, observation, int(router.num_experts))
        return hook

    for index, layer in enumerate(model.model.language_model.layers):
        if layer.mlp.__class__.__name__ != "Qwen3_5MoeSparseMoeBlock":
            raise RuntimeError(
                f"Unexpected MoE module at layer {index}: {layer.mlp.__class__.__name__}"
            )
        num_experts = int(layer.mlp.gate.num_experts)
        reap_state[index] = initialize_pruning_state(num_experts, device="cpu")
        handles.append(layer.mlp.register_forward_pre_hook(capture_route(index, layer.mlp)))
        layer.mlp.experts._reap_capture_callback = capture_routed_outputs(index, num_experts)
        layer.mlp.experts.forward = MethodType(_qwen35_capture_forward, layer.mlp.experts)
        handles.append(layer.mlp.register_forward_hook(make_hook(index, layer.mlp)))

    input_device = model.model.language_model.embed_tokens.weight.device
    total_tokens = 0
    processed_batches = 0
    processed_rows = 0
    checkpoint_path = calibration_dir / "checkpoints" / "latest.pt"
    if RESUME_CHECKPOINT:
        checkpoint_path = pathlib.Path(RESUME_CHECKPOINT)
        reap_state, progress = _load_checkpoint(checkpoint_path, collector)
        processed_batches = progress["processed_batches"]
        processed_rows = progress["processed_rows"]
        total_tokens = progress["total_tokens"]
        if processed_batches > len(batches):
            raise ValueError(
                f"Checkpoint has {processed_batches} batches, but this run has only "
                f"{len(batches)} prepared batches"
            )
        print(
            "resuming "
            f"batches={processed_batches} rows={processed_rows} tokens={total_tokens} "
            f"checkpoint={checkpoint_path}",
            flush=True,
        )
    next_checkpoint_batch = None
    if CHECKPOINT_INTERVAL_BATCHES > 0:
        next_checkpoint_batch = (
            (processed_batches // CHECKPOINT_INTERVAL_BATCHES) + 1
        ) * CHECKPOINT_INTERVAL_BATCHES
    try:
        with torch.inference_mode():
            for batch in batches[processed_batches:]:
                batch_rows = int(batch["input_ids"].shape[0])
                if MAX_SAMPLES is not None:
                    remaining_rows = MAX_SAMPLES - processed_rows
                    if remaining_rows <= 0:
                        break
                    if batch_rows > remaining_rows:
                        batch = {
                            key: value[:remaining_rows]
                            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_rows
                            else value
                            for key, value in batch.items()
                        }
                        batch_rows = remaining_rows
                current["attention_mask"] = batch.get("attention_mask")
                model(
                    **{
                        key: value.to(input_device) if torch.is_tensor(value) else value
                        for key, value in batch.items()
                    },
                    use_cache=False,
                    return_dict=True,
                )
                processed_batches += 1
                processed_rows += batch_rows
                if current["attention_mask"] is not None:
                    total_tokens += int(current["attention_mask"].sum())
                if PROGRESS_INTERVAL > 0 and processed_batches % PROGRESS_INTERVAL == 0:
                    print(
                        f"progress batches={processed_batches} rows={processed_rows} "
                        f"tokens={total_tokens}",
                        flush=True,
                    )
                if (
                    next_checkpoint_batch is not None
                    and processed_batches >= next_checkpoint_batch
                ):
                    _save_checkpoint(
                        checkpoint_path,
                        reap_state=reap_state,
                        collector=collector,
                        processed_batches=processed_batches,
                        processed_rows=processed_rows,
                        total_tokens=total_tokens,
                    )
                    next_checkpoint_batch += CHECKPOINT_INTERVAL_BATCHES
                if MAX_TOKENS is not None and total_tokens >= MAX_TOKENS:
                    break
    finally:
        for handle in handles:
            handle.remove()

    _validate_callback_parity(
        reap_state,
        collector,
        total_tokens=total_tokens,
        top_k=int(config.text_config.num_experts_per_tok),
    )
    state = collector.add_scores_to_state(reap_state, delta=2.0, lambda_=0.5)
    torch.save(state, calibration_dir / "seap_state.pt")
    metadata = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": MODEL_PATH,
        "model_type": config.model_type,
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "split": DATASET_SPLIT,
        "source_rows": source_rows,
        "prepared_source_rows": packing_stats["source_rows"],
        "eligible_source_rows": packing_stats["eligible_source_rows"],
        "eligible_source_tokens": packing_stats["eligible_source_tokens"],
        "skipped_source_rows_over_max_length": packing_stats[
            "skipped_source_rows_over_max_length"
        ],
        "skipped_source_tokens_over_max_length": packing_stats[
            "skipped_source_tokens_over_max_length"
        ],
        "prepared_packed_sequences": prepared_packed_sequences,
        "processed_packed_sequences": processed_rows,
        "processed_batches": processed_batches,
        "total_tokens": total_tokens,
        "batch_size": BATCH_SIZE,
        "full_dataset": FULL_DATASET,
        "model_max_length": MODEL_MAX_LENGTH,
        "truncate": TRUNCATE,
        "max_samples": MAX_SAMPLES,
        "checkpoint_interval_batches": CHECKPOINT_INTERVAL_BATCHES,
        "resume_checkpoint": str(checkpoint_path) if RESUME_CHECKPOINT else None,
        "variance_multiplier": VARIANCE_MULTIPLIER,
        "deltas": [2.0, 4.0],
        "default_delta": 2.0,
        "default_lambda": 0.5,
        "router": "softmax_logits_then_topk_and_topk_renormalization",
        "callback_parity": "exact_per_layer_router_frequency_match",
        "num_moe_layers": len(reap_state),
        "num_routed_experts": config.text_config.num_experts,
        "num_experts_per_token": config.text_config.num_experts_per_tok,
        "shared_expert_preserved": True,
        "seed": SEED,
        "package_versions": _package_versions(),
    }
    collector.save(state, metadata, delta=2.0, lambda_=0.5)
    (log_dir / "run_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
