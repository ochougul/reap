"""Resident four-GPU SEAP calibration runner for Qwen4-Exp and GLM MoE models."""

from __future__ import annotations

import os
import pathlib
import re
import time
from types import MethodType

import torch
import torch.nn.functional as F
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.hooks import remove_hook_from_module
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

from reap.data import load_category_batches, parse_composite_dataset_spec
from reap.pruning_metrics import initialize_pruning_state, update_pruning_state
from reap.router_stability import _route_once
from reap.seap import SeapCollector, collect_seap_observation


MODEL_PATH = os.environ.get(
    "SEAP_MODEL_PATH", "/local/mnt2/workspace/ochougul/artifacts/models/GLM-4.5-Air"
)
OUTPUT_DIR = pathlib.Path(os.environ.get("SEAP_OUTPUT_DIR", "artifacts/seap/calibration"))
DEFAULT_COMPOSITE = (
    "theblackcat102/evol-codealpaca-v1:4096,"
    "Salesforce/xlam-function-calling-60k:4096,"
    "open-r1/Mixture-of-Thoughts[code]:4096,"
    "open-r1/Mixture-of-Thoughts[math]:4096,"
    "open-r1/Mixture-of-Thoughts[science]:4096,"
    "SWE-bench/SWE-smith-trajectories(tool):4096"
)
DATASET_NAME = os.environ.get("SEAP_DATASET_NAME", DEFAULT_COMPOSITE)
DATASET_SPLIT = os.environ.get("SEAP_SPLIT", "train")
BATCH_SIZE = int(os.environ.get("SEAP_BATCH_SIZE", "1"))
MODEL_MAX_LENGTH = int(os.environ.get("SEAP_MODEL_MAX_LENGTH", "2048"))
TRUNCATE = os.environ.get("SEAP_TRUNCATE", "false").lower() in {"1", "true", "yes"}
MAX_TOKENS = int(os.environ.get("SEAP_MAX_TOKENS", "0")) or None
MAX_SAMPLES = int(os.environ.get("SEAP_MAX_SAMPLES", "500"))
SEED = int(os.environ.get("SEAP_SEED", "42"))
VARIANCE_MULTIPLIER = float(os.environ.get("SEAP_VARIANCE_MULTIPLIER", "2.0"))
DELTA = float(os.environ.get("SEAP_DELTA", "2.0"))
LAMBDA = float(os.environ.get("SEAP_LAMBDA", "0.5"))


def _model_class(config):
    return AutoModelForImageTextToText if getattr(config, "model_type", "") == "qwen4_exp" else AutoModelForCausalLM


def build_device_map(config):
    model_class = _model_class(config)
    is_qwen4 = getattr(config, "model_type", "") == "qwen4_exp"
    if is_qwen4:
        # A100s cannot execute this checkpoint's fine-grained FP8 kernels, so
        # Transformers dequantizes to BF16.  The layer-1 n-gram table is about
        # 95 GiB in BF16 and must remain on host RAM; the remaining model is
        # explicitly balanced across the four GPUs.
        device_map = {
            "model.visual": "cpu",
            "model.language_model.embed_tokens": "cuda:0",
            "model.language_model.norm": "cuda:2",
            "lm_head": "cpu",
        }
        for layer in range(48):
            device_map[f"model.language_model.layers.{layer}"] = f"cuda:{layer % 4}"
        # Leave transient conversion headroom on GPU 3 for the packed expert
        # tensors at the end of the checkpoint.
        device_map["model.language_model.layers.47"] = "cuda:2"
        layer_one_prefix = "model.language_model.layers.1"
        for child in (
            "linear_attn", "mlp", "attn_hyper_connection", "mlp_hyper_connection",
            "ple.key_proj", "ple.value_proj", "ple.norm_key", "ple.norm_query",
            "ple.norm_conv", "ple.conv1d",
        ):
            device_map[f"{layer_one_prefix}.{child}"] = "cuda:1"
        device_map[f"{layer_one_prefix}.ple.ple_embedding"] = "cpu"
        device_map.pop(layer_one_prefix)
        return device_map
    with init_empty_weights():
        empty_model = model_class.from_config(config, trust_remote_code=True)
    device_map = infer_auto_device_map(
        empty_model,
        max_memory={0: "78GiB", 1: "78GiB", 2: "78GiB", 3: "78GiB", "cpu": "0GiB"},
        no_split_module_classes=[
            "Glm4MoeDecoderLayer", "Qwen4ExpTextDecoderLayer", "Qwen3_5MoeDecoderLayer"
        ],
        dtype=torch.float8_e4m3fn if is_qwen4 else torch.bfloat16,
    )
    placements = {str(device) for device in device_map.values()}
    if "disk" in placements or not {"0", "1", "2"}.issubset(placements):
        raise RuntimeError(f"SEAP requires a disk-free multi-GPU placement, got {placements}")
    return device_map


def _load_batches(tokenizer):
    components = parse_composite_dataset_spec(DATASET_NAME, default_split=DATASET_SPLIT)
    if components is None:
        components = [type("Component", (), {
            "name": DATASET_NAME, "subset": None, "split": DATASET_SPLIT,
            "num_batches": MAX_SAMPLES if MAX_SAMPLES > 0 else 10**12,
        })()]
    batches, metadata = [], []
    remaining = MAX_SAMPLES if MAX_SAMPLES > 0 else None
    for component in tqdm(components, desc="Preparing calibration datasets", unit="dataset"):
        requested = min(component.num_batches, remaining) if remaining is not None else component.num_batches
        if requested <= 0:
            break
        loaded = load_category_batches(
            dataset_name=component.name, split=component.split, subset=component.subset,
            tokenizer=tokenizer, model_max_length=MODEL_MAX_LENGTH,
            batch_size=BATCH_SIZE, split_by_category=False,
            return_vllm_tokens_prompt=False, truncate=TRUNCATE,
            batches_per_category=requested, pack_samples=False,
        )
        component_batches = next(iter(loaded.values()))
        batches.extend(component_batches)
        metadata.append({"name": component.name, "subset": component.subset,
                         "split": component.split, "requested_batches": requested,
                         "retained_batches": len(component_batches)})
        if remaining is not None:
            remaining -= len(component_batches)
    return batches, metadata


def _expert_activations(experts, hidden_states):
    """Return [experts, tokens, hidden] activations for packed Qwen4 experts."""
    if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
        outputs = []
        for index in range(int(experts.num_experts)):
            gate, up = F.linear(hidden_states, experts.gate_up_proj[index]).chunk(2, dim=-1)
            outputs.append(F.linear(F.silu(gate) * up, experts.down_proj[index]))
        return torch.stack(outputs, dim=0)
    return torch.stack([expert(hidden_states) for expert in experts], dim=0)


def _layer_index(name: str):
    match = re.search(r"(?:model\.)?(?:language_model\.)?layers\.(\d+)(?:\.mlp)?$", name)
    return int(match.group(1)) if match else None


def _keep_qwen_ngram_embedding_on_cpu(model) -> None:
    """Prevent Accelerate from moving Qwen4's 95-GiB n-gram table to a GPU."""
    for name, module in model.named_modules():
        if not name.endswith("ple.ple_embedding") or not hasattr(module, "ngram_embedding"):
            continue
        embedding = module.ngram_embedding
        remove_hook_from_module(embedding, recurse=False)

        def cpu_forward(self, input_ids):
            values = F.embedding(input_ids.to(self.weight.device), self.weight)
            return values.to(device=input_ids.device, dtype=torch.bfloat16)

        embedding.forward = MethodType(cpu_forward, embedding)
        return
    raise RuntimeError("Could not find Qwen4 n-gram embedding module")


def _bridge_qwen_ple_layer(model) -> None:
    """Bridge layer 1 before its PLE runs, since its parent is split manually."""
    layer = model.model.language_model.layers[1]
    target = torch.device("cuda:1")

    def move(value):
        if torch.is_tensor(value):
            return value.to(target)
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if isinstance(value, list):
            return [move(item) for item in value]
        return value

    def pre_hook(module, args, kwargs):
        args = tuple(move(item) for item in args)
        kwargs = {
            key: value if key == "past_key_values" else move(value)
            for key, value in kwargs.items()
        }
        return args, kwargs

    layer.register_forward_pre_hook(pre_hook, with_kwargs=True)


def main() -> None:
    set_seed(SEED)
    started = time.perf_counter()
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model = _model_class(config).from_pretrained(
        MODEL_PATH, device_map=build_device_map(config), torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True, offload_state_dict=True,
        offload_buffers=True, local_files_only=True,
    ).eval()
    if getattr(config, "model_type", "") == "qwen4_exp":
        _keep_qwen_ngram_embedding_on_cpu(model)
        _bridge_qwen_ple_layer(model)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    batches, component_metadata = _load_batches(tokenizer)
    if not batches:
        raise RuntimeError("No calibration batches were retained")

    collector = SeapCollector(OUTPUT_DIR, deltas=(2.0, 4.0), seed=SEED)
    current = {"mask": None}
    reap_state, handles, layer_names = {}, [], {}
    total_tokens = total_examples = 0

    def make_hook(layer, moe):
        router = moe.gate
        def hook(module, args, output):
            hidden = args[0].reshape(-1, args[0].shape[-1])
            mask = current["mask"]
            if mask is not None:
                hidden = hidden[mask.reshape(-1).to(hidden.device).bool()]
            if hidden.numel() == 0:
                return
            route = _route_once(router, hidden, router.top_k, boundary=0)
            observation = collect_seap_observation(
                router, hidden, collector.generator(hidden.device, layer),
                variance_multiplier=VARIANCE_MULTIPLIER, top_k=router.top_k,
            )
            num_experts = int(getattr(router, "num_experts", getattr(router, "n_routed_experts", router.weight.shape[0])))
            collector.observe(layer, observation, num_experts)
            activations = _expert_activations(moe.experts, hidden)
            if layer not in reap_state:
                reap_state[layer] = initialize_pruning_state(num_experts, device="cpu")
            update_pruning_state(
                reap_state[layer], activations=activations, selected_experts=route[4],
                router_logits=route[0], num_experts=num_experts,
                renormalize_router_weights=True,
            )
            del activations
        return hook

    for name, module in model.named_modules():
        if hasattr(module, "experts") and hasattr(module, "gate"):
            layer = _layer_index(name)
            if layer is not None:
                layer_names[layer] = name
                handles.append(module.register_forward_hook(make_hook(layer, module)))

    expected_layers = int(getattr(config, "num_hidden_layers", getattr(getattr(config, "text_config", None), "num_hidden_layers", 0)))
    if expected_layers and len(layer_names) != expected_layers:
        raise RuntimeError(f"Expected {expected_layers} MoE layers, found {layer_names}")

    input_device = model.get_input_embeddings().weight.device
    backbone = getattr(model, "model", model)
    progress = tqdm(batches, desc="Calibrating stability + REAP", unit="example")
    calibration_started = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch in progress:
                mask = batch.get("attention_mask")
                current["mask"] = mask
                inputs = {key: value.to(input_device) if torch.is_tensor(value) else value for key, value in batch.items()}
                backbone(**inputs, use_cache=False, return_dict=True)
                valid_tokens = int(mask.sum()) if mask is not None else int(batch["input_ids"].numel())
                total_tokens += valid_tokens
                total_examples += int(mask.shape[0]) if mask is not None else 1
                progress.set_postfix(tokens=total_tokens, layers=len(layer_names))
                if MAX_TOKENS is not None and total_tokens >= MAX_TOKENS:
                    break
    finally:
        for handle in handles:
            handle.remove()

    if not reap_state or len(reap_state) != len(layer_names):
        raise RuntimeError(f"Missing REAP state for layers: {set(layer_names) - set(reap_state)}")
    state = collector.add_scores_to_state(reap_state, delta=DELTA, lambda_=LAMBDA)
    calibration_seconds = time.perf_counter() - calibration_started
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    collector.save(state, {
        "model_path": MODEL_PATH, "model_type": getattr(config, "model_type", None),
        "dataset_name": DATASET_NAME, "split": DATASET_SPLIT,
        "components": component_metadata, "batch_size": BATCH_SIZE,
        "max_samples": MAX_SAMPLES, "retained_examples": total_examples,
        "total_tokens": total_tokens, "model_max_length": MODEL_MAX_LENGTH,
        "truncate": TRUNCATE, "variance_multiplier": VARIANCE_MULTIPLIER,
        "delta": DELTA, "lambda": LAMBDA, "seed": SEED,
        "num_layers": len(layer_names), "num_experts": 512,
        "calibration_seconds": calibration_seconds,
        "total_seconds": time.perf_counter() - started,
        "examples_per_second": total_examples / max(calibration_seconds, 1e-9),
        "tokens_per_second": total_tokens / max(calibration_seconds, 1e-9),
        "full_composite_requested_examples": sum(c["requested_batches"] for c in component_metadata),
    }, delta=DELTA, lambda_=LAMBDA)


if __name__ == "__main__":
    main()
