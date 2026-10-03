"""Resident four-GPU router-stability pilot for GLM-4.5-Air."""

from __future__ import annotations

import pathlib
import re

import torch
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils import set_seed
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from reap.data import load_category_batches
from reap.router_stability import RouterStabilityCollector, collect_router_observation


MODEL_PATH = "/local/mnt2/workspace/ochougul/artifacts/models/GLM-4.5-Air"
OUTPUT_DIR = pathlib.Path(
    "artifacts/router_stability/glm4.5-air-evol-codealpaca-500k-sharded"
)
VARIANCE_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0)
MAX_TOKENS = 500_000


def build_device_map(config):
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True
        )
    device_map = infer_auto_device_map(
        empty_model,
        max_memory={0: "54GiB", 1: "54GiB", 2: "54GiB", 3: "54GiB", "cpu": "0GiB"},
        no_split_module_classes=["Glm4MoeDecoderLayer"],
        dtype=torch.bfloat16,
    )
    placements = {str(device) for device in device_map.values()}
    required = {"0", "1", "2", "3"}
    if placements != required:
        raise RuntimeError(
            f"Expected resident placement on all four GPUs, got {placements}: {device_map}"
        )
    return device_map


def main() -> None:
    set_seed(42)
    config = AutoConfig.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    device_map = build_device_map(config)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    batches_by_category = load_category_batches(
        dataset_name="theblackcat102/evol-codealpaca-v1",
        split="train",
        subset=None,
        tokenizer=tokenizer,
        model_max_length=2048,
        batch_size=1,
        split_by_category=False,
        return_vllm_tokens_prompt=False,
        truncate=True,
        batches_per_category=256,
    )
    batches = next(iter(batches_by_category.values()))

    collector = RouterStabilityCollector(
        OUTPUT_DIR,
        VARIANCE_MULTIPLIERS,
        seed=42,
        boundary=4,
        max_tokens=MAX_TOKENS,
    )
    generators: dict[str, torch.Generator] = {}
    handles = []
    current_batch = {"id": -1, "mask": None}

    def hook_factory(layer: int, router: torch.nn.Module):
        def hook(module, args, output):
            hidden_states = args[0]
            flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
            mask = current_batch["mask"]
            if mask is not None:
                flat_hidden = flat_hidden[mask.reshape(-1).to(flat_hidden.device).bool()]
            if flat_hidden.numel() == 0:
                return
            key = str(flat_hidden.device)
            if key not in generators:
                generators[key] = torch.Generator(device=flat_hidden.device).manual_seed(
                    42 + layer
                )
            observation = collect_router_observation(
                router,
                flat_hidden,
                collector.variance_multipliers,
                generators[key],
                top_k=router.top_k,
                boundary=collector.boundary,
            )
            collector.observe(layer, current_batch["id"], observation)

        return hook

    for name, module in model.named_modules():
        if hasattr(module, "e_score_correction_bias") and hasattr(module, "n_group"):
            match = re.search(r"model\.layers\.(\d+)", name)
            if match:
                handles.append(
                    module.register_forward_hook(
                        hook_factory(int(match.group(1)), module)
                    )
                )

    input_device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        for batch_id, batch in enumerate(batches):
            current_batch["id"] = batch_id
            current_batch["mask"] = batch.get("attention_mask")
            model_inputs = {
                key: value.to(input_device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            model(**model_inputs, use_cache=False, return_dict=True)
            if collector.max_tokens is not None and collector.total_tokens >= MAX_TOKENS:
                break

    for handle in handles:
        handle.remove()
    collector.finalize()


if __name__ == "__main__":
    main()
