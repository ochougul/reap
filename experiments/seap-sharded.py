"""Resident four-GPU SEAP calibration runner for GLM-4.5-Air.

The REAP state is supplied from a matching calibration run; this process collects
the expensive full-rank stability signal while the model is resident on all GPUs.
"""

from __future__ import annotations

import pathlib
import re
import os

import torch
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils import set_seed
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from reap.data import load_category_batches
from reap.seap import SeapCollector, collect_seap_observation
from reap.router_stability import _route_once
from reap.pruning_metrics import initialize_pruning_state, update_pruning_state


MODEL_PATH = "/local/mnt2/workspace/ochougul/artifacts/models/GLM-4.5-Air"
OUTPUT_DIR = pathlib.Path(os.environ.get("SEAP_OUTPUT_DIR", "artifacts/seap/glm4.5-air-500k"))
MAX_TOKENS = int(os.environ.get("SEAP_MAX_TOKENS", "500000"))


def build_device_map(config):
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    device_map = infer_auto_device_map(
        empty_model,
        max_memory={0: "54GiB", 1: "54GiB", 2: "54GiB", 3: "54GiB", "cpu": "0GiB"},
        no_split_module_classes=["Glm4MoeDecoderLayer"],
        dtype=torch.bfloat16,
    )
    placements = {str(device) for device in device_map.values()}
    if not {"0", "1", "2", "3"}.issubset(placements):
        raise RuntimeError(f"SEAP requires all four GPUs, got {placements}")
    return device_map


def main() -> None:
    set_seed(42)
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
    batches = next(iter(load_category_batches(
        dataset_name="theblackcat102/evol-codealpaca-v1", split="train", subset=None,
        tokenizer=tokenizer, model_max_length=2048, batch_size=1,
        split_by_category=False, return_vllm_tokens_prompt=False,
        truncate=True, batches_per_category=256,
    ).values()))

    collector = SeapCollector(OUTPUT_DIR, deltas=(2.0, 4.0), seed=42)
    total_tokens = 0
    current = {"mask": None, "batch": 0}
    reap_state = {}
    handles = []

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
            obs = collect_seap_observation(
                router, hidden, collector.generator(hidden.device, layer),
                variance_multiplier=2.0, top_k=router.top_k,
            )
            collector.observe(layer, obs, int(router.n_routed_experts))
            activations = torch.stack([expert(hidden) for expert in moe.experts])
            if layer not in reap_state:
                reap_state[layer] = initialize_pruning_state(
                    int(router.n_routed_experts), device="cpu"
                )
            update_pruning_state(
                reap_state[layer],
                activations=activations,
                selected_experts=route[4],
                router_logits=route[0],
                num_experts=int(router.n_routed_experts),
                renormalize_router_weights=True,
            )
            del activations
        return hook

    for name, module in model.named_modules():
        if hasattr(module, "experts") and hasattr(module, "gate"):
            match = re.search(r"model\.layers\.(\d+)", name)
            if match:
                handles.append(module.register_forward_hook(make_hook(int(match.group(1)), module)))

    input_device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        for batch_id, batch in enumerate(batches):
            current["batch"] = batch_id
            current["mask"] = batch.get("attention_mask")
            model(**{key: value.to(input_device) if torch.is_tensor(value) else value
                     for key, value in batch.items()}, use_cache=False, return_dict=True)
            total_tokens += int(current["mask"].sum()) if current["mask"] is not None else 0
            if total_tokens >= MAX_TOKENS:
                break
    for handle in handles:
        handle.remove()

    state = collector.add_scores_to_state(reap_state, delta=2.0, lambda_=0.5)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(state, OUTPUT_DIR / "seap_state.pt")
    collector.save(state, {"total_tokens": total_tokens, "variance_multiplier": 2.0}, delta=2.0, lambda_=0.5)


if __name__ == "__main__":
    main()
