"""Resident four-GPU Fisher-style expert score calibration for GLM-4.5-Air.

One forward + backward pass per batch (parameters frozen) accumulates, in the same
pass, REAP, router stability and the Fisher score E[(G_l . y_e)^2] per layer/expert;
see reap/fisher.py and reap/joint_calibration.py.

Samples are individual (unpacked, untruncated) dataset examples drawn at random
without replacement; examples longer than FISHER_MODEL_MAX_LENGTH are skipped.
"""

from __future__ import annotations

import json
import os
import pathlib
import time

import torch
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils import set_seed
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from reap.data import load_category_batches
from reap.fisher import assistant_label_mask
from reap.joint_calibration import JointCollector


def _flag(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes"}


MODEL_PATH = os.environ.get(
    "FISHER_MODEL_PATH", "/local/mnt2/workspace/ochougul/artifacts/models/GLM-4.5-Air"
)
OUTPUT_DIR = pathlib.Path(os.environ.get("FISHER_OUTPUT_DIR", "artifacts/fisher/glm4.5-air"))
DATASET_NAME = os.environ.get("FISHER_DATASET_NAME", "theblackcat102/evol-codealpaca-v1")
DATASET_SPLIT = os.environ.get("FISHER_SPLIT", "train")
MODEL_MAX_LENGTH = int(os.environ.get("FISHER_MODEL_MAX_LENGTH", "16384"))
SEED = int(os.environ.get("FISHER_SEED", "42"))
TARGET_SAMPLES = int(os.environ.get("FISHER_TARGET_SAMPLES", "12228"))
# Padded tokens per forward/backward (batch = as many samples as fit). 4096 is the
# size validated to fit in memory on 4x80GB; a single longer sample runs alone.
MAX_BATCH_TOKENS = int(os.environ.get("FISHER_MAX_BATCH_TOKENS", "4096"))
# Samples are length-sorted only inside windows of this size, so every multiple of
# WINDOW samples is an unbiased random subset (needed for valid checkpoints).
WINDOW = int(os.environ.get("FISHER_WINDOW", "250"))
CHECKPOINT_EVERY = int(os.environ.get("FISHER_CHECKPOINT_EVERY", "4000"))


def make_batches(samples, pad_id):
    """Group random-order samples into token-budget batches within windows."""
    batches = []
    for start in range(0, len(samples), WINDOW):
        window = sorted(samples[start:start + WINDOW], key=lambda x: x["input_ids"].shape[-1])
        groups, current = [], []
        for sample in window:
            length = sample["input_ids"].shape[-1]
            if current and (len(current) + 1) * length > MAX_BATCH_TOKENS:  # sorted: length is the max
                groups.append(current)
                current = []
            current.append(sample)
        groups.append(current)
        for index, group in enumerate(groups):
            width = max(item["input_ids"].shape[-1] for item in group)
            ids = torch.full((len(group), width), pad_id, dtype=torch.long)
            mask = torch.zeros((len(group), width), dtype=torch.long)
            for row, item in enumerate(group):
                n = item["input_ids"].shape[-1]
                ids[row, :n] = item["input_ids"][0]
                mask[row, :n] = 1
            batches.append({"input_ids": ids, "attention_mask": mask,
                            "window_end": index == len(groups) - 1})
    return batches


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
        raise RuntimeError(f"Fisher calibration requires all four GPUs, got {placements}")
    return device_map


def main() -> None:
    set_seed(SEED)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=build_device_map(config),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.enable_input_require_grads()  # frozen params: give activations a graph
    load_seconds = time.time() - t_load

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    assistant_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
    if assistant_id is None or assistant_id == tokenizer.unk_token_id:
        assistant_id = None
    loaded = next(iter(load_category_batches(
        dataset_name=DATASET_NAME, split=DATASET_SPLIT, subset=None,
        tokenizer=tokenizer, model_max_length=MODEL_MAX_LENGTH, batch_size=1,
        split_by_category=False, return_vllm_tokens_prompt=False,
        truncate=False, batches_per_category=TARGET_SAMPLES, pack_samples=False,
    ).values()))
    if len(loaded) != TARGET_SAMPLES:
        raise RuntimeError(f"expected {TARGET_SAMPLES} samples, loader returned {len(loaded)}")
    samples = [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in loaded]
    lengths = [x["input_ids"].shape[-1] for x in samples]
    pad_id = tokenizer.pad_token_id or 0
    batches = make_batches(samples, pad_id)
    print(f"[fisher] {len(samples)} unpacked samples, {sum(lengths)} tokens, "
          f"mean {sum(lengths) / len(lengths):.0f}, max {max(lengths)}; {len(batches)} batches", flush=True)

    input_device = model.get_input_embeddings().weight.device

    def step(batch, collector):
        input_ids = batch["input_ids"].to(input_device)
        attention_mask = batch["attention_mask"].to(input_device)
        labels = input_ids.clone()
        labels[~assistant_label_mask(input_ids, attention_mask, assistant_id)] = -100
        if collector is not None:
            collector.set_token_mask(attention_mask)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                     use_cache=False, return_dict=True).loss
        loss.backward()
        model.zero_grad(set_to_none=True)
        return float(loss)

    # Preflight: the largest batch by padded tokens must fit before hours are spent.
    largest = max(batches, key=lambda b: b["input_ids"].numel())
    probe = JointCollector(OUTPUT_DIR, seed=SEED, stability=_flag("FISHER_STABILITY", "true"))
    probe.attach(model)
    step(largest, probe)
    probe.detach()
    del probe
    torch.cuda.synchronize()
    print(f"[fisher] preflight ok on largest batch {tuple(largest['input_ids'].shape)}; peak GiB "
          f"{[round(torch.cuda.max_memory_allocated(i) / 2**30, 1) for i in range(torch.cuda.device_count())]}",
          flush=True)

    collector = JointCollector(OUTPUT_DIR, seed=SEED, fisher_mode=os.environ.get("FISHER_MODE", "router_grad"),
                               stability=_flag("FISHER_STABILITY", "true"))
    collector.attach(model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_tokens = done = 0
    next_checkpoint = CHECKPOINT_EVERY
    step_seconds: list[float] = []
    losses: list[float] = []

    def metadata(final: bool) -> dict:
        elapsed = sum(step_seconds)
        return {
            "final": final,
            "samples": done,
            "total_tokens": total_tokens,
            "mean_loss": sum(losses) / max(len(losses), 1),
            "assistant_token_id": assistant_id,
            "label_mask": "assistant tokens" if assistant_id is not None else "all non-pad tokens",
            "dataset_name": DATASET_NAME,
            "split": DATASET_SPLIT,
            "packed": False,
            "truncate": False,
            "random_selection": True,
            "model_max_length": MODEL_MAX_LENGTH,
            "max_batch_tokens": MAX_BATCH_TOKENS,
            "window": WINDOW,
            "seed": SEED,
            "mode": collector.fisher.mode,
            "stability": collector.stability,
            "model_load_seconds": load_seconds,
            "seconds_elapsed": elapsed,
            "seconds_per_sample": elapsed / max(done, 1),
            "peak_memory_gib_per_gpu": [
                torch.cuda.max_memory_allocated(i) / 2**30 for i in range(torch.cuda.device_count())
            ],
            "target_samples": TARGET_SAMPLES,
        }

    def save(directory: pathlib.Path, final: bool) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(collector.state(), directory / "seap_state.pt")
        (directory / "metadata.json").write_text(json.dumps(metadata(final), indent=2))

    for batch in batches:
        torch.cuda.synchronize()
        start = time.time()
        losses.append(step(batch, collector))
        torch.cuda.synchronize()
        step_seconds.append(time.time() - start)
        done += batch["input_ids"].shape[0]
        total_tokens += int(batch["attention_mask"].sum())
        if len(step_seconds) % 20 == 0:
            print(f"[fisher] {done}/{TARGET_SAMPLES} samples, {total_tokens} tokens, "
                  f"{sum(step_seconds) / done:.2f}s/sample, "
                  f"eta {(TARGET_SAMPLES - done) * sum(step_seconds) / done / 3600:.2f}h", flush=True)
        if batch["window_end"] and done >= next_checkpoint and done < TARGET_SAMPLES:
            save(OUTPUT_DIR / f"checkpoint_{done}", final=False)
            print(f"[fisher] saved checkpoint at {done} samples", flush=True)
            next_checkpoint += CHECKPOINT_EVERY
    collector.detach()

    save(OUTPUT_DIR, final=True)
    print(json.dumps(metadata(True), indent=2), flush=True)


if __name__ == "__main__":
    main()
