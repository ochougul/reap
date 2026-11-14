#!/usr/bin/env python3
"""Compare local curve proxies with full-model held-out token KL.

Run ``--stage prepare`` first to freeze 200 calibration-disjoint sequences and
their unpruned logits. Then ``--stage evaluate`` executes 30 random allocations
at each requested budget plus a uniform-REAP anchor. Evaluation checkpoints after
each allocation so the large model sweep can resume safely.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import qwen4_calibrate as calibration
from qwen4_allocate_damage import PROXIES, stable_reap_orders


EXPECTED_CALIBRATION_SHA256 = "042789c5a55407f9205f6d8dbdfdbd446ea72b452442d2835f081e7d0901466a"
DEFAULT_REAP_DIR = pathlib.Path(
    "artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-"
    "12228-16384-unpruned"
)
DEFAULT_CURVE_DIR = pathlib.Path(
    "artifacts/Qwen3.8-Flash-Next-FP8/reap-damage-curves-evol-codealpaca-"
    "12228-16384-unpruned"
)
DEFAULT_OUTPUT = pathlib.Path(
    "artifacts/Qwen3.8-Flash-Next-FP8/reap-damage-heldout-kl-200-seed42"
)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _spearman(x: list[float], y: list[float]) -> float:
    left, right = _rankdata(np.asarray(x)), _rankdata(np.asarray(y))
    result = np.corrcoef(left, right)[0, 1]
    return float(result) if np.isfinite(result) else 0.0


def _make_allocations(seed: int = 42, repeats: int = 30) -> list[dict]:
    rng = np.random.default_rng(seed)
    result = []
    for budget in (6144, 12288):
        for index in range(repeats):
            counts = np.zeros(48, dtype=np.int64)
            for _ in range(budget):
                available = np.flatnonzero(counts < 384)
                selected = int(rng.choice(available))
                counts[selected] += 1
            result.append({
                "allocation_id": f"random-{budget}-{index:02d}",
                "kind": "random",
                "budget": budget,
                "counts": counts.tolist(),
            })
        uniform, remainder = divmod(budget, 48)
        counts = [uniform + (1 if layer < remainder else 0) for layer in range(48)]
        result.append({
            "allocation_id": f"uniform-reap-{budget}",
            "kind": "uniform_reap_anchor",
            "budget": budget,
            "counts": counts,
        })
    return result


def _load_heldout_sequences(tokenizer, reap_dir: pathlib.Path, count: int = 200):
    metadata = json.loads((reap_dir / "metadata.json").read_text())
    saved_manifest = json.loads((reap_dir / "sample_manifest.json").read_text())
    if metadata.get("sample_set_sha256") != EXPECTED_CALIBRATION_SHA256:
        raise RuntimeError("Calibration metadata has an unexpected sample-set digest")
    if saved_manifest.get("sample_set_sha256") != EXPECTED_CALIBRATION_SHA256:
        raise RuntimeError("Calibration manifest has an unexpected sample-set digest")
    raw = calibration._load_raw_dataset(
        "theblackcat102/evol-codealpaca-v1", "train", subset=None
    )
    processor = calibration.CodeAlpacaChatDataset(
        dataset=raw,
        tokenizer=tokenizer,
        max_input_len=16384,
        pack_samples=False,
        split="train",
        split_by_category=False,
        return_vllm_tokens_prompt=False,
        truncate=False,
        batch_size=1,
    )
    calibration_rows = set(saved_manifest["selected_row_indices"])
    permutation = np.random.default_rng(42).permutation(len(raw))
    result, source_rows, lengths, token_hashes = [], [], [], []
    for row_index in permutation:
        row_index = int(row_index)
        if row_index in calibration_rows:
            continue
        encoded = processor._encode_sample(processor._map_fn(raw[row_index]))
        length = int(encoded.shape[-1])
        if length > 16384 or length < 2:
            continue
        ids = encoded.reshape(1, -1).long()
        token_hash = hashlib.sha256(ids[0].numpy().astype("<i8", copy=False).tobytes()).hexdigest()
        result.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
        source_rows.append(row_index)
        lengths.append(length)
        token_hashes.append(token_hash)
        if len(result) == count:
            break
    if len(result) != count or len(set(source_rows)) != count:
        raise RuntimeError(f"Could not create exactly {count} calibration-disjoint held-out samples")
    manifest = {
        "dataset_name": "theblackcat102/evol-codealpaca-v1",
        "split": "train",
        "seed": 42,
        "calibration_sample_set_sha256": EXPECTED_CALIBRATION_SHA256,
        "selected_row_indices": source_rows,
        "token_lengths": lengths,
        "token_id_sha256": token_hashes,
        "selected_examples": len(result),
        "packing": False,
        "truncation": False,
        "padding": False,
        "max_length": 16384,
    }
    manifest["sample_set_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result, manifest


def _logits(output) -> torch.Tensor:
    value = getattr(output, "logits", None)
    if value is None and isinstance(output, dict):
        value = output.get("logits")
    if not torch.is_tensor(value):
        raise RuntimeError("Qwen4 model did not return vocabulary logits")
    return value


def _save_reference_logits(model, samples, input_device, out_dir: pathlib.Path):
    logits_dir = out_dir / "reference_logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(tqdm(samples, desc="Unpruned held-out references", unit="sequence")):
        path = logits_dir / f"{index:03d}.pt"
        if path.exists():
            continue
        with torch.inference_mode():
            result = model(
                input_ids=sample["input_ids"].to(input_device),
                attention_mask=sample["attention_mask"].to(input_device),
                use_cache=False,
                return_dict=True,
            )
        values = _logits(result)[0].detach().to(device="cpu", dtype=torch.float16)
        torch.save(values, path)


def _install_masked_router(router, valid_mask: torch.Tensor):
    original_forward = router.forward
    valid_mask = valid_mask.to(dtype=torch.bool)

    def forward(hidden_states):
        logits, _weights, _indices = original_forward(hidden_states)
        logits_2d = logits.reshape(-1, logits.shape[-1])
        masked_logits = logits_2d.masked_fill(
            ~valid_mask.to(logits_2d.device), -torch.finfo(logits_2d.dtype).max
        )
        probs = masked_logits.float().softmax(dim=-1)
        weights, indices = probs.topk(router.top_k, dim=-1)
        if router.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
        weights = weights * getattr(router, "routed_scaling_factor", 1.0)
        return masked_logits.reshape_as(logits), weights.to(logits.dtype), indices

    router.forward = forward
    return original_forward


def _proxy_values(curve_path: pathlib.Path) -> dict[str, dict[int, np.ndarray]]:
    points = {proxy: {layer: {} for layer in range(48)} for proxy in PROXIES}
    with curve_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            layer, k = int(row["layer"]), int(row["k"])
            for proxy in PROXIES:
                points[proxy][layer][k] = float(row[proxy])
    result = {}
    for proxy in PROXIES:
        result[proxy] = {}
        expected_ks = sorted(points[proxy][0])
        if len(expected_ks) < 2 or expected_ks[0] != 0 or expected_ks[-1] != 384:
            raise ValueError(f"Proxy {proxy} curve grid must include k=0 and k=384")
        for layer in range(48):
            layer_points = points[proxy][layer]
            if sorted(layer_points) != expected_ks:
                raise ValueError(f"Missing curve points for proxy={proxy}, layer={layer}")
            result[proxy][layer] = np.interp(
                np.arange(385), expected_ks,
                [layer_points[k] for k in expected_ks],
            )
    return result


def _kl_metrics(reference_logits: torch.Tensor, candidate_logits: torch.Tensor):
    reference = reference_logits[:-1].to(candidate_logits.device, dtype=torch.float32)
    candidate = candidate_logits[0, :-1].float()
    reference_log_prob = reference.log_softmax(dim=-1)
    candidate_log_prob = candidate.log_softmax(dim=-1)
    kl = (reference_log_prob.exp() * (reference_log_prob - candidate_log_prob)).sum(-1)
    return float(kl.mean().item()), float(kl.sum().item()), int(kl.numel())


def _evaluate_allocations(model, samples, allocations, curve_values, reap_orders, output_dir, input_device):
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "heldout_kl_results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
    else:
        results = {}
    references = [
        torch.load(output_dir / "reference_logits" / f"{index:03d}.pt", map_location="cpu", weights_only=True)
        for index in range(len(samples))
    ]
    for allocation in allocations:
        allocation_id = allocation["allocation_id"]
        if allocation_id in results:
            continue
        originals = []
        for layer, module in sorted(
            ((calibration._layer_index(name), module) for name, module in model.named_modules()
             if hasattr(module, "experts") and hasattr(module, "gate")),
            key=lambda pair: pair[0],
        ):
            if layer is None:
                continue
            valid = torch.ones(512, dtype=torch.bool)
            removed = reap_orders[layer][: int(allocation["counts"][layer])]
            valid[removed] = False
            originals.append((module.gate, _install_masked_router(module.gate, valid)))

        sequence_means, kl_sum, token_count = [], 0.0, 0
        for index, sample in enumerate(tqdm(samples, desc=f"KL {allocation_id}", unit="sequence")):
            with torch.inference_mode():
                output = model(
                    input_ids=sample["input_ids"].to(input_device),
                    attention_mask=sample["attention_mask"].to(input_device),
                    use_cache=False,
                    return_dict=True,
                )
            mean_kl, sum_kl, count = _kl_metrics(references[index], _logits(output))
            sequence_means.append(mean_kl)
            kl_sum += sum_kl
            token_count += count
        for router, original in originals:
            router.forward = original
        results[allocation_id] = {
            "allocation_id": allocation_id,
            "budget": allocation["budget"],
            "kind": allocation["kind"],
            "counts": allocation["counts"],
            "mean_token_kl": kl_sum / max(token_count, 1),
            "mean_sequence_kl": float(np.mean(sequence_means)),
            "sequence_token_kl": sequence_means,
            "evaluated_sequences": len(sequence_means),
            "token_count": token_count,
        }
        results_path.write_text(json.dumps(results, indent=2) + "\n")

    rows = list(results.values())
    correlations = {}
    for budget in (6144, 12288):
        budget_rows = [row for row in rows if row["budget"] == budget and row["kind"] == "random"]
        correlations[str(budget)] = {
            proxy: _spearman(
                [sum(curve_values[proxy][layer][count] for layer, count in enumerate(row["counts"]))
                 for row in budget_rows],
                [row["mean_token_kl"] for row in budget_rows],
            )
            for proxy in PROXIES
        }
    averages = {
        proxy: float(np.mean([correlations[str(b)][proxy] for b in (6144, 12288)]))
        for proxy in PROXIES
    }
    selected = max(averages, key=averages.get)
    summary = {
        "seed": 42,
        "calibration_sample_set_sha256": EXPECTED_CALIBRATION_SHA256,
        "heldout_sample_set_sha256": json.loads((output_dir / "heldout_manifest.json").read_text())["sample_set_sha256"],
        "budgets": [6144, 12288],
        "random_allocations_per_budget": 30,
        "correlations_by_budget": correlations,
        "average_spearman_by_proxy": averages,
        "selected_proxy": selected,
        "allocation_count_completed": len(rows),
        "results": "heldout_kl_results.json",
    }
    (output_dir / "proxy_selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "evaluate"), required=True)
    parser.add_argument("--reap-dir", type=pathlib.Path, default=DEFAULT_REAP_DIR)
    parser.add_argument("--curve-dir", type=pathlib.Path, default=DEFAULT_CURVE_DIR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    calibration._patch_triton_kernel_compat()
    calibration._patch_ep_tp_compat()
    reap_state = args.reap_dir / "calibration_state.pt"
    reap_orders, reap_sha = stable_reap_orders(reap_state)
    source_metadata = json.loads((args.reap_dir / "metadata.json").read_text())
    if source_metadata.get("sample_set_sha256") != EXPECTED_CALIBRATION_SHA256:
        raise RuntimeError("REAP source artifact sample hash is not the required value")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.from_pretrained(calibration.MODEL_PATH, local_files_only=True, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        calibration.MODEL_PATH,
        device_map=calibration._qwen4_device_map(config),
        dtype=None,
        experts_implementation="grouped_mm",
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        offload_state_dict=True,
        offload_buffers=True,
    ).eval()
    calibration._assert_fp8_loaded(model)
    calibration._keep_ngram_embedding_on_cpu(model)
    calibration._bridge_qwen4_ple_layer(model)
    calibration._patch_sensitivity_rotary_device(model)
    tokenizer = AutoTokenizer.from_pretrained(calibration.MODEL_PATH, local_files_only=True, trust_remote_code=True)
    samples, manifest = _load_heldout_sequences(tokenizer, args.reap_dir)
    manifest_path = args.output_dir / "heldout_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["sample_set_sha256"] != manifest["sample_set_sha256"]:
        raise RuntimeError("Existing held-out artifacts use a different sample set")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    input_device = model.get_input_embeddings().weight.device

    if args.stage == "prepare":
        _save_reference_logits(model, samples, input_device, args.output_dir)
        reference_meta = {
            "sample_set_sha256": manifest["sample_set_sha256"],
            "num_sequences": len(samples),
            "token_count": sum(manifest["token_lengths"]),
            "logit_dtype": "float16",
            "logits_path": "reference_logits/{sequence_id}.pt",
            "reap_state_sha256": reap_sha,
            "model_path": calibration.MODEL_PATH,
        }
        (args.output_dir / "reference_metadata.json").write_text(json.dumps(reference_meta, indent=2) + "\n")
        print(json.dumps(reference_meta, indent=2))
        return

    if not (args.output_dir / "reference_metadata.json").exists():
        raise FileNotFoundError("Run --stage prepare before --stage evaluate")
    allocations = _make_allocations()
    (args.output_dir / "allocation_schedule.json").write_text(json.dumps(allocations, indent=2) + "\n")
    curve_values = _proxy_values(args.curve_dir / "curve_summary.csv")
    summary = _evaluate_allocations(
        model, samples, allocations, curve_values, reap_orders, args.output_dir, input_device
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
