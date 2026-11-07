"""Freeze and run the MBPP+ development screen for layerwise allocations.

The serving-time mask keeps the original checkpoint on disk and masks removed
experts before routing.  Functionally this evaluates the retained-expert model
without producing six large copied FP8 checkpoints; it does not measure storage
or latency savings.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import random
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


ROOT = Path("artifacts/Qwen3.8-Flash-Next-FP8/layerwise-reuse-v1")
BASE_MODEL = Path(
    "/home/ubuntu/hf_cache/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/"
    "236dfdf285828023ca3bcd3f37366c58a3469b13"
)
EVALPLUS_ROOT = Path("third-party/evalplus")
SOURCE_ROOT = Path("src")
SERVED_NAME = "Qwen/Qwen3.8-Flash-Next-FP8"
VARIANTS = (
    "U-reap-R9180",
    "T-reap-R9180",
    "P-reap-R9180",
    "U-mixture-lambda-0.75-R9180",
    "T-mixture-lambda-0.75-R9180",
    "P-mixture-lambda-0.75-R9180",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_tasks() -> tuple[dict[str, dict], str]:
    from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash

    return get_mbpp_plus(version="default"), get_mbpp_plus_hash(version="default")


def freeze_split() -> Path:
    tasks, dataset_hash = load_tasks()
    task_ids = sorted(tasks, key=lambda task_id: int(task_id.rsplit("/", 1)[1]))
    shuffled = list(task_ids)
    random.Random(42).shuffle(shuffled)
    payload = {
        "schema_version": 1,
        "benchmark": "EvalPlus MBPP+",
        "dataset_version": "MBPP+ v0.2.0 (EvalPlus vendored default)",
        "dataset_hash": dataset_hash,
        "task_count": len(task_ids),
        "task_id_order": "numeric task ID ascending, then Python random.Random(42).shuffle",
        "screening_seed": 42,
        "screening_task_count": 128,
        "screening_task_ids": shuffled[:128],
        "confirmation_task_count": len(shuffled) - 128,
        "confirmation_task_ids": shuffled[128:],
        "task_id_list_sha256": json_hash(task_ids),
        "screening_ids_sha256": json_hash(shuffled[:128]),
        "confirmation_ids_sha256": json_hash(shuffled[128:]),
        "candidates": list(VARIANTS),
        "generation": {
            "decoding": "greedy",
            "n": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "thinking": False,
            "max_new_tokens": 16384,
            "tensor_parallel_size": 4,
            "expert_parallel": True,
            "max_model_len": 32768,
            "gpu_memory_utilization": 0.90,
        },
        "method": "original checkpoint served with a static pre-routing expert mask derived from each frozen allocation manifest",
        "scope_note": "Measures functional accuracy of retained-expert masks; no compact checkpoint size or serving latency claim.",
    }
    output = ROOT / "development_protocol.json"
    ROOT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2) + "\n"
    if output.exists():
        if output.read_text() != encoded:
            raise RuntimeError(f"Refusing to replace frozen development split: {output}")
    else:
        output.write_text(encoded)
    return output


def load_protocol() -> dict[str, Any]:
    protocol_path = ROOT / "development_protocol.json"
    if not protocol_path.exists():
        raise FileNotFoundError(f"Run --prepare before evaluation: {protocol_path}")
    return json.loads(protocol_path.read_text())


def prepare_mask_and_model(variant: str, attempt: str) -> tuple[Path, Path, dict[str, Any]]:
    allocation_path = ROOT / "allocations" / variant / "manifest.json"
    if not allocation_path.exists():
        raise FileNotFoundError(allocation_path)
    allocation = json.loads(allocation_path.read_text())
    if allocation.get("total_removed") != 9180:
        raise ValueError(f"{variant} does not have exactly 9,180 removals")
    removed = allocation["removed_expert_ids_original"]
    masks: dict[str, list[int]] = {}
    for layer in range(48):
        removed_ids = set(removed[str(layer)])
        if len(removed_ids) != allocation["per_layer_removal_counts"][str(layer)]:
            raise ValueError(f"Layer {layer} removal IDs do not match its count")
        mask = [int(expert not in removed_ids) for expert in range(512)]
        if sum(mask) < 10:
            raise ValueError(f"Layer {layer} retains fewer than top-k experts")
        masks[str(layer)] = mask

    run_root = ROOT / "evaluations" / variant / attempt
    run_root.mkdir(parents=True, exist_ok=True)
    metadata_path = run_root / "nonuniform_metadata.json"
    metadata = {
        "variant": variant,
        "source_allocation_manifest": str(allocation_path),
        "source_allocation_sha256": sha256(allocation_path),
        "original_experts": 512,
        "top_k": 10,
        "total_removed": 9180,
        "total_retained": 48 * 512 - 9180,
        "valid_expert_mask": masks,
        "routing": {
            "mask_before_softmax_and_topk": True,
            "padded_experts_receive_zero_probability": True,
            "redistribute_over_retained_experts_only": True,
        },
    }
    encoded = json.dumps(metadata, indent=2) + "\n"
    if metadata_path.exists():
        if metadata_path.read_text() != encoded:
            raise RuntimeError(f"Existing mask metadata differs; refusing overwrite: {metadata_path}")
    else:
        metadata_path.write_text(encoded)

    # vLLM sees a normal local model directory plus a tiny metadata file.  All
    # model files remain symlinks into the existing HF cache; no large model
    # copy is made.
    model_dir = run_root / "model"
    model_dir.mkdir(exist_ok=True)
    if not BASE_MODEL.is_dir():
        raise FileNotFoundError(BASE_MODEL)
    for source in BASE_MODEL.iterdir():
        target = model_dir / source.name
        if source.name == "nonuniform_metadata.json":
            continue
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError(f"Unexpected existing model symlink: {target}")
        elif target.exists():
            raise RuntimeError(f"Refusing to replace existing model path: {target}")
        else:
            target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    # Place the metadata next to the symlinked config/checkpoint files.
    link_metadata = model_dir / "nonuniform_metadata.json"
    if link_metadata.is_symlink():
        if link_metadata.resolve() != metadata_path.resolve():
            raise RuntimeError(f"Unexpected metadata symlink: {link_metadata}")
    elif not link_metadata.exists():
        link_metadata.symlink_to(metadata_path.resolve())
    elif sha256(link_metadata) != sha256(metadata_path):
        raise RuntimeError(f"Existing model metadata does not match: {link_metadata}")
    return run_root, model_dir, metadata


def port_is_available(port: int) -> bool:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def run_server(model_dir: Path, metadata_path: Path, log_path: Path, port: int) -> tuple[subprocess.Popen, Any]:
    if not port_is_available(port):
        raise RuntimeError(f"Evaluation port {port} is already in use")
    env = os.environ.copy()
    pythonpath = [str(SOURCE_ROOT.resolve())]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env["REAP_QWEN38_MASK_METADATA"] = str(metadata_path.resolve())
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    command = [
        str(Path(sys.executable).with_name("vllm")),
        "serve",
        str(model_dir.resolve()),
        "--gpu-memory-utilization", "0.90",
        "--tensor-parallel-size", "4",
        "--seed", "42",
        "--port", str(port),
        "--max-model-len", "32768",
        "--max-num-seqs", "64",
        "--max-num-batched-tokens", "32768",
        "--override-generation-config", "{}",
        "--trust-remote-code",
        "--enable-expert-parallel",
        "--served-model-name", SERVED_NAME,
    ]
    log_handle = log_path.open("w")
    process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_handle.flush()
            raise RuntimeError(f"vLLM exited early (code {process.returncode}); inspect {log_path}")
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                return process, log_handle
        except requests.RequestException:
            pass
        time.sleep(5)
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    log_handle.close()
    raise TimeoutError(f"vLLM did not become healthy; inspect {log_path}")


def stop_server(process: subprocess.Popen, log_handle: Any) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    log_handle.close()


def generate_and_score(
    variant: str,
    model_dir: Path,
    run_root: Path,
    protocol: dict[str, Any],
    port: int,
    partition: str = "screening",
) -> dict[str, Any]:
    from evalplus.data import get_mbpp_plus
    import openai
    import evalplus.evaluate as evaluator

    ids_key = f"{partition}_task_ids"
    hash_key = f"{partition}_ids_sha256"
    ids = protocol[ids_key]
    expected_count = len(ids)
    all_tasks = get_mbpp_plus(version="default")
    tasks = {task_id: all_tasks[task_id] for task_id in ids}
    if len(tasks) != expected_count:
        raise AssertionError(f"Expected {expected_count} {partition} tasks; got {len(tasks)}")
    sample_path = run_root / "samples.jsonl"
    raw_path = run_root / "samples.raw.jsonl"
    results_path = run_root / "evalplus_results.json"
    summary_path = run_root / "summary.json"
    if summary_path.exists():
        old = json.loads(summary_path.read_text())
        if old.get("task_ids_sha256") != protocol[hash_key]:
            raise RuntimeError(f"Existing result uses a different task split: {summary_path}")
        return old
    if sample_path.exists() or raw_path.exists() or results_path.exists():
        raise RuntimeError(f"Partial pre-existing run found under {run_root}; inspect before resuming")

    server_log = run_root / "vllm.log"
    metadata_path = run_root / "nonuniform_metadata.json"
    process, log_handle = run_server(model_dir, metadata_path, server_log, port)
    try:
        client = openai.OpenAI(
            api_key="none",
            base_url=f"http://127.0.0.1:{port}/v1",
            timeout=300,
            max_retries=0,
        )
        prompt_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"

        def request_one(task_id: str) -> tuple[str, str, str]:
            task = tasks[task_id]
            message = prompt_prefix + f"\n```python\n{task['prompt'].strip()}\n```"
            last_error = None
            for attempt_index in range(3):
                try:
                    response = client.chat.completions.create(
                        model=SERVED_NAME,
                        messages=[{"role": "user", "content": message}],
                        temperature=0.0,
                        top_p=1.0,
                        max_completion_tokens=16384,
                        n=1,
                        seed=42,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    content = response.choices[0].message.content or ""
                    if not content.strip():
                        raise RuntimeError("empty model completion")
                    fenced = re.findall(r"```(?:python|py)?\s*\n(.*?)```", content, flags=re.IGNORECASE | re.DOTALL)
                    code = max(fenced, key=len).strip() if fenced else content.strip()
                    # The request explicitly asks for one self-contained script
                    # in a fenced code block.  Strip the fence here and let the
                    # official EvalPlus sandbox judge the resulting program;
                    # its generic prose sanitizer is quadratic on long outputs.
                    return task_id, content, code
                except Exception as error:  # bounded retries keep task coverage auditable
                    last_error = error
                    if attempt_index < 2:
                        time.sleep(2 * (attempt_index + 1))
            raise RuntimeError(f"generation failed for {task_id} after three attempts: {last_error}")

        completed: dict[str, tuple[str, str]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(request_one, task_id): task_id for task_id in ids}
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                task_id, raw, solution = future.result()
                completed[task_id] = (raw, solution)
                if index % 8 == 0 or index == len(ids):
                    print(f"generated {index}/{len(ids)} tasks", flush=True)

        if set(completed) != set(ids):
            raise RuntimeError(f"generation coverage mismatch: {len(completed)} of {len(ids)}")
        with sample_path.open("x") as samples, raw_path.open("x") as raw_samples:
            for task_id in ids:
                raw, solution = completed[task_id]
                samples.write(json.dumps({"task_id": task_id, "solution": solution}) + "\n")
                raw_samples.write(json.dumps({"task_id": task_id, "solution": raw}) + "\n")
        client.close()
    finally:
        stop_server(process, log_handle)

    counts: dict[str, int] = {}
    with sample_path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                counts[row["task_id"]] = counts.get(row["task_id"], 0) + 1
    if set(counts) != set(ids) or any(count != 1 for count in counts.values()):
        missing = sorted(set(ids) - set(counts))
        unexpected = sorted(set(counts) - set(ids))
        raise RuntimeError(f"Generated sample coverage mismatch: missing={missing}, unexpected={unexpected}")

    # EvalPlus normally requires every task in its dataset.  Scope only its
    # dataset accessors to the pre-frozen screen subset and give that subset a
    # distinct truth-cache key; then use the official MBPP+ execution/scoring.
    split_hash = json_hash({"dataset_hash": protocol["dataset_hash"], "task_ids": ids})
    evaluator.get_mbpp_plus = lambda **kwargs: tasks
    evaluator.get_mbpp_plus_hash = lambda **kwargs: split_hash
    result = evaluator.evaluate(
        dataset="mbpp",
        samples=str(sample_path),
        output_file=str(results_path),
        parallel=16,
        version="default",
    )
    stored = json.loads(results_path.read_text())
    evaluated = stored.get("eval", {})
    if set(evaluated) != set(ids) or len(evaluated) != len(ids):
        raise RuntimeError(
            f"EvalPlus result coverage mismatch: expected {expected_count} task IDs, got {len(evaluated)}"
        )
    plus_solved = sum(
        bool(rows) and rows[0].get("plus_status") == "pass"
        for rows in evaluated.values()
    )
    base_solved = sum(
        bool(rows) and rows[0].get("base_status") == "pass"
        for rows in evaluated.values()
    )
    summary = {
        "variant": variant,
        "partition": partition,
        "task_count": len(evaluated),
        "task_ids_sha256": protocol[hash_key],
        "dataset_hash": protocol["dataset_hash"],
        "mbpp_plus_solved": plus_solved,
        "mbpp_plus_pass_at_1": plus_solved / len(evaluated),
        "mbpp_base_solved": base_solved,
        "mbpp_base_pass_at_1": base_solved / len(evaluated),
        "task_ids_evaluated": sorted(evaluated, key=lambda task_id: int(task_id.rsplit("/", 1)[1])),
        "allocation_manifest_sha256": sha256(ROOT / "allocations" / variant / "manifest.json"),
        "mask_metadata_sha256": sha256(metadata_path),
        "model_config_sha256": sha256(BASE_MODEL / "config.json"),
        "sample_file_sha256": sha256(sample_path),
        "result_file_sha256": sha256(results_path),
        "evalplus_result": str(results_path),
        "vllm_log": str(server_log),
        "vllm_log_sha256": sha256(server_log),
        "model_snapshot": BASE_MODEL.name,
        "prompt_prefix": "Please provide a self-contained Python script that solves the following problem in a markdown code block:",
        "response_extraction": "use longest fenced Python/code block; otherwise evaluate the full response",
        "inference_mode": "static expert mask on original FP8 checkpoint; no copied/pruned weight file",
        "serving_settings": {
            "tensor_parallel_size": 4,
            "expert_parallel": True,
            "max_model_len": 32768,
            "gpu_memory_utilization": 0.90,
            "server_seed": 42,
            "per_mask_server_restart": True,
        },
        "generation_settings": protocol["generation"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_versions": {
            "python": sys.version,
            "torch": __import__("torch").__version__,
            "transformers": __import__("transformers").__version__,
            "vllm": __import__("vllm").__version__,
        },
        "runner_sha256": sha256(Path(__file__).resolve()),
        "evalplus_codegen_sha256": sha256(EVALPLUS_ROOT / "evalplus" / "codegen.py"),
        "evalplus_evaluate_sha256": sha256(EVALPLUS_ROOT / "evalplus" / "evaluate.py"),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def write_paired_comparisons(
    protocol: dict[str, Any],
    attempt: str,
    variants: tuple[str, ...],
    partition: str = "screening",
    metric: str = "plus",
) -> Path:
    import numpy as np

    if metric not in {"plus", "joint"}:
        raise ValueError("metric must be 'plus' or 'joint'")
    outcomes: dict[str, dict[str, int]] = {}
    ids = protocol[f"{partition}_task_ids"]
    expected_count = len(ids)
    ids_hash = protocol[f"{partition}_ids_sha256"]
    for variant in variants:
        result_path = ROOT / "evaluations" / variant / attempt / "evalplus_results.json"
        if not result_path.exists():
            raise FileNotFoundError(f"missing completed evaluation: {result_path}")
        result = json.loads(result_path.read_text())
        if len(result.get("eval", {})) != expected_count:
            raise ValueError(
                f"{variant} does not contain all {expected_count} {partition} tasks"
            )
        outcomes[variant] = {
            task_id: int(
                bool(rows)
                and rows[0].get("plus_status") == "pass"
                and (
                    metric == "plus"
                    or rows[0].get("base_status") == "pass"
                )
            )
            for task_id, rows in result["eval"].items()
        }
        if set(outcomes[variant]) != set(ids):
            raise ValueError(f"{variant} result IDs differ from the frozen {partition} split")

    rng = np.random.default_rng(42)
    output_rows = []
    for i, candidate in enumerate(variants):
        for control in variants[i + 1 :]:
            differences = np.asarray([outcomes[candidate][task] - outcomes[control][task] for task in ids], dtype=np.int8)
            wins = int((differences == 1).sum())
            losses = int((differences == -1).sum())
            boot = rng.choice(differences, size=(10000, len(ids)), replace=True).mean(axis=1)
            discordant = wins + losses
            if discordant:
                tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / (2 ** discordant)
                mcnemar_p = min(1.0, 2.0 * tail)
            else:
                mcnemar_p = 1.0
            output_rows.append({
                "candidate": candidate,
                "control": control,
                "tasks": len(ids),
                "candidate_plus_solved": sum(outcomes[candidate].values()),
                "control_plus_solved": sum(outcomes[control].values()),
                "paired_delta_solved": wins - losses,
                "paired_delta_pass_rate": float(differences.mean()),
                "bootstrap_95_low": float(np.quantile(boot, 0.025)),
                "bootstrap_95_high": float(np.quantile(boot, 0.975)),
                "candidate_only_passes": wins,
                "control_only_passes": losses,
                "exact_mcnemar_p": mcnemar_p,
                "task_ids_sha256": ids_hash,
                "metric": "MBPP+ plus_status" if metric == "plus" else "MBPP base AND MBPP+",
                "interpretation": (
                    "development screening; not confirmation evidence"
                    if partition == "screening"
                    else "development confirmation; not external benchmark evidence"
                ),
            })
    output = ROOT / f"paired_comparisons_{metric}_{partition}_{attempt}.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true", help="Freeze the MBPP+ task split; no inference")
    group.add_argument(
        "--variant",
        choices=VARIANTS,
        action="append",
        help="Run a frozen candidate (repeat to select multiple candidates)",
    )
    group.add_argument("--all", action="store_true", help="Run all six frozen candidates sequentially")
    parser.add_argument(
        "--partition",
        choices=("screening", "confirmation"),
        default="screening",
        help="Use the pre-frozen 128-task screen or disjoint confirmation partition",
    )
    parser.add_argument("--port", type=int, default=18010)
    parser.add_argument("--attempt", default="mbppplus-screen-v3")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps({"development_protocol": str(freeze_split())}, indent=2))
        return
    protocol = load_protocol()
    if len(protocol.get("screening_task_ids", [])) != 128 or len(protocol.get("confirmation_task_ids", [])) != protocol["task_count"] - 128:
        raise ValueError("frozen development protocol has invalid split sizes")
    if protocol.get("dataset_hash") != load_tasks()[1]:
        raise ValueError("current MBPP+ dataset hash differs from frozen protocol")
    variants = VARIANTS if args.all else tuple(args.variant or ())
    if not variants:
        raise ValueError("select at least one --variant or use --all")
    summaries = []
    for index, variant in enumerate(variants):
        run_root, model_dir, _metadata = prepare_mask_and_model(variant, args.attempt)
        summary = generate_and_score(
            variant,
            model_dir,
            run_root,
            protocol,
            args.port + index,
            partition=args.partition,
        )
        summaries.append(summary)
        print(json.dumps({"variant": variant, "solved": summary["mbpp_plus_solved"], "tasks": summary["task_count"], "pass_at_1": summary["mbpp_plus_pass_at_1"]}), flush=True)
    index_path = ROOT / "evaluations" / f"paired_comparison_input_{args.partition}_{args.attempt}.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps({"task_ids_sha256": protocol[f"{args.partition}_ids_sha256"], "partition": args.partition, "summaries": summaries}, indent=2) + "\n")
    if args.all:
        print(json.dumps({"paired_comparisons": str(write_paired_comparisons(protocol, args.attempt, variants, partition=args.partition))}), flush=True)
    elif len(variants) > 1:
        print(json.dumps({"paired_comparisons": str(write_paired_comparisons(protocol, args.attempt, variants, partition=args.partition))}), flush=True)


if __name__ == "__main__":
    main()
