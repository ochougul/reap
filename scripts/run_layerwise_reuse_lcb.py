#!/usr/bin/env python3
"""Run and strictly validate frozen historical LiveCodeBench evaluations.

This script only prepares static expert masks and delegates inference to the
repository's existing ``src/reap/eval.py``. It never loads a model itself.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_TASK_COUNT = 182
EXPECTED_TASK_ID_SHA256 = (
    "ad8b7e21195710bd04fbe4794013aeddb9be2172e3ee2309dbb0caaa2dea5383"
)
START_DATE = "2025-01-01"
END_DATE = "2025-04-30"
SEED = 42
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 16384
TIMEOUT_SECONDS = 120
SCENARIO_STEM = "Scenario.codegeneration_1_0.0"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALLOCATION_ROOT = (
    REPO_ROOT / "artifacts/Qwen3.8-Flash-Next-FP8/layerwise-reuse-v1/allocations"
)
EVAL_SCRIPT = REPO_ROOT / "src/reap/eval.py"


@dataclass(frozen=True)
class Candidate:
    requested: str
    variant: str
    label: str
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    checkpoint: Path
    num_layers: int
    num_experts: int
    top_k: int
    removed_by_layer: dict[int, list[int]]
    run_dir: Path
    port: int


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def cache_roots() -> list[Path]:
    if os.environ.get("HF_DATASETS_CACHE"):
        return [Path(os.environ["HF_DATASETS_CACHE"]).expanduser().resolve()]
    if os.environ.get("HF_HOME"):
        return [(Path(os.environ["HF_HOME"]).expanduser() / "datasets").resolve()]
    return [(Path.home() / ".cache/huggingface/datasets").resolve()]


def frozen_historical_ids() -> tuple[list[str], str, str]:
    """Read IDs/dates from the local release_latest Arrow cache only."""
    from datasets import Dataset

    cache_dirs: list[Path] = []
    for root in cache_roots():
        cache_dir = root / "livecodebench___code_generation_lite"
        if cache_dir.is_dir():
            cache_dirs.append(cache_dir)
    if not cache_dirs:
        raise RuntimeError(
            "Local LiveCodeBench dataset cache not found; refusing network fetch. "
            "Expected livecodebench___code_generation_lite under HF_DATASETS_CACHE."
        )

    matching_infos: list[tuple[Path, dict[str, Any]]] = []
    for cache_dir in cache_dirs:
        for info_path in cache_dir.rglob("dataset_info.json"):
            info = read_json(info_path)
            if (
                info.get("builder_name") == "code_generation_lite"
                and info.get("config_name") == "release_latest"
                # This HF cache layout preserves the builder config name for
                # more than one version_tag, so also select the requested tag.
                and "release_latest-version_tag=release_latest"
                in info_path.parent.parts
                and any(info_path.parent.glob("*.arrow"))
            ):
                matching_infos.append((info_path.parent, info))

    if len(matching_infos) != 1:
        paths = [str(path) for path, _ in matching_infos]
        raise RuntimeError(
            "Expected exactly one cached LiveCodeBench release_latest dataset, "
            f"found {len(paths)}: {paths}"
        )

    data_dir, info = matching_infos[0]
    expected_rows = info.get("splits", {}).get("test", {}).get("num_examples")
    arrow_files = sorted(data_dir.glob("*.arrow"))
    if not arrow_files:
        raise RuntimeError(f"No local Arrow shards found in {data_dir}")

    selected_rows: list[tuple[str, str]] = []
    for arrow_path in arrow_files:
        dataset = Dataset.from_file(str(arrow_path)).select_columns(
            ["question_id", "contest_date"]
        )
        for row in dataset:
            date_value = str(row["contest_date"])
            if START_DATE <= date_value[:10] <= END_DATE:
                selected_rows.append((str(row["question_id"]), date_value))

    ids = [question_id for question_id, _ in selected_rows]
    unique_ids = sorted(set(ids))
    if len(ids) != EXPECTED_TASK_COUNT or len(unique_ids) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            "Cached release_latest task window does not contain exactly "
            f"{EXPECTED_TASK_COUNT} unique tasks (rows={len(ids)}, "
            f"unique={len(unique_ids)}; cache metadata rows={expected_rows})."
        )
    id_hash = sha256_bytes("\n".join(unique_ids).encode("utf-8"))
    if id_hash != EXPECTED_TASK_ID_SHA256:
        raise RuntimeError(
            "Frozen task ID hash mismatch; refusing evaluation. "
            f"Expected {EXPECTED_TASK_ID_SHA256}, got {id_hash}."
        )
    cache_description = f"{data_dir} ({expected_rows} cached rows)"
    return unique_ids, id_hash, cache_description


def resolve_manifest(requested: str, allocation_root: Path) -> Path:
    given = Path(requested).expanduser()
    if given.is_file():
        return given.resolve()
    candidate = allocation_root / requested / "manifest.json"
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(
        f"{requested!r} is neither a manifest file nor a variant below "
        f"{allocation_root}"
    )


def manifest_checkpoint(manifest_path: Path) -> Path:
    experiment_root = manifest_path.parent.parent.parent
    experiment_path = experiment_root / "experiment_manifest.json"
    experiment = read_json(experiment_path)
    model_path = experiment.get("model")
    if not model_path:
        model_path = experiment.get("calibration_metadata", {}).get("model_path")
    if not model_path:
        raise ValueError(f"No source model path recorded in {experiment_path}")
    checkpoint = Path(model_path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Original FP8 checkpoint not found: {checkpoint}")
    return checkpoint


def source_model_dimensions(checkpoint: Path) -> tuple[int, int, int]:
    config = read_json(checkpoint / "config.json")
    model_config = config.get("text_config", config)
    try:
        return (
            int(model_config["num_hidden_layers"]),
            int(model_config["num_experts"]),
            int(model_config["num_experts_per_tok"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Checkpoint config lacks layer/expert/top-k dimensions: {checkpoint}"
        ) from exc


def validate_manifest(
    requested: str,
    manifest_path: Path,
    checkpoint_override: Path | None,
    run_dir: Path,
    port: int,
) -> Candidate:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest is not a JSON object: {manifest_path}")

    variant = str(manifest.get("variant") or manifest.get("allocation") or "")
    if not variant:
        raise ValueError(f"Manifest has no variant/allocation name: {manifest_path}")
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", variant).strip("._-")
    if not label or label in {".", ".."}:
        raise ValueError(f"Unsafe or empty variant name {variant!r}")

    checkpoint = (
        checkpoint_override.resolve()
        if checkpoint_override
        else manifest_checkpoint(manifest_path)
    )
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Original FP8 checkpoint not found: {checkpoint}")
    num_layers, num_experts, config_top_k = source_model_dimensions(checkpoint)

    top_k = int(manifest.get("top_k", -1))
    if top_k != config_top_k:
        raise ValueError(
            f"Manifest top_k={top_k} does not match checkpoint top-k={config_top_k}"
        )
    if int(manifest.get("num_layers", num_layers)) != num_layers:
        raise ValueError("Manifest num_layers conflicts with source checkpoint config")
    if int(manifest.get("num_experts", num_experts)) != num_experts:
        raise ValueError("Manifest num_experts conflicts with source checkpoint config")
    if int(manifest.get("original_experts", num_experts)) != num_experts:
        raise ValueError("Manifest original_experts conflicts with source checkpoint config")
    checkpoint_files = [path for path in checkpoint.iterdir() if path.is_file()]
    if not any(path.name == "model.safetensors.index.json" for path in checkpoint_files):
        raise ValueError(f"Missing model.safetensors.index.json in {checkpoint}")
    if not any(
        re.fullmatch(r"model-\d+-of-\d+\.safetensors", path.name)
        for path in checkpoint_files
    ):
        raise ValueError(f"No sharded FP8 model weights found in {checkpoint}")

    ids_by_layer_raw = manifest.get("removed_expert_ids_original")
    counts_raw = manifest.get("per_layer_removal_counts")
    if not isinstance(ids_by_layer_raw, dict) or not isinstance(counts_raw, dict):
        raise ValueError(f"Manifest lacks exact per-layer removals/counts: {manifest_path}")

    expected_layers = set(range(num_layers))
    try:
        id_layers = {int(layer) for layer in ids_by_layer_raw}
        count_layers = {int(layer) for layer in counts_raw}
    except (TypeError, ValueError) as exc:
        raise ValueError("Manifest has non-integer layer keys") from exc
    if id_layers != expected_layers or count_layers != expected_layers:
        raise ValueError(
            "Manifest must specify removed IDs and removal counts for every "
            f"layer 0..{num_layers - 1}"
        )

    removed_by_layer: dict[int, list[int]] = {}
    for raw_layer, raw_ids in ids_by_layer_raw.items():
        layer = int(raw_layer)
        if not isinstance(raw_ids, list):
            raise ValueError(f"Layer {layer} removals must be a list")
        ids = [int(expert_id) for expert_id in raw_ids]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate removed expert IDs in layer {layer}")
        if any(expert_id < 0 or expert_id >= num_experts for expert_id in ids):
            raise ValueError(f"Out-of-range removed expert ID in layer {layer}")
        declared_count = int(counts_raw.get(str(layer), counts_raw.get(layer, -1)))
        if declared_count != len(ids):
            raise ValueError(
                f"Layer {layer} declares {declared_count} removals but lists {len(ids)}"
            )
        if num_experts - len(ids) < top_k:
            raise ValueError(
                f"Layer {layer} retains fewer than top_k={top_k} experts"
            )
        removed_by_layer[layer] = ids

    total_removed = sum(map(len, removed_by_layer.values()))
    if total_removed != int(manifest.get("total_removed", -1)):
        raise ValueError("Manifest total_removed does not match per-layer ID lists")
    if total_removed != sum(int(value) for value in counts_raw.values()):
        raise ValueError("Manifest total removal counts are inconsistent")
    if num_layers * num_experts - total_removed != int(
        manifest.get("total_retained", num_layers * num_experts - total_removed)
    ):
        raise ValueError("Manifest total_retained is inconsistent")

    return Candidate(
        requested=requested,
        variant=variant,
        label=label,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_sha256=sha256_bytes(manifest_bytes),
        checkpoint=checkpoint,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        removed_by_layer=removed_by_layer,
        run_dir=run_dir,
        port=port,
    )


def verify_eval_protocol() -> None:
    """Fail closed if eval.py's in-code LCB settings drift from the freeze."""
    tree = ast.parse(EVAL_SCRIPT.read_text(), filename=str(EVAL_SCRIPT))
    expected = {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "n": 1,
        "enable_thinking": False,
        "timeout": TIMEOUT_SECONDS,
        "max_tokens": MAX_TOKENS,
    }
    matching_calls: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "get_args_dict":
            continue
        kwargs: dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg is not None:
                try:
                    kwargs[keyword.arg] = ast.literal_eval(keyword.value)
                except (ValueError, TypeError):
                    continue
        if set(expected).issubset(kwargs):
            matching_calls.append(kwargs)
    if not matching_calls or any(
        any(call.get(key) != value for key, value in expected.items())
        for call in matching_calls
    ):
        raise RuntimeError(
            "Existing eval.py does not match frozen LCB protocol constants: "
            f"expected {expected}; found {matching_calls}"
        )


def verify_ports_available(ports: list[int]) -> None:
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
            except OSError as exc:
                raise RuntimeError(f"Requested vLLM port {port} is unavailable") from exc


def prepare_model(candidate: Candidate) -> tuple[Path, Path, str]:
    candidate.run_dir.mkdir(parents=True, exist_ok=False)
    model_dir = candidate.run_dir / "Qwen3.8-Flash-Next-FP8"
    model_dir.mkdir()
    checkpoint_files = sorted(path for path in candidate.checkpoint.iterdir() if path.is_file())
    if not checkpoint_files:
        raise RuntimeError(f"No checkpoint files found in {candidate.checkpoint}")
    if not any(path.name == "model.safetensors.index.json" for path in checkpoint_files):
        raise RuntimeError(f"Missing model.safetensors.index.json in {candidate.checkpoint}")
    if not any(re.fullmatch(r"model-\d+-of-\d+\.safetensors", path.name) for path in checkpoint_files):
        raise RuntimeError(f"No sharded FP8 model weights found in {candidate.checkpoint}")
    removed_by_layer = candidate.removed_by_layer
    # Equal retained-expert counts can be represented directly in vLLM's
    # native expert-parallel layout. The source shards remain symlinked; a
    # small load-time map compacts router rows and expert IDs, avoiding a
    # second ~173-GiB checkpoint and the runtime mask used for non-uniform
    # allocations.
    removal_counts = {len(ids) for ids in removed_by_layer.values()}
    compact_uniform = len(removal_counts) == 1
    source_config = read_json(candidate.checkpoint / "config.json")
    for source in checkpoint_files:
        target = model_dir / source.name
        if source.name == "config.json":
            if not compact_uniform:
                target.symlink_to(source.resolve())
                continue
            text_config = source_config.get("text_config")
            if not isinstance(text_config, dict):
                raise RuntimeError("Expected Qwen4-Exp config with a text_config object")
            retained_counts = {
                candidate.num_experts - len(ids)
                for ids in removed_by_layer.values()
            }
            if len(retained_counts) != 1:
                raise RuntimeError("Uniform compaction requires equal retained counts")
            compact_config = dict(source_config)
            compact_text_config = dict(text_config)
            compact_text_config["num_experts"] = retained_counts.pop()
            compact_config["text_config"] = compact_text_config
            target.write_bytes(stable_json_bytes(compact_config))
        else:
            target.symlink_to(source.resolve())

    if compact_uniform:
        retained_ids = {
            str(layer): [
                expert_id
                for expert_id in range(candidate.num_experts)
                if expert_id not in set(removed_by_layer[layer])
            ]
            for layer in range(candidate.num_layers)
        }
        compact_metadata = {
            "variant": candidate.variant,
            "source_allocation_manifest": str(candidate.manifest_path),
            "source_allocation_sha256": candidate.manifest_sha256,
            "original_experts": candidate.num_experts,
            "retained_experts_per_layer": len(next(iter(retained_ids.values()))),
            "retained_original_ids_by_layer": retained_ids,
            "routing": "compact router and experts; no runtime routing mask",
        }
        metadata_path = model_dir / "expert_compaction.json"
        metadata_path.write_bytes(stable_json_bytes(compact_metadata))
        return model_dir, metadata_path, "native-vllm-expert-parallel-compact"

    valid_masks: dict[str, list[int]] = {}
    for layer in range(candidate.num_layers):
        removed_ids = set(removed_by_layer[layer])
        valid_masks[str(layer)] = [
            0 if expert_id in removed_ids else 1
            for expert_id in range(candidate.num_experts)
        ]
    metadata = {
        "variant": candidate.variant,
        "source_allocation_manifest": str(candidate.manifest_path),
        "source_allocation_sha256": candidate.manifest_sha256,
        "original_experts": candidate.num_experts,
        "top_k": candidate.top_k,
        "total_removed": sum(map(len, removed_by_layer.values())),
        "total_retained": (
            candidate.num_layers * candidate.num_experts
            - sum(map(len, removed_by_layer.values()))
        ),
        "per_layer_removal_counts": {
            str(layer): len(removed_by_layer[layer])
            for layer in range(candidate.num_layers)
        },
        "valid_expert_mask": valid_masks,
        "routing": {
            "mask_before_softmax_and_topk": True,
            "padded_experts_receive_zero_probability": True,
            "redistribute_over_retained_experts_only": True,
        },
    }
    metadata_path = model_dir / "nonuniform_metadata.json"
    with metadata_path.open("xb") as stream:
        stream.write(stable_json_bytes(metadata))
    return model_dir, metadata_path, "masked-original-expert-layout"


def expected_output_paths(results_dir: Path) -> dict[str, Path]:
    generation = results_dir / f"{SCENARIO_STEM}.json"
    return {
        "generation": generation,
        "eval": results_dir / f"{SCENARIO_STEM}_eval.json",
        "eval_all": results_dir / f"{SCENARIO_STEM}_eval_all.json",
    }


def validated_ids(path: Path, expected_ids: set[str]) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        raise RuntimeError(f"Expected LCB output was not created: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
        raise RuntimeError(f"Expected a list of question records in {path}")
    ids = [str(row.get("question_id", "")) for row in data]
    if len(ids) != EXPECTED_TASK_COUNT or len(set(ids)) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"{path.name}: expected {EXPECTED_TASK_COUNT} unique question IDs; "
            f"found {len(ids)} rows and {len(set(ids))} unique IDs"
        )
    if set(ids) != expected_ids:
        missing = sorted(expected_ids - set(ids))[:8]
        unexpected = sorted(set(ids) - expected_ids)[:8]
        raise RuntimeError(
            f"{path.name}: question IDs differ from frozen task set; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if ids != sorted(expected_ids):
        raise RuntimeError(
            f"{path.name}: rows are not in the frozen sorted question-ID order"
        )
    return data, sha256_file(path)


def validate_eval_metrics(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise RuntimeError(f"Expected LCB eval metrics were not created: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list) or len(data) != 3:
        raise RuntimeError(f"Unexpected LCB eval metrics structure in {path}")
    _, indexed_results, metadata = data
    expected_indexes = {str(index) for index in range(EXPECTED_TASK_COUNT)}
    if not isinstance(indexed_results, dict) or set(indexed_results) != expected_indexes:
        found = len(indexed_results) if isinstance(indexed_results, dict) else "non-object"
        raise RuntimeError(
            f"{path.name}: expected per-task eval records for indices 0.."
            f"{EXPECTED_TASK_COUNT - 1}; found {found}"
        )
    if not isinstance(metadata, list) or len(metadata) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"{path.name}: expected {EXPECTED_TASK_COUNT} per-task metadata entries"
        )
    return data[0], sha256_file(path)


def make_command(
    candidate: Candidate,
    model_dir: Path,
    results_dir: Path,
    python_executable: Path,
) -> list[str]:
    return [
        str(python_executable),
        str(EVAL_SCRIPT),
        "--model-name",
        str(model_dir),
        "--vllm-port",
        str(candidate.port),
        "--server-log-file-name",
        str(candidate.run_dir / "vllm-server.log"),
        "--results-dir",
        str(results_dir),
        "--seed",
        str(SEED),
        "--greedy",
        "true",
        "--run-lm-eval",
        "false",
        "--run-evalplus",
        "false",
        "--run-livecodebench",
        "true",
        "--run-math",
        "false",
        "--run-wildbench",
        "false",
        "--livecodebench-seed",
        str(SEED),
        "--livecodebench-temperature",
        str(TEMPERATURE),
        "--livecodebench-top-p",
        str(TOP_P),
    ]


def run_candidate(
    candidate: Candidate,
    expected_ids: set[str],
    python_executable: Path,
    cuda_visible_devices: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "variant": candidate.variant,
        "manifest": str(candidate.manifest_path),
        "manifest_sha256": candidate.manifest_sha256,
        "checkpoint": str(candidate.checkpoint),
        "port": candidate.port,
        "output_dir": str(candidate.run_dir),
        "removed_experts": sum(map(len, candidate.removed_by_layer.values())),
        "pruning_fraction": sum(map(len, candidate.removed_by_layer.values()))
        / (candidate.num_layers * candidate.num_experts),
        "correct_count": None,
        "pass_count": None,
        "pass_at_1": None,
        "task_count": EXPECTED_TASK_COUNT,
        "output_sha256": {},
        "return_code": None,
        "passed_validation": False,
        "inference_mode": None,
    }
    try:
        model_dir, metadata_path, inference_mode = prepare_model(candidate)
        record["inference_mode"] = inference_mode
        results_dir = candidate.run_dir / "results"
        results_dir.mkdir()
        outputs = expected_output_paths(results_dir)
        command = make_command(candidate, model_dir, results_dir, python_executable)
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
                "REAP_VLLM_ENABLE_EXPERT_PARALLEL": "true",
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "PYTHONPATH": str(REPO_ROOT / "src")
                + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
            }
        )
        if inference_mode == "masked-original-expert-layout":
            env["REAP_QWEN38_MASK_METADATA"] = str(metadata_path)
        else:
            env.pop("REAP_QWEN38_MASK_METADATA", None)
        venv_bin = python_executable.parent
        env["PATH"] = str(venv_bin) + os.pathsep + env.get("PATH", "")
        with (candidate.run_dir / "runner.log").open("x") as log:
            process = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record["return_code"] = process.returncode
        record["command"] = command

        # eval.py intentionally catches LCB exceptions; outputs, not exit code,
        # are the authority for completeness.
        generation_rows, generation_hash = validated_ids(
            outputs["generation"], expected_ids
        )
        eval_all_rows, eval_all_hash = validated_ids(outputs["eval_all"], expected_ids)
        _, eval_hash = validate_eval_metrics(outputs["eval"])
        if len(generation_rows) != len(eval_all_rows):
            raise RuntimeError("Generation and eval_all record counts differ")
        if any(
            not isinstance(row.get("graded_list"), list)
            or len(row["graded_list"]) != 1
            or not isinstance(row["graded_list"][0], bool)
            for row in eval_all_rows
        ):
            raise RuntimeError("eval_all is missing exactly one boolean grade per task")
        record["correct_count"] = sum(
            row["graded_list"][0] for row in eval_all_rows
        )
        record["pass_count"] = record["correct_count"]
        record["pass_at_1"] = record["correct_count"] / EXPECTED_TASK_COUNT
        record["output_sha256"] = {
            "generation": generation_hash,
            "eval": eval_hash,
            "eval_all": eval_all_hash,
        }
        if process.returncode != 0:
            raise RuntimeError(
                f"eval.py returned nonzero status {process.returncode}; see runner.log"
            )
        record["passed_validation"] = True
    except Exception as exc:  # Record failures and continue with other variants.
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def paired_lcb_comparisons(
    candidates: list[Candidate],
    records: list[dict[str, Any]],
    expected_ids: set[str],
) -> list[dict[str, Any]]:
    """Compute exact-task paired deltas and problem-bootstrap intervals."""
    import numpy as np

    outcomes: dict[str, dict[str, int]] = {}
    for candidate, record in zip(candidates, records):
        if not record.get("passed_validation"):
            continue
        eval_all_path = candidate.run_dir / "results" / f"{SCENARIO_STEM}_eval_all.json"
        rows, _ = validated_ids(eval_all_path, expected_ids)
        outcomes[candidate.variant] = {
            str(row["question_id"]): int(row["graded_list"][0]) for row in rows
        }

    comparisons: list[dict[str, Any]] = []
    for left_index, candidate in enumerate(candidates):
        left = candidate.variant
        if left not in outcomes:
            continue
        for control_candidate in candidates[left_index + 1 :]:
            control = control_candidate.variant
            if control not in outcomes:
                continue
            task_ids = sorted(outcomes[left])
            differences = np.asarray(
                [outcomes[left][task_id] - outcomes[control][task_id] for task_id in task_ids],
                dtype=np.int8,
            )
            wins = int((differences == 1).sum())
            losses = int((differences == -1).sum())
            discordant = wins + losses
            if discordant:
                tail = sum(
                    math.comb(discordant, count)
                    for count in range(min(wins, losses) + 1)
                ) / (2**discordant)
                mcnemar_p = min(1.0, 2.0 * tail)
            else:
                mcnemar_p = 1.0
            rng = np.random.default_rng(42)
            bootstrap = rng.choice(
                differences, size=(10000, len(differences)), replace=True
            ).mean(axis=1)
            comparisons.append(
                {
                    "candidate": left,
                    "control": control,
                    "candidate_removed": sum(
                        map(len, candidate.removed_by_layer.values())
                    ),
                    "control_removed": sum(
                        map(len, control_candidate.removed_by_layer.values())
                    ),
                    "tasks": len(task_ids),
                    "candidate_solved": sum(outcomes[left].values()),
                    "control_solved": sum(outcomes[control].values()),
                    "paired_delta_solved": wins - losses,
                    "paired_delta_pass_rate": float(differences.mean()),
                    "candidate_only_passes": wins,
                    "control_only_passes": losses,
                    "bootstrap_95_low": float(np.quantile(bootstrap, 0.025)),
                    "bootstrap_95_high": float(np.quantile(bootstrap, 0.975)),
                    "exact_mcnemar_p": mcnemar_p,
                    "task_ids_sha256": EXPECTED_TASK_ID_SHA256,
                }
            )
    return comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen historical LCB for existing layerwise-reuse manifests. "
            "This launches vLLM inference when executed."
        )
    )
    parser.add_argument(
        "targets",
        nargs="+",
        help="One or more variant IDs under --allocation-root or manifest.json paths",
    )
    parser.add_argument(
        "--allocation-root", type=Path, default=DEFAULT_ALLOCATION_ROOT
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional original FP8 checkpoint override; otherwise read experiment_manifest.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New or existing parent directory; candidate directories and summary must not exist",
    )
    parser.add_argument("--port-start", type=int, default=18341)
    parser.add_argument(
        "--python",
        type=Path,
        help="Evaluation environment Python (defaults to .venv-eval/bin/python if present)",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default="0,1,2,3",
        help="CUDA devices to expose to the delegated vLLM evaluator",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # The data-ID audit and the delegated evaluator are both cache-only.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    output_root = args.output_root.expanduser().resolve()
    allocation_root = args.allocation_root.expanduser().resolve()
    checkpoint_override = args.checkpoint.expanduser() if args.checkpoint else None
    python_executable = (
        # Preserve a venv's `bin/python` symlink. Resolving it points at the
        # base interpreter and loses the venv's site-packages when this script
        # launches eval.py in a subprocess.
        Path(os.path.abspath(args.python.expanduser()))
        if args.python
        else REPO_ROOT / ".venv-eval/bin/python"
    )
    if not python_executable.is_file() or not os.access(python_executable, os.X_OK):
        if args.python:
            raise FileNotFoundError(f"Evaluation Python is not executable: {python_executable}")
        python_executable = Path(sys.executable).resolve()
    if not EVAL_SCRIPT.is_file():
        raise FileNotFoundError(f"Existing evaluator not found: {EVAL_SCRIPT}")
    verify_eval_protocol()

    if args.port_start < 1 or args.port_start + len(args.targets) - 1 > 65535:
        raise ValueError("Assigned ports would fall outside 1..65535")
    ports = list(range(args.port_start, args.port_start + len(args.targets)))
    if len(set(args.targets)) != len(args.targets):
        raise ValueError("Duplicate variant/manifest targets are not allowed")

    # Freeze/check IDs before creating any outputs or starting any evaluation.
    expected_ids_list, expected_ids_hash, cache_description = frozen_historical_ids()
    expected_ids = set(expected_ids_list)

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing summary: {summary_path}")

    candidates: list[Candidate] = []
    labels: set[str] = set()
    for index, requested in enumerate(args.targets):
        manifest_path = resolve_manifest(requested, allocation_root)
        raw_manifest = read_json(manifest_path)
        variant = str(raw_manifest.get("variant") or raw_manifest.get("allocation") or "")
        label = re.sub(r"[^A-Za-z0-9_.-]+", "_", variant).strip("._-")
        if not label or label in labels:
            raise ValueError(f"Variant output directory collision: {variant!r}")
        labels.add(label)
        run_dir = output_root / label
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
        candidates.append(
            validate_manifest(
                requested,
                manifest_path,
                checkpoint_override,
                run_dir,
                ports[index],
            )
        )

    verify_ports_available(ports)
    settings = {
        "dataset_release": "release_latest (local cached Arrow files)",
        "dataset_cache": cache_description,
        "date_window_inclusive": [START_DATE, END_DATE],
        "task_count": EXPECTED_TASK_COUNT,
        "task_ids_sha256_sorted_newline_joined": expected_ids_hash,
        "seed": SEED,
        "greedy": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "thinking_enabled": False,
        "max_tokens": MAX_TOKENS,
        "execution_timeout_seconds": TIMEOUT_SECONDS,
        "offline_cache_only": True,
    }
    records = [
        run_candidate(candidate, expected_ids, python_executable, args.cuda_visible_devices)
        for candidate in candidates
    ]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_versions": {
            "python": sys.version,
            "torch": installed_version("torch"),
            "transformers": installed_version("transformers"),
            "vllm": installed_version("vllm"),
            "datasets": installed_version("datasets"),
        },
        "source_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
            "eval_script": sha256_file(EVAL_SCRIPT),
            "livecodebench_runner": sha256_file(
                REPO_ROOT / "third-party/LiveCodeBench/lcb_runner/runner/main.py"
            ),
        },
        "validated_candidate_count": sum(
            bool(record["passed_validation"]) for record in records
        ),
        "candidate_count": len(records),
        "settings": settings,
        "results": records,
        "paired_comparisons": paired_lcb_comparisons(
            candidates, records, expected_ids
        ),
    }
    with summary_path.open("xb") as stream:
        stream.write(stable_json_bytes(summary))
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "validated_candidate_count": summary["validated_candidate_count"],
                "candidate_count": len(records),
            },
            indent=2,
        )
    )
    return 0 if summary["validated_candidate_count"] == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
