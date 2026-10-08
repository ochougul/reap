"""Prune and optionally evaluate a model from a saved SEAP state."""

from __future__ import annotations

import argparse
import shutil
import pathlib
import subprocess
import tempfile
import gc
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate.hooks import remove_hook_from_module

from reap.args import ClusterArgs, DatasetArgs, EvalArgs, ModelArgs, ObserverArgs, PruneArgs, ReapArgs
from reap.eval import run_evaluate
from reap.main import dump_args_to_yaml, smoke_test
from reap.prune import prune


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--compression-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-smoke-test", action="store_true")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--use-server", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-lm-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-evalplus", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-livecodebench", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-wildbench", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-math", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lm-eval-tasks", nargs="+", default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--evalplus-tasks", nargs="+", default=None)
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--server-log-file-name", default="seap-eval-server.log")
    parser.add_argument("--evaluate-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()

    # Never start vLLM from the process that loaded the calibration model.
    # The parent only orchestrates two short-lived children, guaranteeing that
    # CUDA allocations and Accelerate hooks are gone before vLLM starts.
    if args.run_eval and not args.evaluate_only:
        checkpoint_dir = args.output_dir / f".pruned_model_eval_{os.getpid()}"
        argv = list(sys.argv[1:])
        argv = [arg for arg in argv if arg not in {"--run-eval", "--evaluate-only"}]
        argv.extend(["--save-checkpoint", "--checkpoint-dir", str(checkpoint_dir)])
        subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), *argv], check=True)
        try:
            eval_argv = [
                arg for arg in sys.argv[1:]
                if arg not in {"--run-eval", "--save-checkpoint", "--run-smoke-test", "--evaluate-only"}
            ]
            eval_argv.extend(["--evaluate-only", "--checkpoint-dir", str(checkpoint_dir)])
            subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), *eval_argv], check=True)
        finally:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        return

    if args.evaluate_only:
        if args.checkpoint_dir is None:
            raise ValueError("--checkpoint-dir is required with --evaluate-only")
        venv_bin = str(pathlib.Path(sys.executable).parent)
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
        eval_args = EvalArgs(
            use_server=args.use_server,
            run_lm_eval=args.run_lm_eval,
            run_evalplus=args.run_evalplus,
            run_livecodebench=args.run_livecodebench,
            run_wildbench=args.run_wildbench,
            run_math=args.run_math,
            lm_eval_tasks=args.lm_eval_tasks or EvalArgs().lm_eval_tasks,
            evalplus_tasks=args.evalplus_tasks or EvalArgs().evalplus_tasks,
            vllm_port=args.vllm_port,
            server_log_file_name=args.server_log_file_name,
            eval_limit=args.eval_limit,
        )
        run_evaluate(ModelArgs(model_name=str(args.checkpoint_dir)), args.output_dir / "eval", eval_args, args.seed)
        return

    state = torch.load(args.state_path, weights_only=False)
    first_layer = next(iter(state.values()))
    total_experts = int(first_layer["seap_score"].shape[0])
    n_experts_to_prune = int(total_experts * args.compression_ratio)
    if not 0 <= n_experts_to_prune < total_experts:
        raise ValueError("compression ratio must prune at least 0 and fewer than all experts")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    prune_args = PruneArgs(prune_method="seap", n_experts_to_prune=n_experts_to_prune)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = None
    if args.save_checkpoint:
        checkpoint_dir = args.checkpoint_dir or args.output_dir / "pruned_model"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="seap-pruned-")
        checkpoint_dir = pathlib.Path(temporary.name)

    try:
        prune(state, model, prune_args, n_experts_to_prune, checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        if args.run_smoke_test:
            smoke_test(model, tokenizer)

        dump_args_to_yaml(
        checkpoint_dir,
        reap_args=ReapArgs(seed=args.seed),
        ds_args=DatasetArgs(dataset_name="saved_seap_state"),
        obs_args=ObserverArgs(output_file_name=pathlib.Path(args.state_path).name),
        model_args=ModelArgs(model_name=args.model_path),
        eval_args=EvalArgs(
            use_server=args.use_server,
            run_lm_eval=args.run_lm_eval,
            run_evalplus=args.run_evalplus,
            run_livecodebench=args.run_livecodebench,
            run_wildbench=args.run_wildbench,
            run_math=args.run_math,
            lm_eval_tasks=args.lm_eval_tasks or EvalArgs().lm_eval_tasks,
            evalplus_tasks=args.evalplus_tasks or EvalArgs().evalplus_tasks,
            vllm_port=args.vllm_port,
            server_log_file_name=args.server_log_file_name,
            eval_limit=args.eval_limit,
        ),
        prune_args=prune_args,
        cluster_args=ClusterArgs(compression_ratio=args.compression_ratio),
        )

    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
