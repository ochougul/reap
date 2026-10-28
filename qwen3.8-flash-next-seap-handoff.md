# Qwen3.8-Flash-Next SEAP handoff

This document records the Qwen3.8-Flash-Next-FP8 calibration, pruning, and
evaluation work, including the reproducibility settings and results.

## Model and pruning semantics

- Model: `Qwen/Qwen3.8-Flash-Next-FP8`
- Transformers architecture: `qwen4_exp`
- Layers: 48
- Experts per MoE layer: 512
- Experts selected per token: 10
- 25% pruning: remove `int(512 * 0.25) = 128`; retain 384 experts/layer.
- 50% pruning: remove `int(512 * 0.50) = 256`; retain 256 experts/layer.

The retained experts are the highest-scoring experts for each layer. For fused
Qwen4/FP8 modules, pruning slices `gate_up_proj`, `down_proj`, and their
expert-indexed FP8 scale tensors, then updates the router and nested text
configuration. The implementation is ratio-based and therefore works for
other pruning ratios as well.

SEAP scores are computed as:

```text
seap_score = lambda * normalized_reap + (1 - lambda) * stability
```

Therefore λ=0 is stability-only and λ=1 is REAP-only. In the reported runs,
REAP-only means λ=1 and stability-only means λ=0.

## Source and calibration artifacts

Base checkpoint:

```text
/home/ubuntu/hf_cache/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13
```

Primary calibration artifacts:

```text
artifacts/Qwen3.8-Flash-Next-FP8/full-packed-2048/calibration_state.pt
artifacts/Qwen3.8-Flash-Next-FP8/full-packed-2048/router_stability/router_stability_accumulators.npz
```

Calibration used `theblackcat102/evol-codealpaca-v1` train data, seed 42,
sequence length 2048, packed samples, logical batch size 32, forward
microbatch 4, variance multiplier 1.0, and four-way TP plus four-way EP.
The final run contained 26,035 packed sequences and 53,236,097 valid tokens.

Materialized score states:

```text
artifacts/Qwen3.8-Flash-Next-FP8/eval_states/reap/state.pt
artifacts/Qwen3.8-Flash-Next-FP8/eval_states/selection-stability/state.pt
artifacts/Qwen3.8-Flash-Next-FP8/eval_states/lambda-sweep/delta-2/lambda-0.25/state.pt
artifacts/Qwen3.8-Flash-Next-FP8/eval_states/lambda-sweep/delta-2/lambda-0.50/state.pt
artifacts/Qwen3.8-Flash-Next-FP8/eval_states/lambda-sweep/delta-2/lambda-0.75/state.pt
```

The lambda sweep uses the same REAP and stability state and only changes the
offline score combination; it does not require recalibration.

## Reproducing calibration and checkpoints

From the repository root, use the evaluation environment created for this
model (`.venv-eval` in the original runs):

```bash
QWEN4_EXPERT_PARALLEL=1 \
QWEN4_EP_SIZE=4 \
QWEN4_TP_SIZE=4 \
QWEN4_EXPERTS_IMPLEMENTATION=deepgemm \
QWEN4_PACK_SAMPLES=1 \
QWEN4_MODEL_MAX_LENGTH=2048 \
QWEN4_BATCH_SIZE=32 \
QWEN4_FORWARD_BATCH_SIZE=4 \
QWEN4_VARIANCE_MULTIPLIERS=1.0 \
QWEN4_SAVE_ROUTING_EVENTS=0 \
QWEN4_DISABLE_CUDNN_SDP=1 \
QWEN4_OUTPUT_DIR=artifacts/Qwen3.8-Flash-Next-FP8/evol-codealpaca \
PYTHONPATH=src .venv-eval/bin/python experiments/qwen4_calibrate.py
```

Build REAP/stability states from the final calibration artifacts:

```bash
PYTHONPATH=src .venv-eval/bin/python scripts/build_qwen38_eval_states.py \
  artifacts/Qwen3.8-Flash-Next-FP8/full-packed-2048/calibration_state.pt \
  artifacts/Qwen3.8-Flash-Next-FP8/full-packed-2048/router_stability/router_stability_accumulators.npz \
  artifacts/Qwen3.8-Flash-Next-FP8/eval_states

PYTHONPATH=src .venv-eval/bin/python scripts/build_seap_states.py \
  artifacts/Qwen3.8-Flash-Next-FP8/eval_states/selection-stability/state.pt \
  --output-dir artifacts/Qwen3.8-Flash-Next-FP8/eval_states/lambda-sweep \
  --deltas 2 --lambdas 0.25 0.5 0.75
```

Materialize a checkpoint for any ratio and state (the same command handles
25% and 50%):

```bash
PYTHONPATH=src .venv-eval/bin/python scripts/materialize_qwen38_pruned.py \
  /home/ubuntu/hf_cache/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13 \
  artifacts/Qwen3.8-Flash-Next-FP8/eval_states/reap/state.pt \
  artifacts/Qwen3.8-Flash-Next-FP8/eval/reap-50-fixed/model \
  --method reap --compression-ratio 0.50
```

For stability-only use `selection-stability/state.pt` and `--method seap`.
For λ=0.25/0.50/0.75 use the corresponding lambda state and `--method seap`.
The existing 25% and 50% checkpoint directories are:

```text
artifacts/Qwen3.8-Flash-Next-FP8/eval/reap-50-fixed/model
artifacts/Qwen3.8-Flash-Next-FP8/eval/stability-50-fixed/model
artifacts/Qwen3.8-Flash-Next-FP8/eval/lambda-0.25-50/model
artifacts/Qwen3.8-Flash-Next-FP8/eval/lambda-0.5-50/model
artifacts/Qwen3.8-Flash-Next-FP8/eval/lambda-0.75-50/model
```

The 25% checkpoint directories used analogous names without the `-50`
suffix. Each checkpoint was verified to contain the expected expert count.

## Evaluation settings and commands

All evaluations used a four-GPU vLLM server on port 8010 with:

```text
--tensor-parallel-size 4
--enable-expert-parallel
--gpu-memory-utilization 0.90
--max-model-len 32768
--max-num-seqs 64
--max-num-batched-tokens 32768
seed=42
```

EvalPlus used greedy decoding, temperature 0.0, n=1, thinking disabled,
`parallel_tasks=1`, and correctness parallelism 32. The reproducible wrapper
is:

```bash
EVALPLUS_MODEL=/path/to/checkpoint \
EVALPLUS_OUTPUT=artifacts/evalplus_qwen38_variant \
PYTHONPATH=src .venv-eval/bin/python scripts/run_evalplus_qwen38.py
```

This runs MBPP+ and HumanEval+. LiveCodeBench used snapshot dates
2025-01-01 through 2025-04-30, n=1, temperature 0.0, top-p 1.0, greedy
decoding, max generation tokens 16,384, request seed 42, and timeout 120s.
The run is launched through the normal REAP evaluation entrypoint with
`--run-livecodebench`, using the same server settings above.

## Results

### 25% pruning

| Variant | MBPP+ | HumanEval+ | LiveCodeBench |
|---|---:|---:|---:|
| Base | 0.8042328 | 0.9390244 | 124/182 = 0.681319 |
| REAP-only (λ=1) | 0.8015873 | 0.9512195 | 123/182 = 0.675824 |
| Stability-only (λ=0) | 0.8015873 | 0.9451220 | 123/182 = 0.675824 |
| λ=0.25 | 0.8015873 | 0.9451220 | 123/182 = 0.675824 |
| λ=0.50 | 0.8095238 | 0.9451220 | 122/182 = 0.670330 |
| λ=0.75 | 0.8121693 | 0.9451220 | 122/182 = 0.670330 |

### 50% pruning

| Variant | MBPP+ | HumanEval+ | LiveCodeBench |
|---|---:|---:|---:|
| REAP-only (λ=1) | 0.7989418 | 0.9390244 | 111/182 = 0.609890 |
| Stability-only (λ=0) | 0.7936508 | 0.9329268 | 116/182 = 0.637363 |
| λ=0.25 | 0.7830688 | 0.9390244 | 116/182 = 0.637363 |
| λ=0.50 | 0.7883598 | 0.9268293 | 112/182 = 0.615385 |
| λ=0.75 | 0.7883598 | 0.9268293 | 121/182 = 0.664835 |

Base results were reused for 50% and were not rerun.

Result directories:

```text
artifacts/evalplus_qwen38_base
artifacts/evalplus_qwen38_reap25
artifacts/evalplus_qwen38_stability25
artifacts/evalplus_qwen38_lambda025
artifacts/evalplus_qwen38_lambda05
artifacts/evalplus_qwen38_lambda075
artifacts/evalplus_qwen38_reap50
artifacts/evalplus_qwen38_stability50
artifacts/evalplus_qwen38_lambda025_50
artifacts/evalplus_qwen38_lambda05_50
artifacts/evalplus_qwen38_lambda075_50
artifacts/livecode_qwen38_reap50_retry
artifacts/livecode_qwen38_stability50
artifacts/livecode_qwen38_lambda025_50
artifacts/livecode_qwen38_lambda05_50
artifacts/livecode_qwen38_lambda075_50
```

## Checkpoint audit

The similar MBPP+/HumanEval+ numbers are not caused by reusing one checkpoint.
Direct expert-weight hashes differ. The mean overlap of dropped expert sets,
computed layer-by-layer, was:

| Comparison | Mean dropped-set overlap |
|---|---:|
| REAP vs stability | 86.55% |
| λ=0.25 vs stability | 97.22% |
| λ=0.50 vs stability | 94.41% |
| λ=0.75 vs stability | 91.02% |

The overlap is high for some score mixtures but the retained expert tensors
are not identical. This supports the conclusion that the close benchmark
scores are genuine model behavior.

## Code changes included

- Registered Qwen4-Exp model attributes and observer configuration.
- Added fused Qwen expert and FP8-scale slicing.
- Fixed generic ratio-based pruning for 25%, 50%, and other ratios.
- Updated nested Qwen text configuration and router dimensions on save.
- Vectorized routed metric accumulation for 512 experts.
- Added generic router-stability route handling and configurable top-k,
  variance count, and event persistence.
- Added offline SEAP state construction for arbitrary λ values.
- Added Qwen3.8 EvalPlus/LiveCodeBench evaluation support and deterministic
  LiveCodeBench settings.
- Added focused fused-expert pruning tests and Qwen3.8 helper scripts.
