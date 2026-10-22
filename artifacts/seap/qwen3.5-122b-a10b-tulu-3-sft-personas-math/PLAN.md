# Qwen3.5-122B-A10B × Tulu Personas Math SEAP experiment

## Objective

Calibrate REAP and router-stability scores on the full mathematical-reasoning
dataset, derive SEAP lambda/delta states offline, then compare REAP,
stability-only, and SEAP pruning on mathematical-reasoning benchmarks.

This directory is the experiment root. Keep every output from this experiment
under it so that calibration provenance, score states, evaluations, plots, and
reports remain connected.

## Pinned inputs

| Input | Value |
| --- | --- |
| Model | `Qwen/Qwen3.5-122B-A10B` |
| Model revision | `dc4d348443bc740c68e2d77492492c11606384d5` |
| Model architecture | `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe` |
| MoE layout | 48 text layers; 256 routed experts; 8 routed experts per token; 1 shared expert |
| Calibration dataset | `allenai/tulu-3-sft-personas-math` |
| Dataset revision | `46add032f5203fd96efbd54d074198981ed57d25` |
| Dataset split | `train`, 149,960 rows, with a ready-to-use `messages` column |
| Hardware available | four NVIDIA A100 80 GB GPUs |
| Initial seed | `42` |

The dataset is already registered in `src/reap/data.py` and is text-only. The
model is multimodal, but calibration must pass only the text `messages` from
the dataset and must not construct image or video inputs.

## Required compatibility work

The existing SEAP runner supports GLM-4.5-Air only. Do not start a full run
with it for this model. Complete and verify these changes first:

1. Use a Transformers release that supports `qwen3_5_moe`. The repository
   pins `transformers==4.55.0`; the Qwen3.5 model implementation is available
   in Transformers 5.20. Use an isolated calibration environment until it is
   proven compatible with the repository-wide evaluation stack.
2. Add a Qwen3.5 model adapter instead of reusing the Qwen3 adapter blindly:
   register the actual causal-text/MoE module class, its text-layer path,
   router attribute, expert container, expert-count attribute, and top-k
   attribute discovered from a loaded checkpoint.
3. Use `AutoModelForConditionalGeneration` and the model’s processor/tokenizer
   for text-only inputs. Confirm that the model accepts `input_ids` and
   `attention_mask` without vision tensors.
4. Make the SEAP runner model-agnostic. It must discover MoE blocks from the
   adapter, collect full router ranks according to the Qwen3.5 router’s real
   selection semantics, and apply every routed expert to obtain REAP.
5. Extend pruning and evaluation support for the Qwen3.5 architecture. The
   existing `MODEL_ATTRS`, temporary-checkpoint path, vLLM version, and model
   name mapping do not yet include it.

The router implementation must be verified against the model forward pass:
the selected top-8 IDs and applied routing weights must agree exactly before
any calibration is accepted.

## Experiment configuration

Copy [`config.env.example`](config.env.example) to `config.env` and set the
local checkpoint path after downloading the pinned revision. `config.env` is
local run configuration and must not contain credentials.

Run calibration, state construction, and calibration reporting with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash experiments/run-seap-qwen35.sh \
  artifacts/seap/qwen3.5-122b-a10b-tulu-3-sft-personas-math/config.env
```

Initial calibration policy:

- Full dataset: all rows that can be tokenized under the declared length rule.
- Batch size: 1 for the initial full run; increase only after a pilot measures
  peak memory.
- Maximum length: 2,048 tokens.
- Truncation: disabled, matching existing REAP calibration behavior. Record
  the exact retained, skipped-overlength, and valid-token counts. A separate
  truncation-enabled run would be a distinct calibration and must use a
  different directory.
- Noise variance multiplier: 2.0.
- Stability deltas: 2 and 4 from one model pass.
- Default saved state: delta 2, lambda 0.5.
- Router weights: renormalized consistently with the current REAP defaults.

## Phased execution

### 0. Environment and checkpoint preflight

Create an isolated environment with compatible Transformers, Accelerate,
PyTorch, vLLM, and evaluation dependencies. Download the 39-shard model into
the configured local directory; do not write it into this experiment root.

Run these gates before a GPU calibration:

1. Load config, processor/tokenizer, and model on CPU/meta initialization.
2. Print the exact MoE-module paths and router/expert attribute names.
3. Confirm the 48 text layers, 256 routed experts, top-k 8, and one shared
   expert from the live modules.
4. Run one text-only forward pass and compare adapter routing results to model
   routing results for one MoE block.
5. Confirm the sharding map fits all four GPUs without CPU/disk offload.

Write command output and package versions to `logs/preflight.log` and
`reports/environment.json`.

### 1. Calibration smoke run

Use at most 2,048 valid tokens. Verify all 48 MoE layers have finite
`reap_score`, `normalized_reap`, both stability vectors, and `seap_score`, each
of shape `[256]`.

Check these invariants:

- all stability scores lie in `[0, 1]`;
- stability at delta 4 is at least stability at delta 2 elementwise;
- lambda 1 reproduces normalized REAP and lambda 0 reproduces stability;
- state metadata reports source revisions, actual rows, skipped rows, tokens,
  sequence-length policy, router semantics, seed, package versions, and GPU
  layout;
- a 25% prune smoke test removes the lowest 64 of 256 experts in every MoE
  layer and can generate text.

### 2. Full calibration

Run all usable rows in `train` once. Save:

```text
calibration/seap_state.pt
calibration/seap_accumulators.npz
calibration/metadata.json
calibration/logs/
```

The metadata must state whether long rows were skipped. Do not label the result
“full dataset” if rows were skipped; call it “all rows within 2,048 tokens.”

### 3. Offline score sweep

Generate delta 2 and 4 score states for lambdas `0.00, 0.25, 0.50, 0.75,
1.00`. `lambda=1.00` is the REAP-only endpoint, and `lambda=0.00` is
stability-only. Store them as:

```text
sweeps/delta-2/lambda-0.00/state.pt
...
sweeps/delta-4/lambda-1.00/state.pt
```

### 4. Prune and evaluate

First run a baseline unpruned evaluation, then REAP and SEAP states at 25% and
50% expert pruning. Each method/ratio must use the same checkpoint revision,
seed, benchmark settings, and generation parameters.

The initial evaluation suite is mathematical reasoning: `gsm8k`, `math_500`,
and `aime25` through the project’s math runner. Add further benchmarks only
after Qwen3.5 serving and prompt-format compatibility are verified. Evaluate
one method at a time and remove temporary checkpoints after its results are
written; retaining every 122B checkpoint would exceed practical disk space.

### 5. Statistics and figures

Write machine-readable numbers before figures. Required outputs are:

| Output | Location |
| --- | --- |
| Per-layer REAP/stability/SEAP score summary, selection counts, quantiles, Pearson and Spearman correlations | `reports/calibration_statistics.json` and `.csv` |
| Top-10% and pruning-set overlap with REAP for both ratios | `reports/pruning_overlap.json` and `.csv` |
| Pruned expert IDs by layer and method | `reports/pruned_experts.json` |
| Benchmark scores, deltas from unpruned and REAP baselines, runtime, peak memory | `reports/evaluation_summary.csv` and `.json` |
| Score distribution and per-layer score plots | `plots/scores/` |
| Lambda/delta pruning overlap plots | `plots/overlap/` |
| Calibration ranking/correlation heatmaps | `plots/calibration/` |
| Evaluation comparison bars and lambda/ratio curves | `plots/evaluation/` |
| Router selection/balance figures, if router event collection is enabled | `plots/router/` |

## Directory layout

```text
artifacts/seap/qwen3.5-122b-a10b-tulu-3-sft-personas-math/
├── PLAN.md
├── config.env.example
├── logs/
├── calibration/
├── sweeps/
├── evaluations/
├── plots/
└── reports/
```

## Run gates and non-negotiable comparisons

- Never compare methods calibrated on different row populations, sequence
  lengths, truncation policies, seeds, or checkpoint revisions.
- Preserve the shared expert. Only the 256 routed experts participate in
  REAP/SEAP scoring and pruning.
- A Qwen3.5-specific router adapter and a smoke calibration are mandatory
  before the full job.
- Use the implemented formula: `lambda * normalized_reap + (1 - lambda) *
  stability`.
- Generate score/overlap figures immediately after calibration; generate
  benchmark-comparison figures only after all corresponding evaluations finish.

## Status

Planned on 2026-10-16. No model weights, calibration state, pruning, or
evaluation has started for this experiment.
