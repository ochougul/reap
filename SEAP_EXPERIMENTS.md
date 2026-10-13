# Reproducible SEAP experiments

This repository contains three reusable stages:

1. calibration, which produces a per-layer state containing REAP and router-
   stability statistics;
2. offline score construction, which creates states for arbitrary SEAP `lambda`
   and stability `delta` values without another model pass; and
3. pruning/evaluation, which accepts any saved state and any supported
   evaluation task set.

The calibration state and model checkpoint are intentionally not committed.
Store those artifacts outside Git and pass their paths on the command line.

## Calibrate on another dataset

The resident four-GPU GLM runner reads its configuration from environment
variables:

```bash
SEAP_MODEL_PATH=/path/to/model \
SEAP_OUTPUT_DIR=artifacts/seap/my-dataset \
SEAP_DATASET_NAME=my-org/my-dataset \
SEAP_SPLIT=train \
SEAP_FULL_DATASET=true \
SEAP_SEED=42 \
uv run python -u experiments/seap-sharded.py
```

For a bounded run, use `SEAP_BATCHES_PER_CATEGORY`, `SEAP_BATCH_SIZE`, and/or
`SEAP_MAX_TOKENS`. `SEAP_TRUNCATE` controls whether overlong examples are
truncated. The runner saves `seap_state.pt`, which includes `reap_score`,
`normalized_reap`, stability scores for deltas 2 and 4, and a default SEAP
score.

The calibration dataset is independent of the evaluation dataset. Use the
same calibration settings for methods being compared.

## Sweep lambda and delta offline

Given a saved state, construct any number of score states without recalibration:

```bash
uv run python scripts/build_seap_states.py \
  artifacts/seap/my-dataset/seap_state.pt \
  --output-dir artifacts/seap/my-dataset/sweeps \
  --lambdas 0.0 0.25 0.5 0.75 1.0 \
  --deltas 2 4
```

The convention used by the implementation is:

```text
seap_score = lambda * normalized_reap + (1 - lambda) * stability
```

Thus `lambda=1` is REAP-only and `lambda=0` is stability-only. Each output
directory contains `state.pt` and `metadata.json` recording the source state,
lambda, delta, and formula.

## Prune and evaluate on another dataset/task

The pruning helper can use either score:

```bash
# SEAP state
uv run python experiments/prune_seap_state.py \
  --model-path /path/to/model \
  --state-path artifacts/seap/my-dataset/sweeps/delta-2/lambda-0.50/state.pt \
  --output-dir artifacts/seap/eval/my-run \
  --compression-ratio 0.50 \
  --prune-method seap \
  --run-eval \
  --no-run-evalplus --no-run-livecodebench --no-run-math --no-run-wildbench \
  --lm-eval-tasks hellaswag arc_easy

# REAP state or a combined state
uv run python experiments/prune_seap_state.py \
  --model-path /path/to/model \
  --state-path artifacts/seap/my-dataset/seap_state.pt \
  --output-dir artifacts/seap/eval/reap-50 \
  --compression-ratio 0.50 \
  --prune-method reap \
  --run-eval \
  --no-run-evalplus --no-run-livecodebench --no-run-math --no-run-wildbench \
  --lm-eval-tasks hellaswag
```

For EvalPlus, enable it and select `--evalplus-tasks humaneval mbpp`. For
LiveCodeBench, enable `--run-livecodebench`; its date range and generation
settings are defined in `src/reap/eval.py`. `--eval-limit` is available for a
smoke test and should be omitted for a full evaluation.

The helper creates a temporary pruned checkpoint when `--run-eval` is used,
starts evaluation in a fresh process, saves results below `--output-dir/eval`,
and removes the checkpoint afterward. Use `--save-checkpoint` without
`--run-eval` when a persistent checkpoint is desired.

## Files that make this workflow portable

- `experiments/seap-sharded.py`: configurable multi-GPU calibration runner.
- `scripts/build_seap_states.py`: offline lambda/delta state construction.
- `experiments/prune_seap_state.py`: score selection, pruning, and evaluation.
- `src/reap/seap.py`: stability accumulation and SEAP score definition.
- `src/reap/router_stability.py`: deterministic router perturbations/ranking.
- `src/reap/pruning_metrics.py`: REAP accumulation.
- `src/reap/eval.py`: lm-eval, EvalPlus, and LiveCodeBench orchestration.
- `seap_score.md`: algorithm and validation specification.
- `tests/test_seap.py` and `tests/test_router_stability.py`: score/routing tests.

Generated checkpoints, calibration states, evaluation outputs, and plots are
experiment artifacts and should remain outside Git unless a specific result
must be published.
