#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${1:-0}
export REAP_GLM_MODEL_PATH=${REAP_GLM_MODEL_PATH:-/local/mnt2/workspace/ochougul/artifacts/models/GLM-4.5-Air}

output_dir=${ROUTER_STABILITY_OUTPUT_DIR:-/local/mnt2/workspace/ochougul/artifacts/router_stability/glm4.5-air-evol-codealpaca-500k}

uv run python -m reap.layerwise_prune \
  --model_name zai-org/GLM-4.5-Air \
  --dataset_name theblackcat102/evol-codealpaca-v1 \
  --split train \
  --batch_size 1 \
  --batches_per_category 256 \
  --model_max_length 2048 \
  --truncate true \
  --profile false \
  --do_eval false \
  --run_observer_only true \
  --record_pruning_metrics_only true \
  --batch_group_size 8 \
  --collect_router_stability true \
  --router_stability_output_dir "${output_dir}" \
  --router_stability_max_tokens 500000 \
  --router_stability_variance_multipliers 0.25 0.5 1.0 2.0 \
  --router_stability_seed 42 \
  --router_stability_boundary_width 4 \
  --overwrite_observations true
