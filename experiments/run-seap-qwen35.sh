#!/usr/bin/env bash
set -euo pipefail

config_file=${1:-artifacts/seap/qwen3.5-122b-a10b-tulu-3-sft-personas-math/config.env}
if [[ ! -f "$config_file" ]]; then
  echo "Missing experiment configuration: $config_file" >&2
  exit 1
fi
set -a
source "$config_file"
set +a
python_bin=${SEAP_PYTHON:-python}
checkpoint_path="$SEAP_EXPERIMENT_ROOT/calibration/checkpoints/latest.pt"
if [[ "${SEAP_AUTO_RESUME:-true}" == "true" \
  && -z "${SEAP_RESUME_CHECKPOINT:-}" \
  && -f "$checkpoint_path" ]]; then
  export SEAP_RESUME_CHECKPOINT="$checkpoint_path"
  echo "Resuming calibration from $SEAP_RESUME_CHECKPOINT"
fi

"$python_bin" -u experiments/seap-qwen35-sharded.py

"$python_bin" scripts/build_seap_states.py \
  "$SEAP_EXPERIMENT_ROOT/calibration/seap_state.pt" \
  --output-dir "$SEAP_EXPERIMENT_ROOT/sweeps" \
  --lambdas 0.00 0.25 0.50 0.75 1.00 \
  --deltas 2 4

"$python_bin" scripts/report_seap_experiment.py "$SEAP_EXPERIMENT_ROOT" \
  --prune-fractions 0.25 0.50
