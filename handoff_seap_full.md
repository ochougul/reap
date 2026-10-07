# Handoff: full-dataset SEAP calibration and evaluation

This document is for the next agent continuing the SEAP work on a new machine.
The objective is to reproduce the repository, environment, model, dataset, and
four-GPU scoring setup, then run full-dataset calibration and compare pruning
against REAP over a lambda sweep.

## Repository and commit

```text
Fork:   https://github.com/ochougul/reap.git
Branch: router-stability-4gpu
Commit: ddd316c Add four-GPU SEAP scoring and analysis
```

Clone and initialize dependencies:

```bash
git clone https://github.com/ochougul/reap.git
cd reap
git checkout router-stability-4gpu
git submodule update --init --recursive
uv sync
source .venv/bin/activate
```

The project uses Python 3.12, PyTorch 2.7.1, Transformers 4.55.0,
Accelerate, Datasets, Matplotlib, PyArrow, vLLM, and the evaluation packages.
Verify the installation:

```bash
.venv/bin/python - <<'PY'
import torch, transformers, accelerate
print(torch.__version__, transformers.__version__, accelerate.__version__)
print("CUDA:", torch.cuda.is_available(), torch.cuda.device_count())
PY
```

Use `hf auth login` if the model or dataset requires Hugging Face access.

## Hardware and model setup

The successful run used four NVIDIA A100 80 GB GPUs. The model is sharded
across all four GPUs with Accelerate. The original model path was:

```text
/local/mnt2/workspace/ochougul/artifacts/models/GLM-4.5-Air
```

On a new machine, download or copy the model. For example:

```bash
mkdir -p artifacts/models/GLM-4.5-Air
hf download zai-org/GLM-4.5-Air \
  --local-dir artifacts/models/GLM-4.5-Air
```

`experiments/seap-sharded.py` currently has `MODEL_PATH` as a source constant.
Change it to the new local path before running, or make it environment-driven.
Do not assume `/local/mnt2/...` exists on the new machine.

Verify the local checkpoint:

```bash
.venv/bin/python - <<'PY'
from transformers import AutoConfig, AutoTokenizer
p = "artifacts/models/GLM-4.5-Air"
AutoConfig.from_pretrained(p, trust_remote_code=True, local_files_only=True)
AutoTokenizer.from_pretrained(p, trust_remote_code=True, local_files_only=True)
print("checkpoint and tokenizer are readable")
PY
```

## Dataset setup

Dataset:

```text
theblackcat102/evol-codealpaca-v1
split: train
```

The train split contains 111,272 examples with `instruction` and `output`
columns. Cache and verify it:

```bash
.venv/bin/python - <<'PY'
from datasets import load_dataset
ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")
print(len(ds), ds.column_names)
PY
```

The historical scoring run used `batch_size=1`, `model_max_length=2048`,
`truncate=True`, `batches_per_category=256`, and a 500k-token cap. That was a
subset, not the full dataset.

## What was implemented

### `src/reap/seap.py`

Computes full-expert GLM router ranks for original and variance-2 perturbed
inputs. It uses one-based ranks, the `num_experts + 1` sentinel when an expert
becomes ineligible, float64 streaming accumulators, and per-layer REAP
normalization.

The current combination is:

```text
normalized_reap = reap / max(reap in the layer)
SEAP = lambda * stability + (1 - lambda) * normalized_reap
```

The completed run used `lambda=0.5`, `delta=2`, variance multiplier `2.0`, and
seed `42`. Both delta-2 and delta-4 stability accumulators are retained.

### Observer/pruning integration

`src/reap/args.py`, `src/reap/observer.py`, and
`src/reap/layerwise_observer.py` add SEAP collection options. `src/reap/prune.py`
accepts `--prune-method seap` and removes the lowest SEAP scores.

### `experiments/seap-sharded.py`

Loads GLM-4.5-Air resident across all four GPUs. For each MoE layer it collects
full router ranks and re-evaluates all 128 experts to calculate REAP. It saves
per-layer tensors for:

```text
reap_score      [128]
stability_score [128]
seap_score      [128]
```

Environment overrides currently available are `SEAP_MAX_TOKENS` and
`SEAP_OUTPUT_DIR`. The model path and dataset batch count should be
parameterized before the full-dataset run.

## Experiments and data collected

### Existing router-stability run

```text
artifacts/router_stability/glm4.5-air-evol-codealpaca-500k-sharded
```

Approximately 500k tokens were collected. The directory contains the existing
`routing_events.parquet`, full-expert `router_stability_accumulators.npz`, and
`metadata.json`. Do not regenerate it unnecessarily.

### SEAP smoke run

```text
artifacts/seap/smoke2
```

The four-GPU implementation was tested with 32 tokens. It produced 45 MoE
layers × 128 experts with finite REAP, stability, and SEAP tensors.

### SEAP 500k run

```text
artifacts/seap/glm4.5-air-500k
```

This completed with 500,881 valid tokens and 45 MoE layers × 128 experts. It
contains `seap_state.pt`, `seap_accumulators.npz`, and `metadata.json`.

Aggregate statistics were:

```text
Pearson REAP/stability: 0.1972
Spearman correlation:   0.1948
Top-10% overlap:         12.15%
```

Plots are in `artifacts/seap/glm4.5-air-500k/plots`, including the aggregate
three-score graph and one graph per layer. Layer 27 plots include raw and
sum-normalized variants.

No model pruning/evaluation has been run yet for SEAP.

## Smoke command

After fixing `MODEL_PATH`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SEAP_MAX_TOKENS=2048 \
SEAP_OUTPUT_DIR=artifacts/seap/smoke \
.venv/bin/python experiments/seap-sharded.py
```

Validate the state:

```bash
.venv/bin/python - <<'PY'
import torch
s = torch.load("artifacts/seap/smoke/seap_state.pt", weights_only=False)
for layer, v in s.items():
    for key in ("reap_score", "stability_score", "seap_score"):
        assert v[key].shape == (128,)
        assert torch.isfinite(v[key]).all()
print("all layers have finite [128] score tensors")
PY
```

## Next task: full-dataset calibration

Before running the full dataset, update the runner to accept:

```text
SEAP_MODEL_PATH
SEAP_DATASET_NAME
SEAP_SPLIT
SEAP_BATCH_SIZE
SEAP_MAX_TOKENS (unset means no cap)
SEAP_DELTA
SEAP_LAMBDA
```

The dataset processor currently samples without replacement but expects a
batch-count limit. Add an explicit full-dataset path, preferably
`batches_per_category=None` or `process_all_examples`, so all 111,272 examples
are processed exactly once rather than relying on an oversized batch count.
Log the dataset row count, retained encoded examples, valid token count, number
of batches, truncations, and padding.

Run the complete calibration on all four GPUs, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SEAP_OUTPUT_DIR=artifacts/seap/glm4.5-air-full-dataset \
.venv/bin/python experiments/seap-sharded.py
```

Verify every MoE layer has exactly 128 finite entries for REAP, stability, and
SEAP. Verify metadata reflects the actual processed token count.

## Next task: lambda sweep and evaluation

Do not rerun the expensive router pass for every lambda. Once full-dataset
stability and normalized REAP are saved, derive SEAP offline:

```text
seap(lambda) = lambda * stability
             + (1 - lambda) * normalized_reap
```

Evaluate:

```text
REAP baseline
stability-only: lambda=1.0
SEAP: lambda=0.0, 0.25, 0.5, 0.75, 1.0
```

Run delta `2` first; run delta `4` as a second comparison if the full state
retains both stability vectors.

For each method and compression ratio, use identical model, calibration data,
seed, pruning, and evaluation settings. The next agent should build an
orchestration script that:

1. Loads the full score state.
2. Creates a state copy for each lambda/delta.
3. Recomputes `seap_score` offline.
4. Prunes the lowest scores with the existing pruning function.
5. Saves each model with an unambiguous method/delta/lambda directory name.
6. Runs the existing evaluation commands on each saved model.
7. Writes one comparison table.

Record accuracy/perplexity, expert identities pruned per layer, rank
correlation and overlap with REAP, routing balance, calibration/pruning/eval
runtime, peak GPU memory, and exact token metadata.

Suggested output layout:

```text
artifacts/seap/eval/full-dataset/reap-baseline/
artifacts/seap/eval/full-dataset/delta-2.0/lambda-0.00/
artifacts/seap/eval/full-dataset/delta-2.0/lambda-0.25/
artifacts/seap/eval/full-dataset/delta-2.0/lambda-0.50/
artifacts/seap/eval/full-dataset/delta-2.0/lambda-0.75/
artifacts/seap/eval/full-dataset/delta-2.0/lambda-1.00/
```

## Correctness checklist

- Stability is finite and in `[0,1]`.
- Delta-4 stability is never below delta-2 stability.
- Zero-selection experts have a defined score, currently zero.
- Lambda 0 reproduces normalized REAP ranking.
- Lambda 1 reproduces stability ranking.
- Pruning removes the lowest-scoring experts.
- Full perturbed ranks are used, not only perturbed top-k IDs.
- Existing REAP-only behavior remains unchanged.
- Evaluation is run on saved pruned models, not just score files.

The central unresolved task is the full 111,272-example calibration followed by
fair REAP-versus-SEAP pruning and evaluation across the lambda sweep.
