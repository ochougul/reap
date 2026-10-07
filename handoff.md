# Handoff: router-stability observer and next scoring work

## Objective for the next agent

The router-stability data-collection work is complete. The next task is to
define an expert score from the collected router-stability information, make it
usable for pruning in the same way as REAP, and compare pruning methods by
model accuracy and related metrics.

The immediate development sequence should be:

1. Implement a router-stability-derived expert score.
2. Integrate it with the existing layerwise pruning path.
3. Run the score and REAP on the same model, dataset, calibration budget, and
   compression ratios.
4. Evaluate the resulting pruned models using the existing evaluation scripts.
5. Compare accuracy, score rankings, overlap with REAP, and routing/stability
   diagnostics.

Do not regenerate the existing router-stability parquet unless a new
experiment is explicitly required. It already contains the 500k-token
observations.

## Repository state

Current branch:

```text
router-stability-4gpu
```

Relevant recent commits:

```text
30ad9e1  Add router stability collection
7a3f7ca  Add resident sharded router stability runner
e83562a  Document router stability experiments
```

Current working-tree changes at handoff:

- `scripts/analyze_router_stability_layer.py` is a new analysis script added in
  this session.
- `uv.lock` is modified by dependency resolution. The lockfile diff includes
  unrelated resolver/platform changes; inspect it before committing. No source
  observer implementation files were modified by the analysis work in this
  session.

## Existing observer implementation

### `src/reap/router_stability.py`

Important functions/classes:

- `collect_router_observation(...)`
  - Runs the original router input and four perturbed inputs.
  - Perturbations are zero-mean Gaussian noise with variance equal to the
    per-token hidden-state variance multiplied by `0.25`, `0.5`, `1.0`, and
    `2.0`.
- `_glm_route(...)`
  - Implements the GLM-4.5-Air routing calculation used by collection.
- `_route_once(...)`
  - Selects top-12 experts, where top-8 are selected and four are boundary
    experts.
- `RouterStabilityCollector`
  - Streams full-expert aggregate statistics to
    `router_stability_accumulators.npz`.
  - Writes token-level data to `routing_events.parquet`.

The collector’s accumulator tensors have these useful keys, indexed by
`[variance_condition, layer, expert]` unless noted otherwise:

- `abs_change_sum`: sum of absolute perturbed-minus-original router score
  changes across all 128 experts.
- `squared_change_sum`: sum of squared score changes.
- `signed_change_sum`: sum of signed score changes.
- `original_score_sum`: original score sum, indexed `[layer, expert]`.
- `perturbed_score_sum`: perturbed score sums.
- `original_selected_count`: selected counts for original routing,
  `[layer, expert]`.
- `perturbed_selected_count`: selected counts under each perturbation.
- `selection_gained_count`: experts not selected originally but selected after
  perturbation.
- `selection_lost_count`: experts selected originally but not after
  perturbation.

These accumulators are the best source for a full-expert score because they
cover every expert, not only the 12 experts written per token to parquet.

### `src/reap/layerwise_observer.py`

`LayerwiseMoEObserver` already calls `collect_router_observation` during the
layerwise pass and forwards the observation to `RouterStabilityCollector`.
It also computes the normal pruning metrics in the same pass. Relevant code is
near the router-stability call around line 660 and the normal router-logit
handling around line 750.

### `src/reap/pruning_metrics.py`

This is the reference implementation for the existing REAP-compatible state.
`update_pruning_state(...)` computes and accumulates:

- `expert_frequency`
- `pairwise_expert_frequency`
- `ean_sum` and `ean_mean`
- `weighted_ean_sum`
- `reap`
- `weighted_expert_frequency_sum`
- `max_activations`

The current REAP definition in code is effectively:

```text
ean_norm = ||selected expert activation||
reap_i = mean(ean_norm * router_weight_i)
```

The `OnlineStatsTracker` values are later converted to means by
`LayerwiseMoEObserver.report_state()`.

The new router-stability score should ideally follow the same per-layer,
per-expert output convention as `reap`, so it can be passed through the same
pruning and comparison machinery.

## Router calculation details

For GLM-4.5-Air, with hidden state `h` and router weight `W`:

```text
logits = h @ W.T
scores = sigmoid(logits)
corrected = scores + e_score_correction_bias
```

The GLM router then applies grouped expert eligibility. Only eligible experts
are ranked. The stored ranking value is:

```text
ranking_value = corrected score for an eligible expert
               = sigmoid(logit) + correction_bias
```

Ineligible experts are represented as `-inf` before top-k selection. The top-12
ranking values are sorted descending; the first eight IDs are the selected
experts and the final four are boundary experts.

The applied top-8 weights are calculated from the uncorrected `scores`, not
from `ranking_values`:

```text
selected_scores = scores[selected_ids]
weights = selected_scores / sum(selected_scores)
weights *= routed_scaling_factor
```

Therefore, `top12_ranking_values` are not probabilities. The parquet does not
store raw full router scores, raw logits, hidden states, eligibility masks, or
boundary-expert applied weights.

## Existing model/checkpoint and dataset

Model checkpoint:

```text
/local/mnt2/workspace/ochougul/artifacts/models/GLM-4.5-Air
```

Dataset/config used for the recorded run:

```text
dataset: theblackcat102/evol-codealpaca-v1
split: train
model_max_length: 2048
batch_size: 1
batches_per_category: 256
max router-stability tokens: 500000
seed: 42
variance multipliers: 0.25 0.5 1.0 2.0
boundary width: 4
top-k: 8
experts: 128
```

The resident four-GPU runner is:

```text
experiments/router-stability-sharded.py
```

It has model and output paths as Python constants. The single-GPU layerwise
runner is:

```text
experiments/router-stability-layerwise.sh
```

Use `.venv` for Python commands. The needed packages (`torch`, `pyarrow`,
`numpy`, `matplotlib`, `safetensors`, etc.) are already available there.

## Existing router-stability artifacts

Output directory:

```text
artifacts/router_stability/glm4.5-air-evol-codealpaca-500k-sharded
```

Files:

- `routing_events.parquet` — approximately 3.6 GB, 22.5M rows.
- `router_stability_accumulators.npz` — approximately 715 KB.
- `metadata.json` — completed-run metadata.

`metadata.json` currently records:

```json
{
  "variance_multipliers": [0.25, 0.5, 1.0, 2.0],
  "boundary_width": 4,
  "top_k": 8,
  "total_tokens": 500000,
  "num_layers": 46,
  "num_experts": 128,
  "noise": "zero_mean_gaussian_relative_to_input_variance"
}
```

For GLM-4.5-Air, block 0 is not an MoE block. The recorded routing rows are
for the MoE layers; layer 20 has 500,000 rows.

### Parquet schema

Each row represents one token at one MoE layer:

- `batch_id: int64`
- `token_position: int64`
- `layer: int64`
- `top12_expert_ids: list[list[int64]]`, shape conceptually `[5, 12]`
- `top12_ranking_values: list[list[double]]`, shape `[5, 12]`
- `top8_applied_weights: list[list[double]]`, shape `[5, 8]`

The five conditions are ordered:

```text
0 original
1 variance 0.25
2 variance 0.5
3 variance 1.0
4 variance 2.0
```

Use predicate pushdown when reading the large file:

```python
import pyarrow.parquet as pq

path = "artifacts/router_stability/glm4.5-air-evol-codealpaca-500k-sharded/routing_events.parquet"
table = pq.read_table(
    path,
    columns=["batch_id", "token_position", "layer", "top12_expert_ids"],
    filters=[("layer", "=", 20)],
)
```

The IDs are compact unsigned 8-bit values and ranking values/weights were
quantized to float16 before writing, even though pyarrow may display nested
values as int64/double.

## Analysis added this session

New script:

```text
scripts/analyze_router_stability_layer.py
```

It reads the existing parquet and does not run a model or modify the data. It
counts the first eight IDs under each condition and computes:

```text
selection_probability[condition, expert] = selected_count / token_count
delta = probability[condition] - probability[original]
```

The probabilities sum to 8 across the 128 experts because eight experts are
selected per token. The mean probability across experts is therefore
`8 / 128 = 0.0625` (6.25%).

Run it with:

```bash
MPLCONFIGDIR=/tmp/reap-mpl-cache \
.venv/bin/python scripts/analyze_router_stability_layer.py \
  --parquet artifacts/router_stability/glm4.5-air-evol-codealpaca-500k-sharded/routing_events.parquet \
  --layer 20 \
  --output-dir artifacts/router_stability/glm4.5-air-evol-codealpaca-500k-sharded/analysis_layer20
```

Generated outputs:

- `analysis_layer20/expert_selection_probability.csv`
- `analysis_layer20/expert_selection_probability_heatmap.png`
- `analysis_layer20/expert_selection_probability_lines.png`
- `analysis_layer20/expert_selection_probability_delta.png`
- `analysis_layer20/summary.json`

Layer-20 summary from the existing 500k rows:

- Original selection probability range: `1.765%` to `22.084%`.
- Original mean across experts: `6.25%`.
- Original cross-expert standard deviation: approximately `2.963` percentage
  points.
- Mean absolute change versus original:
  - variance `0.25`: `0.279` percentage points
  - variance `0.5`: `0.473` percentage points
  - variance `1.0`: `0.780` percentage points
  - variance `2.0`: `1.175` percentage points
- Correlation of expert probabilities with original:
  - variance `0.25`: `0.995`
  - variance `0.5`: `0.983`
  - variance `1.0`: `0.944`
  - variance `2.0`: `0.840`

These are routing-preference/stability observations, not causal expert
importance conclusions.

## Correction-bias checkpoint lookup

The layer-20 correction vector was read without loading the model using
`safetensors.safe_open`:

```text
checkpoint shard: model-00021-of-00047.safetensors
tensor key: model.layers.20.mlp.gate.e_score_correction_bias
shape: [128]
dtype: float32
```

Statistics:

```text
min: 18.140549 (expert 59)
max: 18.414001 (expert 8)
mean: 18.350094
std: 0.047853
```

The complete expert-indexed vector is available from the safetensor. To read
only this tensor:

```python
import json
from pathlib import Path
from safetensors import safe_open

model = Path("/local/mnt2/workspace/ochougul/artifacts/models/GLM-4.5-Air")
key = "model.layers.20.mlp.gate.e_score_correction_bias"
index = json.loads((model / "model.safetensors.index.json").read_text())
shard = model / index["weight_map"][key]
with safe_open(str(shard), framework="numpy", device="cpu") as f:
    bias = f.get_tensor(key)
```

## Example routing flip

One concrete layer-20 event is:

```text
batch_id=0, token_position=0, layer=20
```

Original selected IDs:

```text
122, 25, 110, 94, 22, 19, 126, 91
```

Variance-2.0 selected IDs:

```text
122, 70, 18, 112, 126, 79, 22, 87
```

Five of the eight selected experts change. The retained experts are `122`,
`126`, and `22`. The parquet’s ranking values for this event are corrected
ranking values, while the top-8 applied weights are the actual mixture weights.

## Recommended next score implementation

The next score should be explicitly defined before implementation. Candidate
ingredients available from the completed run include:

- original selection frequency;
- perturbed selection frequency;
- selection gain/loss counts;
- retention probability;
- score-change magnitude from `abs_change_sum` or
  `squared_change_sum`;
- original/perturbed score means;
- applied-weight statistics from parquet (only for selected top-8 and only
  token-level, so expensive to aggregate at full scale);
- existing REAP activation norm and router-weight statistics from the normal
  observer state.

For a first implementation, prefer a simple, auditable score with one
well-defined direction: higher score means “more important / preserve this
expert.” Store it per layer as a tensor of shape `[num_experts]`, matching the
existing `reap` output shape. Avoid mixing quantities with incompatible scales
without normalization.

Potential stability features to evaluate separately before combining:

```text
retention_i(v) = retained selections for expert i at variance v
                 / original selections for expert i

gain_i(v) = gained_count_i(v) / total_tokens

loss_i(v) = lost_count_i(v) / total_tokens

instability_i(v) = abs_change_sum[v, layer, i] / total_tokens
```

Be careful: a high selection frequency does not necessarily mean high causal
importance, and a high perturbation sensitivity can mean either important
specialization or router fragility. Keep frequency, stability, and activation-
based importance as separate baselines in the first comparison.

## Recommended comparison experiment

Use the same:

- checkpoint;
- calibration dataset and split;
- tokenization and sequence length;
- number of calibration tokens;
- pruning ratio/compression ratio;
- layerwise pruning/evaluation code;
- evaluation tasks and decoding/evaluation settings.

Start with layer 20 for fast score validation, then run all MoE layers.
Compare at least:

1. Existing REAP score.
2. Selection-frequency score.
3. Router-stability score with each feature separately.
4. Combined router-stability score, if justified.

Record:

- expert ranking and rank correlation with REAP;
- top-k overlap with REAP;
- selected/pruned expert identities by layer;
- calibration score statistics;
- downstream accuracy/perplexity/evaluation results;
- routing balance after pruning;
- runtime and memory overhead.

The existing layerwise entry point is:

```text
src/reap/layerwise_prune.py
```

The standard observer-only/calibration path uses `--run_observer_only true`
and `--record_pruning_metrics_only true`. For actual pruning, use the normal
`--prune_method reap` path as the baseline and add a new method/configuration
with the same output/state conventions.

Before any expensive run, test the score on a small calibration subset and
verify:

- score shape and device;
- no NaNs or infinities;
- score ordering is deterministic under a fixed seed;
- pruning removes the intended experts;
- existing REAP behavior is unchanged.

## Important cautions

- Do not call `top12_ranking_values` probabilities. They include the GLM
  correction bias and can be around 18–19.
- The parquet does not contain raw router scores for all 128 experts. Use the
  accumulator `.npz` for full-expert router-score changes.
- The parquet’s four perturbation conditions are paired with the same token,
  but the actual random noise vectors were not stored.
- `metadata.json.num_layers` is the maximum layer index plus one, not the
  number of layers with routing rows.
- Layer 20 means the model block index used by the observer. GLM block 0 is not
  an MoE routing layer.
- Do not treat the mean signed probability change as meaningful: because every
  token selects exactly eight experts, signed changes sum to zero across
  experts. Use per-expert changes, mean absolute changes, retention, and
  paired-token statistics instead.
- The analysis script only reads the existing parquet; it does not modify the
  data or run new model experiments.
