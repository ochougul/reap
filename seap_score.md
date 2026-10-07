# SEAP Score

## Goal

SEAP combines REAP expert importance with router stability. The intended
behavior is:

- experts that are stable under router-input perturbations should be favored;
- experts with large REAP scores should be favored;
- the tradeoff is controlled by `lambda`.

The initial experiments use only the variance-`2.0` router perturbation and
run separately with fixed `delta` values of `2` and `4`.

## Definition

For token `t`, expert `i`, and the variance-2 perturbation, define:

- `r_orig`: the expert's original router rank;
- `r_pert`: its rank after the variance-2 perturbation.

The absolute rank displacement is:

```text
x_i,t = abs(r_pert_i,t - r_orig_i,t)
```

Thus, an expert moving from rank 6 to rank 9 has `x = 3`. An expert moving
from rank 9 to rank 6 also has `x = 3`. The score does not distinguish
promotions from demotions; both are treated as routing instability.

For a fixed configurable `delta`, the token-level stability contribution is:

```text
stability_i,t(delta) = exp(-max(0, x_i,t - delta))
```

Consequences:

- `x <= delta` gives stability `1.0`;
- displacement beyond `delta` decays exponentially;
- with `delta=2`, `x=3` gives `exp(-1) = 0.3679`;
- with `delta=4`, `x=3` still gives `1.0`.

The expert stability score is the mean over tokens for which the expert was
originally selected:

```text
stability_i(delta)
  = sum_t stability_i,t(delta) / original_selected_count_i
```

Only originally selected experts contribute to the denominator. Experts with
zero original selections should receive a defined fallback (normally score
zero) and must not cause a divide-by-zero.

## Combining with REAP

REAP and stability have different scales, so normalize REAP within each MoE
layer before combining:

```text
normalized_reap_i = reap_i / max_j(reap_j)
```

If the layer maximum is zero, use an all-zero normalized REAP tensor.

The SEAP score is:

```text
seap_i(lambda, delta)
  = lambda * stability_i(delta)
  + (1 - lambda) * normalized_reap_i
```

Interpretation:

- `lambda=1`: stability only;
- `lambda=0`: normalized REAP only;
- intermediate values: combined score;
- higher SEAP means “preserve this expert.”

Recommended first lambda sweep:

```text
0.0, 0.25, 0.5, 0.75, 1.0
```

Different lambda values can be evaluated offline as long as stability and
REAP remain stored separately.

## Streaming state

Do not write top-12 routing events for the SEAP run. Compute rank information
in memory during each batch and discard it after updating accumulators.

For each MoE layer, maintain:

```text
stability_sum:           [num_experts]
original_selected_count: [num_experts]
reap_score:              [num_experts]
```

For GLM-4.5-Air, `num_experts=128`.

At finalization, produce:

```text
stability_score: [128]
reap_score:      [128]
seap_score:      [128]  # for the configured lambda and delta
```

The implementation may retain only the accumulators and final score tensors;
per-token displacement does not need to be persisted.

For each original selected expert `i` in each token:

```text
original_selected_count[i] += 1
x = abs(perturbed_rank[i] - original_rank[i])
stability_sum[i] += exp(-max(0, x - delta))
```

Then:

```text
stability_score = stability_sum / original_selected_count
```

Use float64 accumulators where practical, then save final scores as float32.

## Rank calculation requirements

The original and perturbed ranks must use the same router-selection semantics
as the model. For GLM-4.5-Air:

1. Compute router logits with the router matrix multiplication.
2. Apply sigmoid to obtain raw router scores.
3. Add `e_score_correction_bias` for ranking.
4. Apply grouped-router eligibility.
5. Rank eligible experts by corrected ranking values.

The rank calculation must be done for every originally selected expert, even
when the expert is no longer in the perturbed top-k. Computing only the
perturbed top-k IDs is insufficient to distinguish rank 9 from rank 20.

If an originally selected expert is no longer eligible or cannot be assigned
an exact perturbed rank, use an explicit sentinel policy. The recommended
initial policy is:

```text
perturbed_rank = num_experts + 1
```

This treats disappearance from the eligible set as a severe demotion. Document
and test this policy because it affects stability scores.

Ranks should be one-based (`1` is best). Ties should use the same deterministic
ordering as the model's `topk` operation.

## Perturbation setup

Use exactly one perturbed router input:

```text
variance multiplier: 2.0
noise: zero-mean Gaussian
noise standard deviation per token: sqrt(input_variance * 2.0)
```

The original and perturbed routing observations must use the same hidden state
and the same model/router weights. The perturbation should be generated with a
fixed seed for reproducibility.

Run separate experiments for:

```text
delta=2
delta=4
```

Because `delta` is fixed per run, a single `[128]` stability accumulator is
enough. If both deltas are desired, run calibration twice or maintain two
temporary stability accumulators during one pass.

## Existing code to reuse

Relevant files:

- `src/reap/router_stability.py` — existing router calculation and
  perturbation generation; `collect_router_observation` and `_route_once` are
  useful references.
- `src/reap/layerwise_observer.py` — layerwise MoE observer and per-layer
  expert activation handling.
- `src/reap/pruning_metrics.py` — existing REAP accumulation and state
  conventions.
- `src/reap/layerwise_prune.py` — layerwise calibration, pruning, and
  evaluation entry point.
- `experiments/router-stability-sharded.py` — existing four-GPU GLM-4.5-Air
  runner.
- `experiments/router-stability-layerwise.sh` — existing single-GPU
  router-stability configuration.

Existing REAP is calculated in `update_pruning_state` as:

```text
activation_norm_i,t = ||expert_activation_i,t||_2
reap_contribution_i,t = activation_norm_i,t * router_weight_i,t
reap_i = mean over tokens routed to expert i of reap_contribution_i,t
```

The new score should match the existing per-layer/per-expert output shape and
direction: larger score means the expert should be preserved.

## Model and data setup

Checkpoint:

```text
/local/mnt2/workspace/ochougul/artifacts/models/GLM-4.5-Air
```

Dataset:

```text
theblackcat102/evol-codealpaca-v1
split=train
model_max_length=2048
batch_size=1
batches_per_category=256
calibration token budget=500000
seed=42
```

Use `.venv` for all Python commands. The existing router-stability output is:

```text
artifacts/router_stability/glm4.5-air-evol-codealpaca-500k-sharded
```

The current parquet contains original plus four perturbations, but the new
SEAP collector should use only the variance-2 condition and update scores
directly rather than relying on the parquet.

## Output and pruning integration

The score state should be saved in the same per-layer format as the existing
observer state, for example:

```text
state[layer]["stability_score"]
state[layer]["reap_score"]
state[layer]["seap_score"]
```

The pruning path should select experts with the lowest score for removal. The
score direction must be verified with a small synthetic test before running a
large experiment.

Suggested configuration fields:

```text
collect_seap: bool = false
seap_delta: float = 2.0
seap_lambda: float = 0.5
seap_variance_multiplier: float = 2.0
```

The exact argument names can follow the repository's existing argument style.
`delta` and `lambda` must be recorded in experiment metadata so results are
unambiguous.

## Validation checklist

Before expensive pruning runs, verify:

- stability scores are finite and lie in `[0, 1]`;
- experts with zero original selections have a defined score;
- `delta=4` scores are never lower than `delta=2` scores for the same expert;
- `lambda=0` reproduces normalized REAP ranking;
- `lambda=1` reproduces stability ranking;
- an expert with no rank movement has stability `1.0`;
- an expert demoted by three positions gives `0.3679` at `delta=2`;
- score ordering is deterministic under a fixed seed;
- pruning removes the lowest-scoring experts;
- existing REAP behavior remains unchanged.

## Comparison plan

Compare the following using identical calibration and evaluation settings:

1. Existing REAP.
2. Stability-only (`lambda=1`).
3. SEAP with `delta=2` and several lambda values.
4. SEAP with `delta=4` and the same lambda values.

Record:

- downstream accuracy/perplexity;
- expert rankings and rank correlation with REAP;
- top-k overlap with REAP;
- selected/pruned expert identities by layer;
- router balance after pruning;
- runtime and memory overhead.

The stability score is a routing robustness signal, not automatically a causal
importance measure. The accuracy comparison is required to determine whether
combining it with REAP improves pruning decisions.
