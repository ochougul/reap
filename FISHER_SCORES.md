# Fisher expert scores and the 12,228-sample joint calibration (REAP + stability + Fisher)

## Motivation

REAP scores are in activation-norm units, so they are not comparable across layers.
That is why allocating pruning by inverse layer-summed REAP was unreliable. The
Fisher-style score prices every routed expert in **loss units**, which are comparable
across layers, and it is computed from one forward and one backward pass instead of
the old leave-one-expert-out masking sweep.

## What the Fisher score is

For MoE layer `l` and routed expert `e`:

    score(l, e) = (1 / N) * sum over tokens t of ( G_l(t) . y_e(t) )^2
    y_e(t) = g_e(t) * f_e(x_t)        # router-weighted expert output, only if e is in t's top-k

`G_l(t) = dL/d(MoE layer output)` is the loss gradient at the layer output, from one
backward pass with all parameters frozen. `G . y_e` is the first-order Taylor estimate
of the loss change if expert `e`'s contribution were deleted. It is squared so that
positive and negative effects on different tokens do not cancel. `N` is the number of
counted tokens, so the score also grows with how often an expert is routed.

Reading it: **high = the loss is sensitive to this expert on the calibration data;
low = removing it is predicted to change the loss little.** Caveats: first-order only;
single-expert (interactions when pruning many are not modeled); ignores router
re-selection after removal; calibration-specific (code-assistant tokens); the squared
score hides direction (`fisher_signed` keeps the net signed value); early-layer
rankings are noisy at small sample counts.

No expert recomputation is needed. Since `y_e = w * f_e(x)`, `G . y_e = w * dL/dw`
where `w` is the token's routing weight, and autograd already computes `dL/dw`
when backpropagating into the router. A tensor hook on the gate's `topk_weights`
reads it. `FisherCollector(mode="recompute")` recomputes `y_e` explicitly and is kept
as an independent reference; tests/test_fisher.py checks both modes agree.

## One pass, three scores

`src/reap/joint_calibration.py::JointCollector` collects, in the same forward and
backward pass over each batch:

- **REAP**: expert-output row norms from forward hooks on the experts the stock MoE
  already runs (scattered into a per-layer buffer with no host sync), weighted by the
  renormalized router softmax. Identical definition to `update_pruning_state`
  (`renormalize_router_weights=True`); no dense all-experts-on-all-tokens recompute.
- **Stability**: `collect_seap_observation` on the gate input (deltas 2.0 and 4.0,
  variance multiplier 2.0), accumulated on the GPU. SEAP score uses lambda 0.5.
- **Fisher**: the router-gradient hook above.

The stock MoE implementation is used (no fused or sorted dispatch).
Joint collection added roughly 12% over plain forward+backward in smoke tests; the full run took 0.95 s/sample.

## Data and batching

- Dataset `theblackcat102/evol-codealpaca-v1`, split `train`, seed 42.
- **Unpacked, untruncated, random** samples: one dataset example per sample, drawn
  without replacement by the loader's existing seeded `random.randint` sampling
  (`load_category_batches(..., pack_samples=False)`; the `pack_samples` pass-through
  was added). Examples longer than `FISHER_MODEL_MAX_LENGTH` are skipped and replaced
  by other random draws, never truncated.
- **Earlier SEAP/REAP states used packed sequences.** `pack_samples` defaulted to
  `True`, so each "sample" there was a 2048-token concatenation of several examples
  (about 2,040 tokens) and only the last example's response was counted as loss tokens
  by the last-assistant-marker rule. This run's states are therefore **not sample-for-sample
  comparable** with `artifacts/seap/glm4.5-air-full-dataset`.
- Exactly 12,228 samples (6,736,187 tokens, mean 551, max 5,497). The runner builds the
  full sample list first and fails if the count differs.
- **Max length 5,500, not 16,384.** At 16,384 the preflight hit a 6,837-token example
  and ran out of memory (a 5,497-token sample already peaks at about 75 of 79 GB per
  GPU). Gradient checkpointing is not an option because it breaks the backward hooks.
  This excludes roughly the longest 0.2% of the dataset.
- Batching: up to 4,096 padded tokens per forward/backward (a longer single sample
  runs alone). Samples are length-sorted only inside windows of 250, so every multiple
  of 250 samples is an unbiased random subset and checkpoints are valid.
- Loss: next-token cross-entropy on assistant tokens only (everything up to the last
  `<|assistant|>` marker is masked).
- A preflight runs the largest batch before the main loop, so a memory failure shows up
  in minutes.

## Run command

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
FISHER_MODEL_PATH=/path/to/GLM-4.5-Air-patched \
FISHER_OUTPUT_DIR=artifacts/fisher/glm4.5-air-12228 \
FISHER_MODEL_MAX_LENGTH=5500 \
PYTHONPATH=src python -u experiments/fisher-sharded.py
```

| Variable | Default | Meaning |
|---|---|---|
| `FISHER_TARGET_SAMPLES` | 12228 | exact number of random samples |
| `FISHER_MODEL_MAX_LENGTH` | 16384 | longer examples are skipped, not truncated |
| `FISHER_MAX_BATCH_TOKENS` | 4096 | padded-token budget per forward/backward |
| `FISHER_WINDOW` | 250 | length-sorting window (checkpoint granularity) |
| `FISHER_CHECKPOINT_EVERY` | 4000 | save a state every N samples (at window boundaries) |
| `FISHER_MODE` | router_grad | or `recompute` (reference, slower) |
| `FISHER_STABILITY` | true | collect the stability score |
| `FISHER_SEED` | 42 | sampling and perturbation seed |

## Outputs

`artifacts/fisher/glm4.5-air-12228/` (artifacts are gitignored):

- `seap_state.pt`, `metadata.json`: final state (`final: true`).
- `checkpoint_4000/`, `checkpoint_8000/`, `checkpoint_12000/`: intermediate states and
  metadata (`final: false`).
- `inverse_fisher_allocation.png`: allocation plot described below.
- `run.log`.

The state is `{layer: {...}}` for the 45 MoE layers (1..45; layer 0 is dense), each
with: `reap`, `reap_score`, `normalized_reap`, `stability_score`,
`stability_score_delta_2`, `stability_score_delta_4`, `stability_scores`, `seap_score`,
`fisher_score`, `fisher_signed`, `ean_sum`, `expert_frequency`, `total_tokens`.
It is compatible with `experiments/prune_seap_state.py` for `seap` and `reap`;
`fisher_score` is not yet a selectable prune method.

## Results of the 12,228-sample run

- 3.2 h, 0.95 s/sample, peak 72 to 75 GiB per GPU, mean loss on assistant tokens 1.837.
- All scores finite, Fisher scores non-negative, every layer saw the same 6,736,187 tokens.
- Convergence, Spearman rank correlation of the checkpoint with the final state
  (median over layers, worst layer in brackets):

| Samples | REAP | Stability | Fisher |
|---|---|---|---|
| 4,000 | 0.999 (0.992) | 0.999 (0.998) | 0.980 (0.912) |
| 8,000 | 1.000 (0.998) | 1.000 (0.999) | 0.992 (0.949) |

  REAP and stability settle by 4k samples; Fisher is slower. This compares prefixes
  to the final run, not independent seeds, so it understates the true uncertainty.
- Fisher scores differ by up to about 11% between two runs on identical data (bf16 backward nondeterminism).

## Inverse-Fisher allocation (illustration, not yet evaluated)

Layer sensitivity `S_l = sum_e fisher_score(l, e)`; experts pruned in layer `l` are
proportional to `1 / S_l`, scaled to a 25% budget (1,440 of 5,760 experts, largest-remainder
rounding). `S_l` spans 345x across layers (1.7e-08 to 5.9e-06). Result: layers 4-13 get 0,
layers 14-26 rise from 2 to 41, layers 27-45 get most of the budget (peak 96 of 128 at
layer 35; layer 40 only 18). This is far more aggressive than the 25-50% per-layer bound
that worked in earlier REAP experiments, and early-layer scores are the noisiest, so a
bounded or softened variant (clipping, `1/sqrt(S)`) should be tried before any eval.
No pruned model has been evaluated with Fisher scores yet.

## Tests and code

- `tests/test_fisher.py`: router-grad vs recompute equivalence, first-order prediction of the
  loss change when an expert is dropped, padding exclusion, assistant-label mask,
  sorted-MoE equivalence, and joint-collector agreement with the legacy dense REAP,
  stability, and SEAP computation.
- `src/reap/fisher.py`, `src/reap/joint_calibration.py`, `experiments/fisher-sharded.py`.
- `src/reap/fast_moe.py` (sorted expert dispatch) exists and is tested
  but is **not used**: the stock MoE code was kept on purpose.

## Not done yet

- `fisher` is not wired into `experiments/prune_seap_state.py` (`choices` and `score_key`).
- No seed-stability study for Fisher, no pruned-model accuracy eval.
