# Test non-uniform expert pruning using the existing full calibration

## Objective and scope

Build and evaluate allocations for Qwen3.8-Flash-Next-FP8 that:

1. Improve coding accuracy at the existing exact pruning budget of 9,180 / 24,576 routed experts (37.353515625%).
2. Increase the pruning budget while meeting a declared accuracy target.

Reuse the completed full-dataset calibration. Do not rerun `experiments/qwen4_calibrate.py`, collect another full observer state, or retrain the model. New inference and code-execution evaluations are necessary to test accuracy. The first phase generates allocations entirely offline from saved statistics; targeted expert interventions are a later phase if those allocations fail or their errors need diagnosis.

Treat the methods below as hypotheses. No saved score establishes that an expert is dispensable, and no allocation is a demonstrated improvement until evaluated.

## 1. Inputs and existing evidence

Run from `/home/ubuntu/ochougul/reap`, using `PYTHONPATH=src .venv-eval/bin/python`. The `.venv` interpreter currently lacks torch.

Base model:

```text
/home/ubuntu/hf_cache/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13
```

Authoritative statistics:

```text
artifacts/Qwen3.8-Flash-Next-FP8/full-packed-2048/calibration_state.pt
artifacts/Qwen3.8-Flash-Next-FP8/full-packed-2048/metadata.json
artifacts/Qwen3.8-Flash-Next-FP8/full-packed-2048/router_stability/router_stability_accumulators.npz
```

Derived ranking states:

```text
artifacts/Qwen3.8-Flash-Next-FP8/eval_states/reap/state.pt
artifacts/Qwen3.8-Flash-Next-FP8/eval_states/selection-stability/state.pt
artifacts/Qwen3.8-Flash-Next-FP8/eval_states/lambda-sweep/delta-2/lambda-0.75/state.pt
```

The model has 48 MoE layers, 512 routed experts per layer, and top-k=10. The calibration used 53,236,097 valid tokens from evol-codealpaca, with 2,048-token packed sequences. The final state contains per-layer tensors for `reap`, `expert_frequency`, `weighted_ean_sum`, `weighted_expert_frequency_sum`, `ean_sum`, `ean_mean`, `max_activations`, `total_tokens`, `pairwise_expert_frequency`, and `moe_residual_contribution`.

Historical LiveCodeBench results, all reported on the 182-problem 2025-01-01 through 2025-04-30 snapshot:

| Ranking | Allocation | Removed | Passed |
|---|---|---:|---:|
| None | Original | 0 | 124 |
| REAP | Uniform | 9,180 | 123 |
| Stability | Uniform | 9,180 | 115 |
| REAP | Bounded inverse summed REAP | 9,180 | 118 |
| Stability | Bounded inverse summed REAP | 9,180 | 120 |
| REAP | Uniform | 12,288 | 111 |
| Stability | Uniform | 12,288 | 116 |
| Mixture, lambda=0.75 | Uniform | 12,288 | 121 |
| Stability | Inverse summed REAP, minimum 200 retained | 12,288 | 112 |

The three uniform 50% result files exist outside the directory covered by the research-direction note:

```text
artifacts/livecode_qwen38_reap50_retry/Scenario.codegeneration_1_0.0_eval.json
artifacts/livecode_qwen38_stability50/Scenario.codegeneration_1_0.0_eval.json
artifacts/livecode_qwen38_lambda075_50/Scenario.codegeneration_1_0.0_eval.json
```

Their pass counts were checked while preparing this plan. Before using them as experimental controls, verify checkpoint provenance, retained expert IDs, benchmark IDs, prompt settings, precision and routing against the new pipeline. Rerun a control if its provenance cannot be established. Older broken `reap-25`/`stability-25` result directories also exist; do not infer validity from a directory name.

Existing 37.35% controls and bounded allocations are under `artifacts/Qwen3.8-Flash-Next-FP8/eval/`. Use the original research note for their exact paths and results, not its recommendation.

## 2. What can and cannot be recovered offline

For layer l and expert e, extract:

```text
r[l,e] = reap[l,e]
f[l,e] = expert_frequency[l,e]
T[l]   = total_tokens[l]
c[l,e] = weighted_ean_sum[l,e] / T[l]
       approximately equals r[l,e] * f[l,e] / T[l]
```

`r` is the mean router-weighted output norm conditional on selection. `c` is the unconditional contribution-magnitude proxy per calibration token. Verify the identity numerically; it holds in the final artifact to approximately 1e-7 maximum relative error by layer.

Reuse the existing mixture state exactly. Its ranking score is `0.75 * (reap / layer_max_reap) + 0.25 * stability`; stability is selection retention under the saved perturbation. It is not causal pruning sensitivity.

Offline data supports expert ranking, sorted score tails, contribution concentration, and temporal consistency checks. It does not support exact output reconstruction after pruning, replacement-expert quality, final-logit KL, gradients, or causal accuracy changes. Routing events were not saved. Calibration ran `model.model` and skipped the vocabulary projection.

Critical trap: `pairwise_expert_frequency[i,j]` is `f[i] + f[j]`, not co-selection counts. This was checked in the saved state. Do not use it to claim expert redundancy or construct a co-activation graph. `max_activations` is a maximum activation element, not a maximum expert contribution norm or a reliable safeguard for rare critical experts.

Frequency weighting alone is unlikely to change the existing layer importance ordering: summed r and summed c correlate about 0.996 across these 48 layers. The new hypothesis concerns the distribution of candidate removals within each layer.

## 3. Freeze the evaluation protocol before candidate selection

Create `artifacts/Qwen3.8-Flash-Next-FP8/layerwise-reuse-v1/` with an immutable experiment manifest. Preserve all existing artifacts and unrelated worktree changes. Record source hashes, git diff/commit, package versions, dataset revisions, task IDs, prompts, scores, tie rules, pruning counts, original expert IDs, and all evaluator settings.

Use three evaluation roles:

1. Development: MBPP+ tasks, deterministic seed-42 split into 128 screening tasks and all remaining tasks for confirmation. Use execution pass@1 to select allocations; preserve task IDs and the exact split. Do not pick on prompt perplexity alone.
2. Historical comparison: the existing 182-task LiveCodeBench snapshot. Run only frozen finalists and matched controls. This set has already been repeatedly inspected, so improvement here is exploratory evidence.
3. Fresh confirmation: freeze a non-overlapping LiveCodeBench snapshot, initially 2025-05-01 through 2025-08-31, before observing new candidate results. Verify dataset availability and no task-ID overlap; if unavailable, select and record another non-overlapping interval before running candidates. Also report HumanEval+ as a secondary task family. Neither confirmation set is used to tune allocation parameters.

For screening ties, keep candidates within one solved task of the best and use the larger development partition to resolve them. Prefer the simpler method/uniform control if confirmation remains tied. Do not declare equivalence from a tie on 128 tasks.

Default operational accuracy target, since none was numerically specified by the user:

- Historical target: at least 123/182, matching uniform REAP at 37.35%.
- Fresh-set target: at least the pass@1 of the matched uniform REAP 37.35% control on that same fresh set.
- Report original-model accuracy as a second reference; do not silently replace the target with an easier threshold.

Report paired per-problem differences and 95% problem-bootstrap intervals (10,000 resamples, fixed seed). Use exact McNemar comparisons where helpful. A positive point estimate or a met point target is an observed result; if intervals are inconclusive, label it accordingly. Do not describe a permitted accuracy drop as "same accuracy." Any non-inferiority margin must be declared separately before evaluations, not selected from results.

## 4. Offline audit and cumulative removal curves

Implement a new CPU-only analysis/allocation script. Do not load model weights in this phase.

For each ranking s (first REAP, then the existing lambda=0.75 mixture), sort experts ascending with original expert ID as the tie breaker. Denote this order by pi[l]. A removal count m means removing exactly the first m entries. Save this order in the manifest.

Compute one-expert-resolution curves for every layer and all m=0..502:

```text
C_score[l,m] = sum_{j < m} s[l,pi[l,j]] / sum_e s[l,e]
C_mass[l,m]  = sum_{j < m} c[l,pi[l,j]] / sum_e c[l,e]
```

The denominator is the original full layer's total and stays fixed across budgets. Reject non-finite or negative scores; explicitly flag zero-total layers instead of allowing division by zero to fabricate importance. For this artifact, verify positive totals and report them.

Save the discrete increments, score gaps, routing frequency removed, removed score/mass fractions, and proposed retained counts. Plot cumulative removal curves and their increments, especially around 37.35%, 45%, and 50% pruning.

Interpretation: `C_score` measures the fraction of the layer's ranking score removed; `C_mass` measures the fraction of its expected contribution magnitude removed. Both are proxy objectives. Equal weighting across layers assumes comparable damage for equal fractional removal; this is an assumption to test, not an established sensitivity correction.

These curves explicitly charge for each candidate expert, rather than assigning a pruning ratio from a whole-layer total. An expert with a large proxy cost creates a large increment. An expert with small proxy scores but large accuracy impact can still be missed; section 9 handles this failure mode.

### Reuse intermediate checkpoints to check consistency

The calibration saved cumulative snapshots at 10,016 and 20,000 packed sequences. Each has rank0 files at the root and rank1/rank2/rank3 files in their respective directories:

```text
calibration_state_checkpoint_000010016.pt
calibration_state_checkpoint_000020000.pt
```

Each checkpoint wraps its data in `checkpoint['state']` and still contains OnlineStatsTracker objects. Load with `PYTHONPATH=src`. Intermediate expert output statistics are EP-local: merge all four ranks before comparing with the final merged state. Mirror `_all_reduce_ep_state` in `experiments/qwen4_calibrate.py` on CPU: sums for sufficient statistics, count-weighted means for trackers, maxima for max fields. Do not average rank means or treat root-only expert output sums as complete.

Checked behavior: routing token counts and frequencies reside on rank0; other ranks have zero for these fields. At 10,016 and 20,000 sequences, merged layer token counts are 20,480,870 and 40,896,429. The final count is 53,236,097. Verify across all layers. Router-stability files cover the full routing stream on each rank: root alone has 532,360,970 selected pairs per layer at the end. Do not sum those replicated files.

After checking monotonic cumulative sufficient statistics, derive three disjoint temporal segments by subtracting sums and counts: first snapshot; second minus first; final minus second. Derive segment REAP from segment weighted sums / selection counts. Never subtract means. Do not difference max fields. Use these segments for ranking/allocation consistency, not as independent test sets or formal confidence intervals.

Report expert-order agreement near removal boundaries, retained-set agreement, and layer-count changes between segments. A layer whose suggested capacity changes sharply across segments is an uncertain proposal. The final full state remains the allocation input; do not cherry-pick the segment with the best benchmark result.

## 5. Allocate an exact budget, one expert at a time

For a total R, define uniform removal counts:

```text
u[l] = R // 48 + int(l < R % 48)
```

First test counts within a declared exploration radius of 64 experts of the uniform allocation:

```text
lower[l] = max(0, u[l] - 64)
upper[l] = min(502, u[l] + 64)
sum_l m[l] = R
```

The radius bounds the deviation from a known control; it is not an assertion that 64 experts have equal importance or a removal batch size. It is a fixed initial search restriction, not an optimality claim. Only consider radius 128 as a separately named follow-up after improvement is confirmed on development data. Always include uniform as a feasible allocation.

Generate two allocation methods:

- `tail-score`: minimize sum_l C_score[l,m[l]] at exact R.
- `tail-mass`: minimize sum_l C_mass[l,m[l]] at exact R, retaining the same ranking order pi.

Use deterministic dynamic programming over layer and total removals, with the allowed integer counts per layer. Resolve equal costs deterministically and document the rule. Each state transition chooses an integer count; it does not average the cost of a group of 16.

For `tail-score`, sorted increments are nondecreasing, so a heap of the next eligible expert per layer is an equivalent faster solver; test agreement with DP on toy cases. `tail-mass` increments need not be monotonic under a fixed REAP/mixture order, so use DP rather than assuming that greedy marginal selection is optimal.

At every solution, verify the exact count, bounds, top-k feasibility, and original expert identity. The objective should be no worse than uniform under the same proxy; that is a solver check, not evidence of better accuracy.

Keep within-layer selection fixed in this phase. Allowing the allocator to bypass a high-cost expert and remove a later one changes expert ranking and must be a distinct later experiment.

### Offline feasibility check already performed

A CPU-only heap calculation of `tail-score`, using the actual full-state scores and radius 64, produced the following feasible allocations. Exact totals, bounds, and improvement over uniform in the proxy objective were checked. These are not model accuracy results and have not been materialized or evaluated.

| Ranking | Removed R | Experts transferred between layers | Retained range | Uniform proxy sum | Candidate proxy sum |
|---|---:|---:|---:|---:|---:|
| REAP | 9,180 | 809 | 256–385 | 11.746992 | 11.600639 |
| Mixture | 9,180 | 590 | 272–378 | 12.552456 | 12.484123 |
| REAP | 12,288 | 505 | 192–295 | 17.149444 | 17.082758 |
| Mixture | 12,288 | 293 | 210–289 | 18.299322 | 18.282040 |

Transferred experts means half the sum of absolute per-layer count differences from uniform. Reproduce these numbers as an implementation check. The proxy reductions are modest, especially for the mixture at 50%; large accuracy gains should not be assumed. This experiment tests whether even these modest reductions predict better behavior.

## 6. Initial experiment matrix: allocation versus ranking

At R=9,180, screen these six configurations, then confirm the leading candidate for each ranking on the remaining development tasks:

| ID | Ranking | Allocation |
|---|---|---|
| U-R | REAP | Uniform |
| T-R | REAP | tail-score |
| P-R | REAP | tail-mass |
| U-M | Existing lambda=0.75 mixture | Uniform |
| T-M | Existing lambda=0.75 mixture | tail-score |
| P-M | Existing lambda=0.75 mixture | tail-mass |

Compare T-R/P-R with U-R and T-M/P-M with U-M. This isolates allocation benefit. Then compare the overall winner with the strongest uniform result at that budget. Deduplicate identical retained masks before inference.

After choosing one allocation rule on development data, create one permutation control: shuffle that candidate's removal counts among layers with seed 42 while retaining each layer's own expert ranking. Preserve the count multiset and exact total. This tests whether the chosen layer identities matter beyond simply introducing variation. One permutation is a limited control, not a distributional significance test.

Record the old bounded inverse-REAP result as a historical reference. It is not the default policy. If reevaluated, use its actual saved mask rather than recomputing from an assumed rounded rate.

## 7. Implementation and routing correctness

Implement isolated scripts/modules where possible, preserving user changes already present in `src/reap/prune.py`, `src/reap/eval.py`, and the untracked non-uniform files. Reuse these helpers:

```text
scripts/materialize_qwen38_nonuniform.py
src/reap/model_util.py: slice_fused_expert_weights
src/reap/nonuniform_routing.py
src/reap/vllm_nonuniform.py
tests/test_nonuniform_routing.py
tests/test_vllm_nonuniform.py
```

The current materializer cannot accept arbitrary allocation masks; add a manifest-driven path or a separate materializer. Slice fused weights, every expert-indexed FP8 scale tensor, and router rows consistently. Always derive each independent candidate from the original checkpoint. Preserve expert identities in both original and compact indexing.

Keep top-k=10. Mask pruned/padded experts before both top-k and softmax, with probability normalization over valid selected experts. Keep shared experts and the rest of the model unchanged. Store compact artifacts as canonical; padded artifacts are a serving representation.

The existing mask patch loads static masks when workers initialize; it is not an implemented hot-swap API. Either use static checkpoint/server evaluation for the small candidate matrix or implement and verify explicit mask updates on all workers before using dynamic interventions. Disable/refresh prefix caches, KV caches and CUDA graphs as necessary when changing masks; results from different masks must not share stale state. Do not assume changing the JSON file updates live workers.

Before the experiment matrix:

- Test exact budgets, prefix identity, tie handling, zero/non-finite input handling, and a toy curve containing one very expensive expert. Check the allocator preserves it when a cheaper feasible allocation exists.
- Verify scales and weights refer to the same original expert IDs; inspect real checkpoint tensors, not just metadata counts.
- Verify full-mask/reference and uniform-mask/materialized outputs on fixed prompts within a declared precision tolerance; then verify short greedy generations. Resolve substantive differences before interpreting allocation results.
- Audit final result counts and task IDs explicitly. `src/reap/eval.py` currently catches some LiveCodeBench failures; successful process exit alone does not establish a completed evaluation.

Existing `scripts/validate_qwen38_nonuniform.py` asserts only the old 25%/50% totals. Update the validation path to use the manifest's exact R rather than that hard-coded set.

Use the established evaluation settings: TP=4, EP enabled, seed=42, greedy n=1, temperature=0, top-p=1, thinking disabled, model context 32,768; LiveCodeBench max generation 16,384 and execution timeout 120s. Record precision and software version. The current LiveCodeBench wrapper hard-codes the old date interval; add explicit date arguments for fresh confirmation or use the vendored runner with the same prompt and seed setup. Never label the old interval as fresh data.

## 8. Higher pruning at the declared target

Freeze the selected allocation method from development comparisons. Evaluate it and matched uniform rankings at these exact budgets:

| Requested rate | Exact removed R |
|---:|---:|
| Existing 37.35% point | 9,180 |
| 40% | 9,830 |
| 45% | 11,059 |
| 50% | 12,288 |

Rates other than the existing point use round(24,576 * requested_rate); report actual achieved percentages. Uniform counts may differ by one across layers. Do not use floor(512 * rate) independently per layer and call it a matched budget.

At each budget compare uniform and non-uniform with the same ranking, for both REAP and the existing mixture. This yields four configurations per budget after choosing the allocation method. Existing 50% uniform artifacts may be reused after provenance verification, but score them on the same newly frozen development/confirmation tasks as the candidate.

The lambda=0.75 50% uniform result is particularly relevant: its historical score is already 121/182. A new 50% allocation scoring 112 would not be competitive simply because it beats uniform REAP by one problem.

Use development results to bracket the target boundary, then add 42.5% (R=10,445) or 47.5% (R=11,674) when those intervals need refinement. If 50% meets the development target, test 55% (R=13,517), then 60% (R=14,746) as needed. These are bounded follow-ups, not an instruction to launch every rate immediately. Accuracy may be non-monotonic: a failed lower-budget point is not mathematical proof that every higher-budget point fails.

Select and freeze one finalist for goal 1 and one for goal 2 on development data. Evaluate both against their matched controls on historical and fresh confirmation data. Do not move to a different candidate because fresh confirmation failed and still treat the same set as untouched.

Report the measured accuracy-versus-pruning frontier. For goal 2, report the largest tested budget meeting the declared target; do not interpolate an untested success point or claim a global optimum.

## 9. Targeted check for a critical expert missed by the saved scores

Run this phase if offline allocations fail to improve development accuracy, or if a candidate's predicted proxy improvement conflicts with measured outcomes. This uses limited new inference, not new full calibration.

Choose four layers that donate the most capacity and four that receive the most relative to the matched uniform control, breaking ties by layer index. If fewer layers change, use those layers. In each chosen layer initially test four candidate deletions and four candidate restorations, one expert at a time in the current pruned model (at most 64 individual interventions).

Candidate deletion pool: retained experts with the lowest frozen ranking score and largest disagreements between ranking score and c, using segment disagreement to identify uncertain cases. Candidate restoration pool: removed experts with highest c, highest frozen ranking scores, and strongest segment disagreement. Take candidates from the entire removed set, not just the next expert at the boundary. Use a deterministic interleaving of these lists with deduplication and record the exact selection rule before inference. This is a shortlist, not an exhaustive importance audit.

Use a frozen small set of development prompts/reference code completions to compare completion-token NLL, saving per-problem losses. Cache teacher logits and measure KL only if a correct full-distribution or explicitly documented approximation is available; top-token logprobs alone are not exact KL. Confirm promising changes with execution pass@1 on development tasks. Do not use the final benchmark to choose experts.

For each intervention measure deletion cost or restoration benefit against the current model. Test promising deletion/restoration pairs jointly at the exact original global budget before accepting them. An individually safe deletion can become harmful after another deletion; remeasure after accepted changes. If small groups are used to reduce screening cost, split damaging groups and test members individually; neither block averages nor group splitting guarantee discovery of all interactions.

Label these variants `selection-refined`, since they alter which experts survive as well as layer counts. Include a same-layer deletion/restoration control where possible: if improvement can be obtained at unchanged layer counts, it is evidence for better expert selection, not for non-uniform allocation. Separate gains from selection and allocation in the final report.

If the shortlist fails, report that limitation and the next required evidence. Do not claim that aggregate calibration data proved all removed experts safe. A complete causal audit would require broader interventions or richer new observations.

## 10. Deliverables and completion criteria

Deliver under the new experiment root:

```text
experiment_manifest.json
input_audit.json
expert_statistics.csv
layer_removal_curves.npz
segment_consistency.json
allocations/<variant>/manifest.json
evaluations/<variant>/<split>/...
paired_comparisons.csv
accuracy_vs_pruning.csv
accuracy_vs_pruning.png
final_report.md
```

The report must answer:

1. Does non-uniform allocation improve accuracy over a matched uniform ranking at exactly 9,180 removals? Does it beat the strongest uniform control?
2. What is the highest tested pruning rate meeting the declared accuracy target, and how does that compare with uniform pruning?
3. Which layers gain/lose experts, how consistent is that across calibration segments, and do results support those choices?
4. Are gains explained by allocation, expert selection, or both? What happened to the permutation control and any individual expert probes?
5. What evidence is statistically inconclusive, and which proposed mechanisms remain unmeasured?

Report logical expert-parameter savings separately from actual serving memory and latency. Padding to the widest layer can erase part of the memory advantage; top-k remains 10, so fewer stored experts does not by itself imply fewer active expert FLOPs. Include compact/padded sizes if discussing deployment benefits.

A valid negative result completes a hypothesis test. Do not claim either user goal was achieved unless the corresponding accuracy/budget comparison supports it. Start by implementing the offline audit, curves, and manifests; run the six-configuration 37.35% development screen before spending compute on the larger budget sweep.
