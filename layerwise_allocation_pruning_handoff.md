# Layerwise allocation pruning handoff

## TL;DR

Per-expert RMS-L2 sensitivity measures the root-mean-square change in a layer's weighted MoE output when one routed expert is replaced by the eleventh candidate and the router weights are renormalized. For layer allocation, sum the 512 expert scores, give lower-sum layers a larger share of the pruning budget (inverse-sum allocation), cap removals per layer, and choose the actual experts within each layer by ascending REAP score.

## Why this method

The initial plan was to estimate pruning sensitivity from a loss gradient, but backward propagation through the model's FP8 grouped expert matmul was not available. Rather than use a teacher model or multiple full-model counterfactual forwards, the calibration run uses a forward-only local-output proxy. The proxy captures how much the weighted mixture at a MoE layer changes under a one-expert replacement. It is a pruning heuristic, not a coding-accuracy objective or an exact estimate of jointly removing many experts.

The RMS-L2 metric uses the **difference between two weighted output vectors**, not the difference between their scalar norms. It therefore responds both to output-magnitude changes and to direction changes (including rotations at unchanged vector length). These effects are combined in one scalar and are not reported separately. The implementation takes an L2 norm over hidden dimensions and then an RMS over token occurrences; it does **not** divide by hidden width:

\[
s_{\ell,e}=\sqrt{\frac{1}{N_{\ell,e}}\sum_{t:e\in\mathrm{Top10}(t)}
\left\|\mathbf y^{\,\mathrm{replace}(e\to11)}_{\ell,t}-\mathbf y_{\ell,t}\right\|_2^2}.
\]

Here, \(\mathbf y\) is the router-weighted sum of expert outputs for a token, the counterfactual replaces one of the selected ten experts with candidate 11, and the remaining router weights are renormalized. \(N_{\ell,e}\) is the number of calibration token occurrences for that expert in that layer. **Correction to a shorthand explanation given earlier:** the RMS is over occurrence-level squared L2 norms, not over the hidden coordinates themselves.

## Calibration protocol and data

- Model: unpruned `Qwen3.8-Flash-Next-FP8` (48 MoE layers, 512 experts/layer, top-10 routing).
- Dataset: `theblackcat102/evol-codealpaca-v1`, `train` split.
- Sampling: exactly 12,228 eligible rows, selected randomly without replacement with seed 42; selected row indices and token-ID hashes are recorded in the sample manifest.
- Sequence handling: maximum length 16,384; no truncation, packing, or padding; batch and forward-batch size 1.
- Run totals: 7,684,711 tokens; longest sequence 16,384; calibration took about 45,694 seconds (12.7 hours).
- In the same calibration run/forward, the saved state includes per-expert REAP and contribution statistics, router stability accumulators, MoE-output/residual-norm ratios, and local-output RMS-L2 scores. Router stability uses top-k 10, Gaussian router-input perturbation variance multiplier 1.0, boundary width 4, and seed 42; per-token events are not saved.
- Router stability and local-output counterfactuals add scoring around the calibration forward; they do not require a separate teacher model or backward pass.

Calibration artifact root (local, git-ignored):

`artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-12228-16384-unpruned/`

The metadata records sample-set SHA-256 `042789c5a55407f9205f6d8dbdfdbd446ea72b452442d2835f081e7d0901466a`. Detailed metadata is in `metadata.json`; selected examples are in `sample_manifest.json`; per-layer/per-expert state is in `calibration_state.pt` and the consolidated results in `local_output_sensitivity_results.pt`.

## How the allocation works

1. For each layer, sum all 512 per-expert RMS-L2 scores: \(S_\ell=\sum_e s_{\ell,e}\).
2. Allocate the requested global removal budget in proportion to \(1/S_\ell\). Lower-sum layers therefore receive more removals. If a layer reaches its cap, hold it at the cap and redistribute its excess quota among the remaining layers. Integer quotas use floor plus largest-remainder allocation so the global total is exact.
3. Having determined each layer's count, remove that many experts from the **ascending-REAP prefix within that layer** (ties by original expert ID). REAP ranks expert identities; it does not set the layer quotas.

Thus this is not a global ranking of every expert's RMS-L2 score. The layer quota comes from the inverse sum, while the per-layer expert list comes from REAP. The cumulative RMS-L2 curve is a visualization/proxy and is not an exact multi-expert damage curve.

## 50% allocation with 384 cap

The total budget is 12,288 of 24,576 experts. The per-layer cap is 384, so each layer retains at least 128 experts; 14 layers hit the cap. The realized count ranges from 67 to 384.

| Layer | Pruned | Layer | Pruned | Layer | Pruned | Layer | Pruned |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 316 | 1 | 384 | 2 | 184 | 3 | 303 |
| 4 | 256 | 5 | 136 | 6 | 259 | 7 | 384 |
| 8 | 384 | 9 | 296 | 10 | 384 | 11 | 384 |
| 12 | 384 | 13 | 384 | 14 | 384 | 15 | 237 |
| 16 | 277 | 17 | 352 | 18 | 384 | 19 | 350 |
| 20 | 186 | 21 | 169 | 22 | 221 | 23 | 182 |
| 24 | 384 | 25 | 244 | 26 | 384 | 27 | 384 |
| 28 | 384 | 29 | 384 | 30 | 292 | 31 | 204 |
| 32 | 208 | 33 | 276 | 34 | 291 | 35 | 226 |
| 36 | 141 | 37 | 133 | 38 | 120 | 39 | 99 |
| 40 | 96 | 41 | 91 | 42 | 67 | 43 | 203 |
| 44 | 103 | 45 | 108 | 46 | 214 | 47 | 72 |

The full CSV also includes layer score sums and experts retained.

## LiveCodeBench results

All reported evaluations use the same frozen set of 182 `release_latest` tasks in the inclusive date window 2025-01-01 through 2025-04-30 (sorted task-ID SHA-256 `ad8b7e21195710bd04fbe4794013aeddb9be2172e3ee2309dbb0caaa2dea5383`). Settings: seed 42, greedy decoding, n=1, temperature 0, top-p 1.0, maximum generation 16,384 tokens, and 120-second execution timeout. Uneven allocations use the existing per-layer expert-mask path (mask before top-k/softmax and renormalize over retained experts); uniform REAP uses native vLLM expert-parallel compaction.

| Pruning target | Allocation | Pass@1 |
|---:|---|---:|
| 25% | Uniform REAP | 87/182 (47.80%) |
| 25% | Inverse layer-sum, cap 502 | 111/182 (60.99%) |
| 50% | Uniform REAP | 36/182 (19.78%) |
| 50% | Inverse layer-sum, cap 502 | 38/182 (20.88%) |
| 50% | Inverse layer-sum, cap 400 | 45/182 (24.73%) |
| 50% | Inverse layer-sum, cap 384 | **46/182 (25.27%)** |

The 25% runs had 29 wins / 5 losses / 148 ties for inverse-sum vs uniform REAP (exact McNemar \(p\approx3.86\times10^{-5}\)). For the newest 50% cap-384 run, paired outcomes were:

| Cap-384 vs. | Wins | Losses | Ties | Exact McNemar p |
|---|---:|---:|---:|---:|
| Cap 400 | 12 | 11 | 159 | 1.000 |
| Cap 502 | 15 | 7 | 160 | 0.134 |
| Uniform REAP | 21 | 11 | 150 | 0.110 |

The cap-384 run is numerically best among the 50% variants, but these paired differences are not statistically significant at 0.05 on this task set. In that run, one isolated grading subprocess exceeded eight minutes of CPU time; it was stopped and recorded as a `TestRunnerError` (LiveCodeBench question ID `3765`), so that record counts as incorrect. The run still validated all 182 task records. Treat 46/182 as the recorded result, with at most a one-task uncertainty if that pathological grading case is rerun under a different timeout setup.

For comparison, the calibration proxy cost (sum of per-expert RMS-L2 scores for selected removals) at 50% was 3,165.366 for uniform REAP, 2,107.834 for inverse-sum cap 502, 2,252.241 for cap 400, and 2,288.441 for cap 384. These proxy totals are not exact multi-expert damage estimates; tighter caps constrain the inverse allocation and can increase the proxy cost.

## Cap-384 artifacts

These generated artifacts are in the local, git-ignored `artifacts/` tree; the small code/scripts and this handoff are what get pushed.

- Allocation graph: `artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-12228-16384-unpruned/evaluations/lcb-p50-inverse-sum-cap384/inverse_sum_rms_l2_pruned_experts_by_layer_p50_cap384.png`
- Allocation CSV: `artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-12228-16384-unpruned/evaluations/lcb-p50-inverse-sum-cap384/inverse_sum_rms_l2_pruning_allocation_p50_cap384.csv`
- Exact expert-ID manifest: `artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-12228-16384-unpruned/evaluations/lcb-p50-inverse-sum-cap384/allocation_manifest.json`
- Evaluation summary: `artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-12228-16384-unpruned/evaluations/lcb-p50-inverse-sum-cap384/summary.json`
- Validated task-level evaluation outputs: `artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-12228-16384-unpruned/evaluations/lcb-p50-inverse-sum-cap384/nonuniform-rmsl2-inverse-layer-sum-p50-r12288-cap384/results/`
- Other calibration plots: `artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-12228-16384-unpruned/local_output_sensitivity_rms_l2_overlaid_layers.png` and `artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-12228-16384-unpruned/rms_l2_summed_score_by_layer.png`.

## Code map and rerun pointers

- `experiments/qwen4_calibrate.py`: one-run calibration, REAP, router stability, and top-11 local-output RMS-L2 accumulation.
- `scripts/plot_layer_summed_rms_l2.py`: sum the 512 expert scores by layer and plot/export them.
- `scripts/plot_inverse_sum_rms_l2_pruning_allocation.py`: compute capped inverse-sum quotas and save the layer-allocation plot/CSV.
- `scripts/build_inverse_sum_rms_l2_allocation_manifest.py`: convert layer counts plus ascending-REAP prefixes into exact retained/removed expert IDs.
- `scripts/build_uniform_reap_allocation_manifest.py`: build uniform REAP comparison manifests.
- `scripts/run_layerwise_reuse_lcb.py`: freeze/check the task set, prepare model layout, run vLLM evaluation, and validate all outputs.
- `src/reap/vllm_nonuniform.py`: per-layer runtime mask for uneven retained counts.
- `src/reap/vllm_expert_compaction.py`: load-time router/expert compaction for uniform retained counts, enabling native expert parallelism without copying the full checkpoint.
- `src/reap/eval.py` and `src/sitecustomize.py`: select the appropriate mask or compaction integration for an evaluation run.

Typical allocation/manifest commands for the 50%-cap-384 case (from the repository root):

```bash
python scripts/plot_inverse_sum_rms_l2_pruning_allocation.py \
  --scores artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-12228-16384-unpruned/rms_l2_summed_score_by_layer.csv \
  --plot artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-12228-16384-unpruned/evaluations/lcb-p50-inverse-sum-cap384/inverse_sum_rms_l2_pruned_experts_by_layer_p50_cap384.png \
  --allocation-csv artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-12228-16384-unpruned/evaluations/lcb-p50-inverse-sum-cap384/inverse_sum_rms_l2_pruning_allocation_p50_cap384.csv \
  --pruning-fraction 0.5 --max-pruned-per-layer 384

python scripts/build_inverse_sum_rms_l2_allocation_manifest.py \
  --state artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-12228-16384-unpruned/calibration_state.pt \
  --metadata artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-12228-16384-unpruned/metadata.json \
  --allocation-csv artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-12228-16384-unpruned/evaluations/lcb-p50-inverse-sum-cap384/inverse_sum_rms_l2_pruning_allocation_p50_cap384.csv \
  --output artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-12228-16384-unpruned/evaluations/lcb-p50-inverse-sum-cap384/allocation_manifest.json \
  --pruning-fraction 0.5 --max-pruned-per-layer 384
```

The calibration weights/state, dataset cache, and evaluation outputs are machine-local prerequisites and are intentionally not versioned. Do not interpret this handoff as claiming coding quality beyond the measured LiveCodeBench slice or the RMS-L2 proxy as true multi-expert pruning damage.
