# LiveCodeBench pruning comparison

## Evaluation protocol

All rows use Qwen3.8-Flash-Next-FP8 on the same frozen LiveCodeBench `release_latest` set: 182 tasks, contest dates 2025-01-01 through 2025-04-30 inclusive, and sorted task-ID SHA-256 `ad8b7e21195710bd04fbe4794013aeddb9be2172e3ee2309dbb0caaa2dea5383`. Decoding is seed 42, greedy, n=1, temperature 0, top-p 1.0, and max generation length 16,384 tokens; execution timeout is 120 seconds. All listed runs validated the 182 task IDs and one grade per task.

The base row uses the original FP8 checkpoint, with no pruning mask or expert compaction. Uniform REAP runs retain equal expert counts by layer. Uneven allocations use layer masks and redistribute router weights among retained experts. The inverse-sum variants allocate layer counts inversely to each layer's summed per-expert RMS-L2 score, then remove experts in ascending REAP order within each layer. The RMS-L2 DP variant instead minimizes the summed local-output RMS-L2 proxy subject to per-layer ascending-REAP-prefix constraints.

## Results

“Paired outcomes” are `base-only solved / pruned-only solved (ties)` on the same 182 tasks. Exact McNemar p-values compare each pruned run with the unpruned base.

| Pruning | Variant | Per-layer removals | Pass@1 | Δ vs. base | Paired outcomes | Exact p vs. base |
|---:|---|---|---:|---:|---:|---:|
| 0% | Unpruned base | 0; all 512 experts/layer | **126/182 (69.23%)** | — | — | — |
| 25% | Inverse layer-sum + REAP prefix | 26–372; cap 502 not reached | **111/182 (60.99%)** | −8.24 pp | 24 / 9 (149) | 0.0135 |
| 25% | Uniform REAP | 128/layer | 87/182 (47.80%) | −21.43 pp | 43 / 4 (135) | 2.78×10⁻⁹ |
| 25% | RMS-L2 DP + REAP prefix | 0–502; 7 layers at cap 502 | 67/182 (36.81%) | −32.42 pp | 61 / 2 (119) | 4.37×10⁻¹⁶ |
| 50% | Inverse layer-sum, cap 384 + REAP | 67–384; 14 layers at cap | **46/182 (25.27%)†** | −43.96 pp | 80 / 0 (102) | 1.65×10⁻²⁴ |
| 50% | Inverse layer-sum, cap 400 + REAP | 65–400; 13 layers at cap | 45/182 (24.73%) | −44.51 pp | 81 / 0 (101) | 8.27×10⁻²⁵ |
| 50% | Inverse layer-sum, cap 502 + REAP | 57–502; 6 layers at cap | 38/182 (20.88%) | −48.35 pp | 89 / 1 (92) | 1.47×10⁻²⁵ |
| 50% | Uniform REAP | 256/layer | 36/182 (19.78%) | −49.45 pp | 90 / 0 (92) | 1.62×10⁻²⁷ |

Each 25% run removes 6,144 of 24,576 experts; each 50% run removes 12,288. The cap-384 allocation was the highest-scoring measured 50% variant, while the 25% inverse-layer-sum allocation was the highest-scoring measured 25% variant. The paired differences between the cap-384 run and the other 50% variants were not statistically significant against one another (p=1.000 vs cap 400, 0.134 vs cap 502, and 0.110 vs uniform REAP).

† In the cap-384 evaluation, one isolated grading subprocess exceeded eight minutes of CPU time and was stopped. The task was recorded as a `TestRunnerError` and counted incorrect; all 182 task records still validated. Thus 46/182 is the recorded score, with at most one task of uncertainty on a clean rerun. Other rows above had no `TestRunnerError` records. The 25% RMS-L2 DP score is from its validated retry; the earlier attempt failed validation and is excluded. That DP retry used the same task-ID hash and stated evaluation settings, but an earlier evaluator source revision.

## Result artifacts

Artifacts are generated locally under `artifacts/` (git-ignored):

- Base metrics: `artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-12228-16384-unpruned/evaluations/lcb-base-unpruned-same-settings/results/Scenario.codegeneration_1_0.0_eval.json`
- Base per-task grades: same directory, `Scenario.codegeneration_1_0.0_eval_all.json`
- Run summaries, manifests, and complete result paths: `artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-12228-16384-unpruned/evaluations/`

For calibration details, allocation rationale, reproducibility commands, and the cap-384 per-layer count table, see [layerwise allocation pruning handoff](layerwise_allocation_pruning_handoff.md).
