# Qwen3.8-Flash-Next-FP8 layerwise expert-pruning: final handoff

## TL;DR

- **Best result: inverse layer-sum of per-expert RMS-L2 (top-11 candidate), 111/182 LiveCodeBench at 25% pruning** (base 126, uniform REAP 87). It uses one forward-only calibration run and no damage curves.
- Anything that carries each layer's **output scale** lands within noise of it: absolute-L2 damage curves gave 103 (DP) and 105 (power 4, cap 256). A layer's REAP sum is the same signal (Spearman 0.999 with the RMS-L2 sum).
- Removing the scale hurts: **relative-L2** curves gave 98. Flattening the allocation with a router-stability term hurt too: **94**.
- **Correction to an earlier hypothesis:** extreme skew (14 layers at the 384 cap) is *not* by itself what fails. That DP scored 103, above uniform (p=0.02) and tied with inverse-sum (p=0.27). The RMS-L2 *DP* (67) is the only clearly bad allocator.
- n=182 tasks, one greedy run each: differences below about 10 tasks are not distinguishable. Nothing here beat inverse-sum with significance; the stability blend and relative-L2 runs lost to it significantly.

## Method logic (TL;DR)

| Idea | Logic |
|---|---|
| Per-expert RMS-L2 (top-11) | For each token and each selected expert e in the top-10, swap e for the 11th-ranked expert, renormalize router weights, and measure the L2 change of the layer's weighted MoE output. Score = RMS over token occurrences. Forward-only, no backward, no teacher. |
| Inverse layer-sum allocation | Layer score S = sum of its 512 expert scores. Quota proportional to 1/S (low-S layers lose more), capped per layer, largest-remainder rounding to the exact budget. |
| Experts within a layer | Remove in ascending-REAP order (stable by expert ID). REAP picks identities; RMS-L2 only sets per-layer counts. |
| Damage curves | Remove the k lowest-REAP experts of one layer (k = 0..384 step 32), measure local output damage. Used as cost curves for an exact-budget DP over layers. |
| Why scale matters | Absolute damage = relative damage x output norm. Relative curves treat tiny-output mid layers and huge-output late layers alike and push cuts into late layers. |
| Power / cap / blend | Cost^p (p>1) spreads removals; a lower cap bounds concentration; blend mixes counts with uniform. |
| Stability blend | Score = 0.5*stability/mean + 0.5*REAP-sum/mean; stability = fraction of selected experts still selected under router perturbation. |

## Experiments

LiveCodeBench: frozen 182-task set, 2025-01-01..2025-04-30, greedy n=1, seed 42, 16,384 max tokens, 120 s timeout. "vs inv-sum" = tasks only inverse-sum solved / only the row solved, exact McNemar p.

### 25% pruning (6,144 experts)

| Allocation | Proxy / settings | Layer counts | Pass@1 | vs inv-sum (p) |
|---|---|---|---:|---|
| Unpruned base | n/a | 0 | **126** | n/a |
| **Inverse layer-sum (best)** | RMS-L2 sum, cap 502 | 26-372 | **111** | n/a |
| Abs-L2 damage, power 4 | cap 256 | up to 256 | 105 | 19/13 (0.38) |
| Abs-L2 damage DP | power 1, cap 384, 14 layers at cap | 0-384 | 103 | 24/16 (0.27) |
| Relative-L2 damage, power 4 | cap 256 | up to 256 | 98 | 23/10 (0.035) |
| Inverse 0.5 stability + 0.5 REAP | cap 502 | 58-183 | 94 | 28/11 (0.0095) |
| Uniform REAP | 128/layer | 128 | 87 | not recorded here |
| RMS-L2 DP | min summed proxy | skewed | 67 | not recorded here |

By difficulty (easy/medium/hard, of 45/55/82): base 45/43/38, inv-sum 44/34/33, abs256 41/34/30, rel256 43/34/21, stability 41/33/20, uniform 44/26/17. Hard tasks separate the runs.

### 50% pruning (12,288 experts)

| Allocation | Pass@1 |
|---|---:|
| Inverse layer-sum, cap 384 | 46 |
| Inverse layer-sum, cap 400 | 45 |
| Inverse layer-sum, cap 502 | 38 |
| Uniform REAP | 36 |

### Generated but not evaluated

| Allocation | Notes |
|---|---|
| Abs-L2, power 4, cap 384 | Pearson 0.95 with inverse-sum; closest damage-curve allocation to it |
| Relative-L2, power 4, cap 384 | Not evaluated; the cap-256 version scored 98 |
| Relative-L2, cross-layer p90 winsorized, power 4, cap 384 | Pearson -0.06 vs inverse-sum |
| Relative x REAP layer sum (`--reap-layer-scale 1`) | Previewed only (not saved as a manifest) |
| Abs-L2 p90-winsorized DP | Manifest exists; no evaluation recorded |

## Takeaways (TL;DR)

1. **Scale-aware layer score + REAP prefix is the robust recipe.** Inverse-sum, abs-L2 DP and abs-L2 power 4 are within noise of each other (103-111).
2. **Relative-only metrics fail**: they erase the layer-scale information (98 and a depth profile opposite to inverse-sum).
3. **Flattening toward uniform loses accuracy**: stability blend 94 and uniform 87 versus 111 for concentrated mid-layer pruning (layers 7-14, 24-29 pruned most; 36-47 least).
4. Open, untested: cap layers 0/39/45/46 at inverse-sum counts (abs-L2 allocations prune them harder); an end-to-end KL probe per layer; a rerun of inverse-sum to measure run-to-run spread.

## How to reproduce the best run (inverse-sum, RMS-L2 top-11, no damage curves)

All paths relative to the repo root; `CAL=artifacts/Qwen3.8-Flash-Next-FP8/local-output-sensitivity-evol-codealpaca-12228-16384-unpruned`.

**0. Calibration (about 12.7 h; skip if `$CAL` already exists).** `experiments/qwen4_calibrate.py` with `QWEN4_SENSITIVITY_MODE=1` fixes: dataset `theblackcat102/evol-codealpaca-v1` train, 12,228 random rows (seed 42), max length 16,384, no truncation/packing/padding, batch 1, sample-set SHA-256 `042789c5a55407f9205f6d8dbdfdbd446ea72b452442d2835f081e7d0901466a`. It writes `calibration_state.pt` (REAP, per-expert `local_output_sensitivity_rms_l2`) and `local_output_sensitivity_results.pt`. The exact multi-GPU launch command is not recorded in the repo.

```bash
QWEN4_SENSITIVITY_MODE=1 PYTHONPATH=src .venv-eval/bin/python experiments/qwen4_calibrate.py
```

**1. Per-layer sums of the 512 expert RMS-L2 scores.**

```bash
python scripts/plot_layer_summed_rms_l2.py     # writes $CAL/rms_l2_summed_score_by_layer.csv (+ png)
```

**2. Capped inverse-sum quotas (25%, cap 502).**

```bash
python scripts/plot_inverse_sum_rms_l2_pruning_allocation.py \
  --scores $CAL/rms_l2_summed_score_by_layer.csv \
  --plot $CAL/evaluations/lcb-p25-rms-l2-inverse-sum/inverse_sum_rms_l2_pruned_experts_by_layer.png \
  --allocation-csv $CAL/evaluations/lcb-p25-rms-l2-inverse-sum/inverse_sum_rms_l2_pruning_allocation.csv \
  --pruning-fraction 0.25 --max-pruned-per-layer 502
```

**3. Expert-ID manifest (ascending-REAP prefix per layer).**

```bash
python scripts/build_inverse_sum_rms_l2_allocation_manifest.py \
  --state $CAL/calibration_state.pt --metadata $CAL/metadata.json \
  --allocation-csv $CAL/evaluations/lcb-p25-rms-l2-inverse-sum/inverse_sum_rms_l2_pruning_allocation.csv \
  --output $CAL/evaluations/lcb-p25-rms-l2-inverse-sum/allocation_manifest.json \
  --pruning-fraction 0.25 --max-pruned-per-layer 502
```

Check: total 6,144; per-layer counts begin `125, 155, 73, 120, 101, 54, 102, 233, 325, ...`, max 372. The original manifest's SHA-256 is `32554eddf6886246d31a86954277005e29d51902ea3d35f35feb34ab8d3c67f8`; regenerated bytes may differ only if the script embeds paths or timestamps, so compare the counts first.

**4. Evaluate (vLLM, masked original expert layout).** The runner takes manifest paths, an output root that must not already exist, and the original FP8 checkpoint. The argument list below comes from `scripts/run_layerwise_reuse_lcb.py --help`; I did not recover the exact original command line.

```bash
.venv-eval/bin/python scripts/run_layerwise_reuse_lcb.py \
  $CAL/evaluations/lcb-p25-rms-l2-inverse-sum/allocation_manifest.json \
  --output-root $CAL/evaluations/lcb-p25-rms-l2-inverse-sum-rerun \
  --checkpoint /home/ubuntu/hf_cache/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13 \
  --cuda-visible-devices 0,1,2,3
```

Expected: 111/182 give or take a few tasks (vLLM numerics are not bit-stable; the cap-384 50% run also recorded one grading timeout).

## Reproducing the other allocations

| Allocation | Command (from repo root) |
|---|---|
| Damage curves | `experiments/qwen4_damage_curves.py` (see its argparse/docstring for settings; output `artifacts/.../reap-damage-curves-evol-codealpaca-12228-16384-unpruned/curve_summary.csv`) |
| Abs-L2 pow 4 cap 256 | `python experiments/qwen4_allocate_damage.py --curves <curve_summary.csv> --total-pruned 6144 --proxy mean_absolute_l2 --power 4 --max-pruned-per-layer 256 --output out.json --plot out.png` |
| Relative-L2 | same with `--proxy mean_relative_l2` |
| Winsorized | add `--winsorize-percentile 90` |
| REAP-scale multiplicative | add `--reap-layer-scale 1` |
| Blend with uniform | add `--blend-lambda 0.5` |
| Stability + REAP inverse-sum | `python scripts/build_stability_reap_inverse_sum_allocation.py --output manifest.json` |

The allocator writes layer counts plus REAP-prefix IDs under `layers`. The evaluator manifests used here also carry `per_layer_removal_counts` and `removed_expert_ids_original` (the keys `run_layerwise_reuse_lcb.py` reads); the evaluator-ready files are in each `evaluations/lcb-*` directory. Use `/usr/bin/python` for the allocator and plots if the repo venv lacks matplotlib.

## Caveats

- One greedy run per allocation on 182 tasks; run-to-run spread was not measured.
- The local-output proxies measure one-layer, one-expert (RMS-L2) or one-layer, k-expert (damage curves) effects, not joint multi-layer damage.
- Results are for this LiveCodeBench slice only.

## Code map

- `experiments/qwen4_calibrate.py`: single-run calibration (REAP, router stability, top-11 RMS-L2) and the damage-curve mode.
- `experiments/qwen4_damage_curves.py`, `experiments/qwen4_validate_damage_proxy.py`: damage curves and proxy validation.
- `experiments/qwen4_allocate_damage.py`: exact-budget DP allocator with `--power`, `--max-pruned-per-layer`, `--blend-lambda`, `--winsorize-percentile`, `--reap-layer-scale`.
- `scripts/build_stability_reap_inverse_sum_allocation.py`: stability + REAP inverse-sum allocation.
- `scripts/plot_layer_summed_rms_l2.py`, `scripts/plot_inverse_sum_rms_l2_pruning_allocation.py`, `scripts/build_inverse_sum_rms_l2_allocation_manifest.py`, `scripts/run_layerwise_reuse_lcb.py`: the best-run pipeline above.
- Earlier docs: `layerwise_allocation_pruning_handoff.md`, `livecodebench_pruning_comparison.md`.
