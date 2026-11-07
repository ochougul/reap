# Layerwise Non-Uniform Pruning: Research Direction

## Executive summary

We evaluated several layerwise expert-pruning strategies for Qwen3.8-Flash-Next-FP8. The experiments separated two decisions:

1. Which experts to remove within each layer.
2. How to distribute the pruning budget across layers.

All experiments use the authoritative calibration artifact:

`artifacts/Qwen3.8-Flash-Next-FP8/full-packed-2048/calibration_state.pt`

The calibration dataset was `theblackcat102/evol-codealpaca-v1`.

Evaluation used the same 182-problem LiveCodeBench snapshot (`2025-01-01` through `2025-04-30`) with greedy decoding.

## Results at a glance

| Experiment | Layer-budget rule | Expert-ranking rule | Overall pruning | Accuracy |
|---|---|---|---:|---:|
| Baseline | None | None | 0% | **124/182 = 68.13%** |
| Stability uniform | Same pruning in every layer | Stability | 37.35% | **115/182 = 63.19%** |
| REAP uniform | Same pruning in every layer | REAP | 37.35% | **123/182 = 67.58%** |
| Stability non-uniform | Inverse `moe_residual_contribution` | Stability | 50% | **86/182 = 47.25%** |
| Stability + REAP-sum non-uniform | Inverse summed REAP | Stability | 50% | **98/182 = 53.85%** |
| Stability + REAP-sum, minimum 128 | Inverse summed REAP, minimum 128 experts/layer | Stability | 50% | **98/182 = 53.85%** |
| Stability + REAP-sum, minimum 200 | Inverse summed REAP, minimum 200 experts/layer | Stability | 50% | **112/182 = 61.54%** |
| Stability bounded | Inverse summed REAP, 25%–50% per layer | Stability | 37.35% | **120/182 = 65.93%** |
| REAP bounded | Inverse summed REAP, 25%–50% per layer | REAP | 37.35% | **118/182 = 64.84%** |

## Experiment progression

### 1. Uniform pruning controls

The first comparison held the layer budget constant and changed only the within-layer expert-ranking criterion.

- Uniform stability pruning: **63.19%**.
- Uniform REAP pruning: **67.58%**.

At the same 37.35% global pruning rate, REAP expert ranking was better than stability ranking.

### 2. Stability-ranked non-uniform pruning

We then allocated more pruning to layers considered less important. The initial layer-importance signal was inverse `moe_residual_contribution`; experts within each layer were still selected using stability.

At 50% global pruning:

- Minimum retained: 94 experts.
- Maximum retained: 467 experts.
- Accuracy: **47.25%**.

This showed that unconstrained layerwise allocation can make some layers too narrow.

### 3. Summed REAP as layer importance

We defined layer importance as:

```text
layer_importance[layer] = sum(per-expert REAP scores in that layer)
```

The inverse of this value determined the relative pruning allocation:

- High summed REAP -> more important layer -> prune less.
- Low summed REAP -> less important layer -> prune more.

With stability used for within-layer expert selection and 50% total pruning:

- Some layers retained only 92 experts.
- Accuracy improved to **53.85%**, but remained poor.

### 4. Minimum-retained constraints

We kept the same 50% global pruning budget while imposing a minimum number of retained experts per layer.

#### Minimum 128 experts

- Retained range: 128–415.
- Accuracy: **53.85%**.

This did not improve over the unconstrained summed-REAP allocation.

#### Minimum 200 experts

- Retained range: 200–399.
- Accuracy: **61.54%**.

Retained experts by layer:

```text
[204, 200, 200, 200, 200, 200, 200, 200,
 200, 200, 200, 200, 200, 200, 200, 235,
 200, 200, 200, 200, 274, 240, 215, 258,
 200, 200, 200, 200, 200, 200, 218, 260,
 263, 237, 237, 299, 391, 362, 392, 398,
 370, 379, 399, 342, 368, 375, 374, 398]
```

### 5. Bounded 25%–50% layerwise pruning

The strongest stability-based non-uniform design used independent per-layer pruning bounds:

- Minimum pruning per layer: 25%.
- Maximum pruning per layer: 50%.
- Layer allocation: inverse summed REAP.
- Expert selection: lowest stability scores.
- Overall pruning: 37.35%.

Each layer retained between 256 and 384 experts. This produced the best non-uniform result:

**120/182 = 65.93%**

### 6. REAP-ranked bounded pruning

The final experiment used REAP for both decisions:

- Layer allocation: inverse summed REAP.
- Within-layer expert ranking: lowest raw REAP scores.
- Per-layer pruning bounds: 25%–50%.
- Overall pruning: 37.35%.

Result:

**118/182 = 64.84%**

This was slightly below stability-ranked bounded pruning, but substantially better than unconstrained 50% non-uniform pruning.

Retained experts by layer:

```text
[319, 271, 269, 293, 313, 256, 316, 305,
 305, 300, 291, 277, 285, 276, 292, 329,
 313, 297, 300, 313, 342, 331, 322, 337,
 272, 270, 256, 257, 267, 283, 323, 338,
 339, 330, 330, 350, 381, 372, 382, 384,
 374, 377, 384, 365, 374, 376, 376, 384]
```

## Research conclusions

1. Layerwise allocation matters. Unconstrained 50% non-uniform pruning caused severe accuracy degradation because several layers became too narrow.

2. A minimum-retained-expert constraint is important. Raising the minimum from 128 to 200 materially improved accuracy.

3. Bounded non-uniform pruning is the strongest non-uniform design tested. Restricting every layer to 25%–50% pruning produced the best non-uniform result: **65.93%**.

4. For expert-level decisions, stability currently works better than REAP under the bounded non-uniform allocation:

   - Stability-ranked bounded: **65.93%**.
   - REAP-ranked bounded: **64.84%**.

5. For uniform pruning, REAP worked better than stability:

   - REAP uniform: **67.58%**.
   - Stability uniform: **63.19%**.

6. The best measured result remains uniform REAP pruning at 37.35%: **67.58%**, only 0.55 percentage points below the unpruned baseline.

## Recommended next direction

Use the following as the primary non-uniform configuration for follow-up work:

```text
Layer allocation:
    inverse summed REAP

Within-layer expert selection:
    stability score

Per-layer pruning bounds:
    25% minimum pruning
    50% maximum pruning

Global pruning:
    approximately 37.35%
```

This configuration achieved **120/182 = 65.93%**, the best non-uniform result.

The uniform REAP run should remain the primary control because it achieved the highest compressed-model accuracy at the same global pruning level: **123/182 = 67.58%**.

## Reproducibility artifacts

- Calibration artifact: `artifacts/Qwen3.8-Flash-Next-FP8/full-packed-2048/calibration_state.pt`
- Best non-uniform allocation metadata: `artifacts/Qwen3.8-Flash-Next-FP8/eval/reapsum-bounded-25to50/allocation_summary.json`
- Best non-uniform result: `artifacts/Qwen3.8-Flash-Next-FP8/eval/reapsum-bounded-25to50/livecodebench/Scenario.codegeneration_1_0.0_eval.json`
- REAP-bounded result: `artifacts/Qwen3.8-Flash-Next-FP8/eval/reap-reapsum-bounded-25to50/livecodebench/Scenario.codegeneration_1_0.0_eval.json`
- Minimum-200 allocation: `artifacts/Qwen3.8-Flash-Next-FP8/eval/stability-reapsum-nonuniform-min200-50/allocation_summary.json`

Serving used padded checkpoints with valid-expert routing masks. Padded experts were excluded from top-k selection, softmax, and probability redistribution; compact checkpoints remain the canonical pruning artifacts.
