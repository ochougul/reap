# Non-uniform expert pruning — TL;DR

## Why this idea came up

Qwen3.8-Flash-Next is a large mixture-of-experts model: it has 48 MoE layers, 512 routed experts per layer, and selects 10 experts per token. Uniform pruning removes the same number of experts from every layer, even though layers can differ in how important their experts are. The motivating question was whether we could keep more capacity in sensitive layers and remove more from tolerant layers, improving the accuracy-versus-pruning trade-off without changing the overall removal budget.

The strongest uniform control so far removed 9,180 of 24,576 experts (37.35%) using REAP ranking and scored 123/182 on the historical LiveCodeBench set, versus 124/182 unpruned. That small observed gap made layerwise allocation worth testing; it did not establish that non-uniform pruning would be better.

## What we are trying to achieve

Allocate an exact global number of removals across layers, then choose which experts to remove within each layer. The intended outcome is either:

- better coding accuracy than a matched uniform-pruning control at the same removal budget; or
- a larger removal budget while meeting a predeclared accuracy target.

The comparison must use matched prompts, tasks, routing, precision, and evaluation settings. Proxy scores generate candidates; only inference-and-execution benchmarks can show whether a candidate actually preserves coding accuracy.

## What the experiments say so far

- Unconstrained non-uniform allocation was risky: at 50% overall pruning, allocating by inverse MoE contribution and ranking experts by stability scored 47.25% (86/182).
- A bounded allocation based on inverse summed layer REAP, with 25%–50% pruning allowed per layer, did much better. Stability-ranked expert removal scored 65.93% (120/182); REAP-ranked removal scored 64.84% (118/182).
- The uniform REAP control remains stronger at the same 37.35% budget: 67.58% (123/182). Therefore, the current results do **not** show that non-uniform pruning improves accuracy over that control.

These are measured results on one historical 182-problem benchmark, not proof of generalization. They motivate a more controlled follow-up rather than a claim of a winning pruning rule.

## Current direction and caveats

The follow-up analysis reuses saved calibration statistics to construct per-expert removal curves and exact-budget layer allocations, comparing REAP and a REAP/stability mixture. It keeps allocations near the uniform control initially, uses deterministic tie-breaking, and plans development screening plus held-out confirmation. Contribution and stability statistics are heuristics: neither alone proves that an expert is safe to remove, and pairwise routing totals must not be mistaken for co-activation counts.

A separate in-progress calibration collects REAP, router stability, and local RMS-L2 MoE-output changes together on the same examples and model forwards. RMS-L2 measures the size of a local weighted-sum output change when a routed expert is swapped for the 11th candidate; it is not a loss-gradient estimate or an exact multi-expert accuracy effect. Its scores may help rank candidates, but benchmark evaluation remains necessary.

Finally, non-uniform pruning can save stored expert parameters, but a padded serving checkpoint may erase some storage savings. Since routing still selects 10 experts per token, fewer stored experts do not by themselves imply fewer active expert FLOPs or lower latency. Report model size and measured serving performance separately.

## Related notes

- [Research direction and historical results](layerwise_non_uniform_pruning_research_direction.md)
- [Calibration-reuse experiment plan](layerwise_pruning_reuse_calibration_execution_plan.md)
