# Handoff: Full GLM-4.5-Air Layerwise SEAP Calibration

## Objective

Run the full calibration pass for the composite coding calibration dataset, using
the latest routed-output callback implementation and the validated batch size of
32. The output will be used for non-uniform, layerwise MoE expert pruning.

## Repository and model

- Repository: `/local/mnt2/workspace/ochougul/reap`
- Model: `/local/mnt2/workspace/ochougul/artifacts/models/GLM-4.5-Air`
- GPUs: `0,1,2,3` (the runner requires all four GPUs and fails if one is absent)
- Python: use the repository virtual environment/interpreter that has the local
  Transformers installation.

## Transformers compatibility

This calibration runner is not compatible with a stock Transformers 4.55.0
installation. That release implements GLM-4.5-Air's routed experts as a plain
`ModuleList` loop and does not call the routed-output callback used by this
runner. Running the script without the callback integration can produce a
forward pass while silently omitting the per-routed-expert observations.

The calibration run was validated against the newer GLM MoE implementation in
Transformers 5.20.2, with one local callback addition in
`transformers/integrations/moe.py`. Therefore there is currently no released
Transformers version that makes this exact callback-based calibration work
fully out of the box: use Transformers 5.20.2 (or a compatible newer release)
and apply the callback patch described below.

To adapt the runner to the pinned project environment (Transformers 4.55.0),
port the same callback contract into
`transformers/models/glm4_moe/modeling_glm4_moe.py`: after each routed expert
forward, invoke `experts._reap_capture_callback` with
`(expert_ids, expert_outputs, permutation, top_k)`. The callback must receive
only active routed token/expert pairs, before router weighting is applied.
Validate the port with `tests/test_layerwise_seap.py` and a small calibration
run before using a full dataset.

## Latest implementation

The main runner is `experiments/seap-sharded.py`. It now:

1. Loads GLM-4.5-Air across all four GPUs using Accelerate's device map.
2. Reuses the model's existing grouped-MoE forward pass. The local Transformers
   package has a callback inserted in
   `.venv/lib/python3.12/site-packages/transformers/integrations/moe.py`, after
   expert down projection/post-expert normalization and before router weighting.
3. Collects only routed, active token-expert pairs; it does not recompute all
   experts and does not use manual expert blocking.
4. Collects routing frequencies, EAN/REAP/SEAP statistics, router stability, and
   layerwise MoE residual contribution ratios.
5. Displays a tqdm progress bar with batches, samples, and valid tokens.
6. Saves periodic checkpoints every 5,000 processed samples under
   `<output>/checkpoints/seap_state_5000.pt`, etc.

Important: the local Transformers patch is required for the routed-output
callback. Do not replace the environment with an unpatched Transformers version
before running.

## Calibration settings

Validated configuration:

- `SEAP_BATCH_SIZE=32`
- `SEAP_MODEL_MAX_LENGTH=1024`
- `SEAP_TRUNCATE=false` (overlong examples are skipped, matching REAP defaults)
- `SEAP_FULL_DATASET=true`
- `SEAP_MAX_SAMPLES=0`
- `SEAP_MAX_TOKENS=0`
- checkpoint interval: 5,000 samples
- seed: 42
- GPU memory budget: 54 GiB per GPU, no CPU offload

Batch-size smoke timings on 500 samples at 1,024 tokens:

| Implementation/configuration | Batch size | Calibration time |
|---|---:|---:|
| Old manual expert recomputation | 16 | approximately 2:33 per 100 samples at 512 tokens |
| New routed-output callback | 16 | approximately 5:06 per 500 samples at 1,024 tokens |
| New routed-output callback | 24 | 3:29 |
| New routed-output callback | 32 | 3:13 |

Batch size 32 completed without OOM and is the current recommended setting.
The full dataset is expected to contain about 105,413 usable samples at 1,024
tokens; approximately 5,859 of the 111,272 rows are skipped by
`truncate=false`. Expected calibration duration is approximately 11 hours 20
minutes, plus roughly one minute for model loading/startup.

## SEAP calculation

For each layer `l`, each token contributes the scalar residual ratio:

```text
r_l(token) = ||moe_output_l(token)||_2 /
             (||residual_hidden_state_l(token)||_2 + dtype_epsilon)
```

The layer contribution is the mean over valid, non-padding tokens:

```text
contribution_l = mean_token(r_l(token))
```

The observer stores the numerator and count as
`moe_residual_contribution_sum` and
`moe_residual_contribution_count`, then materializes
`moe_residual_contribution` at the end of calibration.

The per-expert SEAP score is computed as:

```text
normalized_reap_l,e = reap_score_l,e / max_e(reap_score_l,e)
seap_score_l,e = 0.5 * normalized_reap_l,e +
                 0.5 * stability_score_l,e(delta=2)
```

Experts with the lowest `seap_score` are pruned within each layer.

## Non-uniform layerwise pruning

`layerwise_seap` uses the global removal budget but allocates different numbers
of removals to different layers. Layer priorities are inverse contribution:

```text
priority_l = 1 / max(contribution_l, epsilon)
```

Lower-contribution MoE layers therefore receive a larger pruning allocation.
Integer allocations use deterministic capped/largest-remainder allocation and
always retain at least one expert per layer. The total number of removed experts
still exactly matches the requested global compression budget.

For GLM-4.5-Air there are 45 layers and 128 experts per layer, or 5,760 total
experts. A 50% global pruning target removes 2,880 experts, distributed across
layers according to the inverse residual-contribution priorities. Within each
layer, the allocated experts are the lowest-SEAP-score experts.

## Smoke-test evidence

The callback implementation was compared against the old implementation using
500 identical samples at 512 tokens:

- all 45 layers populated;
- expert and pairwise routing frequencies identical;
- residual contribution ratios bit-for-bit identical;
- all 45 layers selected the same 64 lowest-scoring experts in a 50%-per-layer
  ranking check;
- unit tests: `3 passed` in `tests/test_layerwise_seap.py`.

## Full-run command

Use the plain dataset name together with `SEAP_FULL_DATASET=true`. This makes the
loader exhaust the complete dataset rather than using the smoke-test composite
specification ending in `:125`.

```bash
cd /local/mnt2/workspace/ochougul/reap

CUDA_VISIBLE_DEVICES=0,1,2,3 \
SEAP_MODEL_PATH=/local/mnt2/workspace/ochougul/artifacts/models/GLM-4.5-Air \
SEAP_DATASET_NAME='theblackcat102/evol-codealpaca-v1' \
SEAP_SPLIT=train \
SEAP_OUTPUT_DIR=artifacts/seap/glm4.5-air-full-coding-b32-1024 \
SEAP_BATCH_SIZE=32 \
SEAP_MAX_SAMPLES=0 \
SEAP_MAX_TOKENS=0 \
SEAP_FULL_DATASET=true \
SEAP_MODEL_MAX_LENGTH=1024 \
SEAP_TRUNCATE=false \
SEAP_CHECKPOINT_INTERVAL=5000 \
SEAP_GPU0_MAX_MEMORY=54GiB \
SEAP_GPU1_MAX_MEMORY=54GiB \
SEAP_GPU2_MAX_MEMORY=54GiB \
SEAP_GPU3_MAX_MEMORY=54GiB \
SEAP_CPU_MAX_MEMORY=0GiB \
python -u experiments/seap-sharded.py
```

Before starting, verify all four GPUs are available with `nvidia-smi`. Monitor
the tqdm progress bar and preserve the output directory and checkpoints if the
run is interrupted. At completion, verify:

- `seap_state.pt`
- `seap_accumulators.npz`
- `metadata.json`
- periodic checkpoint files under `checkpoints/`
- all 45 layers have finite `moe_residual_contribution` and `seap_score` data.

## Instruction to the next agent

Please start the full composite coding-dataset calibration now using the command
above, keep the progress visible, and report the first checkpoint, completion
time, processed sample/token count, and final output path when the run finishes.
