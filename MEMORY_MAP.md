# REAP memory map

This is the maintained working memory for this repository. Update it whenever
we change code, artifacts, experiment status, assumptions, or verification.
Record the commit, date, evidence, and any remaining uncertainty. Prefer the
implementation and executable tests over historical handoff documents when
they disagree.

## Snapshot

- Recorded: 2026-10-16
- Branch: `router-stability-4gpu`
- HEAD: `1cac49c` — `Add full GLM calibration state for SEAP experiments`
- Merge base with `main`: `1970473` (`Add layerwise calibration observer (#17)`)
- Branch delta: 12 commits, 31 changed paths, 2,866 insertions and 20 deletions.
- Working tree when recorded: only untracked `.zsh_history`; it is unrelated
  shell history and must not be committed.

## Purpose of the branch

The branch adds two related research workflows for GLM-4.5-Air MoE experts:

1. **Router-stability collection**: measure how routing changes under
   variance-scaled Gaussian perturbations.
2. **SEAP scoring**: combine per-expert REAP saliency with routing stability,
   then prune and evaluate saved score states reproducibly.

The intended research question is whether the stability signal selects a
meaningfully different and better set of experts than REAP alone.

## Change history from `main`

| Commit | Change |
| --- | --- |
| `30ad9e1` | Added router-stability collection. |
| `7a3f7ca` | Added resident, sharded four-GPU router runner. |
| `e83562a` | Documented the router-stability workflow. |
| `ddd316c` | Added four-GPU SEAP calibration/scoring and analysis utilities. |
| `7037613` | Added a continuation handoff. |
| `8e1e089` | Added a verified SEAP prune/evaluation workflow. |
| `64a2d85` | Split pruning and evaluation into separate processes to release CUDA state before vLLM. |
| `0092ece`, `56fdfef` | Aligned SEAP sampling defaults with REAP calibration. |
| `f989f8f` | Added the full-dataset calibration mode. |
| `bde6d4a` | Added portable configuration, offline lambda/delta sweeps, and `SEAP_EXPERIMENTS.md`. |
| `1cac49c` | Added the completed full GLM calibration state. |

## Current implementation map

| Area | Files | Current behavior |
| --- | --- | --- |
| Router perturbation | `src/reap/router_stability.py` | Executes the original route plus four zero-mean Gaussian perturbations with relative input-variance multipliers `0.25, 0.5, 1.0, 2.0`. It preserves GLM group eligibility and correction bias when calculating ranking values. |
| Router output | `RouterStabilityCollector` in the same file | Streams token/layer events to Parquet and full-expert aggregate arrays to NPZ. Per-token output retains top-8 selected experts plus four boundary experts. |
| SEAP observation | `src/reap/seap.py` | Computes full, one-based router ranks for experts selected by the original route. A perturbed expert made ineligible receives rank `num_experts + 1`. |
| SEAP accumulation | `SeapCollector` in the same file | For selected expert `i`, accumulates `exp(-max(abs(perturbed_rank - original_rank) - delta, 0))`; averaged scores are in `[0, 1]`. Scores are retained for deltas 2 and 4. |
| SEAP formula | `src/reap/seap.py`, `scripts/build_seap_states.py` | **Authoritative convention:** `seap_score = lambda * normalized_reap + (1 - lambda) * stability`. Therefore `lambda=1` is REAP-only and `lambda=0` is stability-only. |
| Layerwise path | `src/reap/args.py`, `observer.py`, `layerwise_observer.py`, `layerwise_prune.py` | Adds CLI/config fields for router-stability and SEAP collection, and saves their state after calibration. |
| Resident four-GPU calibration | `experiments/seap-sharded.py` | Shards GLM-4.5-Air across GPUs 0–3, hooks each MoE block, collects full-rank stability and REAP activations, then writes `seap_state.pt`, accumulators, and metadata. Environment variables configure model, dataset, split, batch size, full-dataset mode, truncation, token cap, seed, and output path. |
| Offline state sweeps | `scripts/build_seap_states.py` | Derives any lambda/delta state from one calibration state without another model forward pass. |
| Pruning/evaluation | `experiments/prune_seap_state.py`, `src/reap/prune.py`, `src/reap/eval.py` | Selects `seap_score` or REAP, removes the lowest scores, creates a checkpoint, and evaluates it in a fresh child process. Temporary checkpoints are removed after evaluation unless explicitly saved. |
| Tests | `tests/test_seap.py`, `tests/test_router_stability.py` | Cover rank collection, score combination, perturbation tensor shapes, token caps, and router aggregate update behavior. |

## Calibration artifact currently committed

`artifacts/seap/glm4.5-air-full-dataset/` is versioned by commit `1cac49c`:

- `metadata.json` reports GLM-4.5-Air calibration on
  `theblackcat102/evol-codealpaca-v1`, train split.
- It reports full-dataset mode, batch size 1, no truncation, sequence length
  2,048, seed 42, deltas 2 and 4, variance multiplier 2, and 50,116,980 valid
  tokens.
- `seap_state.pt` and `seap_accumulators.npz` are present. Their tensors have
  not been independently loaded in this environment because neither `uv` nor
  PyTorch is installed here.

The latest commit establishes that calibration as complete. It does **not**
establish completed pruning or benchmark results from that full state.

## Operational workflow

1. Calibrate with `experiments/seap-sharded.py`, normally using four GPUs and
   `SEAP_FULL_DATASET=true` for a complete data pass.
2. Generate comparable score states with `scripts/build_seap_states.py` for
   chosen deltas and lambdas.
3. Use `experiments/prune_seap_state.py` once with `--prune-method reap` as
   the baseline and once per SEAP state with `--prune-method seap`.
4. Keep model, calibration data, compression ratio, seed, and evaluation task
   settings identical across comparisons.
5. Record task metrics, pruned expert identities, REAP/SEAP rank correlation
   and overlap, runtime, GPU memory, and exact calibration metadata.

## Planned experiment: Qwen3.5 mathematical reasoning

- Root: `artifacts/seap/qwen3.5-122b-a10b-tulu-3-sft-personas-math/`
- Model: `Qwen/Qwen3.5-122B-A10B` at revision
  `dc4d348443bc740c68e2d77492492c11606384d5`.
- Dataset: `allenai/tulu-3-sft-personas-math`, train split, revision
  `46add032f5203fd96efbd54d074198981ed57d25` (149,960 rows).
- The model has 48 text MoE layers, 256 routed experts, top-k 8, and one
  shared expert. The shared expert must not be pruned.
- This model uses `Qwen3_5MoeForConditionalGeneration`. The branch now has a
  Qwen3.5 router adapter, packed-expert REAP accumulation, a dedicated
  four-GPU runner, and packed-expert pruning support. A compatible environment
  and smoke calibration remain required before a full run.
- [The experiment plan](artifacts/seap/qwen3.5-122b-a10b-tulu-3-sft-personas-math/PLAN.md)
  defines the calibration, offline lambda/delta sweep, mathematical evaluation,
  statistics, and graph outputs.

## Known inconsistencies and risks

- The lambda formula was inconsistent across the handoff and score design
  notes. On 2026-10-16 both documents were corrected to match source:
  `lambda=1` is REAP-only and `lambda=0` is stability-only.
- `SEAP_EXPERIMENTS.md` says calibration states should remain outside Git, but
  the full GLM state is committed. Treat this as an intentional, current branch
  fact until the repository policy is changed deliberately.
- The resident runner defaults `SEAP_MODEL_PATH` to a machine-specific
  `/local/mnt2/...` path. Set `SEAP_MODEL_PATH` explicitly elsewhere.
- The full state has only metadata-level verification in this workspace. Run
  the SEAP/router test files and load every layer’s tensors after provisioning
  the project environment.
- The historical handoff reports a smaller 500k-token experiment and no SEAP
  evaluation at that time. It cannot be used as evidence for results from the
  later full-dataset state.

## Verification record

| Date | Check | Result |
| --- | --- | --- |
| 2026-10-16 | Git history, branch diff, tracked artifacts, source, and documentation inspected | Completed. |
| 2026-10-16 | `uv run pytest -q tests/test_seap.py tests/test_router_stability.py` | Not run: `uv` is absent. |
| 2026-10-16 | Load full calibration tensors with PyTorch | Not run: system Python has no `torch`; no project virtual environment was found. |

## Update log

| Date | Update | Evidence |
| --- | --- | --- |
| 2026-10-16 | Created this map from the branch diff, source, metadata, commit history, and existing handoff/docs. Flagged stale lambda documentation. | `1cac49c`; current source in `src/reap/seap.py` and `scripts/build_seap_states.py`. |
| 2026-10-16 | Corrected the lambda formula and endpoint labels in `seap_score.md` and `handoff_seap_full.md`. | Matches `src/reap/seap.py`, `scripts/build_seap_states.py`, and `SEAP_EXPERIMENTS.md`. |
| 2026-10-16 | Created the planned Qwen3.5 × Tulu Personas Math experiment root and pinned its model/dataset revisions. | Hugging Face model and dataset metadata; experiment `PLAN.md`. |
| 2026-10-16 | Began Qwen3.5 support: exact top-k router weights, sparse packed-expert REAP accumulation, Qwen3.5 pruning, dataset revision pinning, and the four-GPU calibration runner. | Upstream Transformers Qwen3.5 implementation and local source changes. |
| 2026-10-16 | Provisioned `.venv-qwen35` (PyTorch 2.8.0+cu128; Transformers 5.20.0) and downloaded the pinned 39-shard Qwen3.5-122B-A10B checkpoint (234 GB) to `artifacts/models/Qwen3.5-122B-A10B`. | Four A100 80 GB GPUs visible; config/model preflight reports `qwen3_5_moe`, 48 MoE layers, 256 routed experts, top-k 8. |
| 2026-10-16 | Qwen3.5 support implementation and focused tests completed. | `.venv-qwen35/bin/python -m pytest -q tests/test_pruning_metrics.py tests/test_router_stability.py tests/test_seap.py`: 8 passed. |
| 2026-10-16 | Qwen3.5 smoke calibration completed and reporting/sweeps were generated. It processed 2 retained rows / 4,093 tokens (all remaining source rows were intentionally excluded by smoke limits). | `smoke/calibration/{seap_state.pt,seap_accumulators.npz,metadata.json}`, 10 offline sweep states, CSV/JSON reports, 48 layer plots, summary and overlap plots. |
| 2026-10-16 | Independently validated smoke state. | 48 layers; all five score vectors are finite `[256]`; stability scores are in `[0,1]`; delta 4 is elementwise at least delta 2. |
| 2026-10-16 | Started the full Qwen3.5 calibration in tmux session `reap-qwen35-full` after smoke validation. The first managed-shell launch was interrupted while it was still tokenizing and packing the full dataset; no full-calibration state had been written. | `tmux list-sessions`; live log `/tmp/reap-qwen35-full.log`. |
| 2026-10-16 | Corrected the full-run tmux launch. The first tmux attempt used a login shell that reset `HOME` to `/usr2/amitraj`, causing a Hugging Face cache permission error before dataset preparation. The new `reap-qwen35-full` session inherits the workspace cache path and has `remain-on-exit` enabled. | Pane process `experiments/seap-qwen35-sharded.py` live after restart; `/tmp/reap-qwen35-full.log`. |
| 2026-10-16 | Restarted the full calibration under `reap-qwen35-full` with `HOME=/local/mnt/workspace/amitraj` and explicit repository-local Hugging Face cache directories. | Session created 06:59 PDT; model loading active with 35–68 GB allocated per A100. |
| 2026-10-16 | Added a noninvasive 15-minute watchdog in tmux session `reap-qwen35-monitor`; it records calibration-process presence, log freshness, GPU allocation/utilization, and workspace disk space. | `/tmp/reap-qwen35-monitor.log`; initial check found the full calibration process live. |
| 2026-10-16 | Full calibration dataset preparation completed: the runner packed all 149,960 source samples, then entered the model-forward stage. | Full-run pane reports the final packing messages and optimized-kernel fallbacks; process is live at ~800% CPU with resident model memory on all four GPUs and observed rotating GPU compute activity. |
| 2026-10-19 | Sampled 1,000 uniformly selected calibration rows with the pinned Qwen tokenizer to refine workload estimation. Mean chat-templated length was 1,392 tokens (median 1,306; p90 1,909; p99 2,522), estimating 208.8M tokens or approximately 102,000 packed 2,048-token sequences. | Low-priority local sample using the pinned dataset revision and local checkpoint tokenizer. |
| 2026-10-21 | Fetched Onkar’s callback-based calibration work (`7902354`) and Qwen4 pilot (`b9f2973`). The callback path avoids recomputing routed expert outputs and supports checkpoints/progress, but its released implementation targets GLM; the Qwen4 pilot uses a different model class and still manually derives Qwen expert outputs. It cannot replace the active Qwen3.5 runner unchanged. | `git fetch --all --prune`; inspected both commits and Qwen3.5’s installed packed-MoE forward implementation. |
| 2026-10-21 | Stopped the original full Qwen3.5 calibration at the user’s request after approximately 135.8 hours, before it emitted a checkpoint or final state. Its log was preserved at `/tmp/reap-qwen35-full-stopped-20261021.log`. | tmux pane exited with SIGINT; all four GPUs returned to 0 MiB use. |
| 2026-10-21 | Implemented Qwen3.5 routed-output capture, eliminating the runner’s duplicate selected-expert projections. Focused tests passed (8), and real-model callback smoke states were finite across all 48 layers with the delta invariant preserved. Batch-size pilots completed at 1, 2, 4, and 8; batch 8 was selected for the replacement full run. | `fast-smoke`, `fast-batch2`, `fast-batch4`, and `fast-batch8` artifacts; batch-8 one-batch pilot completed in 66.08 seconds including 31 seconds of model loading. |
| 2026-10-21 | Started replacement full Qwen3.5 calibration in tmux session `reap-qwen35-full`, using callback capture and batch size 8. | `callback-b8.env`; live log `/tmp/reap-qwen35-callback-b8.log`. |
| 2026-10-21 | Stopped the replacement batch-8 callback calibration at the user’s request before any calibration result was accepted, freeing all GPUs for separate experiments. | tmux `reap-qwen35-full` interrupted with SIGINT; GPU use returned to 0 MiB. |
| 2026-10-21 | Stopped the Qwen3.5 calibration monitor and removed the stopped calibration tmux pane so separate experiments can use the machine without stale experiment sessions. Calibration remains paused; no restart is authorized. | `tmux kill-session -t reap-qwen35-monitor`; `tmux kill-session -t reap-qwen35-full`; all four A100 GPUs report 0 MiB and 0% utilization. |
| 2026-10-21 | Completed an exact length scan of all 149,960 pinned Tulu Personas Math rows with the local Qwen3.5 tokenizer and its chat template. Calibration sequences have median 1,327 tokens (mean 1,400.8; p90 1,923; p95 2,171; p99 2,814). At a 1,792-token non-truncating cap, 128,206 rows (85.49%) and 77.28% of calibration tokens remain; at 1,536, 105,432 rows (70.31%) and 59.39% of tokens remain. | `length-analysis/summary.json`, `retention_by_max_length.csv`, and `retention_by_max_length.png`; reproducible utility `experiments/analyze_qwen35_sequence_lengths.py`. |
| 2026-10-21 | Four-GPU one-batch capacity pilots passed with the callback calibration path: batch 10 at a 1,792-token cap processed 17,913 valid tokens; batch 12 at a 1,536-token cap processed 18,424 valid tokens. Both runs exited cleanly and released all GPU memory. These are capacity checks, not accepted calibration states. | `capacity-pilot-b10-l1792/calibration/logs/run_summary.json`; `capacity-pilot-b12-l1536/calibration/logs/run_summary.json`; all A100s at 0 MiB after completion. |
| 2026-10-21 | Validated 2,048-token batch capacity with actual 100-row callback calibration smokes. With PyTorch's standard allocator, batch 8 and 9 both failed on their second forward because fragmented reserved blocks prevented a 7.58 GiB / 8.53 GiB allocation. With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, batch 10 processed all 100 rows in 10 forwards, wrote a finite 48-layer state, and took 217.24 seconds. Peak memory was 80,035 MiB (78.159 GiB, 98.624%) on GPU 2, leaving 1,117 MiB (1.091 GiB). Batch 10 is the largest validated 2,048-token setting; batch 11 has no practical headroom. | `smoke100-b10-l2048-expandable/smoke_report.json`, `gpu_peak_summary.csv`, `gpu_memory_over_time.png`, and runner log; prepared but did not launch `callback-b10-l2048.env`. |
| 2026-10-21 | Added per-GPU mapper limits and validated the balanced 2,048-token batch-10 configuration without `PYTORCH_CUDA_ALLOC_CONF`: GPU limits 59/64/64/64 GiB map contiguous decoder layers 0–10, 11–23, 24–36, and 37–47. The 100-row smoke passed all 10 forwards in 207.94 s with finite 48-layer scores. Peaks were 77,945 / 75,969 / 76,131 / 77,095 MiB, leaving at least 3.13 GiB on every GPU. | `smoke100-b10-l2048-map59-64/smoke_report.json` and `gpu_peak_summary.csv`; prepared but did not launch `callback-b10-l2048-balanced.env`. |
| 2026-10-21 | Increased the balanced 2,048-token batch capacity to 11. A 100-packed-sequence smoke completed all 10 forwards in 207.84 seconds with finite score vectors. Peak GPU use was 80,265 / 77,361 / 77,477 / 79,345 MiB; GPU 0 retains 887 MiB. Batch 12 is not viable with this full-length workload because the additional 2,048-token sequence exceeds that margin. | `smoke100-b11-l2048-map59-64/smoke_report.json` and `gpu_peak_summary.csv`; prepared but did not launch `callback-b11-l2048-balanced.env`. |
| 2026-10-21 | Tested batch 12 with the balanced 59/64/64/64 GiB map and no allocator override. Its first 2,048-token packed forward passed, but the second failed: GPU 0 needed 11.37 GiB with 11.07 GiB free and 11.94 GiB reserved but unallocated. Batch 11 remains the highest stable configuration under the requested allocator policy. | `smoke100-b12-l2048-map59-64/calibration/logs/runner.log`; all GPUs returned to 0 MiB. |
| 2026-10-21 | Refined the batch-10 duration estimate using the full exact length distribution and simulation of the runner’s fixed-2,048-token packing. It produces about 74,853–74,993 packed sequences (74,939 in a seed-42 shuffle), or 7,494 model forwards at batch 10. The balanced batch-10 smoke measured 17.694 forward seconds per batch after model loading, estimating 36.83 forward hours; budget about 37–38 hours including dataset packing, load, state serialization, sweeps, and reports. | `length-analysis/lengths.npz`; 10 packing-order simulations; `smoke100-b10-l2048-map59-64` timing. |
| 2026-10-22 | Added restartable Qwen3.5 calibration checkpoints. The runner atomically saves raw REAP state, SEAP stability accumulators, selected-expert counts, and device RNG states every 100 batches by default. A resume validates model, dataset, batching, sequence preparation, perturbation, seed, and score settings before continuing from the saved batch. The launcher automatically selects `calibration/checkpoints/latest.pt` when it exists; set `SEAP_AUTO_RESUME=false` for a deliberate fresh run. | `src/reap/seap.py`; `experiments/seap-qwen35-sharded.py`; `experiments/run-seap-qwen35.sh`; `callback-b11-l2048-balanced.env`. |
| 2026-10-22 | Added a Qwen callback parity gate. Before final state output, each of the 48 layers must have exactly `total_tokens × top_k` callback captures and an exact per-expert frequency match with independent router observations. | `experiments/seap-qwen35-sharded.py`; successful two-process checkpoint-resume smoke. |
| 2026-10-22 | Verified checkpoint resume end to end: stage one saved after 11 packed sequences; stage two in a new process resumed at batch 1 and completed 22 sequences / 45,034 tokens. Final metadata records 2 processed batches and exact callback parity. Focused regression suite passed 9 tests. | `checkpoint-resume-smoke`; `.venv-qwen35/bin/python -m pytest -q tests/test_seap.py tests/test_pruning_metrics.py tests/test_router_stability.py`. |
| 2026-10-22 | Validated batch 12 at a 1,792-token non-truncating cap with the balanced 59/64/64/64 GiB map. The 100-row callback smoke completed 9 forwards / 179,122 valid tokens in 191.44 s, produced finite scores for all 48 layers, passed callback parity, and released all GPUs. Peak use was 79,117 / 76,681 / 76,825 / 78,247 MiB, leaving 2.74 GiB on GPU 0 at the smallest margin. | `smoke100-b12-l1792-map59-64/smoke_report.json`; `gpu_samples.csv`; `run_summary.json`. |
| 2026-10-22 | Tested batch 16 at the same 1,792-token cap and balanced map. It failed in the first forward on GPU 1: an additional 896 MiB allocation was requested with 860 MiB free (2.59 GiB reserved but unallocated). Batch 12 is the highest validated batch size at this cap without an allocator override. The failed process released all GPU memory. | `capacity-pilot-b16-l1792-map59-64/calibration/logs/runner.log`. |
| 2026-10-22 | Started the authorized full Qwen3.5 calibration in tmux session `reap-qwen35-b12-l1792`, then stopped it during initialization before any forwards after finding that packed calibration sequences cut source samples at their boundaries despite `SEAP_TRUNCATE=false`. | tmux session log `/tmp/reap-qwen35-b12-l1792-balanced.log`; no calibration state or checkpoint written. |
| 2026-10-22 | Corrected packed calibration to carry an eligible source sample to the next sequence when it does not fit, and to skip only source rows longer than the cap with `truncate=false`. Added packing statistics to Qwen metadata. The corrected batch-12, 1,792-token 100-sequence smoke passed in 9 forwards / 128,864 valid tokens with finite 48-layer scores and callback parity; peaks left at least 3.62 GiB free. Restarted the full tmux run using this path. | `src/reap/data.py`; `experiments/seap-qwen35-sharded.py`; `tests/test_data_packing.py`; `smoke100-b12-l1792-nontruncating/smoke_report.json`. |
| 2026-10-22 | Stopped and removed the restarted batch-12 full-calibration tmux session at the user’s request before accepting any calibration result, so other experiments can use all four GPUs. | `tmux kill-session -t reap-qwen35-b12-l1792`; all GPUs report 0 MiB and 0% utilization. |
