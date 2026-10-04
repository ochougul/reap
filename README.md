# Router-weighted Expert Activation Pruning (REAP)

## Updates
* 2026-03-30: We have added a memory-efficient layer-wise (block-wise) calibration observer for pruning large models on a single GPU (see `experiments/pruning-layerwise-cli.sh` on how to run layer-wise calibration).
* 2026-03-19: We have released our data calibration mix recipe for agentic reasoning REAP-compressed models published on HuggingFace (see [details](#huggingface-checkpoints)).
* 2026-03-11: For REAP saliency, top-k router logits are now correctly renormalized to sum to 1. See `args.py:ObserverArgs.renormalize_router_weights`. Generally, you can expect a modest improvement for most model/datasets with this fix in place. On non-agentic coding evaluations across ERNIE-4.5-21B-A3B-PT, Qwen3-30B-A3B, Mixtral-8x7B-Instruct-v0.1, GLM-4.5-Air,and Llama-4-Scout-17B-16E-Instruct, REAP achieves a mean decrease in accuracy of 1.9% vs. 2.6% without logit normalization. Our prior large-scale results and all Cerebras checkpoints on previously calibrated with normalized logits. 
* 2026-03-01: REAP has been accepted to ICLR 2026, see you in Rio! 

## Summary
<img src="./fig/reaper.png" align="right" alt="REAP the experts" width="400">
This repository contains code required to reproduce the expert pruning and merging methods used in the paper: <a href="https://arxiv.org/abs/2510.13999">REAP the Experts: Why Pruning Prevails for One-Shot MoE compression</a>
<br></br> 
Expert pruning and merging can be used to reduce the memory overhead of Sparsely-activated Mixture-of-Experts (SMoE) LLMs. Our novel expert pruning criterion, Router-weighted Expert Activation Pruning (REAP) considers both router gate-values and expert activation norms. Across a diverse set of SMoE models ranging from 20B to 1T parameters, REAP consistently outperforms merging and other pruning methods on generative benchmarks, especially at 50% compression. Notably, our method achieves near-lossless compression on code generation and tool-calling tasks with Qwen3-Coder-480B and Kimi-K2, even after pruning 50% of experts.   
<br></br>

**Our main contributions are as follows:**
- We prove that expert merging introduces *irreducible error* due to the loss of the router's independent, input-dependant modulation of the expert outputs resulting in *functional subspace collapse*, substantially reducing the functional output space of the compressed SMoE layer. In contrast, in expert pruned SMoEs the router maintains independent control over the remaining experts;
- We introduce REAP, a novel expert pruning saliency criterion, which selects experts to prune which contribute minimally to the layer output by considering both the router gate-values and average activation norm of the experts;
- Across diverse SMoE architectures ranging from 20B to 1T parameters and a suite of generative evaluations, we demonstrate the significant and consistent advantage of REAP over existing expert pruning and merging approaches, particularly at 50% compression. Notably, our method achieves near-lossless compression on code generation tasks after pruning 50% of experts from Qwen3-Coder-480B and Kimi-K2.

## Results
**Compressed GLM-4.5 Air (left) & Qwen3-30B-A3B (right) on non-agentic coding, mathematical reasoning, creative writing, and multiple choice (MC) benchmarks using a variety of compression methods.**
![Qwen/GLM](./fig/combined-all-tasks_qwen_and_glm.png)

**Accuracy vs. parameters across all models and compression methods for non-agentic coding (left) and multiple choice (MC) (right)**
![Accuracy vs. params](./fig/combined_performance_vs_params.png)

**Large-scale pruned SMoEs on agentic, non-agentic coding, tool-use tasks, and multiple choice (MC) benchmarks.**
![large-scale pruned SMoE results](./fig/large-scale-moe.png)


## <img src="./fig/hf-transparent.png" alt="Cerebras REAP checkpoints" width='20'>  HuggingFace checkpoints
[REAP model collection on HuggingFace](https://huggingface.co/collections/cerebras/cerebras-reap), including pruned versions of GLM4.6, GLM4.5-Air, Qwen3-Coder-480B, Qwen3-Coder-30B, MiniMax-M2, Kimi-Linear, DeepSeek-V3.2. More to come, stay tuned!

To reproduce our HuggingFace checkpoints, we are releasing the calibration data mix used for agentic reasoner models:

- General coding dataset (4096 samples): [theblackcat102/evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1)
- Reasoning dataset (12288 samples, equal parts from code, math, science subsets): [open-r1/Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts)
- Single-turn tool calling dataset (4096 samples): [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- Agentic coding / multi-turn tool calling dataset (4096 samples, tool split): [SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories)

Our full calibration set is composed of 24576 samples from the sources above at maximal sequence length equal to 16384 tokens. Composite calibration dataset can be specified in the command line as follows:

```bash
bash experiments/pruning-cli.sh 0 Qwen/Qwen3-30B-A3B reap 42 0.25 "theblackcat102/evol-codealpaca-v1:4096,Salesforce/xlam-function-calling-60k:4096,open-r1/Mixture-of-Thoughts[code]:4096,open-r1/Mixture-of-Thoughts[math]:4096,open-r1/Mixture-of-Thoughts[science]:4096,SWE-bench/SWE-smith-trajectories(tool):4096" true true false false false
```

Note that the dataset subsets are specified in square brackets and the split name is specified in curly brackets.

## Installation

### venv
To build the project and setup a virtual environment install `uv` and run:
```bash
bash scripts/build.sh
```

### Docker
Alternatively, use docker:
```bash
docker compose up --build -d
docker compose exec app bash
```

The `docker-compose.yaml` file is setup to mount the default huggingface cache (`~/.cache/huggingface`), but if you use a different cache directory then we suggest updating the mount path to avoid excessive container storage sizes.  

### Configuration
Copy `.env.template` and rename as `.env`. Populate the empty fields.

For WildBench, copy `config/wildbench_prod_env_XXXX.example`. Update the copied subdir name with the port used to launch vLLM, defaults to 800X where X is rank of the first GPU used to run the eval script. I.e, `wildbench_prod_env_8000` if running `eval.py` with `CUDA_VISIBLE_DEVICES=0,1,2,3`. In the copied subdir, update `credentials.conf` with your OpenAI API key and in `model_deployments.yaml` substitute `base_url:XXXX` with the port selected. i.e., `http://0.0.0.0:XXXX/v1/ -> http://0.0.0.0:8000/v1/` for the example above. 


### Adding a new model
Add model attribute names to `MODEL_ATTRS` in `src/reap/model_util.py`. Each entry is identified by the class name of the model `model.__class__.__name__` as key. The values correspond to the following:
- `moe_block`: Attribute name of SMoE submodule in the decoder module. 
- `*_proj`: Attribute names for the expert projections. 
- `experts`: Attribute of the ModuleList containing the experts in the SMoE module. 
- `fused`: If true, the model uses a FusedMoE layer. Of the currently supported models, only Llama-4 is fused. 
- `router`: Attribute name of the router/gate layer in the SMoE module. 
- `num_experts`: The key in the huggingface config containing the number of experts per layer. (ie., `num_experts` if `model.config.num_experts` contains this value)
- `num_experts_per_tok`: The key in the huggingface config containing the number of experts activated per token. 


## Reproducing Experiments

### Expert Merging

To run merging experiments, use:

```bash
bash experiments/merging-cli.sh <CUDA_DEVICES> [MODEL_NAME] [MERGE_METHOD] [SEED] [COMPRESSION_RATIO] [DATASET_NAME] [RUN_LM_EVAL] [RUN_EVALPLUS] [RUN_LIVE_CODE_BENCH] [RUN_MATH] [RUN_WILDBENCH] [SINGLETON_SUPER_EXPERTS] [SINGLETON_OUTLIER_EXPERTS]
```

- `CUDA_DEVICES`: e.g. `0` or `0,1`
- `MODEL_NAME`: (default: Qwen/Qwen3-30B-A3B)
- `MERGE_METHOD`: `hc_smoe`, `m_smoe`, or `submoe` (default: hc_smoe)
- `SEED`: (default: 42)
- `COMPRESSION_RATIO`: (default: 0.25)
- `DATASET_NAME`: (default: theblackcat102/evol-codealpaca-v1)
- `RUN_*`: Flags control which evaluations to run (true/false)
- `SINGLETON_SUPER_EXPERTS` and `SINGLETON_OUTLIER_EXPERTS` force super and outlier experts into singleton clusters, respectively. See [Unveiling Super Experts in Mixture-of-Experts Large Language Models](https://arxiv.org/abs/2507.23279) paper for definitions.

Example:
```bash
bash experiments/merging-cli.sh 0 Qwen/Qwen3-30B-A3B hc_smoe 42 0.25 theblackcat102/evol-codealpaca-v1 true true true false false
```

### Expert Pruning

To run pruning experiments, use:

```bash
bash experiments/pruning-cli.sh <CUDA_DEVICES> [MODEL_NAME] [PRUNING_METHOD] [SEED] [COMPRESSION_RATIO] [DATASET_NAME] [RUN_LM_EVAL] [RUN_EVALPLUS] [RUN_LIVE_CODE_BENCH] [RUN_MATH] [RUN_WILDBENCH] [SINGLETON_SUPER_EXPERTS] [SINGLETON_OUTLIER_EXPERTS]
```

- `PRUNING_METHOD`: e.g. `reap` will use the REAP expert saliency criterion for expert pruning. 
- Other arguments are similar to merging.

Example:
```bash
bash experiments/pruning-cli.sh 0 Qwen/Qwen3-30B-A3B frequency 42 0.25 theblackcat102/evol-codealpaca-v1 true true true false false
```

### Router stability data collection

Router stability measures how much an MoE router's routing changes when its
input is perturbed. The collector runs the original router input and four
zero-mean Gaussian perturbations. The perturbation variance is relative to the
per-token input variance, using multipliers `0.25`, `0.5`, `1.0`, and `2.0`.
This is a data-collection method only; it does not assign a pruning score.

For every routing event, the collector records the eight selected experts plus
the next four boundary experts. It records correction-adjusted ranking values
for all twelve experts and the final normalized/scaled applied weights for the
selected eight. Full 128-expert tensors are kept only in streaming aggregate
accumulators and are not written per token.

#### Prerequisites

Use the `router-stability-layerwise` branch for the single-GPU implementation
or the `router-stability-4gpu` branch for the resident four-GPU runner. Install
the project environment from the repository root:

```bash
uv sync --dev
```

The project declares `pyarrow` as a runtime dependency. It is required to write
and read the Parquet event table. Confirm the environment before starting a
large run:

```bash
uv run python -c "import pyarrow, pandas; print(pyarrow.__version__)"
```

Set `REAP_GLM_MODEL_PATH` to a local GLM-4.5-Air checkpoint. The calibration
dataset used by the provided scripts is
`theblackcat102/evol-codealpaca-v1`; Hugging Face access and sufficient local
disk are required.

#### Single-GPU, layerwise run

This path loads and processes one layer at a time and is intended for a single
GPU. The first argument selects the GPU. The default output directory is
`artifacts/router_stability/glm4.5-air-evol-codealpaca-500k`.

```bash
REAP_GLM_MODEL_PATH=/path/to/GLM-4.5-Air \
ROUTER_STABILITY_OUTPUT_DIR=artifacts/router_stability/glm4.5-air-evol-codealpaca-500k \
bash experiments/router-stability-layerwise.sh 0
```

The script currently processes up to 256 batches, at most 500,000 valid input
tokens, with sequence length 2,048, and variance multipliers
`0.25 0.5 1.0 2.0`. Change these values in the script or pass equivalent
arguments through `reap.layerwise_prune` when running a custom experiment.

#### Resident four-GPU run

This path keeps the full model resident across four GPUs and avoids repeated
CPU/GPU layer offloading. It requires four GPUs with enough free memory. The
runner uses GPUs `0,1,2,3`, the local model path configured in
`experiments/router-stability-sharded.py`, and writes to
`artifacts/router_stability/glm4.5-air-evol-codealpaca-500k-sharded`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
uv run python -u experiments/router-stability-sharded.py \
  > artifacts/router_stability/sharded-pilot.log 2>&1
```

To run it detached:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
nohup uv run python -u experiments/router-stability-sharded.py \
  > artifacts/router_stability/sharded-pilot.log 2>&1 &
echo $! > artifacts/router_stability/sharded-pilot.pid
```

The four-GPU script has its model and output paths as Python constants. Edit
those constants before using a different checkpoint or output directory.

#### Output files

Each completed run creates three files in its output directory:

- `routing_events.parquet`: one row per token per recorded MOE layer. Columns
  are `batch_id`, `token_position`, `layer`, `top12_expert_ids`,
  `top12_ranking_values`, and `top8_applied_weights`. The nested condition
  dimension is ordered as original, variance `0.25`, `0.5`, `1.0`, and `2.0`.
- `router_stability_accumulators.npz`: streaming full-expert aggregates for
  each variance and layer, including absolute, squared, and signed changes;
  original and perturbed score sums; selected counts; and selection gains and
  losses. These aggregates cannot be joined back to individual tokens.
- `metadata.json`: variance settings, boundary width, top-k, token count, layer
  count, expert count, and noise definition. Its presence indicates that
  `finalize()` completed.

For GLM-4.5-Air, transformer block `0` is not an MoE block, so the first data
rows are normally `layer == 1`. Do not interpret `metadata.json`'s
`num_layers` as the number of layers with routing rows; it is the maximum layer
index plus one.

#### Inspecting a completed run

Use predicate pushdown to inspect only the first recorded MOE layer without
loading the full multi-gigabyte table:

```python
import pyarrow.parquet as pq

path = "artifacts/router_stability/glm4.5-air-evol-codealpaca-500k-sharded/routing_events.parquet"
table = pq.read_table(path, filters=[("layer", "=", 1)])
print(table.num_rows)
print(table.schema)
print(table.slice(0, 3).to_pandas())
```

The stored values are deliberately compact: expert IDs are written as unsigned
8-bit values and ranking values/weights are quantized to float16 before being
written. The table does not contain raw router logits, full score vectors,
router inputs, generated noise, token text, eligibility masks, or boundary
expert applied weights.

---

## Source Directory Structure

The `src/reap` directory contains the main codebase:

- **args.py**: Argument dataclasses for experiment configuration.
- **cluster.py**: Clustering algorithms for grouping experts.
- **data.py**: Dataset loading and processing utilities.
- **eval.py**: Evaluation routines for models and experiments.
- **main.py**: Main entry point for running merging experiments and pipelines.
- **merge.py**: Core logic for merging experts in Mixture-of-Experts (MoE) models.
- **metrics.py**: Distance and similarity metrics for model analysis.
- **model_util.py**: Utilities for model introspection and manipulation.
- **observer.py**: Hooks and classes for collecting model activations.
- **permute.py**: Permutation and alignment utilities for expert weights.
- **prune.py**: Main entry point for expert pruning.
- **restricted_cluster.py**: Clustering with constraints (e.g., max cluster size).

### Models Subdirectory

- **models/**: Contains patched model definitions and configurations for select architectures that do not return router_logits in the SMoE module forward method. (e.g., GLM, ERNIE).


## Citation
Please consider using the following citation if you found this work useful:
```
@inproceedings{
    lasby2026reap,
    title={{REAP} the Experts: Why Pruning Prevails for One-Shot MoE compression},
    author={Mike Lasby and Ivan Lazarevich and Nish Sinnadurai and Sean Lie and Yani Ioannou and Vithursan Thangarasa},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=ukGxWd2aDG}
}
```
