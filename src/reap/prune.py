from __future__ import annotations
import time
import logging
import dataclasses
import pathlib
import time
from typing import Any
import gc
import yaml

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser

from accelerate.utils import set_seed
from accelerate.hooks import remove_hook_from_module


from reap.args import (
    ReapArgs,
    ModelArgs,
    EvalArgs,
    PruneArgs,
    ObserverArgs,
    DatasetArgs,
    ClusterArgs,
)
from reap.data import DATASET_REGISTRY
from reap.cluster import (
    get_penalty_vector,
    hierarchical_clustering,
    dynamic_frequency_penalized_clustering,
)
from reap.model_util import (
    get_moe,
    assert_merge,
    MODEL_ATTRS,
    patched_model_map,
    get_super_expert_indices,
    slice_fused_expert_weights,
)
import shutil

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def allocate_layerwise_pruning_counts(
    observer_data: dict[int, dict[str, Any]],
    total_experts_to_prune: int,
    *,
    epsilon: float = 1e-12,
) -> dict[int, int]:
    """Allocate a global pruning budget using inverse MoE contribution.

    Lower-contribution layers receive a larger pruning fraction. Allocation is
    deterministic, respects the one-expert-per-layer lower bound, and returns
    integer counts summing exactly to ``total_experts_to_prune``.
    """
    layers = list(observer_data)
    if not layers:
        return {}

    sizes = {
        layer: int(observer_data[layer]["expert_frequency"].shape[0])
        for layer in layers
    }
    maximum = sum(max(size - 1, 0) for size in sizes.values())
    if not 0 <= total_experts_to_prune <= maximum:
        raise ValueError(
            f"Requested {total_experts_to_prune} expert removals, but the valid "
            f"range is [0, {maximum}] while retaining at least one expert per layer."
        )

    priorities = {}
    for layer in layers:
        contribution = observer_data[layer].get("moe_residual_contribution")
        if contribution is None:
            raise ValueError(
                "layerwise_seap requires 'moe_residual_contribution' for every "
                f"MoE layer; missing layer {layer}."
            )
        value = float(torch.as_tensor(contribution).item())
        if not torch.isfinite(torch.tensor(value)) or value < 0:
            raise ValueError(
                f"Invalid MoE residual contribution for layer {layer}: {value}"
            )
        priorities[layer] = 1.0 / max(value, epsilon)

    counts = {layer: 0 for layer in layers}
    remaining = total_experts_to_prune
    active = [layer for layer in layers if sizes[layer] > 1]

    # Water-fill capped layers first, then use largest remainder allocation.
    while remaining and active:
        denominator = sum(sizes[layer] * priorities[layer] for layer in active)
        raw = {
            layer: remaining * sizes[layer] * priorities[layer] / denominator
            for layer in active
        }
        capped = [
            layer for layer in active if raw[layer] >= sizes[layer] - 1
        ]
        if capped:
            for layer in capped:
                cap = sizes[layer] - 1
                counts[layer] = cap
                remaining -= cap
                active.remove(layer)
            continue

        floors = {layer: min(int(raw[layer]), sizes[layer] - 1) for layer in active}
        for layer, count in floors.items():
            counts[layer] += count
        remaining -= sum(floors.values())
        if remaining:
            ranked = sorted(
                active,
                key=lambda layer: (raw[layer] - floors[layer], -layers.index(layer)),
                reverse=True,
            )
            for layer in ranked:
                if not remaining:
                    break
                if counts[layer] < sizes[layer] - 1:
                    counts[layer] += 1
                    remaining -= 1
        break

    if remaining:
        raise RuntimeError("Failed to allocate the complete layerwise pruning budget")
    return counts


def prune(
    observer_data,
    model,
    prune_args,
    n_experts_to_prune,
    pruned_model_dir,
):
    """
    Prune the model based on the observer data and clustering.
    """
    model_attrs = MODEL_ATTRS[model.__class__.__name__]

    for layer in observer_data:
        if "expert_proba" not in observer_data[layer]:
            # Calculate expert probabilities if not already present
            observer_data[layer]["expert_proba"] = (
                observer_data[layer]["expert_frequency"]
                / observer_data[layer]["total_tokens"]
            )

    if prune_args.perserve_super_experts or prune_args.perserve_outliers:
        super_expert_idx = get_super_expert_indices(observer_data, include_last_layers=prune_args.perserve_outliers)
        metrics = [
            "expert_proba",
            "ean_sum",
            "ean_mean",
            "weighted_expert_frequency_sum",
            "weighted_ean_sum",
            "reap",
            "reap_l2",
            "weighted_ean_sum_l2",
            "seap",
            "stability_score",
        ]
        for layer in observer_data:
            super_experts_in_layer = super_expert_idx[super_expert_idx[:, 0] == layer][:, 1]
            if len(super_experts_in_layer) > 0:
                for metric in metrics:
                    if metric in observer_data[layer]:
                        observer_data[layer][metric][super_experts_in_layer] = float("inf")

    layer_pruning_counts = None
    if prune_args.prune_method == "layerwise_seap":
        layer_pruning_counts = allocate_layerwise_pruning_counts(
            observer_data, n_experts_to_prune
        )

    retained_counts = {}
    for layer in tqdm(observer_data, "Pruning layers..."):
        num_experts = observer_data[layer]["expert_frequency"].shape[0]
        layer_n_experts_to_prune = (
            layer_pruning_counts[layer]
            if layer_pruning_counts is not None
            else n_experts_to_prune
        )
        if prune_args.prune_method == "ean_ca":
            ean = torch.zeros(num_experts, device=model.device, dtype=torch.float32)
            for i in range(num_experts):
                ean[i] = torch.linalg.norm(
                    observer_data[layer]["routed_characteristic_activation"][i], dim=-1
                ).sum()
            _, experts_to_prune = torch.topk(
                ean, layer_n_experts_to_prune, largest=False
            )
        else:
            prune_method = prune_args.prune_method
            if prune_method == "frequency":
                prune_method = "expert_frequency"
            elif prune_method == "seap":
                prune_method = "seap_score"
            elif prune_method == "layerwise_seap":
                prune_method = "seap_score"
            saliency_data = observer_data[layer].get(prune_method)
            if saliency_data is None:
                raise ValueError(
                    f"Prune method {prune_args.prune_method} not found in observer data for layer {layer}. "
                    f"Available keys: {list(observer_data[layer].keys())}"
                )
            _, experts_to_prune = torch.topk(
                saliency_data, layer_n_experts_to_prune, largest=False
            )

        retained_expert_indicies = [
            i for i in range(num_experts) if i not in experts_to_prune
        ]
        retained_counts[layer] = len(retained_expert_indicies)
        # prune experts
        moe = get_moe(model, layer)
        # Some Qwen4/FP8 loaders advertise a fused expert implementation but
        # materialize ``experts`` as a ModuleList of individual experts. The
        # branch must follow the object that will be serialized; otherwise we
        # only shrink the config/router while all original expert modules
        # remain in the checkpoint.
        is_fused_experts = model_attrs["fused"] and hasattr(
            getattr(moe, model_attrs["experts"]), "gate_up_proj"
        )
        if not is_fused_experts:
            all_experts = getattr(moe, model_attrs["experts"])
            retained_experts = [all_experts[i] for i in retained_expert_indicies]
            retained_experts = torch.nn.ModuleList(retained_experts)
            setattr(moe, model_attrs["experts"], retained_experts)
            if model.__class__.__name__.lower() == "Ernie4_5_MoEForCausalLM".lower():
                # transformers version >=4.54
                # prune expert score correction bias too
                moe.moe_statics.e_score_correction_bias.data = (
                    moe.moe_statics.e_score_correction_bias.data[
                        :, retained_expert_indicies
                    ]
                )

            # prune router
            router = getattr(moe, model_attrs["router"])
            router.weight.data = router.weight.data[retained_expert_indicies, :]
            if getattr(router, "bias", None):
                router.bias.data = router.bias.data[retained_expert_indicies]
            router.out_features = len(retained_expert_indicies)
            if hasattr(router, "e_score_correction_bias"):
                router.e_score_correction_bias.data = (
                    router.e_score_correction_bias.data[retained_expert_indicies]
                )
            if hasattr(router, "n_routed_experts"):
                router.n_routed_experts = len(retained_expert_indicies)
            if hasattr(moe, "num_experts"):
                moe.num_experts = len(retained_expert_indicies)
            setattr(moe, model_attrs["router"], router)
        else:
            # Prune packed expert tensors.  This covers Llama-4 and
            # Qwen4ExpTextExperts, including FP8 weight_scale_inv siblings.
            router = getattr(moe, model_attrs["router"])
            slice_fused_expert_weights(moe.experts, retained_expert_indicies)
            moe.num_experts = len(retained_expert_indicies)
            router.weight.data = router.weight.data[retained_expert_indicies].clone()
            router.out_features = len(retained_expert_indicies)
            if hasattr(router, "num_experts"):  # transformers >= 4.54+
                router.num_experts = len(retained_expert_indicies)
            if hasattr(moe.experts, "num_experts"):
                moe.experts.num_experts = len(retained_expert_indicies)
            top_k = getattr(moe, "num_experts_per_tok", None)
            if top_k is not None and top_k > len(retained_expert_indicies):
                raise ValueError(
                    f"Cannot retain {len(retained_expert_indicies)} experts with "
                    f"top-k routing set to {top_k}"
                )

    # patch config and dump
    logger.info("Saving pruned model...")
    retained_experts = len(retained_expert_indicies)
    setattr(model.config, model_attrs["num_experts"], retained_experts)
    # Qwen4-Exp is multimodal: the text MoE configuration is nested under
    # ``config.text_config``.  Updating only the wrapper config would leave a
    # reloaded checkpoint configured for the original 512 experts.
    for nested_config_name in ("text_config", "language_config"):
        nested_config = getattr(model.config, nested_config_name, None)
        if nested_config is not None and hasattr(
            nested_config, model_attrs["num_experts"]
        ):
            setattr(nested_config, model_attrs["num_experts"], retained_experts)
    if model.__class__.__name__ == "Glm4MoeForCausalLM":
        # GLM's current Transformers/vLLM implementations normally assume one
        # routed-expert count globally. Preserve the actual per-layer counts so
        # a compatible runtime can construct non-uniform MoE layers.
        num_layers = int(getattr(model.config, "num_hidden_layers", 0))
        model.config.layer_n_routed_experts = [
            retained_counts.get(layer, None) for layer in range(num_layers)
        ]
        model.config.n_routed_experts = max(retained_counts.values())
    if model.__class__.__name__ == "Ernie4_5_MoeForCausalLM":  # remote-code verson
        model.config.moe_capacity = [
            retained_experts,
            retained_experts,
            retained_experts,
        ]

    pruned_model_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    model.save_pretrained(pruned_model_dir)
    end = time.time()
    logger.info(
        f"Pruned model saved to {pruned_model_dir} in {end - start:.2f} seconds"
    )
    return pruned_model_dir


def get_pruned_model_dir(
    results_dir: pathlib.Path,
    n_experts_to_prune: int,
    total_experts: int,
    prune_args: PruneArgs,
    seed: int,
    renorm: bool,
    name_prefix: str = None,
) -> pathlib.Path:
    """Generate output directory path for pruned model."""
    compression_ratio_str = f"{(n_experts_to_prune / total_experts):.2f}"
    name_prefix = "" if name_prefix is None else name_prefix
    pruned_model_name = f"{name_prefix}{prune_args.prune_method}"

    if prune_args.perserve_super_experts:
        pruned_model_name += "-perserve_super"
    elif prune_args.perserve_outliers:
        pruned_model_name += "-perserve_outlier"
    if renorm:
        pruned_model_name += f"-renorm_{str(renorm).lower()}"
    pruned_model_name += f"-seed_{seed}"
    pruned_model_name += f"-{compression_ratio_str}"

    pruned_model_dir = results_dir / "pruned_models" / pruned_model_name
    logger.info(f"Using seed {seed}, pruned model dir: {pruned_model_dir}")

    return pruned_model_dir


def main():
    from reap.main import (
        record_activations,
        smoke_test,
        create_results_directory,
        dump_args_to_yaml,
    )
    from reap.eval import run_evaluate

    parser = HfArgumentParser(
        (
            ReapArgs,
            DatasetArgs,
            ObserverArgs,
            ModelArgs,
            EvalArgs,
            PruneArgs,
            ClusterArgs,
        )
    )
    reap_args, ds_args, obs_args, model_args, eval_args, prune_args, cluster_args = (
        parser.parse_args_into_dataclasses()
    )
    if prune_args.perserve_super_experts and prune_args.perserve_outliers:
        raise ValueError("Only one of perserve_super_experts or perserve_outliers can be set to True.")
    set_seed(reap_args.seed)
    results_dir = create_results_directory(model_args.model_name, ds_args.dataset_name)

    # get local patched model if req'd
    model_name = patched_model_map(model_args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    # record activations or load previously recorded activations
    logger.info(
        f"Running observer to collect activation data for model {model_args.model_name} on dataset {ds_args.dataset_name}."
    )
    observer_data = record_activations(
        model,
        tokenizer,
        reap_args,
        model_args,
        ds_args,
        obs_args,
        results_dir,
    )
    if reap_args.run_observer_only:
        logger.info(
            "Observer run completed. Exiting after collecting activation data since "
            "`run_observer_only` is set to True."
        )
        return

    # pruning
    logger.info("Start of pruning")
    layer_expert_counts = {
        layer: len(data["expert_frequency"])
        for layer, data in observer_data.items()
    }
    if prune_args.prune_method == "layerwise_seap":
        # The compression ratio is global across all MoE layers.  For GLM-4.5-Air,
        # 0.5 therefore means removing 50% of the 45 * 128 experts, not 64 total.
        total_experts = sum(layer_expert_counts.values())
    else:
        total_experts = next(iter(layer_expert_counts.values()))
    n_experts_to_prune = prune_args.n_experts_to_prune
    if n_experts_to_prune is None:
        if cluster_args.compression_ratio is None:
            raise ValueError(
                "Either n_experts_to_prune or compression_ratio must be set for pruning."
            )
        else:
            # Calculate n_experts_to_prune from compression_ratio
            n_experts_to_prune = int(total_experts * cluster_args.compression_ratio)
            logger.info(
                f"Calculated n_experts to prune: {n_experts_to_prune} from compression_ratio: {cluster_args.compression_ratio}"
            )

    pruned_model_dir = get_pruned_model_dir(
        results_dir, n_experts_to_prune, total_experts, prune_args, reap_args.seed, obs_args.renormalize_router_weights
    )
    if (
        pruned_model_dir.exists()
        and list(pruned_model_dir.glob("*.safetensors"))
        and not prune_args.overwrite_pruned_model
    ):
        logger.info(
            f"Pruned model directory {pruned_model_dir} already exists and contains pruned model files. "
            "Skipping pruning step."
        )
    else:
        logger.info(f"Pruning model to {total_experts - n_experts_to_prune} experts...")
        prune(
            observer_data,
            model,
            prune_args,
            n_experts_to_prune,
            pruned_model_dir,
        )
        logger.info("pruning completed.")

        # smoke test
        if reap_args.smoke_test:
            logger.info("Running smoke test on the merged model...")
            try:
                smoke_test(model, tokenizer)
            except Exception as e:
                logger.error(f"Smoke test failed: {e}")
                pass

        tokenizer.save_pretrained(pruned_model_dir)
        # Preserve any local custom modeling files for remote-code checkpoints;
        # this is applicable to all model families, not only GLM-4.5-Air.
        for source_file in pathlib.Path(model_name).glob("modeling_*.py"):
            target_file = pruned_model_dir / source_file.name
            shutil.copy2(source_file, target_file)
            logger.info("Copied %s to %s", source_file.name, pruned_model_dir)

        logger.info("Pruning completed.")

        dump_args_to_yaml(
            pruned_model_dir,
            reap_args,
            ds_args,
            obs_args,
            model_args,
            eval_args,
            prune_args,
            cluster_args,
        )

    # eval
    if reap_args.do_eval:
        remove_hook_from_module(model, recurse=True)
        model.to("cpu")
        del model
        del observer_data
        torch.cuda.empty_cache()
        gc.collect()
        model_args.model_name = pruned_model_dir
        run_evaluate(model_args, pruned_model_dir / "eval", eval_args, reap_args.seed)


if __name__ == "__main__":
    main()
