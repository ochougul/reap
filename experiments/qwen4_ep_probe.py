import os
import pathlib
import sys
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
import time
from transformers import AutoConfig, AutoModelForImageTextToText
from transformers.distributed import DistributedConfig
from transformers.distributed.tensor_parallel import ColwiseParallel
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qwen4_calibrate import _patch_triton_kernel_compat

_old_validate = ColwiseParallel.validate_param
_old_shard = ColwiseParallel.shard_param
def _validate(self, module, param, mesh, parameter_name=None):
    meta = module._parameters.get(param)
    if meta is not None and meta.ndim < 2:
        # FP8Embedding.weight_scale is a replicated scalar, not an embedding
        # matrix.  Transformers 5.20's generic colwise validator assumes every
        # parameter has at least two dimensions.
        return
    return _old_validate(self, module, param, mesh, parameter_name)
def _shard(self, module, param, mesh):
    meta = module._parameters.get(param)
    if meta is not None and meta.ndim < 2:
        return
    return _old_shard(self, module, param, mesh)
ColwiseParallel.validate_param = _validate
ColwiseParallel.shard_param = _shard

path = os.environ.get(
    "QWEN4_MODEL_PATH",
    "/home/ubuntu/hf_cache/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/"
    "236dfdf285828023ca3bcd3f37366c58a3469b13",
)
_patch_triton_kernel_compat()
dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
cfg = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
dc = DistributedConfig(tp_size=4, fsdp_size=1, ep_size=4)
if rank == 0:
    print(dc, flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    path,
    distributed_config=dc,
    dtype=None,
    experts_implementation=os.environ.get("QWEN4_EP_IMPL", "deepgemm"),
    attn_implementation="eager",
    trust_remote_code=True,
    local_files_only=True,
    low_cpu_mem_usage=True,
    offload_state_dict=True,
).eval()

# Transformers 5.20 applies the EP rule to FP8 expert weights but misses the
# sibling scale parameters.  Keep this probe explicit so we can validate the
# native kernel with local scale shards.
for module in model.modules():
    if not hasattr(module, "gate_up_proj_scale_inv"):
        continue
    weight = module.gate_up_proj
    scale = module.gate_up_proj_scale_inv
    if isinstance(weight, DTensor) and not isinstance(scale, DTensor):
        local_experts = weight.to_local().shape[0]
        rank_start = dist.get_rank() * local_experts
        local_scale = scale.detach()[rank_start : rank_start + local_experts].to(weight.device)
        sharded = DTensor.from_local(
            local_scale, weight.device_mesh, [weight.placements[0]],
            shape=scale.shape, stride=scale.stride(),
        )
        module._parameters["gate_up_proj_scale_inv"] = torch.nn.Parameter(
            sharded, requires_grad=False
        )
        module._parameters["down_proj_scale_inv"] = torch.nn.Parameter(
            DTensor.from_local(
                module.down_proj_scale_inv.detach()[rank_start : rank_start + local_experts].to(weight.device),
                weight.device_mesh, [weight.placements[0]], shape=module.down_proj_scale_inv.shape,
                stride=module.down_proj_scale_inv.stride(),
            ), requires_grad=False
        )
    for parameter_name in ("gate_up_proj_scale_inv", "down_proj_scale_inv"):
        parameter = getattr(module, parameter_name, None)
        if isinstance(parameter, DTensor) and parameter.dtype != torch.float32:
            module._parameters[parameter_name] = torch.nn.Parameter(
                DTensor.from_local(
                    parameter.to_local().float(), parameter.device_mesh,
                    parameter.placements, shape=parameter.shape,
                    stride=parameter.stride(),
                ), requires_grad=False
            )
if rank == 0:
    print("impl", model.get_experts_implementation(), flush=True)
    print("devices", {str(p.device) for p in model.parameters()}, flush=True)
    print("dtypes", sorted({str(p.dtype) for p in model.parameters()}), flush=True)
    for name, module in model.named_modules():
        if hasattr(module, "gate_up_proj_scale_inv"):
            print("expert", name, "weight", module.gate_up_proj.shape,
                  module.gate_up_proj.dtype, "scale", module.gate_up_proj_scale_inv.shape,
                  module.gate_up_proj_scale_inv.dtype, "block", module.block_size,
                  "is_ep", getattr(module, "_is_expert_parallel", None),
                  "weight_type", type(module.gate_up_proj).__name__,
                  "scale_type", type(module.gate_up_proj_scale_inv).__name__,
                  "scale_local", (module.gate_up_proj_scale_inv.to_local().shape
                                   if isinstance(module.gate_up_proj_scale_inv, DTensor) else None), flush=True)
            break
dist.barrier()
if rank == 0:
    print("loaded", flush=True)
dist.barrier()
ids = torch.full((4, 128), 1, dtype=torch.long, device=f"cuda:{rank}")
mask = torch.ones_like(ids)
dist.barrier()
started = time.perf_counter()
with torch.inference_mode():
    output = model.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
torch.cuda.synchronize()
elapsed = time.perf_counter() - started
print(f"rank={rank} forward_seconds={elapsed:.3f} output={tuple(output.last_hidden_state.shape)}", flush=True)
dist.barrier()
dist.destroy_process_group()
