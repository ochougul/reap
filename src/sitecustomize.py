"""Enable REAP's vLLM patch only for explicitly opted-in servers."""

import os

if os.environ.get("REAP_QWEN38_MASK_METADATA"):
    if os.environ.get("REAP_QWEN38_COMPACTION_METADATA"):
        raise RuntimeError("Qwen3.8 mask and compact-expert modes are mutually exclusive")
    from reap.vllm_nonuniform import install_vllm_qwen38_mask

    install_vllm_qwen38_mask()

if os.environ.get("REAP_QWEN38_COMPACTION_METADATA"):
    from reap.vllm_expert_compaction import install_vllm_qwen38_expert_compaction

    install_vllm_qwen38_expert_compaction()
