"""Enable REAP's vLLM patch only for explicitly opted-in servers."""

import os

if os.environ.get("REAP_QWEN38_MASK_METADATA"):
    from reap.vllm_nonuniform import install_vllm_qwen38_mask

    install_vllm_qwen38_mask()
