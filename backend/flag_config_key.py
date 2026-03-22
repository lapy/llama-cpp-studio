"""Map CLI long flags to stable model config keys (per engine)."""

from __future__ import annotations

from typing import Dict, Optional

# Shared overrides (llama.cpp + ik_llama)
LLAMA_FLAG_TO_CONFIG_KEY: Dict[str, str] = {
    "--typical": "typical_p",
    "--typical-p": "typical_p",
    "--json-schema": "json_schema",
    "--override-tensor": "override_tensor",
    "-ot": "override_tensor",  # not long but listed in mapping sources
    "--alias": "model_alias",
    "--mirostat-tau": "mirostat_tau",
    "--mirostat-eta": "mirostat_eta",
    "--mirostat-ent": "mirostat_tau",
    "--mirostat-lr": "mirostat_eta",
    "--mla-use": "mla_attn",
    "--attention-max-batch": "attn_max_batch",
    "--fused-moe": "fused_moe",
    "--no-fused-moe": "fused_moe",
}

LM_DEPLOY_FLAG_TO_CONFIG_KEY: Dict[str, str] = {
    "--session-len": "session_len",
    "--max-batch-size": "max_batch_size",
    "--tp": "tensor_parallel",
    "--quant-policy": "quant_policy",
    "--enable-prefix-caching": "enable_prefix_caching",
    "--chat-template": "chat_template",
    "--tool-call-parser": "tool_call_parser",
    "--reasoning-parser": "reasoning_parser",
    "--server-name": "server_name",
    "--server-port": "server_port",
    "--model-name": "model_name",
    "--log-level": "log_level",
    "--download-dir": "download_dir",
    "--revision": "revision",
    "--hf-overrides": "hf_overrides",
    "--cache-max-entry-count": "cache_max_entry_count",
    "--cache-block-seq-len": "cache_block_seq_len",
    "--max-prefill-token-num": "max_prefill_token_num",
    "--model-format": "model_format",
    "--max-log-len": "max_log_len",
    "--vision-max-batch-size": "vision_max_batch_size",
    "--speculative-draft-model": "speculative_draft_model",
    "--speculative-num-draft-tokens": "speculative_num_draft_tokens",
    "--rope-scaling-factor": "rope_scaling_factor",
    "--num-tokens-per-iter": "num_tokens_per_iter",
    "--max-prefill-iters": "max_prefill_iters",
    "--dist-init-addr": "dist_init_addr",
    "--dllm-block-length": "dllm_block_length",
    "--dllm-denoising-steps": "dllm_denoising_steps",
    "--dllm-confidence-threshold": "dllm_confidence_threshold",
    "--node-rank": "node_rank",
    "--max-concurrent-requests": "max_concurrent_requests",
    "--proxy-url": "proxy_url",
}


def _snake_from_long_flag(flag: str) -> str:
    s = flag.lstrip("-").replace("-", "_")
    return s


def flag_to_config_key(flag: str, engine: str) -> str:
    """Primary long flag -> config key stored in model YAML."""
    if flag in LLAMA_FLAG_TO_CONFIG_KEY and engine in ("llama_cpp", "ik_llama"):
        return LLAMA_FLAG_TO_CONFIG_KEY[flag]
    if engine == "lmdeploy":
        if flag in LM_DEPLOY_FLAG_TO_CONFIG_KEY:
            return LM_DEPLOY_FLAG_TO_CONFIG_KEY[flag]
        return _snake_from_long_flag(flag)
    if flag in LLAMA_FLAG_TO_CONFIG_KEY:
        return LLAMA_FLAG_TO_CONFIG_KEY[flag]
    return _snake_from_long_flag(flag)


def choose_primary_flag(long_flags: list) -> Optional[str]:
    """Pick one --long flag to represent a help line (prefer value-taking forms)."""
    if not long_flags:
        return None
    negs = [f for f in long_flags if f.startswith("--no-")]
    poss = [f for f in long_flags if f not in negs]
    if len(long_flags) == 2 and negs and poss:
        return negs[0]
    return long_flags[-1]
