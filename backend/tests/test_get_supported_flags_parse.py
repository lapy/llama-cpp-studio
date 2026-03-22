"""Unit tests for parsing llama-server --help (flag extraction)."""

import re

# Mirrors backend.llama_swap_config.get_supported_flags long-flag extraction
LONG_FLAG_RE = re.compile(r"--[a-zA-Z0-9][a-zA-Z0-9-]*")


def test_extracts_flags_from_cuda_preamble_and_comma_aliases():
    """Realistic snippet: diagnostics, section headers, comma-separated options."""
    snippet = """
ggml_cuda_init: found 3 CUDA devices (Total VRAM: 89100 MiB):
----- common params -----

-h,    --help, --usage                  print usage and exit
-c,    --ctx-size N                     size of the prompt context
-kvo,  --kv-offload, -nkvo, --no-kv-offload
                                        whether to enable KV cache offloading
-fa,   --flash-attn [on|off|auto]       set Flash Attention use
--typical, --typical-p N                locally typical sampling
"""
    flags = set(LONG_FLAG_RE.findall(snippet))
    assert "--help" in flags
    assert "--usage" in flags
    assert "--ctx-size" in flags
    assert "--kv-offload" in flags
    assert "--no-kv-offload" in flags
    assert "--flash-attn" in flags
    assert "--typical" in flags
    assert "--typical-p" in flags


def test_does_not_match_http_urls():
    text = "see https://example.com/path --real-flag value"
    flags = set(LONG_FLAG_RE.findall(text))
    assert "--real-flag" in flags
    assert not any("http" in f or "example" in f for f in flags)
