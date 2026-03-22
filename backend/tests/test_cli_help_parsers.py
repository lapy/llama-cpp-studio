"""Unit tests for --help parsers."""

from backend.cli_help_parsers import (
    parse_llama_help_to_sections,
    parse_lmdeploy_api_server_help,
)


def test_parse_llama_snippet_ctx_and_help():
    text = """
----- common params -----
-h,    --help, --usage                  print usage and exit
-c,    --ctx-size N                     size of the prompt context
"""
    sections = parse_llama_help_to_sections(text, "llama_cpp")
    assert sections
    keys = {p["key"] for s in sections for p in s["params"]}
    assert "ctx_size" in keys
    assert "usage" in keys or "help" in keys


def test_parse_lmdeploy_snippet_port_and_backend():
    text = """
options:
  --server-port SERVER_PORT
                        Server port. Default: 23333. Type: int
  --backend {pytorch,turbomind}
                        Set the inference backend. Default: turbomind. Type: str
"""
    raw = parse_lmdeploy_api_server_help(text)
    keys = {p["key"] for p in raw}
    assert "server_port" in keys
    assert "backend" in keys
