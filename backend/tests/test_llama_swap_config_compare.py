"""Regression: llama-swap pending config should ignore benign ordering differences."""

import yaml

from backend.llama_swap_manager import (
    _configs_semantically_equal,
    summarize_llama_swap_yaml_diff,
)


def test_semantic_equality_ignores_groups_members_order():
    a = """
healthCheckTimeout: 600
models:
  m1:
    cmd: "bash -c 'cd /w && LD_LIBRARY_PATH=/b ./llama-server --model /m --port ${PORT} --alias m1'"
  m2:
    cmd: "bash -c 'cd /w && LD_LIBRARY_PATH=/b ./llama-server --model /m2 --port ${PORT} --alias m2'"
groups:
  concurrent_models:
    swap: false
    exclusive: false
    members: [m2, m1]
"""
    b = """
healthCheckTimeout: 600
models:
  m1:
    cmd: "bash -c 'cd /w && LD_LIBRARY_PATH=/b ./llama-server --model /m --port ${PORT} --alias m1'"
  m2:
    cmd: "bash -c 'cd /w && LD_LIBRARY_PATH=/b ./llama-server --model /m2 --port ${PORT} --alias m2'"
groups:
  concurrent_models:
    swap: false
    exclusive: false
    members: [m1, m2]
"""
    assert _configs_semantically_equal(a, b)
    assert summarize_llama_swap_yaml_diff(a, b) == []


def test_semantic_equality_ignores_flag_order_after_port():
    cmd_a = (
        "bash -c 'cd /w && LD_LIBRARY_PATH=/b ./llama-server --model /m --port ${PORT} "
        "--alias x --temp 0.7 --ctx-size 4096'"
    )
    cmd_b = (
        "bash -c 'cd /w && LD_LIBRARY_PATH=/b ./llama-server --model /m --port ${PORT} "
        "--ctx-size 4096 --alias x --temp 0.7'"
    )
    ya = yaml.dump({"models": {"q": {"cmd": cmd_a}}}, sort_keys=False)
    yb = yaml.dump({"models": {"q": {"cmd": cmd_b}}}, sort_keys=False)
    assert _configs_semantically_equal(ya, yb)
    assert summarize_llama_swap_yaml_diff(ya, yb) == []


def test_real_diff_still_detected():
    a = "models:\n  q:\n    cmd: \"bash -c 'cd /w && LD_LIBRARY_PATH=/b ./llama-server --model /m --port ${PORT} --temp 0.7'\"\n"
    b = "models:\n  q:\n    cmd: \"bash -c 'cd /w && LD_LIBRARY_PATH=/b ./llama-server --model /m --port ${PORT} --temp 0.9'\"\n"
    assert not _configs_semantically_equal(a, b)
    lines = summarize_llama_swap_yaml_diff(a, b)
    assert any("q" in line for line in lines)
