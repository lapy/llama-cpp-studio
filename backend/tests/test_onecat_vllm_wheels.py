"""Release-wheel selection for 1Cat-vLLM GitHub assets.

The rule is declared by the definition's release recipe and executed by the
shared ``prereqs.resolve_release_assets`` selector.
"""

import pytest

from backend.engines.definitions import load_default_engine_definitions, materialize_plan
from backend.engines.python.release_assets import select_wheel_urls


def _rules():
    definition = load_default_engine_definitions().definitions["1cat_vllm"]
    step = materialize_plan(definition, "release").by_id("resolve_release")
    options = dict(step.with_)
    return options["wheel_groups"], options["error"]


def _release(tag: str, *asset_names: str) -> dict:
    return {
        "tag_name": tag,
        "assets": [
            {
                "name": name,
                "browser_download_url": f"https://example.test/{name}",
            }
            for name in asset_names
        ],
    }


def _select(release):
    groups, error = _rules()
    return select_wheel_urls(release, groups, error=error)


def test_selects_bundled_1cat_vllm_wheel():
    tag, wheels = _select(
        _release(
            "v1.3.0",
            "1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl",
            "_20260817165150_715_218.png",
        )
    )
    assert tag == "v1.3.0"
    assert wheels == ["https://example.test/1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl"]


def test_selects_legacy_two_wheel_layout():
    tag, wheels = _select(
        _release(
            "v0.0.3",
            "flash_attn_v100-26.2-cp312-cp312-linux_x86_64.whl",
            "vllm-0.0.3.dev0+g72bb24e2d.d20260430.cu128-cp312-cp312-linux_x86_64.whl",
        )
    )
    assert tag == "v0.0.3"
    assert wheels == [
        "https://example.test/flash_attn_v100-26.2-cp312-cp312-linux_x86_64.whl",
        "https://example.test/vllm-0.0.3.dev0+g72bb24e2d.d20260430.cu128-cp312-cp312-linux_x86_64.whl",
    ]


def test_prefers_bundled_wheel_when_both_layouts_present():
    _, wheels = _select(
        _release(
            "v1.0.0",
            "flash_attn_v100-26.2-cp312-cp312-linux_x86_64.whl",
            "vllm-1.0.0-cp312-cp312-linux_x86_64.whl",
            "1cat_vllm-1.0.0-cp312-cp312-linux_x86_64.whl",
        )
    )
    assert wheels == ["https://example.test/1cat_vllm-1.0.0-cp312-cp312-linux_x86_64.whl"]


def test_rejects_release_without_installable_wheel():
    with pytest.raises(RuntimeError, match="1cat_vllm or vllm wheel"):
        _select(_release("v1.3.0", "_20260817165150_715_218.png"))
