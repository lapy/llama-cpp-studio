"""Release-wheel selection for 1Cat-vLLM GitHub assets."""

import pytest

from backend.onecat_vllm_manager import OneCatVllmManager


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


def test_selects_bundled_1cat_vllm_wheel():
    tag, wheels = OneCatVllmManager._select_release_wheels(
        _release(
            "v1.3.0",
            "1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl",
            "_20260817165150_715_218.png",
        )
    )
    assert tag == "v1.3.0"
    assert wheels == [
        "https://example.test/1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl"
    ]


def test_selects_legacy_two_wheel_layout():
    tag, wheels = OneCatVllmManager._select_release_wheels(
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
    _, wheels = OneCatVllmManager._select_release_wheels(
        _release(
            "v1.0.0",
            "flash_attn_v100-26.2-cp312-cp312-linux_x86_64.whl",
            "vllm-1.0.0-cp312-cp312-linux_x86_64.whl",
            "1cat_vllm-1.0.0-cp312-cp312-linux_x86_64.whl",
        )
    )
    assert wheels == [
        "https://example.test/1cat_vllm-1.0.0-cp312-cp312-linux_x86_64.whl"
    ]


def test_rejects_release_without_installable_wheel():
    with pytest.raises(RuntimeError, match="1cat_vllm or vllm wheel"):
        OneCatVllmManager._select_release_wheels(
            _release("v1.3.0", "_20260817165150_715_218.png")
        )
