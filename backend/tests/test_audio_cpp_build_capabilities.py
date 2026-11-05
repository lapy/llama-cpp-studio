"""Installed build settings constrain the embedded audio.cpp UI."""

import pytest

from backend.engines.audio_cpp.build_capabilities import constrain_ui_params, server_ui_available


@pytest.mark.parametrize(
    "value, expected", [(None, False), (False, False), ("OFF", False), (True, True), ("ON", True)]
)
def test_ui_requires_explicit_install_build_setting(value, expected):
    assert server_ui_available({"build_config": {"build_server_frontends": value}}) is expected


def test_registry_forces_ui_off_without_mutating_cached_scan():
    sections = [
        {
            "params": [
                {"key": "ui", "default": True, "supported": True},
                {"key": "ui_management", "default": True},
                {"key": "threads", "default": 8},
            ]
        }
    ]
    constrained = constrain_ui_params(sections, {"build_config": {"build_server_frontends": False}})
    for row in constrained[0]["params"][:2]:
        assert row["forced_value"] is False
        assert row["default"] is False
        assert row["supported"] is False
    assert sections[0]["params"][0]["default"] is True
    assert constrained[0]["params"][2] == sections[0]["params"][2]


@pytest.mark.parametrize("flag", ["--ui", "--ui=true", "--ui-management"])
def test_custom_arguments_cannot_bypass_ui_install_policy(flag):
    from backend.engines.audio_cpp.runtime import _custom_args

    with pytest.raises(ValueError, match="Studio-owned"):
        _custom_args(flag)


def test_ui_restriction_does_not_change_a_model_option_with_same_name():
    sections = [{"params": [{"key": "ui", "scope": "request_option", "default": True}]}]
    assert constrain_ui_params(sections, {}) == sections


def test_ui_capable_install_preserves_advertised_settings():
    sections = [{"params": [{"key": "ui", "default": True}]}]
    assert (
        constrain_ui_params(sections, {"build_config": {"build_server_frontends": True}})
        == sections
    )
