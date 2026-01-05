import pytest
from fastapi import HTTPException

from backend.routes import models as models_routes


def test_validate_lmdeploy_config_clamps_context_length():
    manifest_entry = {
        "max_context_length": 4096,
        "metadata": {}
    }
    payload = {"session_len": 12288, "max_prefill_token_num": 16384}

    result = models_routes._validate_lmdeploy_config(payload, manifest_entry)

    assert result["session_len"] == 4096
    assert result["max_prefill_token_num"] == 16384  # No longer clamped
    assert result["effective_session_len"] == 4096


def test_validate_lmdeploy_config_parses_tensor_split_string():
    manifest_entry = {"metadata": {}}
    payload = {"tensor_split": "40, 30 ,30"}

    result = models_routes._validate_lmdeploy_config(payload, manifest_entry)

    assert result["tensor_split"] == [40.0, 30.0, 30.0]


def test_validate_lmdeploy_config_rejects_non_object():
    manifest_entry = {"metadata": {}}

    with pytest.raises(HTTPException):
        models_routes._validate_lmdeploy_config("invalid", manifest_entry)


def test_validate_lmdeploy_config_applies_rope_scaling():
    manifest_entry = {"max_context_length": 32768, "metadata": {}}
    payload = {
        "rope_scaling_mode": "yarn",
        "rope_scaling_factor": 3.5,
        "session_len": 1024,
    }

    result = models_routes._validate_lmdeploy_config(payload, manifest_entry)

    assert result["session_len"] == 32768
    assert result["rope_scaling_mode"] == "yarn"
    assert result["rope_scaling_factor"] == 3.5
    assert result["effective_session_len"] == 114688


def test_validate_lmdeploy_config_clamps_rope_scaling_factor():
    manifest_entry = {"max_context_length": 32768, "metadata": {}}
    payload = {
        "rope_scaling_mode": "yarn",
        "rope_scaling_factor": 9,
    }

    result = models_routes._validate_lmdeploy_config(payload, manifest_entry)

    assert result["rope_scaling_factor"] == pytest.approx(4.0)
    assert result["effective_session_len"] == 131072


def test_validate_lmdeploy_config_normalizes_hf_overrides_from_json():
    manifest_entry = {"metadata": {}}
    payload = {
        "hf_overrides": '{"rope_scaling": {"rope_type": "yarn", "factor": 2}}'
    }

    result = models_routes._validate_lmdeploy_config(payload, manifest_entry)

    assert result["hf_overrides"]["rope_scaling"]["rope_type"] == "yarn"
    assert result["hf_overrides"]["rope_scaling"]["factor"] == 2


def test_validate_lmdeploy_config_rejects_invalid_hf_overrides():
    manifest_entry = {"metadata": {}}
    payload = {
        "hf_overrides": 123
    }

    with pytest.raises(HTTPException):
        models_routes._validate_lmdeploy_config(payload, manifest_entry)

