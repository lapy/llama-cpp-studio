import pytest
from fastapi import HTTPException

from backend.routes import models as models_routes


def test_validate_lmdeploy_config_clamps_context_length():
    manifest_entry = {
        "max_context_length": 4096,
        "metadata": {}
    }
    payload = {"context_length": 12288}

    result = models_routes._validate_lmdeploy_config(payload, manifest_entry)

    assert result["context_length"] == 4096


def test_validate_lmdeploy_config_parses_tensor_split_string():
    manifest_entry = {"metadata": {}}
    payload = {"tensor_split": "40, 30 ,30"}

    result = models_routes._validate_lmdeploy_config(payload, manifest_entry)

    assert result["tensor_split"] == [40.0, 30.0, 30.0]


def test_validate_lmdeploy_config_rejects_non_object():
    manifest_entry = {"metadata": {}}

    with pytest.raises(HTTPException):
        models_routes._validate_lmdeploy_config("invalid", manifest_entry)

