"""LMDeploy-related helpers in models routes (hf_overrides normalization, etc.)."""

import pytest
from fastapi import HTTPException

from backend.routes import models as models_routes


def test_normalize_hf_overrides_parses_json_string():
    payload = '{"rope_scaling": {"rope_type": "yarn", "factor": 2}}'
    out = models_routes._normalize_hf_overrides(payload)
    assert out["rope_scaling"]["rope_type"] == "yarn"
    assert out["rope_scaling"]["factor"] == 2


def test_normalize_hf_overrides_empty_string():
    assert models_routes._normalize_hf_overrides("") == {}
    assert models_routes._normalize_hf_overrides(None) == {}


def test_normalize_hf_overrides_invalid_json():
    with pytest.raises(HTTPException) as exc:
        models_routes._normalize_hf_overrides("{not json")
    assert exc.value.status_code == 400


def test_normalize_hf_overrides_rejects_non_object():
    with pytest.raises(HTTPException):
        models_routes._normalize_hf_overrides(123)


def test_normalize_hf_overrides_nested_dict():
    out = models_routes._normalize_hf_overrides(
        {"a": {"b": 1}, "c": "x"}
    )
    assert out["a"]["b"] == 1
    assert out["c"] == "x"


def test_normalize_hf_overrides_rejects_empty_key():
    with pytest.raises(HTTPException):
        models_routes._normalize_hf_overrides({"": 1})


def test_normalize_hf_overrides_rejects_list_value():
    with pytest.raises(HTTPException):
        models_routes._normalize_hf_overrides({"x": [1, 2]})


def test_coerce_positive_int():
    assert models_routes._coerce_positive_int(42) == 42
    assert models_routes._coerce_positive_int("1024") == 1024
    assert models_routes._coerce_positive_int("1,024") == 1024
    assert models_routes._coerce_positive_int(None) is None
    assert models_routes._coerce_positive_int(0) is None
    assert models_routes._coerce_positive_int(True) is None


def test_apply_prompt_reservation():
    assert models_routes._apply_prompt_reservation(10000) == 1808
    assert models_routes._apply_prompt_reservation(9000) == 1024
    assert models_routes._apply_prompt_reservation(100) == 100
