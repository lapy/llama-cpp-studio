"""Unit tests for backend.utils.coercion."""

from backend.utils.coercion import coerce_json_dict, coerce_positive_int


def test_coerce_json_dict_falsy():
    assert coerce_json_dict(None) == {}
    assert coerce_json_dict("") == {}


def test_coerce_json_dict_dict_copy_behavior():
    d = {"a": 1}
    out = coerce_json_dict(d, copy=True)
    assert out == d
    assert out is not d


def test_coerce_json_dict_dict_no_copy_identity():
    d = {"a": 1}
    assert coerce_json_dict(d, copy=False) is d


def test_coerce_json_dict_from_json_object_string():
    assert coerce_json_dict('{"x": 2, "y": "z"}') == {"x": 2, "y": "z"}


def test_coerce_json_dict_invalid_string():
    assert coerce_json_dict("{not json") == {}


def test_coerce_json_dict_non_object_json():
    assert coerce_json_dict("[1, 2]") == {}


def test_coerce_positive_int_commas_and_cap():
    assert coerce_positive_int("4,096") == 4096
    assert coerce_positive_int(2_000_000_000) is None
