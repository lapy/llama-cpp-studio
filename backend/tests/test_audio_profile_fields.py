"""Request forms derive their fields from engine contracts, with no field catalog."""

import pytest

from backend.engines.audio_cpp.spec_fields import field_groups_from_contract


@pytest.mark.parametrize("kind, value", [("float", 0.5), ("int", 3), ("bool", False), ("string", "new")])
def test_unknown_engine_field_keeps_declared_type_and_default(kind, value):
    groups = field_groups_from_contract({
        "family": "unseen_family",
        "options": {"request": [{"name": "unseen_field", "type": kind, "default": value}]},
    })
    field = groups[0]["fields"][0]
    assert field["key"] == "unseen_field"
    assert field["type"] == kind
    assert field["default"] == value
    assert field["nested"] is True


def test_no_fields_are_invented_for_an_empty_contract():
    assert field_groups_from_contract({"family": "unknown", "options": {}}) == []
