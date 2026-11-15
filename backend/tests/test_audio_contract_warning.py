"""Regression pin: the audio contract keeps the JSON key ``schema`` without
shadowing Pydantic's deprecated ``BaseModel.schema`` attribute.

A field literally named ``schema`` made every backend test emit
``UserWarning: Field name "schema" in "AudioContractResponse" shadows an
attribute in parent "_OpenModel"``. Only the Python field name (``schema_``)
avoids the shadowing; the wire key stays ``schema``. pytest.ini escalates a
reintroduction of that warning to an error.
"""

from __future__ import annotations

import warnings

from backend.models.engine_contracts import AudioContractResponse

SHADOW_WARNING = "shadows an attribute"


def test_audio_contract_builds_and_serializes_without_shadow_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = AudioContractResponse(
            schema="audio-cpp.studio-contract/v0",
            engine_version="release-0.9.4",
        )
        payload = model.model_dump(by_alias=True)

    shadow = [str(item.message) for item in caught if SHADOW_WARNING in str(item.message)]
    assert shadow == []
    assert payload["schema"] == "audio-cpp.studio-contract/v0"
    assert "schema_" not in payload


def test_audio_contract_python_field_does_not_shadow_base_model():
    field = AudioContractResponse.model_fields["schema_"]
    assert field.alias == "schema"
    assert "schema" not in AudioContractResponse.model_fields
    # The published model schema (what FastAPI serves) keeps the wire key.
    assert "schema" in AudioContractResponse.model_json_schema()["properties"]


def test_audio_contract_populates_by_field_name_and_alias():
    assert AudioContractResponse.model_validate({"schema": "v0"}).schema_ == "v0"
    by_name = AudioContractResponse(schema_="v0")
    assert by_name.schema_ == "v0"
    assert by_name.model_dump(by_alias=True)["schema"] == "v0"
