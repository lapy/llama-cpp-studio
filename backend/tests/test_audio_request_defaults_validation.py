"""Family-specific request defaults validation."""

import pytest

from backend.audio.request_defaults_validation import validate_saved_request_defaults


@pytest.mark.parametrize(
    ("family", "task", "config", "message"),
    [
        (
            "heartmula",
            "gen",
            {"speech_defaults": {"text": "unused"}},
            "task_defaults",
        ),
    ],
)
def test_validate_saved_request_defaults_rejects_mismatched_families(
    family, task, config, message
):
    errors = validate_saved_request_defaults(task=task, family=family, config=config)
    assert errors
    assert any(message in err for err in errors)


def test_validate_saved_request_defaults_accepts_qwen3_natural_language_instructions():
    errors = validate_saved_request_defaults(
        task="vdes",
        family="qwen3_tts",
        inspection={"instructions_policy": "openai_instruct"},
        config={
            "speech_defaults": {
                "instructions": "A calm, kind, motherly narrator with gentle pacing",
            }
        },
    )
    assert errors == []


def test_validate_saved_request_defaults_accepts_omnivoice_canonical_attributes():
    errors = validate_saved_request_defaults(
        task="tts",
        family="omnivoice",
        inspection={"instructions_policy": "soft_tags"},
        config={
            "speech_defaults": {
                "instructions": "female, young adult, moderate pitch, british accent",
            }
        },
    )
    assert errors == []


def test_validate_saved_request_defaults_accepts_heartmula_task_defaults_tags():
    errors = validate_saved_request_defaults(
        task="gen",
        family="heartmula",
        config={"task_defaults": {"tags": "pop, upbeat, drums, bright"}},
    )
    assert errors == []


def test_validate_rejects_speech_defaults_when_inspect_routes_to_tasks_run():
    """Inspection/help signals override family heuristics for endpoint/defaults key."""
    errors = validate_saved_request_defaults(
        task="tts",
        family="qwen3_tts",
        config={"speech_defaults": {"temperature": 0.7}},
        inspection={"tasks": [{"task": "tts"}, {"task": "vc"}], "preferred_api_endpoint": "tasks"},
        model_profile={
            "sections": [
                {
                    "params": [
                        {"name": "task-route"},
                        {"name": "source-audio"},
                    ]
                }
            ]
        },
    )
    assert errors
    assert any("task_defaults" in err for err in errors)


def test_validate_accepts_task_defaults_when_inspect_routes_to_tasks_run():
    errors = validate_saved_request_defaults(
        task="tts",
        family="custom_family",
        config={"task_defaults": {"text": "hello", "options": {"task_route": "tts"}}},
        inspection={"tasks": [{"task": "tts"}, {"task": "vc"}], "preferred_api_endpoint": "tasks"},
        model_profile={
            "params": [
                {"name": "task-route"},
                {"name": "source-audio"},
            ]
        },
    )
    assert errors == []


def test_spec_request_fields_replace_curated_hints(tmp_path):
    from backend.audio.task_profiles import request_field_groups_for

    spec_dir = tmp_path / "model_specs"
    spec_dir.mkdir()
    (spec_dir / "heartmula.json").write_text(
        '{"family":"heartmula","schema_version":1,"tasks":["gen"],'
        '"options":{"request":[{"name":"tags","type":"string",'
        '"description":"Comma-separated style tags from the model spec."}]}}',
        encoding="utf-8",
    )
    groups = request_field_groups_for("gen", "heartmula", source_path=str(tmp_path))
    tags = next(field for group in groups for field in group["fields"] if field["key"] == "tags")
    assert "model spec" in tags["hint"].lower()
