"""Schema v2 reference-definition spike tests.

Proves the plan's hard gate — "the worst-case engine must be expressible
BEFORE Phase 3" — by loading backend/engines/definitions/reference/
(lmdeploy = best case, sglang_v100 = worst case) and checking:

* descriptor equivalence with the live ENGINE_REGISTRY (v2 sections must
  not change the API payload),
* golden launch argv matching backend/proxy/launch_spec.py exactly,
* env requirements matching the V100 managed-CUDA rules,
* recipe validation (typed primitives, condition algebra, output wiring),
* the negative cases the loader must reject at startup.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from backend.engines.definitions import (
    INSTALL_PRIMITIVES,
    REFERENCE_DEFINITIONS_DIR,
    DefinitionError,
    compile_profile_argv,
    compile_profile_env,
    load_default_engine_definitions,
    load_engine_definitions,
)
from backend.engines.definitions.schema import validate_definition
from backend.engines.registry import ENGINE_REGISTRY, SHIPPED_ENGINE_IDS


@pytest.fixture(scope="module")
def reference():
    return load_engine_definitions(REFERENCE_DEFINITIONS_DIR)


BASE_LMDEPLOY = {
    "schema_version": 2,
    "id": "lmdeploy",
    "label": "LMDeploy",
    "repository_source": "LMDeploy",
    "install_kind": "python_venv",
    "runtime_kind": "lmdeploy",
    "scanner_kind": "lmdeploy_help",
    "active_path_fields": ["venv_path"],
    "artifact_formats": ["safetensors"],
    "package_kinds": ["hf_snapshot"],
    "tasks": ["text-generation"],
    "input_modalities": ["text"],
    "output_modalities": ["text"],
}


def _base(**extra):
    raw = copy.deepcopy(BASE_LMDEPLOY)
    raw.update(extra)
    return raw


# ---------------------------------------------------------------------------
# Reference definitions: load, stay registry-equivalent, compile goldens
# ---------------------------------------------------------------------------


def test_reference_definitions_load_with_both_worst_and_best_case(reference):
    assert sorted(reference.ids()) == ["lmdeploy", "sglang_v100"]


def test_reference_definitions_stay_registry_equivalent(reference):
    for engine_id in reference.ids():
        definition = reference.get(engine_id)
        assert definition.to_api_dict() == ENGINE_REGISTRY[engine_id].to_api_dict(), engine_id


def test_default_shadow_registry_untouched_by_reference_directory():
    registry = load_default_engine_definitions()
    # Every shipped engine (generated + definition-owned) has a definition;
    # the harness personas are the only additions.
    assert set(registry.ids()) - {"launchcraft", "launchforge", "nativecraft", "yamlcraft"} == set(
        SHIPPED_ENGINE_IDS
    )
    cut_over = {
        "lmdeploy": {"release", "source", "sync", "remove", "remove_version"},
        "vllm": {"release", "source", "sync", "remove", "remove_version"},
        # Both sglang variants execute all four flows; the fork's steps are
        # fork-specific (v100 env, checkout patches, builder exec) but the
        # vocabulary and executor are shared.
        "sglang": {"release", "source", "sync", "remove", "remove_version"},
        # v100's release intentionally has no recipe: it routes to the
        # source flow at the manager level (legacy behavior), so its plan
        # set is source/sync/remove.
        "sglang_v100": {"source", "sync", "remove", "remove_version"},
        # sglang-sxm2 is the second SM70 source-only fork: same four flows,
        # its own asset directory and provisioner preamble.
        "sglang_sxm2": {"source", "sync", "remove", "remove_version"},
        # 1Cat-vLLM executes all four flows: the release wheel-URL install,
        # the venv-scoped removal, and the wheel-forge source/sync plans
        # (rust bootstrap + conditional flash-attention stay composite
        # engine-local steps inside the shared vocabulary).
        "1cat_vllm": {"release", "source", "sync", "remove", "remove_version"},
        # The venv engines' fifth recipe "remove_version" is the DELETE
        # route's pinned flow (guarded tree + registered-aware forget),
        # executed by routes/llama_versions.py itself. Every engine now
        # carries a remove_version recipe — the DELETE route has no
        # engine-ID branches left.
        # Unsloth llama.cpp installs prebuilt archives only; its release is
        # the first native-family cut-over (download → verify → extract).
        "unsloth_llama": {"release", "remove_version"},
        # audio.cpp: the PREBUILT branch of a release install, the shared
        # DELETE flow, and (Phase 7) the SOURCE build and in-place SYNC — the
        # definition owns the step order and the native host executes it; the
        # legacy manager flows are deleted. The endpoint still picks
        # prebuilt-vs-source before scheduling.
        "audio_cpp": {"prebuilt", "remove_version", "source", "sync"},
        # llama.cpp and ik_llama.cpp share one source-build function; both
        # definitions now execute it as the "source" recipe, with the
        # characterization pins as the preservation gate.
        "llama_cpp": {"source", "remove_version"},
        "ik_llama": {"source", "remove_version"},
    }
    for definition in registry.all():
        if definition.id == "nativecraft":
            # Third native fork: source-build proof only.
            assert definition.schema_version == 2
            assert set(dict(definition.recipes)) == {"source"}
            continue
        if definition.id in {"launchcraft", "launchforge", "yamlcraft"}:
            # The authoritative fixtures (S4 control plane, R3 launch) are v2
            # BY CONSTRUCTION — never Python-authored to cut over; YAML is
            # their only source. Engine-wide remove only; version-scoped
            # delete is deliberately unproven for the pip-wheel persona.
            assert definition.schema_version == 2
            assert set(dict(definition.recipes)) == {"release", "source", "sync", "remove"}
            continue
        # Only engines whose Phase 3 cutover has shipped carry v2 recipes;
        # descriptors stay registry-equivalent either way (tested above).
        if definition.id in cut_over:
            assert definition.schema_version == 2
            assert set(dict(definition.recipes)) == cut_over[definition.id]
        else:
            assert definition.schema_version == 1
            assert definition.runtime_profiles == ()
            assert definition.recipes == ()


def test_lmdeploy_golden_argv_matches_legacy_launcher(reference):
    """The declared profile argv ==
    [serve, api_server, <hf_id positional>, --server-port, <PORT>]."""
    definition = reference.get("lmdeploy")
    argv = compile_profile_argv(definition, model_ref="Org/Model")
    assert argv == ["serve", "api_server", "Org/Model", "--server-port", "{{port}}"]


def test_sglang_v100_llm_golden_argv_matches_legacy_launcher(reference):
    """launch_spec._compile_module + sglang_server_argv(python_bin, "llm")[1:]."""
    definition = reference.get("sglang_v100")
    argv = compile_profile_argv(definition, profile="llm", model_ref="Org/Model")
    assert argv == ["-m", "sglang.launch_server", "--model-path", "Org/Model", "--port", "{{port}}"]


def test_sglang_v100_diffusion_golden_argv_matches_legacy_launcher(reference):
    """Second runtime profile: cli.main has no __main__ guard, so the fork
    goes through -c and the serve subcommand (engines/sglang/runtime.py)."""
    definition = reference.get("sglang_v100")
    argv = compile_profile_argv(definition, profile="diffusion", model_ref="Org/Model")
    assert argv == [
        "-c",
        "from sglang.cli.main import main; main()",
        "serve",
        "--model-type",
        "diffusion",
        "--model-path",
        "Org/Model",
        "--port",
        "{{port}}",
    ]


def test_extra_argv_appended_after_owned_flags(reference):
    argv = compile_profile_argv(
        reference.get("lmdeploy"),
        model_ref="Org/Model",
        extra_argv=["--cache-max-entry-count", "0.9"],
    )
    assert argv[-2:] == ["--cache-max-entry-count", "0.9"]
    assert argv.index("--server-port") < argv.index("--cache-max-entry-count")


def test_missing_model_reference_is_rejected(reference):
    with pytest.raises(DefinitionError, match="requires a model reference"):
        compile_profile_argv(reference.get("lmdeploy"), model_ref="  ")


def test_unknown_profile_is_rejected(reference):
    with pytest.raises(DefinitionError, match="no runtime profile"):
        compile_profile_argv(reference.get("lmdeploy"), model_ref="X", profile="turbo")


def test_default_profile_resolution(reference):
    # Single profile -> unambiguous; two profiles -> the marked default.
    compile_profile_argv(reference.get("lmdeploy"), model_ref="X")
    argv = compile_profile_argv(reference.get("sglang_v100"), model_ref="X")
    assert argv[:2] == ["-m", "sglang.launch_server"]


# ---------------------------------------------------------------------------
# Env requirement goldens (launch_spec._compile_module V100 rules)
# ---------------------------------------------------------------------------


def test_sglang_v100_requires_managed_cuda(reference):
    result = compile_profile_env(reference.get("sglang_v100"), {}, profile="llm")
    assert result["missing_required"] == ["CUDA_HOME"]


def test_sglang_v100_env_defaults_match_legacy_launcher(reference):
    result = compile_profile_env(
        reference.get("sglang_v100"), {"CUDA_HOME": "/opt/studio/cuda/12.8"}, profile="llm"
    )
    assert result["missing_required"] == []
    assert result["env"]["FLASHINFER_DISABLE_VERSION_CHECK"] == "1"
    assert result["env"]["TORCH_CUDA_ARCH_LIST"] == "7.0"
    assert result["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert result["cwd"] == "venv_root"
    assert result["prepend_bin_to_path"] is True


def test_lmdeploy_env_has_no_required_variables(reference):
    result = compile_profile_env(reference.get("lmdeploy"), {})
    assert result["missing_required"] == []
    assert result["env"] == {}
    assert result["prepend_bin_to_path"] is True
    assert result["cwd"] == "none"


# ---------------------------------------------------------------------------
# Recipes: the worst case is expressible over the shared primitive vocabulary
# ---------------------------------------------------------------------------


def test_sglang_v100_source_recipe_covers_the_real_install_flow(reference):
    recipes = dict(reference.get("sglang_v100").recipes)
    assert "source" in recipes
    # The fork has no release channel: settings carry source_repo/branch only.
    assert "release" not in recipes
    steps = recipes["source"]
    uses = [step.get("uses") for step in steps]
    assert "vcs.apply_patch" in uses  # CUDA-python pin, marlin BF16, all-reduce JIT
    assert "cgroup.select_concurrency" in uses  # memory-aware MAX_JOBS
    step_ids = [step["id"] for step in steps]
    assert len(step_ids) == len(set(step_ids))
    for step in steps:
        if "uses" in step:
            assert step["uses"] in INSTALL_PRIMITIVES


def test_settings_defaults_match_python_side_defaults(reference):
    from backend.venv_install_settings import default_install_settings

    for engine_id in ("lmdeploy", "sglang_v100"):
        declared = dict(reference.get(engine_id).settings_defaults)
        effective = default_install_settings(engine_id)
        # Every declared value is the effective one; the effective map may also
        # carry derived keys (the version-setting key default).
        for key, value in declared.items():
            assert effective[key] == value, (engine_id, key)


def test_model_sources_and_discovery_declared(reference):
    for engine_id in reference.ids():
        definition = reference.get(engine_id)
        assert definition.model_sources == ("hf_repo",)
        assert dict(definition.discovery)["parser"] == ENGINE_REGISTRY[engine_id].scanner_kind


# ---------------------------------------------------------------------------
# Negative cases: v2 vocabulary must reject precisely
# ---------------------------------------------------------------------------


def test_v1_document_cannot_carry_v2_sections():
    raw = _base(recipes={"release": [{"id": "s", "uses": "venv.create"}]})
    raw["schema_version"] = 1
    with pytest.raises(DefinitionError, match="require schema_version 2"):
        validate_definition(raw, source="x.yaml")


def test_unsupported_future_version_rejected():
    with pytest.raises(DefinitionError, match="unsupported schema_version 3"):
        validate_definition(_base(schema_version=3), source="x.yaml")


def test_unknown_primitive_rejected():
    raw = _base(recipes={"release": [{"id": "s", "uses": "pip.do_magic"}]})
    with pytest.raises(DefinitionError, match="unknown primitive 'pip.do_magic'"):
        validate_definition(raw, source="x.yaml")


def test_step_needs_exactly_one_of_uses_or_run():
    raw = _base(recipes={"release": [{"id": "s", "uses": "venv.create", "run": ["echo"]}]})
    with pytest.raises(DefinitionError, match="exactly one of 'uses' or 'run'"):
        validate_definition(raw, source="x.yaml")


def test_requires_must_reference_produced_output():
    raw = _base(
        recipes={
            "release": [
                {"id": "a", "uses": "venv.create", "produces": ["venv"]},
                {"id": "b", "uses": "publish.install", "requires": ["ghost"]},
            ]
        }
    )
    with pytest.raises(DefinitionError, match="requires undeclared output 'ghost'"):
        validate_definition(raw, source="x.yaml")


def test_duplicate_step_id_rejected():
    raw = _base(
        recipes={
            "release": [
                {"id": "a", "uses": "venv.create"},
                {"id": "a", "uses": "venv.create"},
            ]
        }
    )
    with pytest.raises(DefinitionError, match="duplicate step id 'a'"):
        validate_definition(raw, source="x.yaml")


def test_input_context_requires_are_allowed():
    raw = _base(recipes={"release": [{"id": "a", "uses": "venv.create", "requires": ["input.x"]}]})
    definition = validate_definition(raw, source="x.yaml")
    assert dict(definition.recipes)["release"][0]["id"] == "a"


def test_condition_inside_recipe_step_validated():
    raw = _base(
        recipes={
            "release": [
                {"id": "a", "uses": "venv.create", "when": {"operator_missing": {}}},
            ]
        }
    )
    with pytest.raises(DefinitionError, match="unknown condition operator"):
        validate_definition(raw, source="x.yaml")


def test_flag_model_binding_requires_flag():
    raw = _base(
        runtime_profiles={
            "default": {
                "default": True,
                "executable": {"kind": "venv_console_script", "name": "lmdeploy"},
                "model_binding": {"mode": "flag", "source": "huggingface_id"},
                "port_binding": {"flag": "--server-port"},
            }
        }
    )
    with pytest.raises(DefinitionError, match="model_binding.flag"):
        validate_definition(raw, source="x.yaml")


def test_console_script_profile_requires_executable_name():
    raw = _base(
        runtime_profiles={
            "default": {
                "executable": {"kind": "venv_console_script"},
                "model_binding": {"mode": "positional", "source": "huggingface_id"},
                "port_binding": {"flag": "--server-port"},
            }
        }
    )
    with pytest.raises(DefinitionError, match="executable.name"):
        validate_definition(raw, source="x.yaml")


def test_profile_rejects_unknown_keys():
    raw = _base(
        runtime_profiles={
            "default": {
                "executable": {"kind": "venv_python"},
                "model_binding": {"mode": "positional", "source": "huggingface_id"},
                "port_binding": {"flag": "--port"},
                "sudo": True,
            }
        }
    )
    with pytest.raises(DefinitionError, match="unknown keys"):
        validate_definition(raw, source="x.yaml")


def test_env_names_validated():
    raw = _base(
        runtime_profiles={
            "default": {
                "executable": {"kind": "venv_python"},
                "model_binding": {"mode": "positional", "source": "huggingface_id"},
                "port_binding": {"flag": "--port"},
                "env": {"required": ["lowercase-var"]},
            }
        }
    )
    with pytest.raises(DefinitionError, match="invalid name 'lowercase-var'"):
        validate_definition(raw, source="x.yaml")


def test_env_cwd_vocabulary():
    raw = _base(
        runtime_profiles={
            "default": {
                "executable": {"kind": "venv_python"},
                "model_binding": {"mode": "positional", "source": "huggingface_id"},
                "port_binding": {"flag": "--port"},
                "env": {"cwd": "repo_root"},
            }
        }
    )
    with pytest.raises(DefinitionError, match="env.cwd"):
        validate_definition(raw, source="x.yaml")


def test_family_profile_declares_only_the_mechanism():
    """A family profile names WHO composes the launch; it declares no argv."""
    raw = _base(
        install_kind="cmake",
        runtime_kind="llama_server",
        scanner_kind="llama_help",
        active_path_fields=["binary_path"],
        artifact_formats=["gguf"],
        package_kinds=["single_file"],
        runtime_profiles={
            "default": {
                "default": True,
                "launch_family": "gguf_llama_server",
                "port_binding": {"flag": "--port"},
            }
        },
    )
    definition = validate_definition(raw, source="x.yaml")
    spec = definition.profile("")
    assert spec["launch_family"] == "gguf_llama_server"
    assert spec["port_binding"] == {"flag": "--port"}
    assert "executable" not in spec
    assert "model_binding" not in spec


def test_unknown_launch_family_rejected():
    raw = _base(
        runtime_profiles={
            "default": {
                "launch_family": "sorcery",
                "port_binding": {"flag": "--port"},
            }
        }
    )
    with pytest.raises(DefinitionError, match="launch_family .* not a known kind"):
        validate_definition(raw, source="x.yaml")


def test_family_profile_requires_port_binding():
    raw = _base(runtime_profiles={"default": {"launch_family": "gguf_llama_server"}})
    with pytest.raises(DefinitionError, match="requires 'port_binding'"):
        validate_definition(raw, source="x.yaml")


@pytest.mark.parametrize(
    "key,value",
    [
        ("executable", {"kind": "native_artifact", "name": "binary_path"}),
        ("model_binding", {"mode": "positional", "source": "huggingface_id"}),
        ("env", {"defaults": {"FOO": "bar"}}),
        ("scan", {"unavailable_hint": "rescan"}),
        ("runtime_switch", {"config_key": "mode", "variants": {"a": {}}}),
    ],
)
def test_family_profile_rejects_mechanism_composed_keys(key, value):
    """Declared-but-ignored keys are the failure mode this vocabulary forbids."""
    profile = {
        "default": True,
        "launch_family": "gguf_llama_server",
        "port_binding": {"flag": "--port"},
        key: value,
    }
    raw = _base(runtime_profiles={"default": profile})
    with pytest.raises(DefinitionError, match="cannot also declare"):
        validate_definition(raw, source="x.yaml")


def test_family_vocabulary_matches_the_implemented_mechanism():
    """The schema vocabulary and the interpreter's mechanism cannot drift."""
    from backend.engines.definitions.schema import LAUNCH_FAMILIES
    from backend.proxy import gguf_launch

    assert LAUNCH_FAMILIES == frozenset({gguf_launch.LAUNCH_FAMILY})


def test_shadow_argv_evaluator_refuses_family_profiles():
    """The dormant shadow evaluator must say so, not KeyError on the binding."""
    raw = _base(
        runtime_profiles={
            "default": {
                "default": True,
                "launch_family": "gguf_llama_server",
                "port_binding": {"flag": "--port"},
            }
        }
    )
    definition = validate_definition(raw, source="x.yaml")
    with pytest.raises(DefinitionError, match="launch family"):
        compile_profile_argv(definition, model_ref="Org/Model")


def test_empty_recipe_rejected():
    with pytest.raises(DefinitionError, match="must not be empty"):
        validate_definition(_base(recipes={"release": []}), source="x.yaml")


# ---------------------------------------------------------------------------
# Typed install plans (Phase 3 seed): materialize + shadow-compare
# ---------------------------------------------------------------------------


def test_lmdeploy_release_plan_order_and_bindings(reference):
    """Shadow comparison against today's shared python-venv install flow
    (backend/engines/python + lmdeploy/installer.py): resolve version,
    create venv, pip install, capture metadata, publish."""
    from backend.engines.definitions.plan import materialize_plan

    plan = materialize_plan(reference.get("lmdeploy"), "release")
    assert [step.id for step in plan.steps] == [
        "resolve_version",
        "create_venv",
        "install_package",
        "capture_metadata",
        "publish",
    ]
    assert plan.required_inputs == ("input.engine_root",)
    install = plan.by_id("install_package")
    assert install.primitive == "pip.install_package"
    assert install.with_["version_from"] == "version"
    assert install.requires == ("venv", "version")


def test_sglang_v100_source_plan_is_dependency_ordered(reference):
    from backend.engines.definitions.plan import materialize_plan

    plan = materialize_plan(reference.get("sglang_v100"), "source")
    order = [step.id for step in plan.steps]
    assert order.index("clone_repository") < order.index("checkout_branch")
    assert order.index("checkout_branch") < order.index("pin_cuda_python")
    assert order.index("install_requirements") < order.index("build_wheel")
    assert order.index("build_wheel") < order.index("install_wheels")
    assert order.index("smoke_test") < order.index("publish")
    assert all(step.primitive is not None for step in plan.steps)  # no raw shell


def test_unknown_recipe_rejected(reference):
    from backend.engines.definitions.plan import materialize_plan

    with pytest.raises(DefinitionError, match="no recipe 'binary'"):
        materialize_plan(reference.get("lmdeploy"), "binary")


def test_materializer_rejects_out_of_order_dependency(reference):
    """The loader would never produce this, but the materializer defends
    the invariant independently (hand-built registries, future APIs)."""
    from backend.engines.definitions.plan import materialize_plan

    definition = reference.get("lmdeploy")
    broken = dict(definition.recipes)
    steps = list(broken["release"])
    install = next(step for step in steps if step["id"] == "install_package")
    steps.remove(install)
    steps.insert(0, install)  # consumes 'venv' before create_venv produces it
    broken["release"] = tuple(steps)
    mutated = replace(definition, recipes=tuple(broken.items()))
    with pytest.raises(DefinitionError, match="before it is produced"):
        materialize_plan(mutated, "release")
