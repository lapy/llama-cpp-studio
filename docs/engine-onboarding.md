# Adding an engine: the definition-only path

Status: current, enforced by
`backend/tests/test_engine_onboarding_is_data_only.py` (2026-11-15).

This page describes how a new engine is onboarded as **data**: one validated
YAML definition, no new per-engine Python module, route, installer class, or
frontend entry. It also states precisely where code is still required and which
generated assets must be refreshed.

Related: [engine host and presentation boundaries](engine-host-and-presentation-boundaries.md),
[engine surface contract matrix](engine-surface-contract-matrix.md),
[YAML onboarding campaign record](yaml-engine-onboarding-campaign.md).

## 1. Add the definition

Create `backend/engines/definitions/<engine_id>.yaml`. The filename stem MUST
equal the declared `id`; the loader rejects mismatches and duplicate ids.

Required keys (schema v1, still required with `schema_version: 2`):
`schema_version`, `id`, `label`, `repository_source`, `install_kind`,
`runtime_kind`, `scanner_kind`, `active_path_fields`, `artifact_formats`,
`package_kinds`, `tasks`, `input_modalities`, `output_modalities`.
Optional: `supports_embeddings`, `experimental`, `maturity_surfaces`, `extends`
(named templates from `_templates.yaml`).

With `schema_version: 2`, these sections are also accepted and validated
(strict: unknown fields refuse startup):

| Section | What it drives |
| --- | --- |
| `lifecycle` | Generic lifecycle record: update source, binary/root/log/state names, URL prefix, status shape, install-settings fallbacks, override env, Python range |
| `settings_defaults` | Persisted install/build-settings defaults served by the generic settings verbs |
| `runtime_profiles` | Launch profiles (`executable`, `model_binding`, `port_binding`, `argv_prefix`, `env`) compiled by the shared launch path |
| `recipes` | Install/source/sync/remove plans over the shared `INSTALL_PRIMITIVES` vocabulary, executed by `GenericVenvInstaller` / native hosts |
| `discovery` | Declared help-probe argv/cwd/env and parser kind for parameter discovery |
| `launch_policy` | CPU support, GPU architectures, GPU-only env invariants, runtime-context key |
| `build_options` / `build_defaults` | Native build-option catalog and per-engine defaults |
| `protocol`, `extensions`, `request_defaults` | Engine protocol flags, declared extension endpoints, request-default vocabularies |
| `catalog`, `model_compat`, `model_sources` | Model-catalog providers, format defaults, accepted model sources |
| `upstream`, `data_dirs`, `source_build`, `release_assets`, `native_host` | Fork identity, managed data dirs, native build/release policy and host selection |
| `ui_kind` | Frontend presentation family (`audio`, `sglang`, `native`, `venv`; empty = generic card) |

A minimal definition (copy the structure of the closest shipped family in
practice):

```yaml
schema_version: 2
id: fixture_echo
label: Fixture Echo
repository_source: Fixture Echo
install_kind: python_venv
runtime_kind: lmdeploy
scanner_kind: lmdeploy_help
active_path_fields: [venv_path]
artifact_formats: [safetensors]
package_kinds: [hf_snapshot]
tasks: [text-generation]
input_modalities: [text]
output_modalities: [text]
ui_kind: venv
lifecycle:
  update_source: {kind: pypi, package: fixture-echo}
  binary_name: fixture-echo
  default_repo_url: https://github.com/example/fixture-echo.git
settings_defaults: {source_repo: https://github.com/example/fixture-echo.git, source_branch: main}
runtime_profiles:
  default:
    default: true
    executable: {kind: venv_console_script, name: fixture-echo}
    argv_prefix: [serve]
    model_binding: {mode: flag, flag: --model, source: huggingface_id}
    port_binding: {flag: --port}
recipes:
  release:
    - {id: venv, uses: venv.create, produces: [venv]}
    - {id: install, uses: pip.install_package, requires: [venv], with: {package: fixture-echo}}
```

## 2. What startup derives (no code)

`backend/main.py` runs two gates on every startup:

1. `verify_engine_definitions()` re-loads and validates every definition,
   refuses drift for engines already in the registry, and registers a **new id**
   as an `authoritative` definition (`register_engine_spec`) into the derived
   `ENGINE_REGISTRY` (backend/engines/registry.py also derives the id-sets,
   capability lookups and descriptor payload from it).
2. `register_definition_lifecycles()` materializes, from the YAML: the generic
   `EngineLifecycle` record (unless a compatibility router already registered a
   definition-derived one) and the persisted install-settings defaults.

The existing generic surfaces then serve the engine with no engine-specific
code:

- `GET /api/engines` — descriptor, with `generic_lifecycle: true` and the
  `generic_verbs` the record actually serves.
- `GET|PUT /api/engines/{engine_id}/build-settings`
- `GET /api/engines/{engine_id}/status`
- `POST /api/engines/{engine_id}/{install,install-source,remove,cancel}`
  (`install-source` only when the definition declares a `source` recipe;
  otherwise 405)
- `GET /api/engines/{engine_id}/check-updates` (when `lifecycle.update_source`
  is declared; otherwise 405)
- `GenericVenvInstaller` executes the declared recipes through shared
  primitives; `compile_profile_argv` / `compile_profile_env` and the launch
  interpreter read `runtime_profiles`; discovery runs the declared probes.
- The frontend needs no local entry: server-backed descriptors feed
  `serverBackedDescriptor` / `presentationForEngine` (label, verbs, generic
  card/dialog) using `generic_lifecycle`, `generic_verbs` and `ui_kind`.

### Native (install-dir) engines

An engine that installs a directory tree instead of a Python virtualenv declares
`native_host` (`cmake_source`, `audio_cpp` or `release_assets`) and, when it is
a YAML-only engine, a `lifecycle` block plus its `recipes`. Startup then builds
the same generic `EngineLifecycle` record, but the installer comes from
`get_native_installer(engine_id)` — the host named by the definition, not the
venv installer and never an engine-id branch. The record serves the same verbs
over `/api/engines/{engine_id}` with no legacy `/api/<segment>` prefix:

| Verb | Native plan |
| --- | --- |
| `install` | the declared `release` recipe through the host |
| `status` / `cancel` / `remove` | the host's install-dir lifecycle |
| `build-settings` | the declared `settings_defaults` surface |
| `check-updates` | only when `lifecycle.update_source` is declared |
| `install-source` | only when a `source` recipe is declared; otherwise 405 |

The installer is resolved lazily and cached on the record, so an in-flight
operation and its cancellation task are the same host instance for every
request. A `native_host` that cannot be resolved is a loud error, never a
silent fallback to the venv installer. `docs/strata-engine.md` records the
planned Strata integration as the worked example: it declares no engine today
because its Terminal-owned model preparation has no host yet.

## 3. What still requires code — declared mechanisms, not engine ids

- `install_kind`, `runtime_kind` and `scanner_kind` are **closed vocabularies**
  (backend/engines/definitions/schema.py). A new engine that joins an existing
  mechanism family is data-only. A genuinely new mechanism — a new inference
  protocol, native build host or help parser — needs one shared, kind-keyed
  implementation. Engine-id branches are never the correct extension point; see
  [engine host and presentation boundaries](engine-host-and-presentation-boundaries.md).
  Concretely, a new `native_host` value is one host factory in
  `backend/engines/adapters.py` (`_NATIVE_HOST_FACTORIES`) plus the shared host
  class; definitions then select it by name.
- The frontend `ENGINE_DESCRIPTORS` table is static for shipped engines (icons,
  copy, specialty panels) and doubles as the offline fallback. It is not
  required for a new engine; specialty UI for a new presentation family needs a
  `ui_kind`/descriptor, which is presentation work, not onboarding plumbing.
- A YAML-only engine serves only the generic `/api/engines/{id}` routes. Legacy
  `/api/<segment>` compatibility prefixes are mounted by converted engines'
  Python modules by decision; new engines do not get one.

## 4. The enforcing test

```powershell
python -m pytest backend/tests/test_engine_onboarding_is_data_only.py -q -p no:cacheprovider
```

The test copies the structure of the shipped `lmdeploy.yaml` into a temp
directory, renames it to a new id (`fixture_echo`) and proves:

1. `load_engine_definitions(<temp dir>)` validates it (and rejects a mutated
   copy with an unknown field, so the schema gate is live);
2. the real startup gates accept it (`verify_engine_definitions` count =
   shipped + 1, registry entry == validated definition, authoritative tier);
3. TestClient calls to `GET /api/engines`, `GET /api/engines/fixture_echo/status`
   and `.../build-settings` succeed with no process spawn and no new route;
4. `runtime_profiles`, `recipes` and `settings_defaults` resolve from declared
   data (compiled argv/env, materialized release plan, coerced settings);
5. an architecture guard finds the fixture id in **no** backend/frontend source
   file (tests/YAML/docs excluded).

Loader directory override: the supported hook is the loader's `directory`
argument (`load_engine_definitions`). The startup gates read the process-wide
singleton `load_default_engine_definitions()`, which has no directory override,
so the test patches exactly that one singleton (`loader._default_registry`) with
shipped + fixture definitions. It does not patch module constants, routes or
lifecycle registration. The fixture lives in `tmp_path`, so no shipped
definition, route, inventory or generated asset changes.

Recommendation (non-blocking): if an alternate definitions directory ever
becomes a supported deployment mode, expose it as an explicit setting inside
`load_default_engine_definitions()`; this test would then drop the singleton
patch. Onboarding into the shipped directory does not need it.

## 5. Refresh generated assets when the definition ships

Adding a definition to the shipped directory changes `GET /api/engines`
payloads and the derived registry snapshot. Refresh and commit:

| Artifact | Command | Needed when |
| --- | --- | --- |
| `docs/engine-parity-inventory.{json,md}` | `python scripts/parity-inventory.py` | Any shipped engine added/removed/renamed (it snapshots the registry) |
| `frontend/src/api/openapi.json` | `npm run openapi:export` | Only if routes or typed models changed — a plain definition adds none |
| `frontend/src/engines/build-defaults.generated.json` | `python scripts/export-build-defaults.py` | Only if the llama build-option catalog changed |
| `docs/legacy-route-inventory.md` | `python scripts/legacy-route-inventory.py` | Only if legacy compatibility route modules/prefixes changed |

Checks (also run in `npm run lint`): `python scripts/engine-literals-ratchet.py`,
`python scripts/module-size-ratchet.py`, `npm run engine-parity:check`,
`npm run openapi:check`, `npm run build-defaults:check`,
`npm run legacy-routes:check`.

## 6. Verification boundary

The acceptance test proves the control-plane/data path: validation, registry
derivation, generic HTTP resolution, profile/recipe resolution and absence of
engine-specific source. It does not execute an install (offline recipe execution
is covered by `backend/tests/test_engine_definitions_generic.py`), run the
frontend (covered by `frontend/src/engines/tests/presentationOverlay.test.js`
and `frontend/src/stores/engines.server.test.js`), or verify real inference.
Native hosts remain covered by their own host tests.
