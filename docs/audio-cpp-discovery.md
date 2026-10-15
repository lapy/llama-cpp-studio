# audio.cpp integration contract and metadata discovery

Studio reads everything it needs to reach and configure an audio.cpp engine from
one composed **integration contract** (schema `audio-cpp.studio-contract/v0`).
The contract is assembled per engine version by
`backend/engines/audio_cpp/integration/`:

- `contract_registry.py` — the Studio-owned operation table (speech,
  transcriptions, transcriptions/details, alignments, batches/transcriptions,
  tasks/run, voices, models, health) with native and public (llama-swap) paths,
  methods, encodings, responses, the deterministic task→operation table, the
  request-defaults bucket per operation, and the client body shape
  (`openai-json` / `multipart-form` / `tasks-run`). This table is a temporary
  shim for upstream issue `0xShug0/audio.cpp#852`: every entry carries
  provenance `studio` and is deleted individually once upstream publishes the
  same facts.
- `composer.py` — overlays the registry with engine observations:
  `--list-loaders --json` (family endpoints, tasks, modes, policies),
  model-aware `--inspect` (`preferred_api_endpoint`, per-task metadata), and
  `model_specs/<family>.json` (typed options and defaults). Observed data
  always beats a `studio` entry; an observation that contradicts the table is
  recorded as a **drift warning**, never silently patched.
- `upstream_adapter.py` — parses the upstream contract document once
  audio.cpp#852 lands (provenance flips to `upstream`).

Modern engine builds are hard-required: activation runs `--list-loaders --json`
through a route gate and refuses any build that does not answer with JSON
(update audio.cpp to v0.9.1 or newer). There are no text-parsing fallbacks, no
heuristic family matching, and no source/docs scraping anywhere in Studio.

`GET /api/audio-cpp/contract` exposes the composed version descriptor
(operations, `task_routing`, `family_routing`, `defaults_keys`, drift warnings,
fingerprint). Per-model composition rides the param-registry payload as
`request_policy` (operation id, api/native/studio paths, body shape,
defaults key, instructions policy and its provenance). The frontend
(`frontend/src/composables/useAudioContract.js`) reads the same descriptor and
never re-derives routes from task or family strings; its offline floor contains
only Studio's own generic route so forms can render before the first load.

## Model metadata

Studio discovers server flags from `audiocpp_server --help`, families from the
loader listing, and model capabilities from model-aware CLI help and
inspection. Engine installation and the parameter scan refresh this catalog.
Model profiles are scanned lazily and cached separately for each engine/model.

Studio does not infer tasks or access restrictions from model names or
organizations. Catalog identity comes from the active engine's catalog, loader
listing, and spec package declarations; missing identity stays unknown. Utility
packages (VAD, aligners, separators) are recognized by exact catalog package
ids, never by fuzzy name matching. There is no curated family profile or
request-field catalog in Studio.

`model_specs/<family>.json` supplies family names, descriptions, tasks,
capabilities, dependencies, voice metadata, and typed request, session, and
load options. New families and options using these existing transports do not
require a Studio family table. A configured `model_spec_override` file or
directory takes precedence over the installed checkout for model metadata as
well as engine execution.

For an option declared by both help and a spec, the structured spec owns its
type, default, range, and explicit enum values. Named enum presets use the
choices expanded by engine help. Model-specific help options missing from the
spec remain available; unrelated global CLI flags are not added to spec-backed
request forms.

Request options appear in Request Defaults, while load and session options
retain their separate server configuration scopes. Saved request defaults are
partial: known supplied values are checked against discovered types, enum
values, numeric ranges, and finite-number requirements. Unknown options remain
preserved for forward compatibility. Required request inputs can still be
supplied at request time.

Profile cache keys include the spec contents, CLI binary identity, load
options, and spec override. Editing a spec or replacing the binary therefore
invalidates the profile without changing its engine version label. An explicit
engine rescan also clears model profiles. Request-default edits do not trigger
inspection. Activation rescans are fingerprint-gated: models are rescanned only
when their profile cache is missing/stale or the contract change touches their
family/task, and probes run with bounded concurrency.

## Routing and process ownership

Studio owns routing and process isolation (model path, host, port, and
generated configuration). Studio-owned CLI flags are declared exactly once as
`STUDIO_OWNED_FLAGS` in the integration registry and filtered from user-supplied
config in both directions. Other advertised server flags are emitted from the
scan registry. New installs leave thread count, device index, and lazy loading
unset so audio.cpp chooses its defaults; explicit saved values continue to take
precedence. Backend selection follows the installed build.

Endpoint resolution is a contract lookup with a strict order: inspect
`preferred_api_endpoint` (or the per-task row), the family's advertised
`api_endpoints` from the loaders JSON, the model spec, then the deterministic
task table. An unknown task always falls through to `tasks_run`. Engine-reported
routes the registry has never seen still resolve (through the generic route) and
are reported as drift, so a new upstream route needs no Studio change once the
engine advertises it.

Instructions require an explicit engine policy (`--inspect` / loaders JSON /
spec), or a declared instruction/caption option in the engine's own option list.
A family name, general speech documentation, or voice-design capability alone
never implies support for an instructions field. Session-voice defaults apply
only to speech tasks, not conversion tasks sharing the same family.

Generic workspace forms use `workspace_request_fields` from the parameter
registry. Model-aware input help supplies top-level inputs; request-option rows
supply nested `options`. Structured spec types, defaults, and constraints
overlay help. Music, conversion, separation, analysis, design, and unknown
tasks share this renderer and serializer. They do not invent input aliases or
extra fields. Without an advertised schema, generic inference is disabled with
a scan hint. Scanned model help can provide fields even without a source
checkout.

Community preset voices are derived from the active checkout's
`webui/native/demo_voices` directory (names and `prompt_text`), not from a
pinned commit or checksum table, so installing preset voices always matches
the engine version in use.

## UI and server frontends

The server UI is disabled unless the active installation explicitly records
`build_server_frontends` enabled. Unknown/prebuilt build settings also default
off. The parameter registry exposes a forced false value, saves clear stale UI
settings, and generated runtime configuration sets both `ui` and
`ui_management` false. UI switches cannot bypass that policy through custom
arguments. The Studio audio workspace is independent of this upstream embedded
UI setting.

## Regression coverage

Contract tests (`backend/tests/test_audio_cpp_integration_contract.py`) pin the
registry, composer resolution order, drift warnings, the activation gate, and
`GET /api/audio-cpp/contract` against a captured v0.9.1 `--list-loaders --json`
payload (`backend/tests/audio_profile_fixtures.py`). Activation-path tests use
an executable fake CLI that answers the JSON probe. Additional coverage
includes a fictional family gaining an option after a spec edit, custom
overrides, enum presets, scope collisions, request validation, and server flags
that share names with CLI flags. Live discovery tests can also run against
`AUDIO_CPP_CLI`, `AUDIO_CPP_MODEL`, and `AUDIO_CPP_SOURCE`, or the local
`data/audio-cpp/src` build and prepared ASR bundles.

Access status is explicitly `gated`, `public`, or `unknown`. Catalog booleans
and package download metadata in the active engine spec provide access evidence;
missing metadata is shown as unknown rather than inferred from an organization.

Compact model-manager catalog rows are completed from exact package declarations
in the active engine's specs. Package download defaults and overrides supply the
repository, revision, access flag, files, and strip prefix. Remote file paths
are kept separately from required installed paths after prefix stripping. When
the engine declares an exact file list, Studio does not add guessed GGUF sidecar
files. Legacy catalog rows without file declarations retain the compatibility
fallback.
