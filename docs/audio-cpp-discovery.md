# audio.cpp metadata discovery

Studio discovers the installed engine's server flags from `audiocpp_server --help`,
families from the loader listing, and model capabilities from model-aware CLI help
and inspection. Engine installation and the parameter scan refresh this catalog.
Model profiles are scanned lazily and cached separately for each engine/model.

Studio does not infer tasks or access restrictions from model names or organizations.
Catalog identity comes from the active engine's catalog, loader listing, and spec
package declarations. Missing identity stays unknown and unverified; generic fuzzy
matching is available only through explicit `AUDIO_CPP_HEURISTIC_DISCOVERY=1`.
There is no curated family profile or request-field catalog in Studio.

`model_specs/<family>.json` supplies family names, descriptions, tasks, capabilities,
dependencies, voice metadata, and typed request, session, and load options. New
families and options using these existing transports do not require a Studio
family table. A configured `model_spec_override` file or directory takes precedence
over the installed checkout for model metadata as well as engine execution.

For an option declared by both help and a spec, the structured spec owns its type,
default, range, and explicit enum values. Named enum presets use the choices
expanded by engine help. Model-specific help options missing from the spec remain
available; unrelated global CLI flags are not added to spec-backed request forms.
The optional `AUDIO_CPP_SOURCE_OPTION_DISCOVERY` fallback supports older engines by
reading loader/source/docs metadata. It does not replace explicit spec metadata.

Request options appear in Request Defaults, while load and session options retain
their separate server configuration scopes. Saved request defaults are partial:
known supplied values are checked against discovered types, enum values, numeric
ranges, and finite-number requirements. Unknown options remain preserved for
forward compatibility. Required request inputs can still be supplied at request
time.

Profile cache keys include the spec contents, CLI binary identity, load options,
and spec override. Editing a spec or replacing the binary therefore invalidates
the profile without changing its engine version label. An explicit engine rescan
also clears model profiles. Request-default edits do not trigger inspection.

Studio owns routing and process isolation (model path, host, port, and generated
configuration). Other advertised server flags are emitted from the scan registry.
New installs leave thread count, device index, and lazy loading unset so audio.cpp
chooses its defaults; explicit saved values continue to take precedence. Backend
selection follows the installed build.

The server UI is disabled unless the active installation explicitly records
`build_server_frontends` enabled. Unknown/prebuilt build settings also default off.
The parameter registry exposes a forced false value, saves clear stale UI settings,
and generated runtime configuration sets both `ui` and `ui_management` false.
UI switches cannot bypass that policy through custom arguments. The Studio audio
workspace is independent of this upstream embedded UI setting.

Regression coverage includes a fictional family gaining an option after a spec
edit, custom overrides, enum presets, scope collisions, request validation, and
server flags that share names with CLI flags. Live discovery tests can also run
against `AUDIO_CPP_CLI`, `AUDIO_CPP_MODEL`, and `AUDIO_CPP_SOURCE`, or the local
`data/audio-cpp/src` build and prepared ASR bundles.

The selected task chooses the fallback API route, even for families advertising
both speech and conversion. An explicit engine request surface takes precedence;
per-task inspection metadata takes precedence over model-level metadata.
Instructions require an explicit engine policy or declared instruction option.
A family name, general speech documentation, or voice-design capability alone
never implies support for an instructions field. Session-voice defaults apply
only to speech tasks, not conversion tasks sharing the same family.

Generic workspace forms use `workspace_request_fields` from the parameter registry.
Model-aware input help supplies top-level inputs; request-option rows supply
nested `options`. Structured spec types, defaults, and constraints overlay help.
Music, conversion, separation, analysis, design, and unknown tasks share this
renderer and serializer. They do not invent input aliases or extra fields.
Without an advertised schema, generic inference is disabled with a scan hint.
Scanned model help can provide fields even without a source checkout.

Access status is explicitly `gated`, `public`, or `unknown`. Catalog booleans and
package download metadata in the active engine spec provide access evidence;
missing metadata is shown as unknown rather than inferred from an organization.

Compact model-manager catalog rows are completed from exact package declarations
in the active engine's specs. Package download defaults and overrides supply the
repository, revision, access flag, files, and strip prefix. Remote file paths are
kept separately from required installed paths after prefix stripping. When the
engine declares an exact file list, Studio does not add guessed GGUF sidecar
files. Legacy catalog rows without file declarations retain the compatibility
fallback.
