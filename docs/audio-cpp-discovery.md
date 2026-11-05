# audio.cpp contract, discovery, and transport

Studio's audio integration composes engine and model metadata into the contract schema `audio-cpp.studio-contract/v0`. The same descriptor drives package discovery, model configuration, request defaults, workspace task selection, and endpoint routing.

## Contract sources and ownership

| Source                                 | Supplies                                                                                                                                                   |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Studio integration registry            | Operation IDs, supported transport shapes, public/native paths, methods, encodings, response kinds, task routing, defaults buckets, and managed CLI flags. |
| `--list-loaders --json`                | Engine family/task/mode declarations, endpoints, capabilities, and policies.                                                                               |
| Model-aware `--inspect` and help       | Preferred/per-task endpoints, model capabilities, inputs, option declarations, and execution constraints.                                                  |
| `model_specs/<family>.json`            | Package identity, tasks, typed load/session/request options, defaults, ranges, dependencies, and voice metadata.                                           |
| Configured model-spec override         | Explicit metadata/execution spec source for the configured model.                                                                                          |
| Available structured upstream contract | Additional observed facts composed by the upstream adapter.                                                                                                |

`backend/engines/audio_cpp/integration/contract_registry.py` defines Studio-owned operations and transport policy. `composer.py` overlays observations and assigns provenance. Observed facts take precedence over the corresponding registry floor. Contradictions produce drift warnings and a descriptor fingerprint rather than silent reinterpretation.

Activation requires a successful structured loader JSON probe. A build without the required interface receives an actionable activation error. Engine and model scans use structured metadata and executable help; a model name or organization does not establish task, family, instruction support, or access permissions.

`GET /api/audio-cpp/contract` returns the version descriptor with operations, task/family routing, defaults keys, fingerprints, and warnings. Model parameter registries also contain `request_policy`: operation identity, relevant paths, body shape, defaults key, instructions policy, and provenance. The frontend's `useAudioContract` reads that descriptor.

## Operation and transport model

The operation table covers speech, transcription, transcription details, alignments, batch transcription, generic tasks, voices, model metadata, and health. Each operation describes:

- Its native engine path and Studio-facing path.
- Allowed method and body encoding.
- Response kind, including JSON/text/audio/stream behavior where declared.
- Whether requests use OpenAI JSON, multipart form data, or generic task-run structure.
- Applicable task/family routing and request-default bucket.
- Instruction behavior and its source.

Endpoint resolution uses model inspection preference/per-task metadata, family-advertised endpoints, model spec declarations, and the deterministic task table. An unknown task resolves through the generic task transport. An advertised route that does not match the operation table appears as drift and can use the generic transport according to the composed policy.

## Model/package discovery

Audio search uses exact package and loader/spec declarations. The normalized catalog records provider/source identity, package kind, artifact format, task/modality/capability information, acquisition method, and access evidence.

Access state is `public`, `gated`, or `unknown`. Missing access metadata stays unknown. Utility packages such as aligners, VAD models, and separators use exact advertised IDs. Built-in utility execution requires inspection to accept the engine-owned ID.

Compact catalog rows acquire their details from matching package declarations. Download defaults and explicit overrides provide repository, revision, access information, remote files, and strip-prefix rules. Remote file identity and required installed paths remain distinct. Exact file declarations determine acquisition without guessing additional companion files.

## Installation and preparation

The active checkout's model-manager interfaces provide package listing, spec-backed snapshot acquisition, and applicable assembly/conversion flows. Studio selects the available manager suited to the package method. Spec-backed acquisition can avoid a Torch environment; a conversion/assembly helper can require its own Python/Torch stack.

Studio owns the package installation task, staging/progress, cancellation integration, validation, and promotion into the installed model directory. A local prepared-directory import supplies an additional acquisition route. Inspection validates the prepared package before it becomes an installed model entry.

Engine installation is separate from package acquisition. A prepared audio model requires a compatible active audio.cpp runtime and any declared load/session assets or dependencies.

## Parameter discovery and precedence

Studio scans server flags from `audiocpp_server --help`, families from loader JSON, and model behavior from model-aware inspection/help and specs. Engine scans refresh the catalog. Model profiles are cached separately for each engine/model.

When help and a spec describe the same option, structured spec metadata owns its type, default, range, and explicit enum values. Named enum presets use help-expanded choices. Model-specific options absent from the spec remain available; unrelated global CLI flags do not populate request forms merely because they appear in global help.

A `model_spec_override` file or directory supplies model metadata and the matching engine-execution override. Profile cache identity includes spec contents, CLI binary identity, load options, and the override. A spec/binary/load-setting change can require inspection even when an engine version label stays the same.

Explicit engine rescan clears relevant model profiles. Activation uses fingerprints and capability differences to choose affected model rescans, with bounded probe concurrency. Request-default edits do not themselves require loading/inspecting a model again.

## Configuration scopes

| Scope                       | Stored/control purpose                                                      |
| --------------------------- | --------------------------------------------------------------------------- |
| Engine server flags         | Process/server behavior discovered from the active executable.              |
| Family/task/mode            | Engine-advertised model role and task selection.                            |
| Backend/device              | Execution target supported by the build/model.                              |
| Load options                | Weight/model loading and resource choices.                                  |
| Session options             | Session-scoped engine configuration.                                        |
| Request defaults            | Partial defaults applied when that request does not supply a value.         |
| Reference assets and voices | File/voice identifiers used by the supported request form.                  |
| Native launch configuration | Studio-managed executable, model paths, port, and generated server sidecar. |

The native compiler creates one model's server configuration and artifacts per real model. Studio declares managed CLI flags centrally as `STUDIO_OWNED_FLAGS`; user settings cannot replace model identity, host, port, and generated-config ownership. Other supported flags use the scan registry.

Thread count, device, and lazy-loading values can remain inherited from the engine when no explicit override exists. Explicit values compile through their supported controls. Build backend and device availability follow the installed runtime and observed metadata.

## Request validation and instructions

Saved request defaults are partial. Known supplied fields validate against discovered JSON types, explicit enums, numeric bounds, and finite-number constraints. Unrecognized option data remains available for compatible engine evolution. Required request inputs can be supplied per invocation.

An instructions field requires an explicit engine policy or an instruction/caption option declared in the model's own option metadata. A family name or broad voice-design capability alone does not establish an instructions contract. Session-voice defaults apply to speech tasks rather than automatically to conversion tasks using the same family.

Native request presets can force supported JSON fields, including nested `options` or request kwargs. They share the same real engine process. Concrete engine-specific audio passthrough retains its explicit parameter-preset restriction. Ordinary API aliases can select the same concrete model transport.

## Audio workspace forms

Speech and Transcribe provide dedicated controls for the applicable operations. Other tasks use a metadata-driven renderer and serializer. `workspace_request_fields` describes top-level inputs; request-option rows describe nested `options`. Spec constraints overlay the discovered fields.

Tasks can include music/generation, conversion, separation, analysis, design, diarization, alignment, speaker operations, or model-specific tasks. The form uses advertised names and body structure. An unavailable input schema disables generic inference with a scan/inspection hint instead of inventing required inputs.

Model selection and supported deep links use `/audio?model=<catalog-id>&tab=<task-panel>`. Models and Model configuration expose Audio navigation for compatible packages. The workspace reads the saved model's task/defaults/assets and keeps per-request input controls distinct from persisted launch configuration.

## Voices and reference assets

Voice choices can come from packaged model voice metadata, model reference assets, and community presets supported by the engine/model. Community preset discovery reads the active checkout's `webui/native/demo_voices` metadata, including names and prompt text.

Reference-audio list/upload/delete endpoints operate on the selected model's reference storage. Voice-reference resolution follows the model/defaults policy and its storage key. Portable configuration export excludes the reference audio file content; full operational backup includes the files when those voices are required.

## Management-origin and inference-origin transport

The management service exposes `/v1/audio/…` and `/v1/tasks/run` transport routes. The native inference origin supplies OpenAI-compatible audio paths, `/audioapi/v1/tasks/run`, and `/upstream/{model}/…` concrete engine passthrough.

Transcription upload transport converts input to WAV only when the engine contract requires it. Speech transport converts a WAV response when the requested output format and engine behavior require conversion. Applicable conversions use `ffmpeg`. Otherwise transport retains the engine's supported request/response form.

Alignments and transcription details use concrete engine-specific routing when required by the operation contract. The target uses the stable native model identity. Errors remain explicit for unsupported model selection, missing assets, invalid paths, absent schema, activation failure, or request-field validation.

## Embedded engine UI policy

The engine's embedded server UI requires an active installation that explicitly records `build_server_frontends` enabled. Unknown/prebuilt build settings default off. The parameter registry, saved configuration policy, generated `ui`/`ui_management` values, and custom-argument validation enforce the applicable build policy.

Studio's Audio workspace uses its own browser/request components and operates independently of the embedded engine UI switch.

## Observability and validation

Contract responses carry fingerprints, provenance, and drift warnings. Activation returns operation/progress data and capability changes; model configuration surfaces relevant review/scan hints. Runtime failures expose task/request details and the real model's process logs.

Tests cover the operation registry, composer precedence, drift warnings, structured activation gate, contract endpoint, model-spec overrides, typed enums/ranges, profile invalidation, request scope conflicts, instructions policy, package acquisition, file declarations, voice handling, server UI policy, and generic forms. Environment-driven discovery tests can use `AUDIO_CPP_CLI`, `AUDIO_CPP_MODEL`, and `AUDIO_CPP_SOURCE` for a suitable actual runtime/model.

Successful metadata composition does not itself measure audio quality, latency, device support, or every family's runtime correctness. Those outcomes require the applicable engine/model/assets and a successful request on the target machine.
