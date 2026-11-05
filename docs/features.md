# Studio feature guide

Studio manages local inference engines, model files, model configuration, audio workflows, and client access through a browser interface. The application serves a Vue frontend and FastAPI management API, and runs a native inference proxy that owns engine subprocesses. Persistent state lives in the configured Studio data directory.

This guide describes the implementation in the working tree. Engine-specific model formats, request fields, device support, and output formats depend on the selected engine build and model. Studio exposes discovered capabilities and validation messages in the relevant screen.

## Workspaces and navigation

| Workspace           | URL                   | Available work                                                                                                                              |
| ------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Models              | `/models`             | Browse installed models, filter and sort, inspect status and files, configure, connect, start, stop, warm, and delete.                      |
| Search              | `/search`             | Search model providers and the normalized package catalog, inspect artifacts, choose quantizations and companions, and download or install. |
| Audio               | `/audio`              | Select an audio model, edit request inputs, manage voice/reference choices, run supported tasks, and inspect outputs.                       |
| Engines             | `/engines`            | Install, build, activate, scan, update, and remove inference engine versions; edit build and install settings.                              |
| Proxy               | `/proxy`              | Observe traffic and processes, manage inference keys and policies, configure live routing, and control serving limits and lifecycle.        |
| System              | `/system`             | Inspect diagnostics, hardware, CUDA installations, configuration backup, and configuration history.                                         |
| Model configuration | `/models/{id}/config` | Edit engine-specific settings, request defaults, assets, aliases, environment, custom arguments, templates, and runtime application.        |

The header contains the Studio orbital mark, workspace navigator, proxy status, activity access, and theme switch. The navigator supports keyboard search. Desktop and mobile layouts expose the same workspaces. Page headings stay compact; the information control opens page guidance. Configuration fields use information tooltips for descriptions and inline text for engine defaults. Focusable controls, status announcements, unsaved-change prompts, and actionable error states support keyboard workflows.

The footer shows compact application/service status and a link to System. Activity notifications carry progress for downloads, builds, installations, scans, and other background work. Views display freshness and connection state so a last observation is distinguishable from a verified live result.

## Inference engines

### Engine matrix

| Engine ID       | Display name      | Model artifacts                                          | Installation and runtime                                                                 |
| --------------- | ----------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `llama_cpp`     | llama.cpp         | GGUF                                                     | Source/CMake build; `llama-server`.                                                      |
| `ik_llama`      | ik_llama.cpp      | GGUF                                                     | Source/CMake build; `llama-server`.                                                      |
| `unsloth_llama` | Unsloth llama.cpp | GGUF                                                     | Published prebuilt distribution; `llama-server`.                                         |
| `lmdeploy`      | LMDeploy          | safetensors/Hugging Face snapshot                        | Managed Python environment; LMDeploy API server.                                         |
| `1cat_vllm`     | 1Cat-vLLM         | safetensors/Hugging Face snapshot                        | Managed Python environment; vLLM-compatible API server.                                  |
| `vllm`          | vLLM              | safetensors/Hugging Face snapshot                        | Managed Python environment; vLLM OpenAI API server.                                      |
| `sglang`        | SGLang            | safetensors/Hugging Face snapshot                        | Managed Python environment; SGLang server and discovered runtime profiles.               |
| `sglang_v100`   | SGLang V100       | safetensors/Hugging Face snapshot                        | Source installation in a managed Python environment with a V100-specific build workflow. |
| `audio_cpp`     | audio.cpp         | Prepared audio packages and discovered GGUF audio models | Source/CMake build; model-scoped `audiocpp_server`.                                      |

The shared engine registry describes artifact formats, package kinds, tasks, input/output modalities, installation type, scanner type, active path fields, and capability hints. Compatibility requires a suitable installed runtime and model; a registry task label does not certify every model/build combination.

### Installation and versions

Engines shows installed versions, active installations, source/build state, and available actions. Multiple installed versions can coexist. Activation selects the installation used by the runtime compiler. Installation, source synchronization, builds, scans, and removals expose progress and failure details.

Source-based engines accept repository and tracking-ref settings. Saved build settings control supported CMake flags and engine-specific options. A version can carry its own build configuration; source synchronization uses that configuration when present. Custom repositories retain their own identity and source-sync workflow. Unsloth installations select a published artifact appropriate to the detected host and verify its checksum.

Python engines use Studio-managed environments. Their settings include release/package selection and source repository/ref choices where supported by the installer. Source installation uses the selected repository and engine-specific build environment. SGLang V100 uses its dedicated source workflow, a managed Python environment, and Studio-managed CUDA 12.8; its installer supplies the selected fork's dependency/build steps while Studio supplies host-tool and CUDA paths.

Engine activation and explicit rescanning populate the parameter catalog. Source build, Python package installation, and prebuilt installation have different hardware and dependency requirements; installer errors and build logs provide the concrete result.

### Installation references and removal

Published or retained launch generations can reference an engine installation or model path. Removal checks those references and reports a conflict when the path is still required. Applicable removal requests expose `retire_launch_references` for an explicit retirement decision; a running reference has its own restriction and requires the relevant process to stop. Retirement affects the available launch/recovery generations, so installation deletion is distinct from simply choosing another active version.

### Parameter discovery

Studio scans executable help and structured engine metadata. Catalog entries describe flags, types, arity, defaults, ranges, enums, sections, and support in the selected build. Model-specific audio profiles and SGLang runtime profiles supply additional configuration fields.

A parameter can be known to the catalog while unsupported by the selected build. The configuration screen labels that state and offers a hide-unsupported control. Parameters with saved overrides remain inspectable. Rescanning refreshes support metadata for the installed executable.

## Discovering and acquiring models

### Search and catalog filters

Search integrates Hugging Face discovery with a normalized model/package catalog. Depending on the selected source, filters include engine, task, input/output modality, feature, provider, source, package kind, install method, release status, language, gated access, and artifact format. Pagination and search results expose repository/package identity and compatible acquisition actions.

GGUF repositories expose quantizations and relevant companion choices. safetensors repositories use logical bundles with model metadata and weight shards. Audio package discovery uses the active engine's package declarations, loader metadata, and model specs. Gated downloads use the configured Hugging Face credential and remain subject to provider access permissions.

### Downloads and installation

Supported acquisition flows include single-file GGUF downloads, sharded/bundled weights, safetensors snapshots, audio package installation, and local prepared-audio directory import. Background task IDs connect progress, cancellation, and completion state to the initiating workflow.

Models keep file ledgers containing artifact roles, sizes when available, remote identities, and companion selections. Update checks compare recorded artifacts with provider metadata. Refresh downloads changed files and reports remote removals. Companion changes use dedicated actions so a projector or draft can change independently of the main weights.

GGUF companion roles include:

| Companion    | Purpose                                                               |
| ------------ | --------------------------------------------------------------------- |
| `mmproj`     | Vision/projector artifact for compatible multimodal models.           |
| MTP draft    | Draft artifact for a compatible multi-token-prediction configuration. |
| DFlash draft | Draft artifact for a compatible speculative-decoding configuration.   |

MTP and DFlash are mutually exclusive selections. Runtime validation checks the paths and engine flags needed for the selected model and companions.

### Model library

The library groups related model entries and exposes card and table layouts. Search, run-state, engine, task, and sorting controls narrow the visible set. Sorting supports name, size, and recent entries. Active filters are visible and can be cleared.

Each entry exposes available runtime actions and an observed state. Table columns keep model identity, runtime status, file size, and actions separate. File size represents stored artifact information; it is distinct from GPU memory use and parameter count.

Start requests acquire the model through the native proxy. Stop unloads that model's process. Warm/preload actions load a model and can issue a small compatible warm-up request. A cold inference request can also trigger loading. Delete workflows identify affected entries/files and expose background progress where applicable.

Model/group deletion can also encounter retained launch-reference checks. The returned conflict identifies the reference/retirement decision rather than implying every file is already deleted.

The UI distinguishes observed running/stopped states from stale, unreachable, or unknown observation quality. A disconnected proxy does not establish that a model is stopped.

## Configuring a model

### Configuration structure and engine switching

Each model has a selected engine and a separate section for each configured engine:

```json
{
  "engine": "llama_cpp",
  "engines": {
    "llama_cpp": { "ctx_size": 8192 },
    "ik_llama": { "ctx_size": 4096 }
  }
}
```

Switching the selected engine retains the other engine sections. The UI displays the chosen engine's supported fields. Configuration screens separate runtime controls from launch/API settings; audio models also expose assets and request defaults.

An explicit save records configuration review. A downloaded model with generated/default settings can require review before its configuration is treated as user-reviewed.

### Parameters, defaults, and custom arguments

Configuration forms use discovered types and controls. Search adds parameters to the editable set; saved overrides and common controls remain visible. Inline help states an engine default when known, while the information tooltip describes the field. Reset actions return an override to its saved or inherited value according to the control.

Custom arguments append raw CLI tokens to the engine command. Studio owns model identity, weights, binding, and runtime-managed flags; compilation rejects conflicting managed options. Engine command previews show the actual executable and ordered arguments, working directory/runtime details when available, environment changes, revision information, and generated sidecars. Sensitive environment values are redacted in previews.

### Import from command

**More actions → Import command** accepts a complete server command, raw flags, and `KEY=value` environment assignments. The preview separates recognized parameters, environment assignments, managed options, unsupported/deprecated entries, and unrecognized tokens. Users choose the entries to import and can put appropriate leftovers into Custom Arguments.

Arity follows catalog metadata:

- Scalar flags consume one value.
- Boolean flags use their positive/negative forms.
- Multi-value flags consume values until the next option and retain order.
- Repeatable flags, JSON-valued flags, comma-delimited enums, and semicolon-delimited enums use their catalog representation.
- Quoting, inline `--flag=value`, and negative numeric values are supported.

For example, `--cuda-graph-bs-decode 1 2 3` imports three values and compiles to three ordered arguments after that flag. Import changes the form; Save persists the selected values.

### Environment and GPU assignment

`runtime_env` sets process environment variables. `runtime_env_unset` suppresses selected variables. An explicit empty string is a set value; empty set/unset collections clear those collections. Removing an override returns control to the inherited/runtime environment. Reserved `LLAMA_STUDIO_*` variables belong to Studio.

GPU controls expose inherited visibility, selected devices, and CPU intent. Selected device order matters. The raw `CUDA_VISIBLE_DEVICES` value and graphical GPU selection describe the same setting. Device UUIDs and explicitly ordered values retain their identity. CPU intent and explicit GPU selection must agree with the engine's supported execution mode.

A process uses the environment and device allocation in its published launch revision. Editing a form or saving desired settings does not mutate a running process's environment.

### Configuration templates

Templates contain a name, description, selected engine, and engine-section snapshots. A template can capture all engine sections or only the active engine. Routing-name inclusion is an explicit option for `model_alias` and additional ordinary aliases.

Applying a template supports these scopes:

| API scope    | Behavior                                                                            |
| ------------ | ----------------------------------------------------------------------------------- |
| `active`     | Merge the template's active section into the target's active engine section.        |
| `all`        | Merge each engine section present in the template into its matching target section. |
| `set_engine` | Merge the template's active section and select that engine on the target.           |

Request-preset configuration is part of the captured engine section. The template screen supports saving, selecting, renaming, describing, applying, and deleting templates. A template application can update the working form or persist configuration according to the requested action; runtime application remains a separate decision.

## Model identity and aliases

Studio distinguishes catalog IDs, stable runtime IDs, ordinary API aliases, request presets, and gateway routing aliases. These names serve different purposes.

| Identity                    | Configuration/location                       | Meaning                                                                  |
| --------------------------- | -------------------------------------------- | ------------------------------------------------------------------------ |
| Catalog ID                  | Model record `id`                            | Selects an installed model in management APIs and URLs.                  |
| Stable runtime ID           | Model record `proxy_name` / runtime identity | Identifies the process, manifests, receipts, and runtime state.          |
| Primary API alias           | Engine config `model_alias`                  | Friendly name clients send as `model`.                                   |
| Additional ordinary aliases | Engine config `swap_aliases`                 | Additional names for the same real model. This is the stored field name. |
| Request-parameter alias     | Engine config `set_params_by_id`             | Named overrides for requests to the same real model.                     |
| Live routing alias          | Proxy settings `routing_aliases`             | Dispatches an alias to one or more target IDs/aliases.                   |

Studio normalizes primary and ordinary API names to safe lower-case names. Request-preset suffixes use letters, numbers, dots, underscores, or hyphens. Model configuration validates name collisions against other catalog entries. `GET /v1/models` lists accessible exposed IDs and aliases according to client permissions.

### Request-parameter presets

In **Model configuration → Launch → Request parameter aliases**, add a suffix and a JSON object. With primary API alias `assistant`, suffix `no-thinking` exposes `assistant:no-thinking`:

```json
{
  "sub_id": "no-thinking",
  "params": {
    "chat_template_kwargs.enable_thinking": false,
    "thinking_effort": "low"
  }
}
```

The engine section stores presets as a list in `set_params_by_id`. An empty suffix selects the base routing name. An empty list removes the presets. Each suffix is unique within that model's preset list.

A request to `assistant:no-thinking` resolves to the same real process as `assistant`. Matching preset values override the incoming JSON request fields. The base name and other aliases keep their own values. The native proxy rewrites the upstream model identity as required by the engine adapter.

Dotted keys address nested request fields; `chat_template_kwargs.enable_thinking` targets one member of `chat_template_kwargs`. A backslash escapes a literal dot in a key. A key ending in `?` applies only when that field is absent. A JSON object supplied as a field value replaces that field's value; dotted keys provide control over individual nested members. `false`, zero, empty strings, arrays, and nested JSON values retain their JSON types. Studio manages the `model` field.

The editor validates JSON object shape, suffix syntax, and duplicate suffixes before save. Presets apply to subsequent JSON inference requests when saved and share the real model's running process. Parameter names and values must be supported by the target engine/model. Native engine-specific audio passthrough explicitly rejects parameter presets because it forwards the request through a concrete-model transport.

### Live multi-target routing

The Proxy page's **Settings & routing** tab edits the default model and routing aliases. An alias maps to a target string or ordered target list:

```json
{
  "default_model": "balanced",
  "routing_aliases": {
    "balanced": ["assistant:thinking", "assistant:no-thinking"]
  }
}
```

The proxy selects successive targets round-robin. A selected request preset applies its own overrides. A single-target alias forwards to that target. A configured default can select an exposed model or routing alias when the request omits a model name.

Routing publication updates an immutable routing snapshot. Accepted requests keep their resolved target, and existing streams continue. A routing edit does not restart engines and does not clear an unrelated pending serving-settings restart. Targets resolve through the model catalog; this is direct target selection rather than an arbitrary recursive routing graph.

## Saving, publishing, and running

Studio tracks three separate states:

| State                     | Meaning                                                             |
| ------------------------- | ------------------------------------------------------------------- |
| Desired configuration     | Saved model settings in the configuration store.                    |
| Published launch revision | Immutable launch specification selected for a future process start. |
| Running revision          | Revision verified for the actual running engine generation.         |

Save updates desired settings. The runtime plan compares launch revisions and reports per-model actions and reasons. Routing-only alias and request-default changes publish through native routing/config observation; engine launch changes require the appropriate runtime action.

| Plan/UI action        | API mode        | Effect                                                                                         |
| --------------------- | --------------- | ---------------------------------------------------------------------------------------------- |
| Restart this model    | `restart_now`   | Publish the selected launch revision; restart the model if it is running and verify readiness. |
| Use on next start     | `next_start`    | Publish the revision for the next actual launch; keep an existing process running.             |
| No launch change      | No apply action | Published/running launch state already matches the relevant desired launch state.              |
| Invalid configuration | No apply action | Show compilation/validation reasons without starting the candidate.                            |

The runtime plan may name its stopped-model action `publish_next_start`; the apply API uses mode `next_start`. Revision-bound requests reject stale desired/published expectations. Idempotency keys identify a single apply intent. Reusing a key with different request content is a conflict.

Applying one model does not unload unrelated models. Native sequential batch apply processes a selected list in order and stops at the first failed outcome. A readiness failure triggers restoration of the selected model's verified revision when that restoration is available. Runtime journals and process receipts support startup reconciliation and truthful outcome reporting.

See [runtime architecture](launch-manifests-plan.md) and [runtime behavior cases](launch-manifests-use-cases.md) for the detailed state model.

## Connecting an application

A model's **Connect** dialog shows the public inference URL, model identifier, cURL/JSON examples, access-key controls, and a test action. The public URL is a client-facing preference; it does not rebind the inference listener. Listen address and proxy port determine the server binding.

With the default local ports:

```bash
curl http://localhost:2000/v1/models

curl http://localhost:2000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"assistant:no-thinking","messages":[{"role":"user","content":"Say hello."}]}'
```

Add `Authorization: Bearer YOUR_INFERENCE_KEY` when inference access requires a key. Select the endpoint and request shape supported by the active engine. Chat, completions, Responses, embeddings, reranking, audio endpoints, and engine-specific passthrough have different contracts; capability discovery and the engine's concrete responses determine availability.

## Proxy workspace and serving controls

### Overview, requests, and events

Overview shows gateway health, running engines, recent traffic, HTTP outcomes, latency/usage metrics, and network binding. Requests provides live/retained metadata with pagination and filters for model, key, state, HTTP status/outcome, and metadata search. Events shows lifecycle activity. Engine details expose process identity, running revision, port, in-flight work, counters, logs, and management links.

Request details can include queue wait, model load time, first response byte, first recognizable content token, stream/generation duration, total duration, bytes, reported tokens, reservations, and failure reason. Missing measurements appear as unavailable. A successful response-header status can coexist with a later stream failure; the detail view preserves both delivered status and final outcome.

### Serving settings

| Setting                       | Default                                               | Meaning                                                                                              |
| ----------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `proxy_port`                  | `2000`                                                | Inference listener port.                                                                             |
| `idle_ttl_seconds`            | `null`                                                | Positive values enable idle model unloading.                                                         |
| `max_body_bytes`              | `null` in settings; effective 256 MiB runtime default | Buffered request-body bound.                                                                         |
| `max_loaded_models`           | `null`                                                | Optional loaded-model count bound.                                                                   |
| `global_concurrency`          | `null`                                                | Optional global request bound; pool capacity also constrains forwarding.                             |
| `per_model_concurrency`       | `null`                                                | Optional per-model request bound.                                                                    |
| `queue_timeout_seconds`       | `30`                                                  | Maximum slot/queue wait.                                                                             |
| `request_timeout_seconds`     | `null`                                                | Optional total deadline including load and stream time.                                              |
| `readiness_timeout_seconds`   | `2700`                                                | Engine readiness budget.                                                                             |
| `drain_timeout_seconds`       | `30`                                                  | Graceful drain/shutdown budget.                                                                      |
| `max_queue_size`              | `100`                                                 | Maximum waiting requests.                                                                            |
| `stream_idle_timeout_seconds` | `null`                                                | Optional inactivity budget between upstream chunks.                                                  |
| `pool_timeout_seconds`        | `10`                                                  | Upstream connection-pool wait.                                                                       |
| `max_connections`             | `200`                                                 | Maximum upstream connections.                                                                        |
| `max_keepalive_connections`   | `40`                                                  | Maximum idle reusable connections; cannot exceed total connections.                                  |
| `history_retention_days`      | `7`                                                   | Request metadata retention age.                                                                      |
| `history_max_rows`            | `100000`                                              | Request metadata retention count.                                                                    |
| `default_output_tokens`       | `4096`                                                | Output reservation/cap applied when required for budgeted text requests without an explicit maximum. |
| `allow_anonymous`             | `true`                                                | Anonymous inference when no inference keys exist.                                                    |
| `default_model`               | `null`                                                | Default target for requests without a model name.                                                    |
| `routing_aliases`             | `{}`                                                  | Live single-/multi-target routing aliases.                                                           |

Serving limits and listener changes take effect through **Restart & apply**. Restart drains running engines and rebuilds the proxy runtime. Start and Stop control the listener; stop closes engine processes owned by that runtime. Model-specific unload, preload, warm, reload, and log actions are also available through the management API.

## Inference access keys and accounting

Management authorization and inference authorization are separate. An inference key grants the scopes and model access in its policy; it does not grant management access. The management session/token controls settings and key administration.

The Access keys tab creates, rotates, revokes, and edits keys. Creation/rotation returns the secret once; key listings expose metadata and usage. Policy updates retain the key identity. Revocation and rotation affect subsequent admissions; accepted requests retain their policy snapshot and key attribution.

Available policy fields include `models`, `scopes`, `expires_at`, `rate_limit_per_minute`, `input_tokens_per_minute`, `output_tokens_per_minute`, `daily_token_budget`, `monthly_token_budget`, `max_output_tokens`, `max_concurrent_requests`, and `token_accounting` (`auto` or `exact`). Scopes include `inference` and `models:read`.

Requests with budgets reserve allowances transactionally before generation. Reservations count against concurrent admissions. Daily and monthly budgets use UTC admission boundaries. For supported engine routes, input token counting adjusts the reservation using the actual request context. Automatic accounting permits a visible UTF-8 estimate for supported simple text requests when exact counting is unavailable; exact accounting requires the engine count. Multimodal/stateful requests require a suitable engine count.

Reported complete usage settles a reservation. Missing or incomplete stream usage retains the reserved charge. Over-cap requests receive a policy rejection and retry information; an explicitly excessive requested output limit is rejected. Embeddings reserve no generated output tokens. Engine support and output-limit enforcement remain requirements of the selected inference runtime.

The accounting database stores client usage, reservations, request metadata, and the statistics observation window. Request telemetry is batched; quota reservations commit before dispatch. Reset stats changes the observation window and samples while retaining request history and key budgets. See [proxy operations](proxy-operations.md) for accounting and timeout details.

## Audio models and workspace

### Discovery and preparation

Audio integration uses a composed engine contract, model specs, loader JSON, and model-aware inspection. Engine activation requires the structured loader interface. Catalog identity, tasks, typed load/session/request options, voice metadata, and instructions policy derive from these sources. Drift warnings describe contradictory observations. Missing metadata stays unknown or presents a scan requirement.

Audio acquisition supports prepared package installation, direct snapshots where supported, composite/converter workflows through the engine's helper, and local prepared directory import. Spec-backed installs use the available spec-aware model manager; conversion workflows use their applicable helper environment. A package passes inspection before promotion into the installed model directory.

### Configuration scopes

| Scope                  | Purpose                                                                       |
| ---------------------- | ----------------------------------------------------------------------------- |
| Family/task/mode       | Choose the engine-advertised model behavior.                                  |
| Backend/device         | Select the supported execution target for the installed build.                |
| Load options           | Configure model loading and resource allocation.                              |
| Session options        | Configure model-session behavior.                                             |
| Request defaults       | Supply overridable defaults for an operation/request.                         |
| Assets/reference audio | Manage model-associated inputs and voice references.                          |
| Launch settings        | Set server options, environment, custom CLI arguments, and API names/presets. |

Request defaults are partial. Supplied known options validate against discovered types, enums, ranges, and finite-number rules. Request-specific values override soft defaults. Required inputs can be supplied for each inference request. Instruction fields appear according to explicit engine policy and metadata.

### Running tasks

The Audio page offers Speech, Transcribe, and metadata-driven generic task panels. Supported tasks can include speech synthesis, transcription, VAD, diarization, separation, generation/music, voice conversion, speech-to-speech, alignment, speaker-related operations, and pipelines. The selected model's contract determines the available controls and request shape.

Speech supports text and available voice choices. Transcription accepts an audio upload and supported transcription options. Generic panels render discovered top-level inputs and nested `options`; they expose an inspection requirement when the engine provides no usable input schema. Results use the operation's JSON, text, audio, or other advertised response form.

Packaged voices, model-associated references, and available community voice presets provide voice inputs. Community preset discovery follows the active checkout's demo-voice metadata. Reference assets have their own upload/list/delete workflow and are separate from portable configuration export.

### Audio endpoints

| Interface                                                        | Purpose                                             |
| ---------------------------------------------------------------- | --------------------------------------------------- |
| Management origin `/v1/audio/speech`                             | Same-origin speech transport.                       |
| Management origin `/v1/audio/transcriptions`                     | Upload transport with WAV conversion when required. |
| Management origin `/v1/audio/voices` and discovered audio routes | Voice and supported engine audio operations.        |
| Management origin `/v1/tasks/run`                                | Same-origin generic task transport.                 |
| Inference origin `/audioapi/v1/tasks/run`                        | Generic task dispatch with upstream path rewriting. |
| Inference origin `/upstream/{model}/…`                           | Concrete engine-specific API passthrough.           |

Speech output conversion and transcription input conversion follow the engine contract and requested format. `ffmpeg` supplies applicable WAV/audio conversions. Engine-specific alignments and transcription-detail transports use a concrete model target and retain explicit parameter-preset restrictions.

See [audio contract and discovery](audio-cpp-discovery.md) for the metadata and routing rules.

## System workspace

### Hardware

**System → Hardware** (`/system#hardware`) displays CPU utilization, used/total memory, disk usage, and discovered GPU memory. Refresh reads service and GPU probes. Unknown information is labeled unknown; a refresh failure retains the last available readings with an error message.

The CUDA card displays detected versions/path and provides installation, version selection, refresh, and uninstall actions. Install/remove work reports through background activity. CUDA installations live in the persistent data directory, with the managed active path supplied to relevant builds/runtimes. An installed toolkit and a working host GPU driver/container runtime are separate requirements.

### Diagnostics

Diagnostics shows live-update connectivity, proxy health observation age, model-runtime observation, and persistence queue/failure state. The downloadable diagnostics bundle contains redacted service information for troubleshooting. It describes observation quality and persistence/operation health rather than treating missing observations as a stopped model.

### Portable backup and restore

**Backup & restore** exports schema-1 JSON containing the portable preferences `public_inference_url` and `proxy_port`, model settings, and configuration templates. It excludes credentials, model weights, reference assets, engine/build installations, executable paths, runtime processes, operation history, published launch generations, and the accounting ledger. The encoded backup has a 1 MiB size bound.

Restore matches portable model references to local catalog entries. A preview exposes keep/add/replace/skip/unresolved choices, mapping needs, current revisions, and a plan ID. It validates payload shape, references, portable values, and selected decisions before applying. Import changes configuration; it does not acquire model files or start engines.

Apply checks the preview revisions under the persistence protocol. A durable multi-document restore journal coordinates settings, templates, and models. Interrupted application reconciles through that journal. If the response is uncertain, the UI offers reconciliation/refresh instead of treating a retry as a new unreviewed mutation.

A portable export is a configuration artifact. A full operational backup also includes the required data volume, keys, reference assets, and accounting database using appropriate filesystem/SQLite backup handling.

### Configuration history

**Change history** shows private configuration snapshots and redacted diffs. Snapshots cover settings, model configurations, and templates; retention is bounded to 100 entries. The UI can restore one selected preference, model configuration, or template with a current-revision check. Restoration changes saved configuration and does not publish a launch or start a process.

Private snapshots live in `config/history/`. Redaction in the displayed diff does not make the private snapshot directory a public export. Treat the data directory as private application state.

## Local model benchmarks

The benchmark management API runs a bounded streaming benchmark against a running compatible text model. This feature is available through the benchmark endpoints; the workspace screens provide model control and connection workflows. Requests contain a prompt of 1–1,000 characters and an output limit of 1–512 tokens (default 64). Results contain the model/proxy name, saved configuration revision/fingerprint, creation time, prompt, output preview, first-token time, duration, completion tokens when reported, throughput when available, and peak observed GPU memory.

The API stores results in `benchmarks.yaml`; listing defaults to 20 entries and supports up to 100. GPU memory is a sampled observation and can include other GPU activity. Results describe their recorded model/configuration and machine; they are not automatically normalized across hardware or engine builds.

## Persistence, activities, and recovery

Configuration documents use serialized mutations and atomic file replacement. Corrupt documents produce explicit errors rather than silent replacement. Persistence errors distinguish a rejected/uncommitted write, an acknowledged replacement with a later failure, and an uncertain result. The UI retains drafts and provides refresh/retry/reconciliation actions appropriate to that outcome.

Long-running operations use durable IDs and resource admission. Duplicate/busy work receives an actionable conflict. A confirmation can bind an operation and state token to a deliberate retry. Starting the application reconciles interrupted operations and runtime journals. Cancellation reaches the owning task/process where the operation supports it; a cancellation response is distinct from proof that every side effect is absent.

SSE supplies task updates, stages, logs, and notifications. Connection errors surface in status and freshness controls. A reconnect refreshes authoritative state; it does not infer a successful installation or model launch from a missing event.

## Deployment and validation boundaries

Local mode restricts management to trusted local/container-bridge peers. Remote mode requires the management token/session policy. `STUDIO_BIND_HOST` controls management binding, `STUDIO_PROXY_LISTEN` controls inference binding, and `proxy_port` controls the inference port. The client-facing public URL is a separate preference. TLS termination belongs at a configured reverse proxy.

Software checks cover frontend logic, runtime compilation, persistence, operation recovery, native process supervision, real CPU fixture HTTP/SSE behavior, and browser navigation/layout/accessibility assertions. A specific model/engine/device combination requires the matching installed runtime, model artifacts, driver/toolkit environment, and a successful readiness/inference check. Build and test success alone does not measure GPU throughput or establish every engine route's compatibility.

Use the [API reference](api-reference.md), [proxy operations guide](proxy-operations.md), and [engineering reference](current-engineering-status.md) for endpoint and implementation details.
