# Native runtime architecture

Studio compiles engine configuration into structured launch specifications and supervises engine processes through its native proxy. Immutable launch generations, published pointers, process receipts, routing snapshots, and apply journals give configuration and runtime state separate identities.

## Responsibilities

| Component                 | Responsibility                                                                                                                               |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Model configuration store | Desired engine selection, parameters, environment, assets, aliases, and request presets.                                                     |
| Launch compiler           | Validated executable, ordered arguments, working directory, environment, sidecars, file identities, and launch revision.                     |
| Proxy model specification | Stable/catalog IDs, exposed aliases, request filters, engine identity, upstream model-name policy, and capabilities.                         |
| Manifest store            | Immutable revision generations, published pointer, receipts, gates, and journals under the model runtime directory.                          |
| Native manager            | Catalog compilation/cache, route snapshots, listener lifecycle, and registry/dataplane construction.                                         |
| Process registry          | Startup coordination, generations, readiness, slots, draining, logs, exits, and process-group shutdown.                                      |
| Apply service             | Reviewed revision checks, per-model publication, selective restart, rollback, idempotency, sequential batch application, and reconciliation. |
| Management operations     | Admission/resource ownership, progress, cancellation, confirmations, and durable operation outcomes.                                         |

## Desired, published, and running

Desired configuration is the saved model configuration. A published pointer selects a launch generation for process creation. A verified receipt records the revision associated with an observed running generation.

These values can intentionally differ. Publishing for next start leaves the current process on its running revision. A matching desired/published revision does not establish that an already-running process uses that revision. The plan, receipt, process state, and action result provide the relevant evidence.

Request routing is a separate concern. Ordinary names, request presets, default targets, and multi-target aliases resolve through the native catalog/routing snapshot. Request-only changes do not change a process environment or device allocation.

## Launch specification

A `LaunchSpec` records:

- Stable model ID, engine ID, and installation identity.
- Concrete executable, ordered argument tokens, and working directory.
- Environment values to set and names to unset.
- Generated artifacts and their digests.
- Relevant executable/model/companion file identities.

The revision is a lower-case SHA-256 digest of canonical launch content. Generated sidecars contribute through artifact digests. The revision field is not part of its own hashed content.

`ProxyModelSpec` holds catalog ID, exposed aliases, filters, upstream model-name behavior, and capability metadata. `CompiledModel` joins launch/routing output and command-preview details. The same compiler serves runtime plans and command previews, so previews describe native launch arguments rather than a separate hand-built shell command.

### Arguments and engine adapters

GGUF adapters choose the active executable, weights or supported HF source, Studio-owned port/model/alias arguments, companions, structured flags, and custom arguments. Python adapters choose the managed environment's executable/module, model source, bind settings, runtime profile, and engine-specific arguments. The audio adapter compiles one-model server configuration and the matching sidecar reference.

Studio rejects conflicting owned flags and invalid model/runtime inputs before publication. Multi-value parameters emit separate ordered argument tokens. JSON-valued, repeatable, and delimited parameters use their catalog representation.

### Environment and devices

Environment compilation uses an allowed process baseline, engine defaults, explicit user set/unset values, and required executable/library paths. Required paths are composed in order and deduplicated. User set/unset entries for the same name conflict. Invalid names, NUL bytes, and reserved `LLAMA_STUDIO_*` keys fail validation.

`runtime_env` values preserve an explicit empty string. `runtime_env_unset` removes selected inherited/default variables. Empty collections clear that collection in saved configuration. GPU intent reconciles graphical device selection, ordered raw CUDA visibility, CPU intent, and engine-specific supported modes. A launch revision describes the environment for process creation, not an in-place mutation of a running engine.

### Storage and privacy

Each stable runtime ID maps to a hashed directory key below `runtime/models/` in the Studio data root. Generations are immutable; publication replaces a pointer atomically. Receipts/journals and referenced generations support comparison and restoration. Runtime directories/files use private permissions where supported by the host filesystem.

Manifest/environment storage can contain operational values needed by an engine. Command previews redact sensitive environment names; that displayed redaction is distinct from protecting the private data directory.

## Runtime planning

`GET /api/runtime/pending` compiles catalog models, observes native registry state, reads published pointers and verified receipts, and returns actionable rows with a plan ID. A row includes IDs, desired/published/running revisions when available, runtime state, action, reasons, and setting differences.

| Condition                                           | Plan action                            |
| --------------------------------------------------- | -------------------------------------- |
| Desired launch equals published launch              | `none`; no actionable row is required. |
| Desired launch differs and model is running/loading | `restart_now`.                         |
| Desired launch differs and model is stopped         | `publish_next_start`.                  |
| Launch compilation fails                            | `invalid` with reasons.                |

`GET /api/runtime/stale` supplies a pending-state check. Unknown/unreachable observations remain distinct from verified stopped state. Apply validates the concrete runtime conditions before taking its action.

## Per-model application

`POST /api/models/{catalog_id}/runtime/apply` accepts:

| Field                                   | Meaning                                                                             |
| --------------------------------------- | ----------------------------------------------------------------------------------- |
| `mode`                                  | `restart_now` or `next_start`.                                                      |
| `idempotency_key`                       | Required identity for a deliberate apply intent.                                    |
| `expected_desired_revision`             | Optional assertion about reviewed desired configuration.                            |
| `expected_published_revision`           | Optional assertion about the published pointer; an explicit `null` asserts absence. |
| `confirm_operation_id`, `confirm_state` | State-bound confirmation when the operation requires deliberate recovery/admission. |

A reviewed revision conflict rejects the operation before the candidate takes effect. Idempotency records bind the key to its request fingerprint. The same intent returns its recorded result; different content under that key conflicts.

### Restart now

The service validates and stages the candidate, coordinates the model's launch/publication gate, stops the selected running generation where required, publishes the revision, starts the model when it is running, and checks readiness. Active work on the selected model can be interrupted. Other model processes retain their identity and active work.

### Next start

The service publishes the selected revision for a future process creation. A running generation continues with its current environment/arguments. Explicit start, demand after unload/TTL, or another genuine process creation uses the selected published generation. A start already in progress uses the coherent generation selected for that attempt.

The UI's **Use on next start** and the plan action `publish_next_start` use API mode `next_start`.

### Outcome and cancellation

Apply results identify the operation and state/outcome. Model-scoped operation read/cancel endpoints expose progress and supported cancellation. A cancellation request is not proof that publication or a process side effect is absent; the journal/result establishes the outcome.

When a candidate fails readiness, the service attempts to restore the selected model's verified revision when available. The reported outcome distinguishes success, failure, rollback, interruption, cancellation, and uncertainty. A missing verified recovery generation limits what can be restored.

## Sequential batch application

`POST /api/runtime/apply-launch` accepts a shared `mode` and ordered model entries with catalog ID, idempotency key, and optional revision expectations. It holds the shared apply lock and delegates each entry to the per-model apply service. The response contains `results` and `stopped_early`.

The batch stops after an unknown model, rejected apply, or failed outcome. Unvisited entries do not run. This is sequential per-model work rather than a global engine restart. The frontend runtime store exposes the corresponding native batch request helper.

## Live routing and request presets

The native manager compiles ordinary API aliases and named request presets into the catalog. Preset filters apply to the selected request alias; a suffix can set nested options such as `chat_template_kwargs.enable_thinking` and `thinking_effort` while sharing the real process.

Proxy-level aliases select a single target or an ordered round-robin target list. The default model can point to such an alias. Resolution retains the selected target name so a routing alias that selects a request preset applies that preset correctly.

A routing publication swaps an immutable snapshot and resets its selection cursor. Already accepted work retains its resolved target. Routing changes do not interrupt streams or clear a serving-settings restart requirement. Filter application copies request data so overrides do not mutate another request's input/default state.

## Process supervision

Concurrent acquisitions share one model-owned startup task. A cancelled waiter releases its interest; other waiters keep the needed start alive. The final unused waiter can trigger cleanup of a cold start. Readiness has its own timeout. A stop/drain coordinates with acquisitions so replacement does not race an incomplete teardown.

The registry tracks starting, ready, draining, stopped, and failed generations, with bounded logs and lifecycle events. In-flight work and model/global slots govern dispatch. Unexpected process exit records failure and permits an actual subsequent start. Process shutdown targets the group, uses a graceful budget, escalates when required, and bounds inherited pipe handling.

## Listener lifecycle and recovery

Start creates/serves the native listener. Stop closes the listener, dataplane, and owned registry processes. Restart rebuilds runtime resources and reads the saved listener/serving settings. A stopped manager does not reuse a closed upstream client or closed registry.

Startup reconciles durable apply journals and operation records. Published pointers, receipts, and observed process identity inform the result; desired configuration alone is not evidence of a running revision. Admission fences and confirmation state prevent automatic duplicate effects when a prior intent has an uncertain result.

See [runtime behavior cases](launch-manifests-use-cases.md), [API examples](api-reference.md#review-and-apply-a-launch-revision), and [proxy operations](proxy-operations.md).
