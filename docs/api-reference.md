# Studio API reference

The management API runs on the Studio application origin: `http://localhost:8080` in the default production/container setup and `http://localhost:8081` during local development. The native inference origin uses port `2000` by default. These origins have separate request authorization roles.

The management route and schema tables below describe the checked-in [OpenAPI document](../frontend/src/api/openapi.json). Interactive documentation is served at `/docs`; `/openapi.json` exposes the running application's schema. The inference proxy is a separate ASGI application, so its forwarded engine routes are described separately here.

## Authorization and request conventions

- Local management mode checks trusted local/container-bridge access. Remote mode accepts the management bearer token or an authenticated browser session with its CSRF policy.
- Inference clients use an inference key and its model/scope/quota policy. `Authorization: Bearer …` is the usual client header. The proxy also accepts `X-API-Key` and Basic-password credentials. GET/HEAD UI navigation can use `key` or `api_key` query credentials and a scoped UI cookie; explicit header credentials take precedence.
- JSON requests use `Content-Type: application/json`. Upload routes use their declared multipart content type.
- Catalog IDs in path segments require URL encoding when they contain slashes or other reserved characters. Runtime/API IDs belong in inference request fields or the relevant proxy model path.
- An operation/task ID identifies work and its progress. An idempotency key identifies a deliberate runtime-apply intent. A revision token identifies reviewed state; these values have different purposes.
- Free-form object endpoints validate additional fields in their route/service implementation. The feature examples below document the principal object contracts that the OpenAPI schema represents as a generic object.

## Native inference interface

| Route                    | Behavior                                                            |
| ------------------------ | ------------------------------------------------------------------- |
| `GET /health`            | Listener health endpoint.                                           |
| `GET /v1/models`         | OpenAI-style model catalog with accessible exposed aliases.         |
| `GET /v1/models/{model}` | OpenAI-style single-model lookup by proxy id, catalog id or alias.  |
| `GET /models`            | Catalog endpoint.                                                   |
| `GET /version`           | vLLM-compatible version discovery; a running engine's answer wins.  |
| `GET /props`             | llama.cpp-compatible server properties; engine answer or fallback.  |
| `GET /v1/props`          | Versioned llama.cpp-compatible server properties.                   |
| `GET /api/tags`          | Ollama-compatible model list.                                       |
| `POST /api/show`         | Ollama-compatible model details (`name` or `model`).                |
| `GET /api/v1/models`     | LM Studio-compatible model list and loaded-instance metadata.       |
| `GET /running`           | Object containing the visible running-model list.                   |
| `/v1/chat/completions`   | Forward the compatible chat request to the resolved engine.         |
| `/v1/completions`        | Forward a compatible text-completion request.                       |
| `/v1/responses`          | Forward a compatible Responses request when the engine provides it. |
| `/v1/embeddings`         | Forward a compatible embedding request.                             |
| `/v1/audio/…`            | Forward the selected engine's supported audio routes.               |
| `/audioapi/v1/tasks/run` | Dispatch a compatible generic audio-task request.                   |
| `/upstream/{model}/…`    | Forward a concrete model-scoped engine API path.                    |
| `/ui/{model}/…`          | Model-scoped engine UI transport with its authorization policy.     |
| `/v/…`                   | Forward a versionless path after route-prefix rewriting.            |

Forwarded engine paths retain their engine-specific method, body, and response contracts. An entry here describes routing capability, not universal endpoint support across all engines. Streaming forwards upstream chunks with cancellation and backpressure. JSON/SSE usage observation is separate from the forwarded response bytes.

The server-wide discovery routes (`/version`, `/props`, `/v1/props`, `/api/tags`, `/api/show`, `/api/v1/models`, `/v1/models/{model}`) do not identify a model in their path. Studio renders them from the accessible catalog so discovery probes never cold-start an engine. For `/version`, `/props` and `/v1/props`, an already-running engine's answer is preferred; a 404/405 or non-JSON reply falls back to the synthesized compatibility shape. `/api/tags`, `/api/show`, `/api/v1/models` and `/v1/models/{model}` are always Studio-owned catalog views. Listing routes require the `models:read` scope and remain subject to a key's model restrictions.

The Studio management API supplies proxy administration: lifecycle, per-model unload/preload/reload, logs, request history, and key policy. Inference keys do not grant these management actions. Model listing and running-model visibility respect the requesting key's restrictions.

## Common management workflows

### Save model configuration

`PUT /api/models/{model_id}/config` accepts the selected engine and engine sections. This example is a configuration object for a GGUF engine; parameter support follows the scanned catalog:

```json
{
  "engine": "llama_cpp",
  "engines": {
    "llama_cpp": {
      "ctx_size": 8192,
      "model_alias": "assistant",
      "runtime_env": { "CUDA_VISIBLE_DEVICES": "0,1" },
      "runtime_env_unset": [],
      "set_params_by_id": [
        {
          "sub_id": "no-thinking",
          "params": {
            "chat_template_kwargs.enable_thinking": false,
            "thinking_effort": "low"
          }
        }
      ]
    }
  }
}
```

Save records desired model configuration. Engine launch changes use the runtime plan/apply flow. API aliases and request presets participate in live native routing. See [model identity and aliases](features.md#model-identity-and-aliases).

### Review and apply a launch revision

1. Read `GET /api/runtime/pending`.
2. Find the model's `catalog_id`, desired/published revision values, action, and reasons.
3. Submit `POST /api/models/{model_id}/runtime/apply` with mode and a unique intent key.
4. Follow the returned result/operation and its model-scoped operation endpoint when applicable.

```json
{
  "mode": "restart_now",
  "idempotency_key": "example-unique-apply-intent",
  "expected_desired_revision": "DESIRED_REVISION_FROM_PLAN",
  "expected_published_revision": "PUBLISHED_REVISION_FROM_PLAN"
}
```

`mode` accepts `restart_now` or `next_start`. An expected published revision can be `null` to assert that no published pointer exists. An omitted expectation does not make that assertion. The plan action `publish_next_start` corresponds to API mode `next_start`.

Batch publication uses `POST /api/runtime/apply-launch`:

```json
{
  "mode": "next_start",
  "models": [
    {
      "catalog_id": "MODEL_CATALOG_ID",
      "idempotency_key": "example-unique-model-intent",
      "expected_desired_revision": "DESIRED_REVISION_FROM_PLAN",
      "expected_published_revision": null
    }
  ]
}
```

Batch entries run sequentially; the result includes `results` and `stopped_early`. A failed entry stops subsequent work. A matching idempotency intent returns its recorded result; conflicting intent content receives a conflict response.

### Configure live routing and serving limits

`GET /api/proxy` returns settings, observed runtime/process state, traffic summary, and network information. `PUT /api/proxy/settings` accepts the `ProxySettingsBody` schema. Supply the desired complete settings object: omitted fields receive that schema's defaults.

```json
{
  "proxy_port": 2000,
  "allow_anonymous": true,
  "default_model": "balanced",
  "routing_aliases": {
    "balanced": ["assistant:thinking", "assistant:no-thinking"]
  },
  "queue_timeout_seconds": 30,
  "max_queue_size": 100,
  "max_connections": 200,
  "max_keepalive_connections": 40
}
```

The response identifies `applied_live_fields`, `routing_revision`, and `restart_required`. Default model, routing aliases, and access-policy publication apply live. Serving/listener changes require the lifecycle action `restart` or `apply`. Available lifecycle actions are `start`, `stop`, `restart`, and `apply`.

### Create or edit an inference key

`POST /api/inference/keys` accepts:

```json
{
  "name": "Desktop client",
  "models": ["MODEL_RUNTIME_ID"],
  "scopes": ["inference", "models:read"],
  "token_accounting": "auto",
  "rate_limit_per_minute": 60,
  "input_tokens_per_minute": 20000,
  "output_tokens_per_minute": 10000,
  "daily_token_budget": 200000,
  "monthly_token_budget": 3000000,
  "max_output_tokens": 2048,
  "max_concurrent_requests": 2
}
```

`name` is required and has a 120-character bound. `expires_at` is an optional expiry value. Omitted/empty model restrictions mean unrestricted model selection within the key's scopes. Default scopes contain `inference`. `token_accounting` is `auto` or `exact`. Optional policy limits use their validated positive bounds; omit a limit when it is not required.

| Policy input                           | Accepted form                                                                              |
| -------------------------------------- | ------------------------------------------------------------------------------------------ |
| `name`                                 | Non-empty string, at most 120 characters.                                                  |
| `models`                               | Array of exposed/model IDs; empty or omitted means no model-list restriction.              |
| `scopes`                               | Non-empty array using `inference` and/or `models:read`; creation defaults to `inference`.  |
| `expires_at`                           | Optional parseable expiry timestamp.                                                       |
| `rate_limit_per_minute`                | Optional positive integer; creation uses a 1–1,000,000 bound.                              |
| Token/budget/output/concurrency limits | Positive integer or `null`; shared policy validation caps supplied integer limits at 10¹². |
| `token_accounting`                     | `auto` or `exact`.                                                                         |

Creation returns metadata and `key`, the secret. Listing returns key metadata and usage. `PATCH /api/inference/keys/{key_id}` updates the permitted policy fields. Rotate returns a usable secret for that key identity; revoke removes the key. See [client policies](proxy-operations.md#client-policies-and-accounting) for admission and settlement semantics.

### Preview and apply a portable restore

`GET /api/config-backup` returns the schema-1 portable configuration document. Restore requests contain `backup`, `decisions`, and `mapping`. Preview returns a plan ID and relevant document revisions. Apply also includes that `plan_id`.

```json
{
  "backup": {
    "schema_version": 1,
    "kind": "llama-cpp-studio-config-backup",
    "application_version": "VERSION",
    "preferences": {},
    "models": [],
    "templates": []
  },
  "decisions": {},
  "mapping": {}
}
```

Use the actual exported document as `backup`; its `limits` and item structure define the portable payload. Decisions use the preview's identifiers and its keep/add/replace/skip choices. Unresolved references require a mapping or skip choice. A response with an uncertain apply outcome uses reconciliation/refresh before another mutation.

### Restore one history item

`POST /api/config-history/{entry_id}/restore` accepts a supported `kind` of `preference`, `model`, or `template`, the target `item_id`, and the `expected_revision` from the diff response. The operation restores saved configuration only.

### Run a local benchmark

```json
{
  "model_id": "MODEL_CATALOG_ID",
  "prompt": "Reply with a short description of the sky.",
  "max_tokens": 64
}
```

Submit to `POST /api/benchmarks/run` for a running compatible model. Read results with `GET /api/benchmarks/{model_id}`. The result records configuration identity, output preview, latency, throughput when available, and observed GPU memory.

## HTTP errors and operation outcomes

| Status        | Typical meaning                                                                                                              |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `400`         | Invalid option, mode, key policy, or operation input.                                                                        |
| `401` / `403` | Authentication, scope, model-access, or administrative permission failure.                                                   |
| `404`         | Unknown model, task, installation, key, history entry, or other resource.                                                    |
| `409`         | Busy resource, stale revision/plan, idempotency conflict, or action requiring state-bound confirmation.                      |
| `413`         | Request/backup/upload size limit.                                                                                            |
| `422`         | Schema validation or model/runtime configuration validation failure.                                                         |
| `429`         | Queue, rate, concurrency, or quota rejection.                                                                                |
| `5xx`         | Persistence, proxy/upstream, installation/runtime, or service failure requiring the returned detail/outcome to be inspected. |

Error shapes vary by endpoint: FastAPI validation errors use `detail`; service errors can also carry `code`, `error`, revision/state tokens, operation IDs, or persistence commitment information. A task cancellation acknowledgement and an uncertain persistence response require their corresponding state checks.

## Management endpoint catalog

The catalog includes every documented OpenAPI operation. The response column lists schema-declared success responses; additional service outcomes follow the endpoint's implementation. A body described as `object` accepts a route-specific JSON contract. Path/query parameters and request schemas follow the tables.

### 1cat-vllm

| Method | Path                            | Operation           | Parameters | Request body                      | Success     |
| ------ | ------------------------------- | ------------------- | ---------- | --------------------------------- | ----------- |
| GET    | `/api/1cat-vllm/build-settings` | Get Build Settings  | —          | —                                 | 200: object |
| PUT    | `/api/1cat-vllm/build-settings` | Save Build Settings | —          | object or null (application/json) | 200: object |
| POST   | `/api/1cat-vllm/cancel`         | Cancel              | —          | object (application/json)         | 200: object |
| GET    | `/api/1cat-vllm/check-updates`  | Check Updates       | —          | —                                 | 200: object |
| POST   | `/api/1cat-vllm/install`        | Install             | —          | object or null (application/json) | 200: object |
| POST   | `/api/1cat-vllm/install-source` | Install Source      | —          | object or null (application/json) | 200: object |
| POST   | `/api/1cat-vllm/remove`         | Remove              | —          | object or null (application/json) | 200: object |
| GET    | `/api/1cat-vllm/status`         | Status              | —          | —                                 | 200: object |

### application

| Method | Path                 | Operation             | Parameters | Request body                    | Success  |
| ------ | -------------------- | --------------------- | ---------- | ------------------------------- | -------- |
| GET    | `/api/access`        | Get Access Policy     | —          | —                               | 200: any |
| GET    | `/api/events`        | Sse Events            | —          | —                               | 200: any |
| POST   | `/api/session`       | Create Session        | —          | SessionLogin (application/json) | 200: any |
| POST   | `/api/tasks/dismiss` | Dismiss Activity Task | —          | —                               | 200: any |

### audio-batches-proxy

| Method | Path                         | Operation                  | Parameters | Request body | Success  |
| ------ | ---------------------------- | -------------------------- | ---------- | ------------ | -------- |
| POST   | `/v1/batches/transcriptions` | Proxy Batch Transcriptions | —          | —            | 200: any |

### audio-openai-proxy

| Method  | Path                               | Operation                       | Parameters           | Request body | Success  |
| ------- | ---------------------------------- | ------------------------------- | -------------------- | ------------ | -------- |
| POST    | `/v1/audio/alignments`             | Proxy Alignments                | —                    | —            | 200: any |
| POST    | `/v1/audio/speech`                 | Proxy Speech                    | —                    | —            | 200: any |
| POST    | `/v1/audio/transcriptions`         | Proxy Transcriptions            | —                    | —            | 200: any |
| POST    | `/v1/audio/transcriptions/details` | Proxy Transcription Details     | —                    | —            | 200: any |
| DELETE  | `/v1/audio/{rest}`                 | Proxy Audio Passthrough Delete  | `rest` path required | —            | 200: any |
| GET     | `/v1/audio/{rest}`                 | Proxy Audio Passthrough Get     | `rest` path required | —            | 200: any |
| HEAD    | `/v1/audio/{rest}`                 | Proxy Audio Passthrough Head    | `rest` path required | —            | 200: any |
| OPTIONS | `/v1/audio/{rest}`                 | Proxy Audio Passthrough Options | `rest` path required | —            | 200: any |
| PATCH   | `/v1/audio/{rest}`                 | Proxy Audio Passthrough Patch   | `rest` path required | —            | 200: any |
| POST    | `/v1/audio/{rest}`                 | Proxy Audio Passthrough Post    | `rest` path required | —            | 200: any |
| PUT     | `/v1/audio/{rest}`                 | Proxy Audio Passthrough Put     | `rest` path required | —            | 200: any |

### audio-tasks-proxy

| Method | Path            | Operation       | Parameters | Request body | Success  |
| ------ | --------------- | --------------- | ---------- | ------------ | -------- |
| POST   | `/v1/tasks/run` | Proxy Tasks Run | —          | —            | 200: any |

### audio.cpp

| Method | Path                                      | Operation                | Parameters           | Request body              | Success  |
| ------ | ----------------------------------------- | ------------------------ | -------------------- | ------------------------- | -------- |
| GET    | `/api/audio-cpp/build-options`            | Get Build Options        | —                    | —                         | 200: any |
| GET    | `/api/audio-cpp/build-settings`           | Get Build Settings       | —                    | —                         | 200: any |
| PUT    | `/api/audio-cpp/build-settings`           | Save Build Settings      | —                    | object (application/json) | 200: any |
| POST   | `/api/audio-cpp/build-source`             | Build Source             | —                    | object (application/json) | 200: any |
| POST   | `/api/audio-cpp/cancel`                   | Cancel                   | —                    | object (application/json) | 200: any |
| GET    | `/api/audio-cpp/check-updates`            | Check Updates            | `ref` query optional | —                         | 200: any |
| DELETE | `/api/audio-cpp/community-voices`         | Remove Community Voices  | —                    | —                         | 200: any |
| GET    | `/api/audio-cpp/community-voices`         | Get Community Voices     | —                    | —                         | 200: any |
| POST   | `/api/audio-cpp/community-voices/install` | Install Community Voices | —                    | —                         | 200: any |
| GET    | `/api/audio-cpp/contract`                 | Get Integration Contract | —                    | —                         | 200: any |
| GET    | `/api/audio-cpp/status`                   | Status                   | —                    | —                         | 200: any |
| POST   | `/api/audio-cpp/update`                   | Update                   | —                    | object (application/json) | 200: any |

### benchmarks

| Method | Path                         | Operation         | Parameters                                       | Request body                        | Success                                               |
| ------ | ---------------------------- | ----------------- | ------------------------------------------------ | ----------------------------------- | ----------------------------------------------------- |
| POST   | `/api/benchmarks/run`        | Run Benchmark     | —                                                | BenchmarkRunBody (application/json) | 200: [BenchmarkResponse](#benchmarkresponse)          |
| GET    | `/api/benchmarks/{model_id}` | Benchmark History | `model_id` path required; `limit` query optional | —                                   | 200: array of [BenchmarkResponse](#benchmarkresponse) |

### config-backup

| Method | Path                           | Operation               | Parameters | Request body | Success                                                  |
| ------ | ------------------------------ | ----------------------- | ---------- | ------------ | -------------------------------------------------------- |
| GET    | `/api/config-backup`           | Download Config Backup  | —          | —            | 200: any                                                 |
| POST   | `/api/config-backup/apply`     | Apply Config Backup     | —          | —            | 200: [BackupApplyResponse](#backupapplyresponse)         |
| POST   | `/api/config-backup/preview`   | Preview Config Backup   | —          | —            | 200: [BackupPreviewResponse](#backuppreviewresponse)     |
| POST   | `/api/config-backup/reconcile` | Reconcile Config Backup | —          | —            | 200: [BackupReconcileResponse](#backupreconcileresponse) |

### config-history

| Method | Path                                     | Operation            | Parameters               | Request body                          | Success                                                     |
| ------ | ---------------------------------------- | -------------------- | ------------------------ | ------------------------------------- | ----------------------------------------------------------- |
| GET    | `/api/config-history`                    | History Entries      | `limit` query optional   | —                                     | 200: array of [HistoryEntryResponse](#historyentryresponse) |
| GET    | `/api/config-history/{entry_id}`         | History Diff         | `entry_id` path required | —                                     | 200: [HistoryDiffResponse](#historydiffresponse)            |
| POST   | `/api/config-history/{entry_id}/restore` | Restore History Item | `entry_id` path required | HistoryRestoreBody (application/json) | 200: [HistoryRestoreResponse](#historyrestoreresponse)      |

### engines

| Method | Path           | Operation               | Parameters | Request body | Success  |
| ------ | -------------- | ----------------------- | ---------- | ------------ | -------- |
| GET    | `/api/engines` | List Engine Descriptors | —          | —            | 200: any |

### gpu

| Method | Path            | Operation    | Parameters | Request body | Success  |
| ------ | --------------- | ------------ | ---------- | ------------ | -------- |
| GET    | `/api/gpu-info` | Get Gpu Info | —          | —            | 200: any |
| GET    | `/api/gpu-list` | Get Gpu List | —          | —            | 200: any |

### inference-access

| Method | Path                                  | Operation            | Parameters             | Request body              | Success  |
| ------ | ------------------------------------- | -------------------- | ---------------------- | ------------------------- | -------- |
| GET    | `/api/inference/keys`                 | Get Inference Keys   | —                      | —                         | 200: any |
| POST   | `/api/inference/keys`                 | Post Inference Key   | —                      | object (application/json) | 201: any |
| DELETE | `/api/inference/keys/{key_id}`        | Remove Inference Key | `key_id` path required | —                         | 204      |
| PATCH  | `/api/inference/keys/{key_id}`        | Patch Inference Key  | `key_id` path required | object (application/json) | 200: any |
| POST   | `/api/inference/keys/{key_id}/rotate` | Rotate Inference Key | `key_id` path required | —                         | 200: any |

### llama-versions

| Method | Path                                        | Operation                       | Parameters                                                                                                                                   | Request body              | Success                                          |
| ------ | ------------------------------------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------ |
| GET    | `/api/llama-versions`                       | List Llama Versions             | —                                                                                                                                            | —                         | 200: any                                         |
| POST   | `/api/llama-versions/build-cancel`          | Build Cancel                    | —                                                                                                                                            | object (application/json) | 200: any                                         |
| GET    | `/api/llama-versions/build-capabilities`    | Get Build Capabilities Endpoint | —                                                                                                                                            | —                         | 200: any                                         |
| GET    | `/api/llama-versions/build-options`         | Get Build Options               | `engine` query optional                                                                                                                      | —                         | 200: any                                         |
| GET    | `/api/llama-versions/build-settings`        | Get Build Settings              | `engine` query optional                                                                                                                      | —                         | 200: any                                         |
| PUT    | `/api/llama-versions/build-settings`        | Update Build Settings           | `engine` query optional                                                                                                                      | object (application/json) | 200: any                                         |
| POST   | `/api/llama-versions/build-source`          | Build Source                    | —                                                                                                                                            | object (application/json) | 200: any                                         |
| GET    | `/api/llama-versions/check-updates`         | Check Updates                   | `source` query optional                                                                                                                      | —                         | 200: [UpdateCheckResponse](#updatecheckresponse) |
| POST   | `/api/llama-versions/cuda-install`          | Install Cuda                    | —                                                                                                                                            | object (application/json) | 200: any                                         |
| GET    | `/api/llama-versions/cuda-logs`             | Get Cuda Logs                   | —                                                                                                                                            | —                         | 200: any                                         |
| GET    | `/api/llama-versions/cuda-status`           | Get Cuda Status                 | —                                                                                                                                            | —                         | 200: any                                         |
| POST   | `/api/llama-versions/cuda-uninstall`        | Uninstall Cuda                  | —                                                                                                                                            | object (application/json) | 200: any                                         |
| POST   | `/api/llama-versions/cuda/cancel`           | Cuda Cancel                     | —                                                                                                                                            | object (application/json) | 200: any                                         |
| POST   | `/api/llama-versions/scan-engine-params`    | Scan Engine Params Route        | —                                                                                                                                            | object (application/json) | 200: any                                         |
| POST   | `/api/llama-versions/update`                | Update Engine                   | —                                                                                                                                            | object (application/json) | 200: any                                         |
| GET    | `/api/llama-versions/verify/{version}`      | Verify Version                  | `version` path required                                                                                                                      | —                         | 200: any                                         |
| POST   | `/api/llama-versions/versions/activate`     | Activate Version Body           | —                                                                                                                                            | object (application/json) | 200: any                                         |
| PUT    | `/api/llama-versions/versions/build-config` | Update Version Build Config     | —                                                                                                                                            | object (application/json) | 200: any                                         |
| POST   | `/api/llama-versions/versions/retry`        | Retry Version Body              | —                                                                                                                                            | object (application/json) | 200: any                                         |
| POST   | `/api/llama-versions/versions/sync`         | Sync Version Body               | —                                                                                                                                            | object (application/json) | 200: any                                         |
| DELETE | `/api/llama-versions/{version_id}`          | Delete Version                  | `version_id` path required; `confirm_operation_id` query optional; `confirm_state` query optional; `retire_launch_references` query optional | —                         | 200: any                                         |

### lmdeploy

| Method | Path                           | Operation           | Parameters | Request body                      | Success     |
| ------ | ------------------------------ | ------------------- | ---------- | --------------------------------- | ----------- |
| GET    | `/api/lmdeploy/build-settings` | Get Build Settings  | —          | —                                 | 200: object |
| PUT    | `/api/lmdeploy/build-settings` | Save Build Settings | —          | object or null (application/json) | 200: object |
| POST   | `/api/lmdeploy/cancel`         | Cancel              | —          | object (application/json)         | 200: object |
| GET    | `/api/lmdeploy/check-updates`  | Check Updates       | —          | —                                 | 200: object |
| POST   | `/api/lmdeploy/install`        | Install             | —          | object or null (application/json) | 200: object |
| POST   | `/api/lmdeploy/install-source` | Install Source      | —          | object or null (application/json) | 200: object |
| POST   | `/api/lmdeploy/remove`         | Remove              | —          | object or null (application/json) | 200: object |
| GET    | `/api/lmdeploy/status`         | Status              | —          | —                                 | 200: object |

### model catalog

| Method | Path                                        | Operation            | Parameters              | Request body              | Success  |
| ------ | ------------------------------------------- | -------------------- | ----------------------- | ------------------------- | -------- |
| POST   | `/api/model-catalog/import`                 | Import Audio Bundle  | —                       | object (application/json) | 200: any |
| POST   | `/api/model-catalog/install`                | Install Catalog Item | —                       | object (application/json) | 200: any |
| POST   | `/api/model-catalog/search`                 | Search Catalog       | —                       | object (application/json) | 200: any |
| POST   | `/api/model-catalog/tasks/{task_id}/cancel` | Cancel Audio Install | `task_id` path required | —                         | 200: any |

### model-config-templates

| Method | Path                                        | Operation       | Parameters                  | Request body                          | Success              |
| ------ | ------------------------------------------- | --------------- | --------------------------- | ------------------------------------- | -------------------- |
| GET    | `/api/model-config-templates`               | List Templates  | —                           | —                                     | 200: array of object |
| POST   | `/api/model-config-templates`               | Create Template | —                           | TemplateCreateBody (application/json) | 200: object          |
| DELETE | `/api/model-config-templates/{template_id}` | Delete Template | `template_id` path required | —                                     | 200: object          |
| PUT    | `/api/model-config-templates/{template_id}` | Update Template | `template_id` path required | TemplateUpdateBody (application/json) | 200: object          |

List responses include `has_routing` and `has_request_aliases` in addition to the stored
snapshot metadata, so clients can enable a routing-only action without downloading every
template body. Template application accepts these `apply_engines` values:

| Value        | Merge behavior                                                                                                    |
| ------------ | ----------------------------------------------------------------------------------------------------------------- |
| `active`     | Merge the template's active section into the target's active engine.                                               |
| `all`        | Merge every engine section present in the template.                                                               |
| `set_engine` | Merge the template's active section and select that engine on the target.                                          |
| `routing`    | Copy only `model_alias`, `swap_aliases`, and `set_params_by_id` into the target's active engine. Requires routing data. |

`routing` leaves engine parameters and the selected engine unchanged. The response reports
`persisted`, `template_id`, `template_name`, and the resulting `config`. A `persist` write
still performs the normal model validation and routing-name collision checks.

### models

| Method | Path                                                         | Operation                      | Parameters                                                                                                                  | Request body                                                                                       | Success  |
| ------ | ------------------------------------------------------------ | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | -------- |
| GET    | `/api/models`                                                | List Models                    | —                                                                                                                           | —                                                                                                  | 200: any |
| POST   | `/api/models/delete-group`                                   | Delete Model Group             | —                                                                                                                           | DeleteGroupRequest (application/json)                                                              | 200: any |
| POST   | `/api/models/download`                                       | Download Huggingface Model     | —                                                                                                                           | object (application/json)                                                                          | 200: any |
| POST   | `/api/models/downloads/cancel`                               | Cancel Download                | —                                                                                                                           | object (application/json)                                                                          | 200: any |
| POST   | `/api/models/gguf/download-bundle`                           | Download Gguf Bundle           | —                                                                                                                           | object (application/json)                                                                          | 200: any |
| GET    | `/api/models/huggingface-token`                              | Get Huggingface Token Status   | —                                                                                                                           | —                                                                                                  | 200: any |
| POST   | `/api/models/huggingface-token`                              | Set Huggingface Token Endpoint | —                                                                                                                           | object (application/json)                                                                          | 200: any |
| GET    | `/api/models/param-registry`                                 | Get Param Registry Endpoint    | `engine` query optional; `model_id` query optional; `rescan` query optional; `family` query optional; `task` query optional | —                                                                                                  | 200: any |
| POST   | `/api/models/quantization-sizes`                             | Get Quantization Sizes         | —                                                                                                                           | object (application/json)                                                                          | 200: any |
| DELETE | `/api/models/safetensors`                                    | Delete Safetensors Model       | —                                                                                                                           | object (application/json)                                                                          | 200: any |
| GET    | `/api/models/safetensors`                                    | List Safetensors Models        | —                                                                                                                           | —                                                                                                  | 200: any |
| POST   | `/api/models/safetensors/download-bundle`                    | Download Safetensors Bundle    | —                                                                                                                           | SafetensorsBundleRequest (application/json)                                                        | 200: any |
| POST   | `/api/models/search`                                         | Search Huggingface Models      | —                                                                                                                           | object (application/json)                                                                          | 200: any |
| GET    | `/api/models/search/{model_id}/file-sizes`                   | Get Search File Sizes          | `model_id` path required; `filenames` query required                                                                        | —                                                                                                  | 200: any |
| DELETE | `/api/models/{model_id}`                                     | Delete Model                   | `model_id` path required; `retire_launch_references` query optional                                                         | —                                                                                                  | 200: any |
| GET    | `/api/models/{model_id}/companions`                          | Get Model Companions           | `model_id` path required                                                                                                    | —                                                                                                  | 200: any |
| GET    | `/api/models/{model_id}/config`                              | Get Model Config               | `model_id` path required                                                                                                    | —                                                                                                  | 200: any |
| PUT    | `/api/models/{model_id}/config`                              | Update Model Config            | `model_id` path required                                                                                                    | object (application/json)                                                                          | 200: any |
| POST   | `/api/models/{model_id}/config/apply-template`               | Apply Model Config Template    | `model_id` path required                                                                                                    | ApplyConfigTemplateBody (application/json)                                                         | 200: any |
| POST   | `/api/models/{model_id}/config/save-template`                | Save Model Config As Template  | `model_id` path required                                                                                                    | SaveConfigTemplateBody (application/json)                                                          | 200: any |
| POST   | `/api/models/{model_id}/connect-test`                        | Connect Test Model             | `model_id` path required                                                                                                    | —                                                                                                  | 200: any |
| POST   | `/api/models/{model_id}/dflash`                              | Update Model Dflash            | `model_id` path required                                                                                                    | object (application/json)                                                                          | 200: any |
| POST   | `/api/models/{model_id}/mtp`                                 | Update Model Mtp               | `model_id` path required                                                                                                    | object (application/json)                                                                          | 200: any |
| POST   | `/api/models/{model_id}/preview-command`                     | Preview Command                | `model_id` path required                                                                                                    | object (application/json)                                                                          | 200: any |
| POST   | `/api/models/{model_id}/projector`                           | Update Model Projector         | `model_id` path required                                                                                                    | object (application/json)                                                                          | 200: any |
| GET    | `/api/models/{model_id}/reference-audio`                     | List Model Reference Audio     | `model_id` path required                                                                                                    | —                                                                                                  | 200: any |
| POST   | `/api/models/{model_id}/reference-audio`                     | Upload Model Reference Audio   | `model_id` path required                                                                                                    | Body_upload_model_reference_audio_api_models__model_id__reference_audio_post (multipart/form-data) | 200: any |
| DELETE | `/api/models/{model_id}/reference-audio/{filename}`          | Delete Model Reference Audio   | `model_id` path required; `filename` path required                                                                          | —                                                                                                  | 200: any |
| POST   | `/api/models/{model_id}/refresh`                             | Refresh Model                  | `model_id` path required                                                                                                    | —                                                                                                  | 200: any |
| POST   | `/api/models/{model_id}/runtime/apply`                       | Apply Model Runtime            | `model_id` path required                                                                                                    | RuntimeApplyBody (application/json)                                                                | 200: any |
| GET    | `/api/models/{model_id}/runtime/apply/{operation_id}`        | Get Model Runtime Apply        | `model_id` path required; `operation_id` path required                                                                      | —                                                                                                  | 200: any |
| POST   | `/api/models/{model_id}/runtime/apply/{operation_id}/cancel` | Cancel Model Runtime Apply     | `model_id` path required; `operation_id` path required                                                                      | —                                                                                                  | 200: any |
| GET    | `/api/models/{model_id}/saved-command`                       | Get Saved Command              | `model_id` path required                                                                                                    | —                                                                                                  | 200: any |
| POST   | `/api/models/{model_id}/start`                               | Start Model                    | `model_id` path required                                                                                                    | ActionConfirmBody or null (application/json)                                                       | 200: any |
| POST   | `/api/models/{model_id}/stop`                                | Stop Model                     | `model_id` path required                                                                                                    | ActionConfirmBody or null (application/json)                                                       | 200: any |

### proxy

| Method | Path                                   | Operation              | Parameters                                                                                                                                                                                           | Request body                         | Success  |
| ------ | -------------------------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ | -------- |
| GET    | `/api/proxy`                           | Get Proxy Admin        | —                                                                                                                                                                                                    | —                                    | 200: any |
| POST   | `/api/proxy/lifecycle/{action}`        | Proxy Lifecycle        | `action` path required                                                                                                                                                                               | —                                    | 200: any |
| DELETE | `/api/proxy/models/{model_id}/logs`    | Clear Proxy Model Logs | `model_id` path required                                                                                                                                                                             | —                                    | 204      |
| GET    | `/api/proxy/models/{model_id}/logs`    | Get Proxy Model Logs   | `model_id` path required; `search` query optional; `level` query optional; `download` query optional; `limit` query optional                                                                         | —                                    | 200: any |
| POST   | `/api/proxy/models/{model_id}/preload` | Preload Proxy Model    | `model_id` path required; `warm` query optional                                                                                                                                                      | —                                    | 200: any |
| POST   | `/api/proxy/models/{model_id}/reload`  | Reload Proxy Model     | `model_id` path required                                                                                                                                                                             | —                                    | 200: any |
| POST   | `/api/proxy/models/{model_id}/unload`  | Unload Proxy Model     | `model_id` path required                                                                                                                                                                             | —                                    | 200: any |
| GET    | `/api/proxy/requests`                  | Get Proxy Requests     | `limit` query optional; `offset` query optional; `model` query optional; `key_id` query optional; `state` query optional; `status` query optional; `outcome` query optional; `search` query optional | —                                    | 200: any |
| PUT    | `/api/proxy/settings`                  | Put Proxy Settings     | —                                                                                                                                                                                                    | ProxySettingsBody (application/json) | 200: any |
| POST   | `/api/proxy/stats/reset`               | Reset Proxy Stats      | —                                                                                                                                                                                                    | —                                    | 200: any |

### runtime

| Method | Path                        | Operation            | Parameters | Request body                       | Success  |
| ------ | --------------------------- | -------------------- | ---------- | ---------------------------------- | -------- |
| POST   | `/api/runtime/apply-launch` | Apply Launch Changes | —          | LaunchApplyBody (application/json) | 200: any |
| GET    | `/api/runtime/pending`      | Runtime Pending      | —          | —                                  | 200: any |
| GET    | `/api/runtime/stale`        | Runtime Stale        | —          | —                                  | 200: any |

### sglang

| Method | Path                              | Operation           | Parameters | Request body                      | Success     |
| ------ | --------------------------------- | ------------------- | ---------- | --------------------------------- | ----------- |
| GET    | `/api/sglang-v100/build-settings` | Get Build Settings  | —          | —                                 | 200: object |
| PUT    | `/api/sglang-v100/build-settings` | Save Build Settings | —          | object or null (application/json) | 200: object |
| POST   | `/api/sglang-v100/cancel`         | Cancel              | —          | object (application/json)         | 200: object |
| GET    | `/api/sglang-v100/check-updates`  | Check Updates       | —          | —                                 | 200: object |
| POST   | `/api/sglang-v100/install`        | Install             | —          | object or null (application/json) | 200: object |
| POST   | `/api/sglang-v100/install-source` | Install Source      | —          | object or null (application/json) | 200: object |
| POST   | `/api/sglang-v100/remove`         | Remove              | —          | object or null (application/json) | 200: object |
| GET    | `/api/sglang-v100/status`         | Status              | —          | —                                 | 200: object |
| GET    | `/api/sglang/build-settings`      | Get Build Settings  | —          | —                                 | 200: object |
| PUT    | `/api/sglang/build-settings`      | Save Build Settings | —          | object or null (application/json) | 200: object |
| POST   | `/api/sglang/cancel`              | Cancel              | —          | object (application/json)         | 200: object |
| GET    | `/api/sglang/check-updates`       | Check Updates       | —          | —                                 | 200: object |
| POST   | `/api/sglang/install`             | Install             | —          | object or null (application/json) | 200: object |
| POST   | `/api/sglang/install-source`      | Install Source      | —          | object or null (application/json) | 200: object |
| POST   | `/api/sglang/remove`              | Remove              | —          | object or null (application/json) | 200: object |
| GET    | `/api/sglang/status`              | Status              | —          | —                                 | 200: object |

### status

| Method | Path                        | Operation              | Parameters | Request body              | Success  |
| ------ | --------------------------- | ---------------------- | ---------- | ------------------------- | -------- |
| GET    | `/api/diagnostics/bundle`   | Diagnostics Bundle     | —          | —                         | 200: any |
| GET    | `/api/live`                 | Live                   | —          | —                         | 200: any |
| POST   | `/api/operations/reconcile` | Reconcile Operations   | —          | —                         | 200: any |
| GET    | `/api/operations/recovery`  | Operation Recovery     | —          | —                         | 200: any |
| GET    | `/api/ready`                | Ready                  | —          | —                         | 200: any |
| GET    | `/api/settings/inference`   | Get Inference Settings | —          | —                         | 200: any |
| PUT    | `/api/settings/inference`   | Put Inference Settings | —          | object (application/json) | 200: any |
| GET    | `/api/status`               | Get System Status      | —          | —                         | 200: any |

### unsloth-llama

| Method | Path                               | Operation     | Parameters | Request body                      | Success     |
| ------ | ---------------------------------- | ------------- | ---------- | --------------------------------- | ----------- |
| POST   | `/api/unsloth-llama/cancel`        | Cancel        | —          | object (application/json)         | 200: object |
| GET    | `/api/unsloth-llama/check-updates` | Check Updates | —          | —                                 | 200: object |
| POST   | `/api/unsloth-llama/install`       | Install       | —          | object or null (application/json) | 200: object |

### vllm

| Method | Path                       | Operation           | Parameters | Request body                      | Success     |
| ------ | -------------------------- | ------------------- | ---------- | --------------------------------- | ----------- |
| GET    | `/api/vllm/build-settings` | Get Build Settings  | —          | —                                 | 200: object |
| PUT    | `/api/vllm/build-settings` | Save Build Settings | —          | object or null (application/json) | 200: object |
| POST   | `/api/vllm/cancel`         | Cancel              | —          | object (application/json)         | 200: object |
| GET    | `/api/vllm/check-updates`  | Check Updates       | —          | —                                 | 200: object |
| POST   | `/api/vllm/install`        | Install             | —          | object or null (application/json) | 200: object |
| POST   | `/api/vllm/install-source` | Install Source      | —          | object or null (application/json) | 200: object |
| POST   | `/api/vllm/remove`         | Remove              | —          | object or null (application/json) | 200: object |
| GET    | `/api/vllm/status`         | Status              | —          | —                                 | 200: object |

## Operation parameter details

Path, query, header, and cookie fields below use the current OpenAPI definitions. A default is explicit only when the schema declares it. JSON body fields appear in the named schema tables or their route-specific workflow description.

| Operation                                                         | Parameter                  | Location | Type            | Required | Default     | Constraints                     |
| ----------------------------------------------------------------- | -------------------------- | -------- | --------------- | -------- | ----------- | ------------------------------- |
| `GET /api/audio-cpp/check-updates`                                | `ref`                      | query    | string or null  | no       | —           | —                               |
| `GET /api/benchmarks/{model_id}`                                  | `model_id`                 | path     | string          | yes      | —           | —                               |
| `GET /api/benchmarks/{model_id}`                                  | `limit`                    | query    | integer         | no       | 20          | minimum: 1; maximum: 100        |
| `GET /api/config-history`                                         | `limit`                    | query    | integer         | no       | 30          | minimum: 1; maximum: 100        |
| `GET /api/config-history/{entry_id}`                              | `entry_id`                 | path     | string          | yes      | —           | —                               |
| `POST /api/config-history/{entry_id}/restore`                     | `entry_id`                 | path     | string          | yes      | —           | —                               |
| `DELETE /api/inference/keys/{key_id}`                             | `key_id`                   | path     | string          | yes      | —           | —                               |
| `PATCH /api/inference/keys/{key_id}`                              | `key_id`                   | path     | string          | yes      | —           | —                               |
| `POST /api/inference/keys/{key_id}/rotate`                        | `key_id`                   | path     | string          | yes      | —           | —                               |
| `GET /api/llama-versions/build-options`                           | `engine`                   | query    | string or null  | no       | —           | —                               |
| `GET /api/llama-versions/build-settings`                          | `engine`                   | query    | string          | no       | "llama_cpp" | —                               |
| `PUT /api/llama-versions/build-settings`                          | `engine`                   | query    | string          | no       | "llama_cpp" | —                               |
| `GET /api/llama-versions/check-updates`                           | `source`                   | query    | string or null  | no       | —           | —                               |
| `GET /api/llama-versions/verify/{version}`                        | `version`                  | path     | string          | yes      | —           | —                               |
| `DELETE /api/llama-versions/{version_id}`                         | `version_id`               | path     | string          | yes      | —           | —                               |
| `DELETE /api/llama-versions/{version_id}`                         | `confirm_operation_id`     | query    | string or null  | no       | —           | —                               |
| `DELETE /api/llama-versions/{version_id}`                         | `confirm_state`            | query    | string or null  | no       | —           | —                               |
| `DELETE /api/llama-versions/{version_id}`                         | `retire_launch_references` | query    | boolean         | no       | false       | —                               |
| `POST /api/model-catalog/tasks/{task_id}/cancel`                  | `task_id`                  | path     | string          | yes      | —           | —                               |
| `DELETE /api/model-config-templates/{template_id}`                | `template_id`              | path     | string          | yes      | —           | —                               |
| `PUT /api/model-config-templates/{template_id}`                   | `template_id`              | path     | string          | yes      | —           | —                               |
| `GET /api/models/param-registry`                                  | `engine`                   | query    | string          | no       | "llama_cpp" | —                               |
| `GET /api/models/param-registry`                                  | `model_id`                 | query    | string or null  | no       | —           | —                               |
| `GET /api/models/param-registry`                                  | `rescan`                   | query    | boolean         | no       | false       | —                               |
| `GET /api/models/param-registry`                                  | `family`                   | query    | string or null  | no       | —           | —                               |
| `GET /api/models/param-registry`                                  | `task`                     | query    | string or null  | no       | —           | —                               |
| `GET /api/models/search/{model_id}/file-sizes`                    | `model_id`                 | path     | string          | yes      | —           | —                               |
| `GET /api/models/search/{model_id}/file-sizes`                    | `filenames`                | query    | string          | yes      | —           | —                               |
| `DELETE /api/models/{model_id}`                                   | `model_id`                 | path     | string          | yes      | —           | —                               |
| `DELETE /api/models/{model_id}`                                   | `retire_launch_references` | query    | boolean         | no       | false       | —                               |
| `GET /api/models/{model_id}/companions`                           | `model_id`                 | path     | string          | yes      | —           | —                               |
| `GET /api/models/{model_id}/config`                               | `model_id`                 | path     | string          | yes      | —           | —                               |
| `PUT /api/models/{model_id}/config`                               | `model_id`                 | path     | string          | yes      | —           | —                               |
| `POST /api/models/{model_id}/config/apply-template`               | `model_id`                 | path     | string          | yes      | —           | —                               |
| `POST /api/models/{model_id}/config/save-template`                | `model_id`                 | path     | string          | yes      | —           | —                               |
| `POST /api/models/{model_id}/connect-test`                        | `model_id`                 | path     | string          | yes      | —           | —                               |
| `POST /api/models/{model_id}/dflash`                              | `model_id`                 | path     | string          | yes      | —           | —                               |
| `POST /api/models/{model_id}/mtp`                                 | `model_id`                 | path     | string          | yes      | —           | —                               |
| `POST /api/models/{model_id}/preview-command`                     | `model_id`                 | path     | string          | yes      | —           | —                               |
| `POST /api/models/{model_id}/projector`                           | `model_id`                 | path     | string          | yes      | —           | —                               |
| `GET /api/models/{model_id}/reference-audio`                      | `model_id`                 | path     | string          | yes      | —           | —                               |
| `POST /api/models/{model_id}/reference-audio`                     | `model_id`                 | path     | string          | yes      | —           | —                               |
| `DELETE /api/models/{model_id}/reference-audio/{filename}`        | `model_id`                 | path     | string          | yes      | —           | —                               |
| `DELETE /api/models/{model_id}/reference-audio/{filename}`        | `filename`                 | path     | string          | yes      | —           | —                               |
| `POST /api/models/{model_id}/refresh`                             | `model_id`                 | path     | string          | yes      | —           | —                               |
| `POST /api/models/{model_id}/runtime/apply`                       | `model_id`                 | path     | string          | yes      | —           | —                               |
| `GET /api/models/{model_id}/runtime/apply/{operation_id}`         | `model_id`                 | path     | string          | yes      | —           | —                               |
| `GET /api/models/{model_id}/runtime/apply/{operation_id}`         | `operation_id`             | path     | string          | yes      | —           | —                               |
| `POST /api/models/{model_id}/runtime/apply/{operation_id}/cancel` | `model_id`                 | path     | string          | yes      | —           | —                               |
| `POST /api/models/{model_id}/runtime/apply/{operation_id}/cancel` | `operation_id`             | path     | string          | yes      | —           | —                               |
| `GET /api/models/{model_id}/saved-command`                        | `model_id`                 | path     | string          | yes      | —           | —                               |
| `POST /api/models/{model_id}/start`                               | `model_id`                 | path     | string          | yes      | —           | —                               |
| `POST /api/models/{model_id}/stop`                                | `model_id`                 | path     | string          | yes      | —           | —                               |
| `POST /api/proxy/lifecycle/{action}`                              | `action`                   | path     | string          | yes      | —           | —                               |
| `DELETE /api/proxy/models/{model_id}/logs`                        | `model_id`                 | path     | string          | yes      | —           | —                               |
| `GET /api/proxy/models/{model_id}/logs`                           | `model_id`                 | path     | string          | yes      | —           | —                               |
| `GET /api/proxy/models/{model_id}/logs`                           | `search`                   | query    | string          | no       | ""          | —                               |
| `GET /api/proxy/models/{model_id}/logs`                           | `level`                    | query    | string          | no       | ""          | —                               |
| `GET /api/proxy/models/{model_id}/logs`                           | `download`                 | query    | boolean         | no       | false       | —                               |
| `GET /api/proxy/models/{model_id}/logs`                           | `limit`                    | query    | integer         | no       | 5000        | —                               |
| `POST /api/proxy/models/{model_id}/preload`                       | `model_id`                 | path     | string          | yes      | —           | —                               |
| `POST /api/proxy/models/{model_id}/preload`                       | `warm`                     | query    | boolean         | no       | false       | —                               |
| `POST /api/proxy/models/{model_id}/reload`                        | `model_id`                 | path     | string          | yes      | —           | —                               |
| `POST /api/proxy/models/{model_id}/unload`                        | `model_id`                 | path     | string          | yes      | —           | —                               |
| `GET /api/proxy/requests`                                         | `limit`                    | query    | integer         | no       | 100         | minimum: 1; maximum: 100        |
| `GET /api/proxy/requests`                                         | `offset`                   | query    | integer         | no       | 0           | minimum: 0                      |
| `GET /api/proxy/requests`                                         | `model`                    | query    | string          | no       | ""          | —                               |
| `GET /api/proxy/requests`                                         | `key_id`                   | query    | string          | no       | ""          | —                               |
| `GET /api/proxy/requests`                                         | `state`                    | query    | string          | no       | ""          | —                               |
| `GET /api/proxy/requests`                                         | `status`                   | query    | integer or null | no       | —           | —                               |
| `GET /api/proxy/requests`                                         | `outcome`                  | query    | string          | no       | ""          | pattern: ^(\|4xx\|5xx\|errors)$ |
| `GET /api/proxy/requests`                                         | `search`                   | query    | string          | no       | ""          | —                               |
| `DELETE /v1/audio/{rest}`                                         | `rest`                     | path     | string          | yes      | —           | —                               |
| `GET /v1/audio/{rest}`                                            | `rest`                     | path     | string          | yes      | —           | —                               |
| `HEAD /v1/audio/{rest}`                                           | `rest`                     | path     | string          | yes      | —           | —                               |
| `OPTIONS /v1/audio/{rest}`                                        | `rest`                     | path     | string          | yes      | —           | —                               |
| `PATCH /v1/audio/{rest}`                                          | `rest`                     | path     | string          | yes      | —           | —                               |
| `POST /v1/audio/{rest}`                                           | `rest`                     | path     | string          | yes      | —           | —                               |
| `PUT /v1/audio/{rest}`                                            | `rest`                     | path     | string          | yes      | —           | —                               |

## Named request and response schemas

Fields use OpenAPI types/constraints plus service-supported values for runtime mode and history item kind. “Required” describes presence; `null` in a type describes a nullable value. A dash in Default means the schema declares no explicit default. Runtime settings and free-form service validation can add constraints described in the feature guide.

### ActionConfirmBody

| Field                  | Type           | Required | Default | Constraints |
| ---------------------- | -------------- | -------- | ------- | ----------- |
| `confirm_operation_id` | string or null | no       | —       | —           |
| `confirm_state`        | string or null | no       | —       | —           |

### ApplyConfigTemplateBody

| Field             | Type            | Required | Default  | Constraints                                      |
| ----------------- | --------------- | -------- | -------- | ------------------------------------------------ |
| `apply_engines`   | string          | no       | "active" | `active`, `all`, `set_engine`, or `routing`      |
| `include_routing` | boolean or null | no       | —        | —                                                |
| `persist`         | boolean         | no       | false    | —                                                |
| `template_id`     | string          | yes      | —        | —                                                |

### BackupApplyResponse

| Field     | Type   | Required | Default | Constraints |
| --------- | ------ | -------- | ------- | ----------- |
| `notice`  | string | yes      | —       | —           |
| `outcome` | string | yes      | —       | —           |
| `plan_id` | string | yes      | —       | —           |

### BackupItemResponse

| Field      | Type           | Required | Default | Constraints |
| ---------- | -------------- | -------- | ------- | ----------- |
| `action`   | string         | yes      | —       | —           |
| `id`       | string         | yes      | —       | —           |
| `kind`     | string         | yes      | —       | —           |
| `local_id` | string or null | no       | —       | —           |
| `reason`   | string or null | no       | —       | —           |

### BackupPreviewResponse

| Field            | Type                        | Required | Default | Constraints |
| ---------------- | --------------------------- | -------- | ------- | ----------- |
| `applicable`     | boolean                     | yes      | —       | —           |
| `items`          | array of BackupItemResponse | yes      | —       | —           |
| `limits`         | map of array of string      | yes      | —       | —           |
| `notice`         | string                      | yes      | —       | —           |
| `plan_id`        | string or null              | yes      | —       | —           |
| `revisions`      | map of string               | yes      | —       | —           |
| `schema_version` | integer                     | yes      | —       | —           |

### BackupReconcileResponse

| Field      | Type            | Required | Default | Constraints |
| ---------- | --------------- | -------- | ------- | ----------- |
| `outcome`  | string          | yes      | —       | —           |
| `replayed` | integer or null | no       | —       | —           |

### BenchmarkResponse

| Field                            | Type            | Required | Default | Constraints |
| -------------------------------- | --------------- | -------- | ------- | ----------- |
| `completion_tokens`              | integer or null | yes      | —       | —           |
| `config_fingerprint`             | string          | yes      | —       | —           |
| `config_revision`                | string          | yes      | —       | —           |
| `created_at`                     | number          | yes      | —       | —           |
| `id`                             | string          | yes      | —       | —           |
| `max_tokens`                     | integer         | yes      | —       | —           |
| `model_id`                       | string          | yes      | —       | —           |
| `output_preview`                 | string          | yes      | —       | —           |
| `peak_observed_gpu_memory_bytes` | integer or null | yes      | —       | —           |
| `prompt`                         | string          | yes      | —       | —           |
| `proxy_name`                     | string          | yes      | —       | —           |
| `time_to_first_token_ms`         | number          | yes      | —       | —           |
| `tokens_per_second`              | number or null  | yes      | —       | —           |
| `total_seconds`                  | number          | yes      | —       | —           |

### BenchmarkRunBody

| Field        | Type    | Required | Default                                      | Constraints                   |
| ------------ | ------- | -------- | -------------------------------------------- | ----------------------------- |
| `max_tokens` | integer | no       | 64                                           | minimum: 1.0; maximum: 512.0  |
| `model_id`   | string  | yes      | —                                            | minLength: 1; maxLength: 300  |
| `prompt`     | string  | no       | "Reply with a short description of the sky." | minLength: 1; maxLength: 1000 |

### Body_upload_model_reference_audio_api_models__model_id__reference_audio_post

| Field  | Type   | Required | Default | Constraints |
| ------ | ------ | -------- | ------- | ----------- |
| `file` | string | yes      | —       | —           |

### CommitSummary

| Field         | Type           | Required | Default | Constraints |
| ------------- | -------------- | -------- | ------- | ----------- |
| `commit_date` | string or null | no       | —       | —           |
| `message`     | string or null | no       | —       | —           |
| `sha`         | string or null | no       | —       | —           |

### DeleteGroupRequest

| Field                      | Type    | Required | Default | Constraints |
| -------------------------- | ------- | -------- | ------- | ----------- |
| `huggingface_id`           | string  | yes      | —       | —           |
| `retire_launch_references` | boolean | no       | false   | —           |

### HTTPValidationError

| Field    | Type                     | Required | Default | Constraints |
| -------- | ------------------------ | -------- | ------- | ----------- |
| `detail` | array of ValidationError | no       | —       | —           |

### HistoryChangeResponse

| Field    | Type   | Required | Default | Constraints |
| -------- | ------ | -------- | ------- | ----------- |
| `after`  | any    | yes      | —       | —           |
| `before` | any    | yes      | —       | —           |
| `path`   | string | yes      | —       | —           |

### HistoryDiffResponse

| Field              | Type                           | Required | Default | Constraints |
| ------------------ | ------------------------------ | -------- | ------- | ----------- |
| `changes`          | array of HistoryChangeResponse | yes      | —       | —           |
| `created_at`       | string                         | yes      | —       | —           |
| `current_revision` | string                         | yes      | —       | —           |
| `document`         | string                         | yes      | —       | —           |
| `id`               | string                         | yes      | —       | —           |
| `reason`           | string                         | yes      | —       | —           |

### HistoryEntryResponse

| Field        | Type   | Required | Default | Constraints |
| ------------ | ------ | -------- | ------- | ----------- |
| `created_at` | string | yes      | —       | —           |
| `document`   | string | yes      | —       | —           |
| `id`         | string | yes      | —       | —           |
| `reason`     | string | yes      | —       | —           |

### HistoryRestoreBody

| Field               | Type                                | Required | Default | Constraints                  |
| ------------------- | ----------------------------------- | -------- | ------- | ---------------------------- |
| `expected_revision` | string                              | yes      | —       | minLength: 1; maxLength: 200 |
| `item_id`           | string                              | yes      | —       | minLength: 1; maxLength: 240 |
| `kind`              | `preference` / `model` / `template` | yes      | —       | —                            |

### HistoryRestoreResponse

| Field      | Type   | Required | Default | Constraints |
| ---------- | ------ | -------- | ------- | ----------- |
| `document` | string | yes      | —       | —           |
| `item_id`  | string | yes      | —       | —           |
| `kind`     | string | yes      | —       | —           |
| `notice`   | string | yes      | —       | —           |
| `outcome`  | string | yes      | —       | —           |
| `revision` | string | yes      | —       | —           |

### LaunchApplyBody

| Field    | Type                         | Required | Default | Constraints |
| -------- | ---------------------------- | -------- | ------- | ----------- |
| `mode`   | `restart_now` / `next_start` | yes      | —       | —           |
| `models` | array of LaunchApplyEntry    | yes      | —       | —           |

### LaunchApplyEntry

| Field                         | Type           | Required | Default | Constraints |
| ----------------------------- | -------------- | -------- | ------- | ----------- |
| `catalog_id`                  | string         | yes      | —       | —           |
| `expected_desired_revision`   | string or null | no       | —       | —           |
| `expected_published_revision` | string or null | no       | —       | —           |
| `idempotency_key`             | string         | yes      | —       | —           |

### ProxySettingsBody

| Field                         | Type                             | Required | Default | Constraints                             |
| ----------------------------- | -------------------------------- | -------- | ------- | --------------------------------------- |
| `allow_anonymous`             | boolean                          | no       | true    | —                                       |
| `default_model`               | string or null                   | no       | —       | —                                       |
| `default_output_tokens`       | integer                          | no       | 4096    | minimum: 1.0; maximum: 1000000.0        |
| `drain_timeout_seconds`       | number                           | no       | 30      | maximum: 3600.0; exclusiveMinimum: 0.0  |
| `global_concurrency`          | integer or null                  | no       | —       | —                                       |
| `history_max_rows`            | integer                          | no       | 100000  | minimum: 100.0; maximum: 1000000.0      |
| `history_retention_days`      | integer                          | no       | 7       | minimum: 1.0; maximum: 365.0            |
| `idle_ttl_seconds`            | number or null                   | no       | —       | —                                       |
| `max_body_bytes`              | integer or null                  | no       | —       | —                                       |
| `max_connections`             | integer                          | no       | 200     | minimum: 1.0; maximum: 100000.0         |
| `max_keepalive_connections`   | integer                          | no       | 40      | minimum: 1.0; maximum: 100000.0         |
| `max_loaded_models`           | integer or null                  | no       | —       | —                                       |
| `max_queue_size`              | integer                          | no       | 100     | minimum: 1.0; maximum: 100000.0         |
| `per_model_concurrency`       | integer or null                  | no       | —       | —                                       |
| `pool_timeout_seconds`        | number                           | no       | 10      | maximum: 3600.0; exclusiveMinimum: 0.0  |
| `proxy_port`                  | integer                          | no       | 2000    | minimum: 1.0; maximum: 65535.0          |
| `queue_timeout_seconds`       | number                           | no       | 30      | maximum: 3600.0; exclusiveMinimum: 0.0  |
| `readiness_timeout_seconds`   | number                           | no       | 2700    | maximum: 86400.0; exclusiveMinimum: 0.0 |
| `request_timeout_seconds`     | number or null                   | no       | —       | —                                       |
| `routing_aliases`             | map of string or array of string | no       | —       | —                                       |
| `stream_idle_timeout_seconds` | number or null                   | no       | —       | —                                       |

### ReleaseSummary

| Field          | Type           | Required | Default | Constraints |
| -------------- | -------------- | -------- | ------- | ----------- |
| `html_url`     | string or null | no       | —       | —           |
| `published_at` | string or null | no       | —       | —           |
| `tag_name`     | string         | yes      | —       | —           |

### RuntimeApplyBody

| Field                         | Type                         | Required | Default | Constraints |
| ----------------------------- | ---------------------------- | -------- | ------- | ----------- |
| `confirm_operation_id`        | string or null               | no       | —       | —           |
| `confirm_state`               | string or null               | no       | —       | —           |
| `expected_desired_revision`   | string or null               | no       | —       | —           |
| `expected_published_revision` | string or null               | no       | —       | —           |
| `idempotency_key`             | string                       | yes      | —       | —           |
| `mode`                        | `restart_now` / `next_start` | yes      | —       | —           |

### SafetensorsBundleRequest

| Field            | Type            | Required | Default | Constraints |
| ---------------- | --------------- | -------- | ------- | ----------- |
| `files`          | array of object | yes      | —       | —           |
| `huggingface_id` | string          | yes      | —       | —           |
| `model_id`       | integer or null | no       | —       | —           |

### SaveConfigTemplateBody

| Field             | Type           | Required | Default | Constraints |
| ----------------- | -------------- | -------- | ------- | ----------- |
| `config`          | object or null | no       | —       | —           |
| `description`     | string         | no       | ""      | —           |
| `engines_scope`   | string         | no       | "all"   | —           |
| `include_routing` | boolean        | no       | false   | —           |
| `name`            | string         | yes      | —       | —           |
| `use_saved`       | boolean        | no       | false   | —           |

### SessionLogin

| Field   | Type   | Required | Default | Constraints |
| ------- | ------ | -------- | ------- | ----------- |
| `token` | string | yes      | —       | —           |

### TemplateCreateBody

| Field             | Type           | Required | Default | Constraints              |
| ----------------- | -------------- | -------- | ------- | ------------------------ |
| `config`          | object or null | no       | —       | —                        |
| `description`     | string         | no       | ""      | —                        |
| `engines_scope`   | string         | no       | "all"   | pattern: ^(all\|active)$ |
| `include_routing` | boolean        | no       | false   | —                        |
| `name`            | string         | yes      | —       | —                        |
| `source_model_id` | string or null | no       | —       | —                        |

### TemplateUpdateBody

| Field         | Type           | Required | Default | Constraints |
| ------------- | -------------- | -------- | ------- | ----------- |
| `description` | string or null | no       | —       | —           |
| `name`        | string or null | no       | —       | —           |

### UpdateCheckResponse

| Field            | Type                   | Required | Default | Constraints |
| ---------------- | ---------------------- | -------- | ------- | ----------- |
| `latest_commit`  | CommitSummary or null  | no       | —       | —           |
| `latest_release` | ReleaseSummary or null | no       | —       | —           |

### ValidationError

| Field   | Type                       | Required | Default | Constraints |
| ------- | -------------------------- | -------- | ------- | ----------- |
| `ctx`   | object                     | no       | —       | —           |
| `input` | any                        | no       | —       | —           |
| `loc`   | array of string or integer | yes      | —       | —           |
| `msg`   | string                     | yes      | —       | —           |
| `type`  | string                     | yes      | —       | —           |

## Source references

- [FastAPI application and router registration](../backend/main.py)
- [Native inference dataplane](../backend/proxy/studio/dataplane.py)
- [Native runtime manager and routing snapshots](../backend/proxy/studio/manager.py)
- [Model management routes](../backend/routes/models.py)
- [Proxy management routes](../backend/routes/proxy_admin.py)
- [Runtime plan and batch apply routes](../backend/routes/runtime.py)
- [Inference-key routes](../backend/routes/inference_keys.py)
- [Portable backup routes](../backend/routes/config_backup.py)
- [Configuration-history routes](../backend/routes/config_history.py)
- [Benchmark routes](../backend/routes/benchmarks.py)
