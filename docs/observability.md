# Observability, audit, benchmark, and GPU validation

Studio exposes three operator-facing observation surfaces plus one manual
hardware runner. They are additive: the existing `/api/status`, proxy request
history, and benchmark fields keep working unchanged.

- `GET /api/metrics` — Prometheus text for the management process.
- `backend/observability/audit.py` — bounded JSONL audit of management mutations.
- `POST /api/benchmarks/run` with `warmup`/`repeats` — repeated benchmark runs
  with per-run and aggregate timings.
- `scripts/gpu-smoke-matrix.py` — a standard-library runner for a manual smoke
  matrix against a started inference endpoint.

These surfaces describe **Studio's** operation. Engine-owned Prometheus values
(SGLang's `--enable-metrics`, vLLM's own statistics, llama.cpp's `--metrics`) stay
engine-owned and are not remapped into `/api/metrics`; see
[`engine-observability-flags.md`](engine-observability-flags.md).

## `GET /api/metrics`

The route is part of the management API, so it uses the same authorization as
every other `/api` route: loopback trust in `local` access mode, and a
management token or session in `remote` mode. Inference keys do not grant
access. The response is `text/plain; version=0.0.4; charset=utf-8` with
`Cache-Control: no-store`.

```bash
curl -fsS http://127.0.0.1:8081/api/metrics
```

A minimal Prometheus scrape job:

```yaml
scrape_configs:
  - job_name: studio
    scheme: http
    static_configs:
      - targets: ["127.0.0.1:8081"]
    authorization:
      type: Bearer
      credentials_file: /etc/studio/management.token
```

### Metric families

| Family | Type | Source | Meaning |
| --- | --- | --- | --- |
| `studio_app_info{version}` | gauge | `backend.version.APP_VERSION` | Always `1`; identifies the build. |
| `studio_process_start_time_seconds` | gauge | process import time | Unix start time of this process. |
| `studio_uptime_seconds` | gauge | process import time | Seconds since this process started. |
| `studio_proxy_requests_total` | counter | proxy accounting summary | Requests in the proxy statistics window. |
| `studio_proxy_request_errors_total` | counter | proxy accounting summary | Failed requests (4xx + 5xx). |
| `studio_proxy_client_errors_total` | counter | proxy accounting summary | 4xx requests. |
| `studio_proxy_server_errors_total` | counter | proxy accounting summary | 5xx requests. |
| `studio_proxy_input_tokens_total` / `..._output_tokens_total` | counter | proxy accounting summary | Token totals that the proxy could attribute. |
| `studio_proxy_requests_active` | gauge | proxy accounting summary | In-flight requests now. |
| `studio_proxy_requests_queued` | gauge | proxy accounting summary | Requests waiting for admission now. |
| `studio_proxy_requests_loading` | gauge | proxy accounting summary | Requests waiting for a model load now. |
| `studio_proxy_requests_streaming` | gauge | proxy accounting summary | Responses streaming now. |
| `studio_proxy_latency_p50_ms` / `p95` | gauge | proxy accounting summary | Completed-request latency in the stats window. |
| `studio_proxy_ttft_p50_ms` / `p95` | gauge | proxy accounting summary | Time-to-first-token in the stats window. |
| `studio_proxy_usage_unknown_total` | counter | proxy accounting summary | Requests whose usage was unknown or only reserved in the stats window. |
| `studio_proxy_stats_since_timestamp_seconds` | gauge | proxy accounting summary | Start of the statistics window (reset changes it). |
| `studio_models_loaded` | gauge | proxy registry snapshot | Generations ready to serve now. |
| `studio_models_starting` | gauge | proxy registry snapshot | Generations loading now. |
| `studio_models_failed` | gauge | proxy registry snapshot | Failed generations still tracked. |
| `studio_models_managed` | gauge | proxy registry snapshot | Tracked generations in every state. |
| `studio_operations_started_total` … `interrupted_total` | counter | `ops_metrics` | Durable operation lifecycle counts for this process. |
| `studio_event_loop_delay_seconds_max` | gauge | `ops_metrics` | Worst scheduling delay observed. |
| `studio_event_loop_stalls_total` | counter | `ops_metrics` | Felt stalls (over the warning threshold). |
| `studio_sse_queue_depth_max` | gauge | `ops_metrics` | Worst progress-subscription queue depth. |
| `studio_cancellation_latency_seconds_max` | gauge | `ops_metrics` | Slowest cancellation handoff. |
| `studio_config_revision` | gauge | `ops_metrics` | Configuration revision loaded now. |
| `studio_persistence_pending_writes` | gauge | `store_io.persistence_status` | Accepted but not yet durable config writes. |
| `studio_persistence_max_pending_writes` | gauge | `store_io.persistence_status` | Gate capacity. |
| `studio_persistence_saturated` | gauge | `store_io.persistence_status` | `1` while the write gate is full. |
| `studio_persistence_recent_failures` | gauge | `store_io.persistence_status` | Bounded in-memory failure history size. |
| `studio_audit_records_written_total` | counter | audit writer | Audit lines written by this process. |
| `studio_audit_records_dropped_total` | counter | audit writer | Records dropped because the queue was full. |
| `studio_audit_write_failures_total` | counter | audit writer | Records that could not be written. |
| `studio_disk_total_bytes` / `free` / `used` | gauge | `psutil.disk_usage(data root)` | Data-root filesystem capacity. |
| `studio_gpu_memory_total_bytes{index,name}` | gauge | cached GPU observation | Per-device total memory. |
| `studio_gpu_memory_free_bytes{index,name}` | gauge | cached GPU observation | Per-device free memory. |

### Semantics

- **Unknown is omitted, never zero.** If the proxy accounting database cannot
  be read, GPU information has not been observed, or the disk stat fails, the
  affected families are absent from that scrape. Zero is only reported when a
  service says the value is zero (for example `studio_models_loaded 0` from a
  running registry snapshot).
- **No blocking probe on the event loop.** The accounting summary and the disk
  stat run on the shared blocking pool. GPU values come from the existing GPU
  cache: if it is cold, a background refresh starts and the scrape proceeds
  without GPU families instead of waiting on `nvidia-smi`/NVML.
- **No work is started by a scrape.** The endpoint does not start the proxy,
  load a model, scan a catalog, flush accounting rows, or write configuration.
  A cold GPU cache may schedule the existing background refresh, but the
  scrape neither waits for it nor reports its result.
- **Counters are process/window scoped.** Operation and audit counters reset
  when the process restarts. Proxy counters reset when the operator resets the
  proxy statistics window (`POST /api/proxy/stats/reset`), which also moves
  `studio_proxy_stats_since_timestamp_seconds`.
- **GPU observations can be stale.** The GPU cache is refreshed opportunistically
  by GPU reads (`GET /api/gpu-info`, runtime estimates) and by a scrape that
  finds it cold. Treat the sample as "last observation", not a live probe.

## Management-mutation audit

Every `/api/*` request whose method is not `GET`, `HEAD`, or `OPTIONS` appends
one JSON object per line to:

```
<STUDIO_DATA_DIR>/logs/audit.jsonl
```

The middleware is pure ASGI and is registered outside management access
control, so a rejected mutation is recorded with its real status and the actor
kind that was presented. It never reads the request body or copies headers.

### Record shape

```json
{
  "timestamp": "2026-11-22T20:31:07.412Z",
  "request_id": "5f0f4b0e8f5a4d6c9a0b1c2d3e4f5061",
  "method": "POST",
  "path": "/api/models/model-1/activate",
  "status": 200,
  "duration_ms": 84.512,
  "actor": "local"
}
```

| Field | Meaning |
| --- | --- |
| `timestamp` | UTC ISO-8601 with milliseconds. |
| `request_id` | Random per-request correlation id generated by Studio. |
| `method` | Upper-case HTTP method. |
| `path` | URL path only. The query string is never recorded. |
| `status` | Response status; `500` when the application raised before starting a response. |
| `duration_ms` | Wall time from middleware entry to response completion. |
| `actor` | `local` (configured local trust), `token` (valid management bearer token), `session` (valid session cookie), or `anonymous` (no valid management credential). |

### What is never recorded

- Request bodies, form data, and file uploads.
- Request or response headers, cookies, and authorization values.
- Query parameters (`?token=…`, `?key=…`).
- Configuration document contents, prompts, model output, or secrets.

A path that embeds a secret-looking assignment (`token=`, `api-key:`,
`password=`, `bearer …`) is rewritten to `redacted`, and any path longer than
512 characters is truncated. The record is rebuilt from an allow-list of fields,
so an added caller field cannot leak into the file.

### Boundedness and rotation

- The middleware only enqueues a serialized line. File I/O happens on one
  daemon writer thread, never on the event loop.
- The queue holds at most 2048 pending records. When it is full, new records are
  dropped and counted (`studio_audit_records_dropped_total`); the request is
  never blocked or failed.
- The active file rotates before a write would exceed **5 MiB**. Five rotated
  files are retained (`audit.jsonl.1` … `audit.jsonl.5`) for a bound of about
  30 MiB. A single line is never split.
- Files are created mode `0600` under a `0700` directory.
- Auditing is best effort. Write failures increment
  `studio_audit_write_failures_total` and are otherwise ignored: observability
  must not fail the mutation it observes.

`GET`, `HEAD`, and `OPTIONS` requests, non-`/api` requests, and the management
API's own read endpoints are not audited.

## Repeated benchmarks

`POST /api/benchmarks/run` accepts two additive request fields:

| Field | Range | Default | Meaning |
| --- | --- | --- | --- |
| `warmup` | 0–3 | `0` | Discarded streaming runs before measurement. They consume the same prompt and `max_tokens` and settle JIT, caches, and scheduling. |
| `repeats` | 1–5 | `1` | Measured streaming runs, executed sequentially. |

```bash
curl -fsS -X POST http://127.0.0.1:8081/api/benchmarks/run \
  -H 'Content-Type: application/json' \
  -d '{"model_id": "org/model", "prompt": "Name a color.", "max_tokens": 16, "warmup": 1, "repeats": 3}'
```

Existing response fields keep their meaning: `time_to_first_token_ms`,
`total_seconds`, `completion_tokens`, `tokens_per_second`,
`peak_observed_gpu_memory_bytes`, `output_preview`, configuration identity, and
the observation fields all describe **the first measured run** (the only run for
the historical default). This keeps old clients and stored rows readable.

Additive fields:

- `warmup` and `repeats` echo the bounded values actually used.
- `runs` — one entry per measured run:
  `{index, time_to_first_token_ms, total_seconds, completion_tokens, tokens_per_second}`.
  Stored rows also retain each run's input-token, delivery-interval, decode
  quality, and engine-metric detail.
- `aggregate` — `{time_to_first_token_ms, total_seconds, tokens_per_second}`,
  each `{min, p50, p95, max}` computed with nearest-rank percentiles over the
  known samples. Unknown samples are excluded; with one run all four values
  equal that run.

Rows stored before this change read back with `warmup: 0`, `repeats: 1`,
`runs: []`, and `aggregate: null`; they were single-run benchmarks with no
per-run detail to reconstruct.

### What a benchmark does and does not prove

- GPU memory is **sampled** while the whole sequence runs (one probe every
  0.5 s, each probe bounded by a 2 s timeout) and reported as the peak of the
  summed *used* bytes across visible devices. It is an observation, not a
  per-run allocation and not a calibrated peak.
- Latency and throughput are **not normalized** across GPU models, drivers,
  quantization, context size, batching, or engine flags. Compare runs only when
  `config_revision`, `config_fingerprint`, engine build, prompt, and
  `max_tokens` match.
- `p50`/`p95` over at most five runs are indicative. They are not a service
  level objective and do not establish statistical stability.
- A benchmark proves that a running model completed these requests on this
  host. It does not prove correctness, maximum batch throughput, multi-GPU
  scaling, or behavior under concurrent load.

## GPU smoke matrix runner

`scripts/gpu-smoke-matrix.py` is a manual, standard-library-only runner for a
host the operator has already started. It makes no CI hardware assumption and
does not install, build, or start anything.

```bash
# 1. Validate the matrix without touching the network.
python scripts/gpu-smoke-matrix.py --matrix gpu-matrix.json --dry-run

# 2. Run it and keep the evidence.
python scripts/gpu-smoke-matrix.py --matrix gpu-matrix.json --output gpu-smoke.json
```

Matrix:

```json
{
  "base_url": "http://127.0.0.1:2000",
  "key_env": "STUDIO_INFERENCE_KEY",
  "prompt": "Reply with a short description of the sky.",
  "max_tokens": 32,
  "timeout_seconds": 120,
  "models": [
    "org/model-name",
    { "id": "org/other-model", "aliases": ["friendly-alias"], "prompt": "Say hello." }
  ]
}
```

- `GET /v1/models` runs once and its advertised ids are recorded.
- One non-streaming chat completion runs for each model id and each declared
  alias. Each result records the matrix model id, the requested name, whether
  it was advertised, status, latency, and the reported `usage` when present.
- The API key is read from the environment variable named by `key_env`; it is
  never written to the report or printed. `--dry-run` does not require the
  secret because it makes no call.
- Exit code `0` means the models endpoint and every chat call succeeded, `1`
  means at least one call failed, `2` means the matrix is invalid.
- The report is evidence of **reachability and basic generation** on that host.
  It does not establish throughput, driver compatibility, or correctness for
  other models.

## Verification

```bash
python -m pytest backend/tests/test_observability_metrics.py \
  backend/tests/test_audit.py \
  backend/tests/test_model_benchmark.py \
  backend/tests/test_gpu_smoke_matrix.py -q
```

The tests use `TestClient`, temporary data roots, and a local fixture server.
They never read a live Studio data directory, a real GPU, or the public network;
GPU and accounting values are injected or omitted in tests, so the tests prove
the exposition and omission rules rather than hardware behavior.
