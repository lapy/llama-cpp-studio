# Native proxy operations

The Proxy page owns gateway configuration, keys, request history and diagnostics.
Models owns starting/stopping models and warming a running text/embedding model.
The native proxy owns the routing, policy, accounting, and process behavior described here.

## Model identity and live routing

Catalog IDs select managed records; stable runtime IDs identify processes and launch generations. A primary API alias and additional ordinary aliases expose names for the same real model. Request presets in `set_params_by_id` expose a suffix such as `assistant:no-thinking` and apply selected JSON overrides to that same process.

A preset can force `thinking_effort` or a nested path such as `chat_template_kwargs.enable_thinking`. A dotted path targets a nested field, a backslash escapes a literal dot, and a trailing `?` marks a default applied only when absent. Boolean false and other JSON types retain their values. The preset editor validates object shape and suffix uniqueness; the `model` parameter remains managed by Studio.

Proxy `routing_aliases` map names to a single target or an ordered target list. A multi-target alias selects targets round-robin. A target can be a real-model alias or request preset. `default_model` selects the target when the request omits a model name, including a routing alias or request preset.

Live publication swaps an immutable routing snapshot. Accepted work retains its target and existing streams continue. Routing changes do not reload unrelated engines. Ordinary alias/request-preset configuration is saved on the model; gateway routing groups are saved under Proxy → Settings & routing.

## Lifecycle and application

Start, Stop, Restart, and Restart & apply control the native listener/runtime. Restart rebuilds runtime resources and drains the proxy's owned engines. Serving limits and port changes require that lifecycle action. Default targets, routing aliases, and applicable access-policy publication apply live.

Per-model start/stop, preload/warm, reload, logs, and runtime-apply APIs provide narrower actions. Launch settings have desired, published, and running revisions. `restart_now` applies to the selected running model; `next_start` publishes for future process creation. Sequential batch apply uses `/api/runtime/apply-launch`, runs entries in order, and stops after failure. See [runtime architecture](launch-manifests-plan.md) for intent/revision and recovery details.

## Request observation

The Requests tab shows live requests and retained metadata, 100 per page. Filter
by key, resolved model, state, exact HTTP status, HTTP outcome group (4xx/5xx), or metadata search. Details include
request ID, client key ID/name, requested alias, resolved model, queue position,
queue wait, model acquisition, time to first response byte, time to first content
token (for supported SSE events), generation/stream duration, total duration,
bytes, reported tokens, reservations, and a failure reason. No prompts,
completion bodies, authorization values, or query-string credentials are stored.
Unknown usage is shown as unknown, not zero. Time to first token is unavailable
when an engine does not send recognizable token events.

Header, query-key and UI-cookie credentials resolve to the same client identity
before model/scope checks. Explicit header credentials take precedence; invalid
credentials do not fall through to cookies. UI cookies are HttpOnly, namespace
scoped, and grant GET/HEAD navigation only. Browser navigation remains subject
to model restrictions and request/concurrency policies. Token-budgeted UI keys
may fetch static UI assets and known metadata endpoints, not arbitrary GET
inference routes.

Client disconnects cancel queued, loading and upstream streaming work, even when
request tracking is disabled. A failure
after response headers cannot change the delivered HTTP status: details retain
both the delivered status and the final outcome (for example 200 and 504).

Request metadata is batched to SQLite approximately once per second and flushed
at shutdown or before history reads. Defaults are seven days and 100,000 rows;
both bounds are configurable. An abrupt process kill can lose the last batch of
telemetry. The in-memory history buffer is capped at 1,000 records; diagnostics
report metadata dropped during a prolonged storage outage. Quota reservations
are committed separately before dispatch and survive that kill. Incomplete reservations from dead processes retain their
estimated charge, release concurrency and become interrupted request records.
Live reservations owned by other local workers are not released.

Reset stats changes the persistent observation window and clears model metric
samples. It preserves request history, key usage and token budgets. Percentiles
cover retained requests within that window, including failures.

## Client policies and accounting

Keys support model restrictions, inference/model-listing scopes, expiry, requests
per rolling minute, separate input/output tokens per rolling minute, UTC daily
and monthly combined token budgets, concurrent requests, and output tokens per
request. Policies can be edited without rotating the secret. Revoking or rotating
a key takes effect on subsequent requests; accepted work keeps its policy
snapshot and remains charged to its original key ID.

Quota admission uses an atomic SQLite transaction. Outstanding reservations
count against every allowance, preventing concurrent requests from independently
spending the same remaining budget. Concurrent admissions can share one FULL-sync
commit; none can forward before that commit succeeds. Daily/monthly totals are
maintained transactionally rather than rescanning a month of requests. Token
minute windows continue to include active reservations during long model loads.
Keys without admission limits record usage in the batched history transaction,
without a synchronous zero-value reservation. Policy edits flush completed usage
before installing new limits; accepted work retains its admission-time policy snapshot.

For supported engines, the proxy reserves the output allowance, loads/acquires
the model, asks its input-token counter to count the actual request, and adjusts
the prompt reservation atomically before generation. If the adjustment exceeds
the budget, no generation is sent and the provisional token charge is released.
`token_accounting: auto` permits a visible UTF-8 byte estimate for simple text
when a counter is unavailable. `token_accounting: exact` refuses that fallback.
Counters currently adapt llama.cpp/ik_llama/unsloth native chat/Responses counting
and completion/embedding tokenization, plus vLLM/1cat_vllm tokenization and
Responses input counting when exposed by the installed version. An adapter does
not imply every engine version supports every route. The actual endpoint and
counting time are visible in request details. See the [token-counting adapters](../backend/proxy/studio/tokenization.py). Output reservation is the requested maximum
across all choices and batch prompts. If omitted, a key's output cap or the gateway default (4096)
is applied to the upstream request. Requests explicitly exceeding the key output
cap are rejected rather than silently changed. Engines must honor output limits.
Reported engine usage replaces the estimate; an unexpectedly larger report is
charged in full and can exhaust the budget for subsequent admissions.

For budgeted chat streams the proxy requests `stream_options.include_usage`.
Usage parsing is incremental and bounded, including final usage after large
responses. It observes an independently decoded copy of gzip/deflate responses
without changing forwarded bytes; binary/audio payloads are not scanned as JSON.
A bounded decompression budget prevents usage observation from expanding
unbounded data. Partial/nonterminal SSE usage never replaces a full reservation. Missing or incomplete usage retains the full reservation after a
partial stream, disconnect, timeout or successful response. Embeddings generate
zero output tokens. Requests rejected before upstream dispatch release their
token reservation; their accepted request-rate slot still counts. Budgets are
attributed to request admission time, including requests spanning UTC boundaries.

Multimodal and stateful conversation requests require a successful engine count
of their actual context. They never fall back to byte estimates. Unsupported
engine versions or rejected count requests fail closed. CPU fixtures verify the
contracts; compatibility with individual GPU engine builds still needs their
hardware smoke tests.
Unrestricted keys retain normal passthrough behavior. Internal Studio traffic and
anonymous traffic have separate identities and do not spend client-key budgets.
Daily/monthly reset times, usage, remaining allowances, active concurrency and
estimated request counts appear beside each key. Last-used tracking lives in the
accounting database, avoiding YAML writes on the inference path.

The database is `config/proxy-accounting.sqlite3` under the Studio data directory.
Keep its WAL alongside it when backing up a running instance; use SQLite's backup
API or stop Studio before copying. Deleting telemetry must not delete this
ledger, because doing so would reset budgets. It is local operational state,
separate from portable configuration exports. Use one local data volume; this
SQLite implementation is not a distributed quota service.

## Overload and timeouts

The queue limit bounds requests waiting for global/model slots. The effective
global concurrency is also bounded by the connection pool size. A second bound
on received requests prevents unlimited concurrent body buffering. Saturation
and queue expiration return 429 with Retry-After. Key quota denials return a
specific policy code and Retry-After. Responses include a generated X-Request-ID;
key requests include remaining-request/day/month headers where configured.

Queue wait, total request deadline (including loading and streaming), stream
inactivity, and connection-pool wait have separate settings. The total deadline
and stream inactivity timeout can be disabled. Connection-pool wait defaults to
10 seconds. Connection reuse defaults to 200 total and 40 idle connections.
Serving limits apply after Restart & apply; key policy edits apply immediately.
Default-model and alias changes publish a versioned immutable routing snapshot
without restarting engines or interrupting admitted streams. A route edit does
not clear a pending restart for unrelated serving settings.

## Supervision

The startup/stop coordination uses process-owned
EnsureReady/run-loop model. Concurrent callers share one startup task. One
cancelled waiter cannot terminate a startup still needed by another; when the
last waiter leaves, Studio cleans up an otherwise unused cold start. Acquires
during a stop wait until draining/teardown finishes. Engine stdout/stderr use
bounded asynchronous pipe reads, never blocking the default thread pool.
Unexpected exits are monitored promptly, shown as failures rather than ready
engines, and retain logs for inspection. A subsequent start launches a fresh
process. Shutdown signals the process group, escalates after the grace period,
and bounds waiting on inherited log pipes. The native registry owns the full process lifecycle.

Upstream forwarding uses aiohttp through the HTTPX transport interface, with
bounded active connections, raw streaming, backpressure, idle read timeouts and
cancellation. Redirects, automatic decompression and shared cookie storage are
disabled. A small connector release hook enforces the separate idle connection
cap; real socket tests cover that hook against the pinned aiohttp dependency.
The request supervisor owns the total deadline so healthy streams do not expire
merely because an idle timeout duration has elapsed.

## Benchmarking

Run a real HTTP synthetic fixture with durable budgets enabled:

```sh
python -m scripts.proxy_benchmark --fixture --accounting quota --requests 256 --concurrency 32
python -m scripts.proxy_benchmark --fixture --accounting tracking --requests 256 --concurrency 32
python -m scripts.proxy_benchmark --fixture --accounting off --requests 256 --concurrency 32
```

Compare a real engine and proxy using the same model/request:

```sh
STUDIO_BENCHMARK_KEY=... python -m scripts.proxy_benchmark \
  --direct http://127.0.0.1:3001 --proxy http://127.0.0.1:2000 \
  --model example --requests 100 --concurrency 8 --output benchmark.json
```

Results include p50/p95 latency, first-byte/first-token latency, requests/second,
byte and reported-token throughput, errors and benchmark-process peak RSS. The
load generator uses a warmed aiohttp connection pool; a throwaway warmup client
would not warm the measured connections. The fixture runs engine, proxy and load
generator in one process, so its RSS is combined and its timings
are not production model-performance measurements. Run several concurrency
levels on target hardware; compare errors as well as successful throughput.

## Native conformance checks

```sh
python -m scripts.proxy_conformance --report conformance.json
```

CI starts real CPU-only fixture engines and compares direct upstream responses
with native proxy responses for JSON, fragmented UTF-8/tool-call SSE, errors,
compressed bytes, binary slow consumers, hop-by-hop headers, and repeated
multipart fields. It also checks upstream cancellation after a client disconnect.
Reports are fixture
contract checks; GPU engine performance still needs validation on target hardware.

## Workspace diagnostics and backup

System owns service diagnostics, portable backup/restore, and configuration
change history. The footer remains a compact live-status strip. Backup and restore use `/system#config-backup`; history uses `/system#history` and hardware uses `/system#hardware`. Proxy HTTP 4xx and 5xx
counters are separate; they cover all HTTP traffic, including missing routes,
authentication rejections and cancellations. They are not engine crash counters.
Use a model's outcome link to inspect the underlying request records.

## Serving-setting reference

The [feature guide's serving settings table](features.md#serving-settings) lists persisted defaults and effective runtime values. `PUT /api/proxy/settings` uses the complete settings schema; omitted values receive schema defaults. Read current settings before constructing a full replacement request. The [API reference](api-reference.md#configure-live-routing-and-serving-limits) supplies a request example and schema.

## Source and diagnostic interfaces

- [Native proxy management endpoints](../backend/routes/proxy_admin.py) expose listener/model lifecycle, logs, statistics reset, request queries, and saved settings.
- [Inference-key endpoints](../backend/routes/inference_keys.py) expose key creation, rotation, revocation, policy changes, and usage.
- [Accounting implementation](../backend/proxy/studio/accounting.py) defines policy validation, durable admission, settlement, retention, and usage queries.
- [Native dataplane](../backend/proxy/studio/dataplane.py) owns transport supervision, authorization, UI transport, filters, and response observations.
- [Runtime manager](../backend/proxy/studio/manager.py) owns compiled catalog resolution and live route snapshots.

Use Proxy request details for a client's HTTP/policy outcome, model logs for engine diagnostics, System diagnostics for observation/persistence health, and operation progress for installation/application work. These signals describe different failure domains.
