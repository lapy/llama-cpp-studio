# audio.cpp integration contract shim — plan (2026-10-15)

Context: upstream issue [0xShug0/audio.cpp#852](https://github.com/0xShug0/audio.cpp/issues/852)
proposes a versioned, machine-readable integration contract owned by each
front-end (server routes, encodings, response/stream shapes, per-model task
discovery). Upstream asked us to hold implementation ("a few more refactors are
underway… I'll ping you"). This plan fills the same five gaps with a
**temporary Studio-side shim** that is shaped so the upstream contract can
later replace it without a second redesign.

The five gaps from the issue, restated as questions Studio must answer from data:

1. What server/CLI configuration is accepted to start a model?
2. Which installed/loaded models expose which tasks, modes, capabilities, options?
3. Which server operation serves each task?
4. How is each request encoded (JSON / multipart / streaming / audio artifacts)?
5. What response, error, and stream-event shapes must a client handle?

## What Studio has today (measured, verified against v0.9.1 prebuilt and the local `linux-cpu-local` build)

Studio already derives most of gaps 1–2 from engine-owned data and does it well:

- Version scan (`scan_audio_cpp_version`): `server --help`, `cli --help`,
  `--list-loaders --json` → server/CLI option sections, families, family→tasks,
  family→modes, `family_endpoints`, `family_policies`, contract fingerprint and
  grade. v0.9.1 CPU prebuilt: grade `full`, 114 families, real JSON.
- Model profile scan (`scan_audio_cpp_model_profile`): model-aware
  `--inspect` + `--help` + `model_specs/<family>.json` → per-model tasks, modes,
  request/session/load options, packaged voices, `preferred_api_endpoint`.

Gaps 3–5 are hardcoded in Studio and duplicated at three layers:

- **Route tables are triplicated.** Canonical constants + decision function in
  `backend/engines/audio_cpp/discovery.py` (`NATIVE_AUDIO_ENDPOINTS` :22-31,
  `resolve_api_endpoint` :610-697 — including `help_force` option-name
  heuristics and a comment baking in llama-swap v250/v260 route behavior);
  per-route handlers in `backend/routes/audio_openai_proxy.py`; `/v1/`-prefix
  validation + `/upstream/{id}` rewrite in `proxy_routing.py`; and a frontend
  mirror re-declaring the task→route table
  (`frontend/src/composables/useAudioInferenceClient.js:106-141`,
  `connectTarget.js:58-62`).
- Server-level endpoint list baked into `capabilities["endpoints"]`
  (`backend/engines/scan/scanner.py` ~867): already **stale** — missing
  `/v1/batches/transcriptions` (which Studio itself proxies) and listing
  `/v1/audio/voices` (which nothing calls) — while `--list-loaders --json`
  (v0.9.1+) reports accurate per-family `api_endpoints` that the parser
  already captures as `family_endpoints`.
- **Request-encoding folklore**: WAV-only upload rule + ffmpeg pipe-WAV header
  workaround (`backend/audio/format_convert.py`), speech response re-encode
  table (`SPEECH_PASSTHROUGH_FORMATS`, codec/muxer pairs),
  `_SPEECH_SWAP_PARAM_KEYS` / `prompt → options.text` / `voice_id → voice`
  renames (`backend/audio/voice_presets.py`), int-vs-float coercion lists
  (`request_defaults.py`). Errors are ad-hoc per route (`_openai_speech_error`).
- **Gap 5 is effectively unimplemented**: the proxy streams bytes verbatim
  (good), but frontend SSE decoding for batch transcriptions is heuristic
  (`data:`/`[DONE]` sniffing), and nothing consumes server event/error shapes.
- **Worst single gap-filler**: `backend/engines/audio_cpp/option_discovery.py`
  regex-parses the audio.cpp **C++ checkout** (`loader.cpp` `CliOptionInfo`
  triples, `session.cpp` allowlists with hardcoded bare keys, docs markdown
  tables with a 1200-char family-attribution window). Silent breakage on any
  upstream source refactor.

So the shim is not new heuristics — it is **one owned, versioned descriptor**
that (a) centralizes the remaining hardcoded knowledge as data with provenance,
and (b) records where each fact came from (upstream probe vs. Studio hardcode).
The Studio-hardcoded entries are the temporary part.

## Shim design

New module: `backend/engines/audio_cpp/integration/` (single import surface).

`contract_registry.py` — a static, versioned operation table, `audio-cpp.studio-contract/v0`:

```python
# Each operation: id, native route (server side), Studio route, method,
# request encodings, response kinds, error shape, and per-field mapping notes.
OPERATIONS = [
  {
    "id": "speech",
    "native_path": "/v1/audio/speech",
    "studio_path": "/v1/audio/speech",
    "encodings": ["json"],
    "responses": ["audio", "json", "sse-events"],
    "errors": "openai-envelope",
    "serves_tasks": ["tts", "clon"],
    "provenance": "studio",          # studio | loaders-json | inspect | spec
  },
  # transcriptions, transcriptions/details, alignments, tasks/run,
  # voices, models, health, batches/transcriptions (studio-only composite)
]
```

Composition pipeline (per active engine version + per model), all cached under
the existing `contract_fingerprint` / model-profile fingerprint scheme:

1. Start from `OPERATIONS` (provenance `studio`).
2. Overlay `capabilities.family_endpoints`, `family_policies`, `family_modes`
   from `--list-loaders --json` (provenance `loaders-json`). v0.9.1 already
   emits `api_endpoints` + `instructions_policy` per family — this alone
   replaces several hardcoded assumptions.
3. Overlay per-model `--inspect` task/mode/`preferred_api_endpoint` and
   `model_specs/<family>.json` option types/defaults (provenance `inspect`/`spec`).
4. Conflict policy: observed beats studio; mismatch raises a drift warning in the
   activation result (reuse `capability_delta`/warnings channel).

Exposure and integration seams (ranked by cleanliness, per code survey):

1. `backend/routes/audio_openai_proxy.py` is already the single HTTP chokepoint
   for all Studio↔audio.cpp inference traffic; per-route handlers become
   registry lookups while `_passthrough`/`_stream_response` stay untouched.
2. `resolve_api_endpoint` / `resolve_defaults_key_for_endpoint`
   (`discovery.py:610-703`) already funnel every consumer
   (`build_request_policy`, `task_profiles`, scanner, `routes/models.py`);
   replace their bodies with contract lookups, keep current logic as fallback.
3. `resolve_audio_upstream_target` (`proxy_routing.py`) — its `/v1/`-prefix
   check becomes a "known routes" validation against the registry.
4. The version-scan `capabilities` dict (`scanner.py:857-894`) already carries
   `family_endpoints`/`family_policies`/`family_modes` and ships to the UI via
   `routes/models.py` — the composed contract blob rides the same payload; the
   stale hardcoded `endpoints` list retires in favor of the composed set.
5. `build_request_policy` (`request_policy.py:63-181`) already prefers
   engine-advertised `preferred_api_endpoint`/`instructions_policy` over
   heuristics — natural home for contract-derived per-task encoding metadata.
6. Gate shim-vs-legacy through the existing `backend/feature_flags.py` idiom
   (same pattern as `audio_cpp_heuristic_discovery`).

Note on the existing "contract" concept: `contract_fingerprint` is only a
sha256 of help-text + loader-list + `model_specs/*.json`, and
`contracts.py` models spec files from the checkout, not the server. Useful as
a drift signal (and its delta/review plumbing is reused for warnings above),
but it must not be confused for the server contract its name suggests — and
note `help_parsers.py:1784` currently *discards* help sections literally
headed "endpoints"/"tasks" that upstream does print.

Served surface:

- `GET /api/audio-cpp/contract` — composed descriptor for the active version
  (version-level; per-model composed lazily with model profile payloads).
- Frontend fetches the task→operation table from that endpoint; the current
  `useAudioInferenceClient.js` fallback table stays only as an offline default
  (delete its duplicate constants and `taskKindFromConfig` table in M2).

Upstream-readiness (the point of the design):

- `fetch_upstream_contract()` adapter stub behind
  `AUDIO_CPP_UPSTREAM_CONTRACT=1`. When #852 lands, its document maps onto the
  same descriptor with provenance `upstream`; every `studio`-provenance entry
  with upstream coverage is deleted. The descriptor is the seam, so the shim is
  throwaway per-entry, not per-layer.

## Milestones — delivered status (2026-10-15)

- ✅ **M1 — Descriptor + registry as data.** `backend/engines/audio_cpp/integration/`
  ships `contract_registry.py` (operations, deterministic task table, coarse
  engine-surface tokens, `DEFAULTS_KEYS`, `STUDIO_OWNED_FLAGS`, per-operation
  `request_defaults_key`/`body_shape`), `composer.py`, and the
  `upstream_adapter.py` stub. Tests pin the registry/composer against the
  captured v0.9.1 loaders-JSON fixture
  (`backend/tests/audio_profile_fixtures.py::V091_LIST_LOADERS_JSON`,
  `backend/tests/test_audio_cpp_integration_contract.py`).
- ✅ **M2 — Single-source routing.** Proxy handlers (`audio_openai_proxy.py`)
  and `proxy_routing.resolve_audio_upstream_target` resolve via the registry
  (`_native_path(op_id)`, `is_native_api_path`, `TASKS_RUN_ALIAS`).
  `GET /api/audio-cpp/contract` serves the composed descriptor. The frontend
  consumes it through `frontend/src/composables/useAudioContract.js`; the local
  task→route table, duplicate `AUDIO_CPP_TASKS_PATH` /
  `LLAMA_SWAP_AUDIO_TASKS_PATH` constants, the `audioApiEndpoint` heuristics,
  and `isBogusAudioConfigKey` are deleted (offline floor = generic route only).
- ✅ **M3 — Composition + drift warnings.** `compose_version_contract` /
  `compose_model_contract` overlay registry ← loaders JSON ← inspect ← spec;
  observed beats studio; conflicts land in `drift_warnings`, ride the activation
  payload and the param-registry payload, and render in Model Config. The stale
  hardcoded `capabilities["endpoints"]` list, text loader-list fallback,
  `option_discovery.py` C++/docs scraping, the
  `AUDIO_CPP_HEURISTIC_DISCOVERY` / `AUDIO_CPP_SOURCE_OPTION_DISCOVERY` flags,
  duplicated reserved-flag lists, heuristic family scoring, and the pinned
  community-voice table are all deleted. Activation now hard-requires
  `--list-loaders --json` (`assert_activation_supported` refuses older builds
  with an update prompt, mirrored by the Model Config update banner).
- ✅ **M4 — Activation speed fixes** (see below): probe-once inspect,
  fingerprint-gated rescans, bounded concurrency, persistent builtin-utility
  cache, async activation returning `{operation_id}` with SSE stage progress,
  and per-subprocess stage timing logs.
- ⏳ **M5 — Upstream adapter** awaits audio.cpp#852. `fetch_upstream_contract()`
  returns nothing until it lands; provenance then flips `studio → upstream` and
  covered entries get deleted individually.

Retirement order as upstream coverage lands: request-encoding folklore
(`voice_presets.py`, `format_convert.py`) → GGUF sidecar guessing → finally
replacing the help-text `contract_fingerprint` semantics. Everything earlier in
the original order (C++/docs scraping, route-table triplication, family/package
heuristics) is already retired.

## Activation slowness — findings (2026-10-15)

Measured on this machine (WSL2, `linux-cpu-local` build commit `a76ec04`, and
fresh v0.9.1 prebuilts), warm cache, 2 registered audio models:

| stage | time |
| --- | --- |
| `scan_engine_version` (3 probes + catalog probe) | 0.26–0.34 s |
| `rescan_model_profiles` (force, 2 models) | 1.2 s |
| full simulated activation | 2.8 s |
| `audiocpp_server/cli --help`, `--list-loaders --json` | 0.02–0.07 s each |
| `model_manager_v2.py list --json` (60 s timeout, offline-capable) | 0.1 s |

So the activation **code path is not slow when binaries exec fast**. The
"forever" comes from amplification factors, each multiplied by ~11–17 subprocess
executions of 44–52 MB binaries during one activation:

1. **Synchronous HTTP activation.** `POST /api/llama-versions/versions/activate`
   holds the request open through scan + all model rescan + proxy start +
   `_wait_for_proxy_ready` (30 s ceiling). The UI just spins; perceived latency
   equals the slowest stage.
2. **Forced full rescan.** `rescan_model_profiles` passes `force=True` for every
   audio model on every activation, even re-activating the same version with an
   unchanged contract. The fingerprint cache exists and is bypassed.
3. **Double inspect probe.** `--inspect --json` emits `key=value` text in both
   v0.9.1 and the local build (JSON is only implemented for `--list-loaders`),
   so `try_parse_json_payload` fails and every model inspect probe runs twice.
   Cost is ~0 extra when warm, doubles the exec count when exec is expensive.
4. **Exec cost in production, not locally.** `HELP_TIMEOUT=90` per probe. Two
   known environments where a 44–52 MB binary exec costs 5–30 s instead of 30 ms:
   - Docker Desktop on Windows with `./data` bind-mounted from a Windows drive:
     binaries load over 9p and Windows Defender re-scans every exec. ~15 probes
     → minutes. This matches the reported symptom under `docker-compose.cuda.yml`.
   - CUDA builds on hosts where GPU init misbehaves (WSL driver mismatch): a
     hanging `audiocpp_server --help` eats 90 s per probe; 3 version probes +
     2×N model probes → ~10–25 min worst case. (Verified v0.9.1 CUDA colab
     binary *fails fast* with `libcublas` missing — that path is fast; the risk
     is a hung driver, not a missing lib.)
5. **Proxy restart loop hides in the logs.** If llama-swap cannot come up,
   `_monitor_process` retries forever every ~7 s (2 s poll + 5 s sleep, 30 s
   ready wait each round); activation itself returns after 30 s but the system
   stays "not ready" indefinitely.

### Fixes (M4)

- Use fingerprint cache: skip version re-scan when binary identity
  (`source_commit` + mtime/size of both binaries) is unchanged; skip per-model
  force rescan unless `contract_changed`, the model's family/task intersects the
  capability delta, or its profile fingerprint/cache is missing/stale.
- Probe once: when inspect output begins with `key=value` lines, parse it
  directly and skip the second fallback probe (fix in
  `_run_audio_cpp_inspect` / `_run_audio_cpp_loaders`).
- Stage timings: log duration per subprocess in the param-scan session so the
  engine dialog shows which command ate the time (session already logs commands).
- Background activation: keep the activation lock, return `{operation_id}` in
  ~1 s, stream scan/profile/proxy stages through the existing progress SSE;
  frontend shows stage text instead of an open spinner.
- Ops note (no code): run with `data` on a Docker volume or inside the WSL
  filesystem, not a Windows bind mount, and add the repo `data/` path to
  Defender exclusions; on CUDA hosts, verify `audiocpp_server --help` exits
  promptly under the compose environment before blaming Studio.

## Non-goals

- No model-spec schema changes; no upstream schema-2 proposals (issue says don't).
- No new heuristics behind `AUDIO_CPP_HEURISTIC_DISCOVERY`; unknown stays unknown.
- No parallel metadata: composed descriptor is cached, fingerprint-keyed, and
  derived from existing scans, not a second source of truth.
