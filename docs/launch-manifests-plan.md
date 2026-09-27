# Launch manifests implementation plan

Prepared 2026-09-27. Planning only; application files are being changed by another agent. Re-read integration points after that work lands and implement against the resulting contracts rather than the line numbers observed during this investigation.

**Objective and release criterion**

Changing model A's startup settings must not unload models B and C. With multiple models running, applying A must leave llama-swap's configuration file and process unchanged, preserve B/C process identities, and allow their existing streams to complete. A stopped model must receive its new settings without being started unnecessarily.

Keep llama-swap responsible for inference routing, on-demand loading, and process ownership. Studio will generate versioned launch manifests, classify pending changes, and coordinate selective application. The first implementation targets Linux/Docker. Other platforms retain the existing path until launcher replacement, locking, and process-tree behavior are tested there.

**All-engine release requirement:** migrate `llama_cpp`, `ik_llama`, `lmdeploy`, `1cat_vllm`, `vllm`, `sglang`, `sglang_v100`, and `audio_cpp` together. There is one deployment-wide transition, with no GGUF-first release, engine-specific enablement, or supported hybrid of legacy and manifest commands after migration. Reviewable implementation changes may land separately behind an inactive deployment-wide flag, but the flag cannot be enabled until every registered engine has passed the release gates. CUDA is a shared runtime dependency, not a ninth inference engine.

The accompanying [use-case specification](launch-manifests-use-cases.md) is normative. It defines save/apply behavior, environment precedence and removal, GPU identity and ordering, failures, and the exact scope of each action. Implement and test both documents together.

**Evidence and scope**

The pinned llama-swap v255 rebuilds its server on configuration reload and shuts down its old local processes. Removing Studio's unload-all call alone does not prevent this. See [reload](https://github.com/mostlygeek/llama-swap/blob/v255/llama-swap.go) and [router shutdown](https://github.com/mostlygeek/llama-swap/blob/v255/internal/router/base.go). The same reload shape is still present in [v260](https://github.com/mostlygeek/llama-swap/blob/v260/llama-swap.go): it constructs a new server and shuts the previous one down. `internal/router/scheduler/fifo.go` is unchanged from v255 through v260.

An isolated experiment using the checksum-verified v255 release and two fake inference servers established that a stable launcher can read a changed per-model manifest, restart A through the existing unload-one API, and preserve B's PID and complete 30-event stream. Editing only A's description in swap YAML stopped both servers. This establishes feasibility, not production GPU, engine compatibility, or failure-recovery coverage.

**Upstream review, v256–v260 (2026-09-27).** Those five releases are 42 commits past v255. None of them make a YAML edit restart one model only, so they do not replace manifests. The image pin is now the checksum-verified v260 Linux binary in the Dockerfile. The v255 experiment above remains the evidence for selective restart.

Carry these constraints into the design now, including the ones that only become active if the pin moves:

| Change | What it does | Requirement for Studio |
| --- | --- | --- |
| TTL idle window (v256, [#1095](https://github.com/mostlygeek/llama-swap/pull/1095)) | A positive TTL starts when the process becomes ready, including before the first request, and resets on restart. v255 starts that window on first use. | Studio does not emit per-model TTL today. A future TTL stays a proxy field and a global apply (use-case C15). Next-start publication is what the following automatic start uses after expiry. |
| `readySince` / `uptimeMs` (v259) | Model status includes when the process last became ready, and for how long, only while it is ready. | Optional display source after an upgrade. v1 running-revision identity stays on Studio receipts and process identity. |
| `globalConcurrencyLimit` (v256, [#1110](https://github.com/mostlygeek/llama-swap/pull/1110)) | Top-level cap on in-flight inference requests across models. Default `0` means unlimited. A non-zero value rejects the excess with HTTP 429. | Proxy configuration, same class as per-model concurrency (C15). Leave it unset unless a fleet-wide cap is an explicit product choice. |
| `cmdStop` hang bound (v258, [#1165](https://github.com/mostlygeek/llama-swap/pull/1165)) | Stop-command output is logged, and a child holding those pipes cannot block unload. | Studio does not generate `cmdStop`. An engine that later needs one requires this behavior or an equivalent bound. |
| Capability discovery (v257, [`capcompat`](https://github.com/mostlygeek/llama-swap/blob/v260/internal/capcompat/design.md)) | On ready, llama-swap fills empty `capabilities` into `/v1/models` and caches them for 30 days. The cache key hashes `cmd`, `proxy`, and `useModelName`. | See the stable-entry rule in section 4. |
| CORS (v257) | An absent `security.cors` block keeps the previous allow-any policy. Any other CORS field without `allowedOrigins` fails config load. Upstream CORS headers are stripped. | Do not emit a `security` block unless browser access to port 2000 is intentionally restricted. |
| Log streams (v259, [#1172](https://github.com/mostlygeek/llama-swap/pull/1172)) | Log lines moved from `GET /api/events` to `GET /api/events/logs?stream=proxy\|upstream\|http`. `logToStdout` accepts a comma-separated stream list; `both` and `none` still work. | Studio's own `/api/events` is unaffected. A client that tails llama-swap's event stream for logs must switch endpoints on upgrade. |
| Client address (v257, [#1130](https://github.com/mostlygeek/llama-swap/issues/1130)) | Metrics prefer `X-Forwarded-For` or `X-Real-IP` and prefix those values with `xff:`. | Relevant only after an upgrade, for activity recorded by llama-swap itself. |

Discovery coverage, if the pin moves to v257 or newer: llama-server and ik_llama.cpp (`owned_by: llamacpp`) supply modalities, tools, and per-slot context from `/props`; vLLM supplies `max_model_len` only; audio.cpp, SGLang, and LMDeploy are cached as misses. Configured capability fields win one field at a time. `tools: false` cannot override a discovered true, because a zero value is indistinguishable from an omitted field. `capabilities.disableAuto: true` is the off switch.

llama-swap's playground, Tailcat, kubeswap, the unified-image audio.cpp MP3 and download changes, DGX Spark detection, and the macOS M6 startup fix are outside this plan.

Runtime manifests cover executable selection, ordered arguments, working directory, environment changes, and launch-time sidecars. They do not make llama-swap aliases, filters, model registration, TTL, or routing policy dynamically editable. Those still require the existing global apply path. Request-time sampling parameters need no restart when supplied in requests and supported by the engine; editing equivalent defaults in swap YAML remains a proxy change.

**1. Separate compilation from execution**

Introduce a pure compiler that produces two independent outputs for a model:

- `LaunchSpec`: everything needed to execute the selected engine, with explicit port and artifact placeholders.
- `ProxyModelSpec`: the stable launcher command and llama-swap-owned configuration, including model identity, aliases, filters, upstream model-name mapping, health endpoint, and proxy policy.

Extract structured arguments before the current command builders turn them into shell strings. Reuse parameter scanning, model compatibility validation, companion-file resolution, and engine-specific environment logic. Do not reconstruct execution arguments by parsing generated `bash -c` commands.

The preview UI must continue showing the actual engine command, environment, and working directory. Show the launcher command only as optional diagnostic information. Previewing or saving a model must not publish a manifest or restart anything.

Proposed modules, adjusted to conventions established by the other agent:

| Module | Responsibility |
| --- | --- |
| `backend/runtime_launch_spec.py` | Typed specs and compiler interfaces |
| `backend/launch_manifest_store.py` | Immutable generations, atomic pointers, locks, receipts, retention |
| `backend/runtime_launcher.py` | Small executable entry point that reads a published manifest and execs an engine |
| `backend/services/model_runtime_apply.py` | Change plans, revision checks, selective apply, rollback, reconciliation |

Keep these responsibilities out of the already-large models router and swap manager. Use the operation supervisor and persistence helpers being introduced by the hardening work; do not create a second background-task framework or competing process owner.

**2. Define the manifest contract**

Use JSON with an explicit schema version. Example, illustrative paths and shortened revision:

```json
{
  "schema_version": 1,
  "model_id": "model-a",
  "engine_id": "llama_cpp",
  "engine_install_id": "source-release-123",
  "executable": "/app/data/llama-cpp/source-release-123/bin/llama-server",
  "argv": [
    "--model", "/app/data/models/model-a.gguf",
    "--port", {"runtime": "port"},
    "--ctx-size", "16384"
  ],
  "cwd": "/app/data/llama-cpp/source-release-123",
  "env": {
    "set": {"CUDA_VISIBLE_DEVICES": "0"},
    "unset": []
  },
  "artifacts": {}
}
```

The executable is separate from arguments; the launcher constructs `argv[0]`. Typed placeholders are substituted only in designated positions, never by arbitrary string interpolation or shell evaluation. Artifact references use the same explicit mechanism and resolve inside their generation directory.

The revision is SHA-256 of canonical launch content and artifact content digests. Normalize object key order but preserve argument order, repeated flags, list order, empty arguments, and values beginning with `-`. Do not reuse the existing command comparison that sorts flags. Exclude timestamps, launch receipts, and the actual assigned port from the hash. Include engine install identity, working directory, environment, and referenced launch configuration content.

Pin generation dependencies to immutable engine/model artifacts where possible. Rebuilds or refreshes that replace a binary or model in place must invalidate its identity even if the path is unchanged. Use existing commit/download ledgers and recorded artifact identities; do not hash multi-gigabyte weights on every save. Refuse deletion of versions/files referenced by a running, published, or retained rollback generation, or explicitly retire those references first. A manifest cannot guarantee rollback after its underlying files have been overwritten.

Environment handling must explicitly set or unset Studio-owned CUDA, library, and engine variables so removing an override does not restore stale values inherited from llama-swap. Preserve required base environment variables deliberately. Use the exact precedence, empty-value, unset, inheritance, and GPU rules in the use-case specification. Adapt current engine environment construction to those rules rather than preserving its inconsistent engine-specific merges. Restrict manifest/receipt permissions and redact sensitive environment values in API responses and command previews.

**3. Store immutable generations and distinguish desired, published, and running state**

Suggested layout outside every llama-swap watched configuration directory:

```text
data/runtime/models/<safe-model-key>/
  generations/<revision>/manifest.json
  generations/<revision>/artifacts/server.json
  active.json
  launch.lock
  launches/<launch-id>.json
```

Derive `<safe-model-key>` from a stable internal ID using a collision-resistant encoding/hash. Do not use a display name, alias, or raw filesystem path supplied by the client. Restrict revision references to validated hashes and prevent traversal or escaping symlinks.

`active.json` contains the published revision. Stage and validate a complete generation, flush its files as required, then atomically replace the pointer. The launcher must read one pointer once and resolve all artifacts from that generation. Never change an existing generation in place.

State meanings:

- **Desired**: compiled from saved Studio configuration; can differ from what is published. A newer saved edit remains pending after an older revision finishes applying.
- **Published**: the revision selected by `active.json`, used for subsequent starts, including on-demand starts after TTL expiry or a crash.
- **Running**: a revision associated with an observed live engine process and verified readiness; can differ from published during deferred application.

Launch receipts record model ID, revision, unique launch ID, PID, process start identity, and timestamp without secrets. Write an atomic receipt immediately before `exec`. It is an attempt record, not proof of readiness. Reconcile it with process identity and the model-specific health/start result before declaring a revision running. Reject stale receipts and PID reuse. Distinguish an exec failure from a successful launch.

Retain the running generation, the published generation, the previous valid generation, and generations referenced by pending operations. Garbage-collect only unreferenced generations. Audio server JSON belongs inside these generations, so a failed apply cannot overwrite sidecars used by the previous configuration.

**4. Keep llama-swap entries stable**

Illustrative generated entry:

```yaml
models:
  model-a:
    cmd: >-
      /app/.venv/bin/python /app/backend/runtime_launcher.py
      --manifest-root /app/data/runtime/models/SAFE_KEY
      --port ${PORT}
    proxy: http://127.0.0.1:${PORT}
```

Resolve the real Studio interpreter and launcher locations for Docker and local development; the example is not a hardcoded deployment requirement. The launcher must operate independently of the engine working directory and should use minimal imports without importing the FastAPI application, creating stores, doing discovery, or accessing the network.

On Linux, validate the pointer/schema, acquire the launch gate, resolve the port/artifacts, construct the environment, change directory, and use `os.execve`. Do not detach, daemonize, or keep a supervisor wrapper between llama-swap and the engine. Preserve stdout/stderr and process-group membership. Release the launch gate before exec and close its descriptor so the engine does not retain the lock. Integrate this ordering with the apply protocol below.

Move engine binaries, runtime flags, launch environment, and engine-specific macros out of swap YAML. Keep only routing-owned fields and a stable launcher invocation. A changing manifest revision must never appear in `cmd`, a swap macro, YAML metadata, or another field that would cause a YAML rewrite. All publish paths must skip writes when the proxy projection is unchanged, including engine activation, model start, regeneration, and startup repair.

The pinned binary is v260, which probes upstream capabilities. Every generated model entry sets `capabilities.disableAuto: true`. A stable `cmd` is exactly the cache key `capcompat` uses, so leaving discovery on would keep context length and modalities from an older generation for up to 30 days after a selective restart. Putting the launch revision into `cmd`, `proxy`, or `useModelName` to bust that cache would rewrite YAML and reload every model, which this plan exists to avoid. Studio remains the source of advertised capabilities. A disk entry missing `disableAuto` is a proxy difference and takes the global apply path.

Use llama-swap's supplied port rather than choosing a second port independently. Verify command quoting for Studio paths with spaces. Maintain each engine's required `useModelName`, health endpoint, and protocol. An engine change that alters these fields is a global proxy change even if its executable fits into a manifest.

**5. Compute an explicit apply plan**

Compare desired and published launch specs separately from desired and deployed proxy specs. Persist/recompute the result after backend restart; an in-memory stale flag is only a UI hint.

| Difference | Action |
| --- | --- |
| Studio metadata or unused engine settings only | No runtime action |
| Launch spec only; model stopped | Publish for next start |
| Launch spec only; model running | Restart this model, or publish for its next start |
| Aliases, filters, health contract, upstream mapping, TTL, registration, groups/profiles/selectors | Global proxy apply |
| Runtime and proxy changes together | Global apply; never silently apply an inconsistent subset |
| No effective difference | No file writes and no restart |

An active engine-version change can produce a set of affected model revisions. Enumerate that set explicitly and restart only its running members when the proxy projection is unchanged. Do not claim independence from physical resource limits: changing A's GPU allocation can prevent A from starting alongside B. Report the conflict or fail/roll back A; do not silently evict unrelated models.

Proposed API contracts:

- Extend `GET /api/llama-swap/pending` with a plan ID, per-model current/desired revisions, actions, reasons, and `requires_proxy_reload`, retaining existing summary fields during migration.
- Add a typed per-model apply endpoint, e.g. `POST /api/models/{model_id:path}/runtime/apply`, accepting the expected desired/published revisions, an idempotency key, and `mode: restart_now | next_start`.
- `restart_now` on a stopped model only publishes. `next_start` on a running model publishes without stopping it, and clearly reports that the running revision is older.
- Return a task/operation ID for work that waits for unload/start/readiness. Distinguish publication, restart, readiness, rollback, and interruption in task results. A retry of the same operation must not restart the model twice.
- Return a revision conflict for stale plans, and a structured global-apply-required response for proxy changes. Never invoke unload-all as an invisible fallback from per-model apply.

**6. Execute selective apply safely**

Coordinate global apply, per-model apply, engine activation/deletion, model deletion, and explicit start/stop using shared operation/resource locks. For v1, serializing configuration applications under the existing global apply lock is acceptable: it does not mean restarting all models. Avoid holding a threading lock across asynchronous network waits. Define lock ordering before adding finer concurrency.

For `restart_now`:

1. Check expected revisions under the coordinator, classify the change again, and record the operation, previous published pointer, and last verified running revision. Select the latter as the service-restoration target when available; the previous pointer might select an untested next-start revision. Validate/stage all launch files before stopping anything.
2. Acquire the per-model exclusive launch gate. The launcher uses its corresponding read gate only while reading/validating its selected generation and writing its receipt. Use a cross-process mechanism with bounded waits. The gate blocks new launchers from selecting a revision during the stop/publish window; it is not an HTTP request-draining mechanism.
3. Determine the model's actual runtime state. If runtime state cannot be established while the proxy is expected to be running, fail/retry rather than treating it as stopped. Unload only this model when running/loading; wait for completion and verify the old owned process has exited.
4. Atomically publish the new pointer and record publication. Release the launch gate before requesting a start or waiting for health, otherwise the new launcher would deadlock.
5. If the model was running, trigger the existing model-specific load path and wait for the new receipt plus readiness, within the engine's configured startup budget. On-demand concurrent starts must converge through llama-swap on the same published revision; do not start a second engine directly from Studio.
6. Record the outcome. Recompute pending state so concurrent newer edits remain visible. Never clear a global stale flag merely because one model succeeded.

For `next_start`, stage and publish under the same pointer/gate discipline without unloading. Explain that the next automatic start, including one caused by TTL or a crash, will use the new revision.

If startup fails, reacquire the gate, stop any candidate process through llama-swap, restore the recorded service-restoration revision (or previous pointer if there was no verified running revision), and release the gate before attempting to restore service. Only restart the restoration revision if the operation originally found the model running. Keep displaced next-start and desired revisions visible as unapplied; never discard them silently. Report both the original failure and rollback result; a successful rollback does not mean the requested change succeeded. If rollback dependencies are unavailable, leave that model stopped with actionable status. B/C remain untouched.

Journal operation phases and revisions before irreversible transitions. On Studio restart, reconcile unfinished applies against pointer, receipt, and actual process state. Do not blindly replay unloads or assume an interrupted operation never published. Cancellation before stopping is safe; after stopping/publishing, finish recovery to a consistent state before releasing the resource lock.

**Request semantics:** v1 offers explicit restart-now or next-start behavior. Restart-now can interrupt requests to the selected model. Existing unrelated streams must survive, but llama-swap's synchronous unload scheduler may briefly delay dispatch of new requests during a slow stop. Do not label this zero-downtime or graceful draining. Adding a reliable drain mode later requires an admission gate covering every inference client, including direct calls to port 2000; polling an in-flight count alone is insufficient. See [upstream unload behavior](https://github.com/mostlygeek/llama-swap/blob/v255/internal/router/scheduler/fifo.go).

**7. Integrate the UI and migration**

Replace the global-only notice with a reviewed action list: “Restart model A,” “Use new settings on next start,” or “Reload proxy — affects all loaded models.” The model configuration page must apply only that model when eligible. Global bulk application may process eligible launch changes sequentially and return per-model results; do not imply multi-model transactional rollback. Mixed proxy changes remain one explicit global action.

Command previews show the resolved engine command from the same compiler used for manifests. The saved/desired command and running revision must be distinguishable. Preserve existing aliases and request-default behavior throughout migration.

Use one deployment-wide migration flag only. All eight engine adapters, all currently deployed model entries, and all relevant audio sidecars move to manifests in the same migration. Never migrate implicitly during model start or save. Fail the preflight if any deployed model cannot be compiled; leave the entire existing deployment untouched. Non-deployed catalog records that lack an installed engine or valid artifacts remain explicitly unavailable and do not gain a route merely because migration ran. No installed instance of every engine is required on each user's machine, but release support and tests for every registered engine are required.

Initial migration changes swap commands, so it requires one scheduled global reload. Build complete manifests for the chosen baseline, capture running models, validate the candidate with the pinned llama-swap validator, and present the impact before the user invokes the existing apply flow. Require an explicit decision about pending saved edits: incorporate them in that reviewed migration or defer migration until resolved; do not infer the running configuration from the latest saved values. Keep legacy YAML and referenced artifacts for rollback.

For any global apply, stage generations before YAML publication and use a journal to recover cross-file partial publication. Do not report success based on `/health` alone: it may still describe the old configuration before the file watcher runs. Use one controlled proxy stop/publish/start for the all-engine migration, with no intermediate hybrid config becoming live. Restore the previously running set sequentially where possible; leave previously stopped models stopped, and report any restoration failures. Switching the feature flag back restores the full legacy deployment through a deliberate global transition, never an engine-by-engine fallback.

**8. Deliver in reviewable increments**

| Change | Deliverable | Completion gate |
| --- | --- | --- |
| 1 | Pure launch/proxy specs for all eight engines; preserved previews | Compiler registry equals engine registry; ordered argv/env/cwd behavior and the use-case contract pass for every engine |
| 2 | Versioned manifest store and Linux exec launcher | Atomicity, malformed input, gate, receipt, signal, environment, and sidecar tests pass |
| 3 | Stable swap entries for every engine behind one inactive flag | All-engine migration/rollback works; runtime edits for every engine leave generated swap YAML byte-identical |
| 4 | Apply planner, per-model API, rollback and reconciliation | Concurrent start/apply, stale revision, failure, and restart tests pass |
| 5 | UI scope/revision handling, environment modes, GPU controls, and all-engine migration preview | Every use-case has a user-visible outcome; sidecar-only edits are detected; no per-engine fallback exists |
| 6 | Pinned-binary integration and one all-engine release | All-engine matrix and migration gates pass before enabling manifests anywhere |

Budget approximately 4–6 engineer-weeks for one tested Linux/Docker release covering all engines, explicit environment/GPU semantics, and the expanded use-case matrix, assuming the ongoing hardening changes are stable and suitable test hardware is available. This is a planning estimate, not an engine-by-engine rollout schedule. No upstream llama-swap fork or complete replacement proxy is required. v256–v260 do not remove that conclusion.

**Acceptance suite**

- Run the actual checksum-pinned llama-swap binary on ephemeral loopback ports with fake engines that expose PID/revision, stream responses, record arguments/environment, delay startup, fail readiness, and spawn child processes. Do not mock the process manager for these checks.
- Start A/B/C; stream from B while applying A. A has the requested revision and new process identity; B/C retain identities; B's stream completes; the proxy PID and YAML bytes/mtime do not change; no unload-all call occurs.
- Repeat for a stopped A, next-start mode, no-op save, unused-engine edit, repeated flags, paths with spaces, environment removal, and audio-sidecar-only changes.
- Test adding/removing models, alias/filter edits, upstream mapping changes, and engine switches that change the proxy contract: all classify as global and cannot enter selective apply.
- Generated entries set `capabilities.disableAuto: true`. After a context-size or modality restart of A, `/v1/models` must not keep A's previous discovered context or modalities. A disk entry missing that field is a global proxy change.
- Inject failures at staging, stop, pointer replacement, exec, health, and rollback. Only A is affected. Test backend interruption at each journal phase and reconcile without inventing success or replaying a completed restart.
- Race requests, explicit start/stop, duplicate apply, newer saves, global apply, and artifact deletion. Assert one owned engine process per model and no mixed manifest/artifact generation. Test launcher gate release and timeout, including cancellation while blocked.
- Verify rollback dependency retention and PID-reuse/stale-receipt rejection. Ensure cleanup cannot delete files referenced by any live or retained generation.
- Verify engine activation changes exactly the intended model set. Run compiler and fake-process lifecycle tests for all eight engine IDs and cross-engine combinations. Real-runtime launch/apply smoke tests cover every engine before the single release; `sglang_v100` needs its supported V100 environment. Fake-engine tests establish lifecycle behavior but do not substitute for engine/GPU compatibility. Missing hardware evidence leaves the all-engine release gate incomplete, rather than enabling a subset.
- Parameterize the numbered use cases over every engine where supported, with explicit unsupported-capability results where appropriate. Verify environment override deletion/empty/unset behavior, GPU ordering and identity, unavailable devices, parallelism validation, and migration abort when any deployed model fails preflight.
- Test UI output for partial bulk success, rollback, pending newer edits, next-start publication, and global-impact warnings. An apply operation is complete only when its declared outcome is observable.

**Handoff to the concurrent hardening work:** preserve its atomic publication, operation supervisor, authentication, cancellation, and structured event contracts. Land this plan independently; resolve integration against those final interfaces. Do not repurpose build/install subprocess ownership for inference processes, which remain children of llama-swap.
