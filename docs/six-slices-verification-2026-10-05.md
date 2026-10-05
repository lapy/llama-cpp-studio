# Six-slice verification — 2026-10-05

Reviewed the working tree on `5d8b042` against the revised CI/recovery plan and its six delivery slices. **The reproduced code defects are now resolved within the checked scope. Production Docker/Node 24 execution remains unverified locally.** This review changed documentation only; independent probes used temporary data and synthetic credentials.

## Final targeted re-check: bearer-token fix confirmed

The original bundle-builder reproduction now returns `Authorization: [redacted]`, without the planted bearer token. Independent checks also confirm complete redaction of Basic authorization, `token: Bearer ...`, ordinary password assignments and URL userinfo. The endpoint regression now includes these planted credential forms.

- Diagnostics tests: **5 passed**, 0.95 seconds.
- Full frontend rerun: **414 passed across 47 files**, 31.04 seconds, including the previously failing audio-upload test. Its earlier intermittent failure remains historical evidence; one green rerun does not establish that the timing issue was repaired.
- `git diff --check`: passed.

The broader backend/browser/build/navigation results from the prior re-check remain the latest evidence for those areas; they were not repeated for this targeted redaction change. No reproduced code blocker remains from this review. Docker smoke, the container navigation runner (including bind-mount ownership), and Node 24 still need execution in the supported CI environment before claiming full production validation.

Logs: `/tmp/studio-six-r3-diagnostics.log` and `/tmp/studio-six-r3-frontend.log`. The sections below preserve earlier findings; their bearer-redaction blocker is now closed.

## Re-check of fixes

| Slice | Updated verdict |
| --- | --- |
| 1. CI/container | Code reviewed; Docker execution and Node 24 validation remain unverified locally. |
| 2. Recovery | Pass: all 14 browser journeys. |
| 3. Diagnostics | Original password/URL cases and failure-record race fixed; a bearer-header redaction regression remains. |
| 4. Apply differences | Reported defects fixed: negative numeric values, inline values and credential argument filtering now work in probes and tests. |
| 5. Navigation | Prior measurement omissions fixed; five local production-backend runs pass. Container measurement is now wired before publishing, but that execution path remains unverified here. |
| 6. Source-build extraction | Missing imports fixed; new default-config CPU and failed-configure tests pass with fake command execution. Checked-command helpers are now used by the workflow. Live compilation remains outside this verification. |

### Remaining P1: bearer credential survives diagnostic sanitization

The new assignment regex runs before the bearer-token regex in `backend/diagnostics.py:redact_text`. For `Authorization: Bearer PLANTED_BEARER_SECRET`, assignment redaction consumes only `Bearer`. The subsequent bearer regex can no longer recognize the token.

Independent reproduction through the actual bundle builder:

```python
store_io._record_persistence_event(
    'failure', RuntimeError('Authorization: Bearer PLANTED_BEARER_SECRET')
)
bundle = build_diagnostics_bundle()
```

The resulting message is **`Authorization: [redacted] PLANTED_BEARER_SECRET`**. Redact the complete authorization value, including its scheme and credential, and add this message form to the bundle regression. A regex reordering alone should be checked against other credential assignment forms as well.

The original `password=PLANTED_PASSWORD` and `https://alice:PLANTED_PASSWORD@example.test/` probes now correctly produce `password=[redacted] https://example.test/`. The Apply probe now returns `{'n_gpu_layers': '-1', 'ctx_size': '4096'}` and excludes the planted API key.

### Latest validation

- **Backend: 1,676 passed, 8 skipped**, 57.45 seconds. Failure events are now recorded inside the write wrapper before its Future completes, resolving the earlier ordering race.
- **Frontend: 413 passed, 1 failed**, 36.01 seconds. `AudioModelConfig.test.js:216`, “uploads a WAV through the hidden file input,” observed no upload call. Its focused file rerun passed all 14 tests. Record this as a timing-sensitive validation failure, not a consistently green full suite.
- **Browser: 14 passed**, 50.3 seconds.
- **Build passed**, 1.15 seconds; compressed asset-size budget **229,865 / 250,000 bytes**.
- **Navigation:** five uncached runs against the built backend with the named Demo safetensors fixture reported document 118.8–198.0 ms, document-plus-asset transfer **232,616 bytes**, and time to control 256.5–345.9 ms. All ceilings pass. Document bytes are now included, APIs excluded, and measurement waits for document load.
- Both shell scripts pass syntax checks; `git diff --check` now passes.

`publish-docker.yml` now installs the measurement browser and invokes `scripts/measure-production-navigation.sh` before pushing the tested image. The script seeds a named model. It bind-mounts a host `mktemp -d` directory (normally mode 0700) without aligning ownership with the non-root image user. Verify this on the actual runner: a host/container UID mismatch will prevent access to `/app/data`. This is a portability concern found by inspection, not an observed container failure; Docker remains unavailable here.

Local Node remains 20.20.2. No remote CI, Docker smoke or live GPU build was claimed. Logs: `/tmp/studio-six-r2-{backend,frontend,browser,build,navigation}.log` and `/tmp/studio-six-r2-audio-rerun.log`. The temporary backend server was stopped afterward.

The original findings and results below are historical; the updated verdicts above supersede them.

| Slice | Verdict | Evidence / remaining work |
| --- | --- | --- |
| 1. Production CI and container | Implemented; runtime verification outstanding | Node 24 checks, mandatory non-fork build, fork step skips, smoke-before-push and pushed digest attestation are present. Shell syntax passes. Docker is unavailable in this WSL distro, so image build, health, asset serving and shutdown could not be exercised. |
| 2. Runtime recovery | Pass within fixture scope | All 14 browser tests pass, including failed Apply, loading versus verified running, startup timeout, stale/unreachable state, proxy recovery, unrelated download task events, cancellation and failure/retry. |
| 3. Diagnostics | Needs fixes | Passwords survive bundle sanitization; full backend suite also exposes a failure-record timing race. |
| 4. Apply differences | Needs fixes | CLI-derived comparison misparses negative values and does not exclude credential flags. |
| 5. Navigation measurement | Partial | Five production-backend samples pass the script's ceilings, but it omits document transfer bytes, counts API resources, lacks deterministic fixture setup and is not invoked by CI. |
| 6. Source-build extraction | Fails independent execution probe | Missing runtime imports break source builds. The extracted workflow still depends on the whole manager rather than the planned explicit dependency contract. |

## Findings

### P1 — Source builds fail after extraction

`backend/engines/llama_cpp/source_build.py:146` calls `BuildConfig()` without importing or defining it. The same module uses `subprocess` at lines 432 and 726, and `shlex` at line 754, without importing them. Deferred annotations hide the missing type names at module import time; they do not fix executable references.

Independent reproduction: create a `LlamaManager`, point `llama_dir` at a temporary directory and call `await manager.build_source('abc123', use_workspace=True)` without a config. It fails before network work with:

```text
Failed to build from source abc123: name 'BuildConfig' is not defined
```

Restore all moved dependencies and test a default-config path plus a CPU build path with command execution faked. Existing delegation and early clone-failure tests do not reach these branches. `run_checked_command` has tests but is not called by the extracted workflow; the production workflow continues to call `host` methods directly. Establish the planned contract or explicitly narrow the architecture claim.

### P1 — The downloadable diagnostic bundle can retain credentials

`backend/diagnostics.py:45` handles selected query parameters but preserves URL userinfo. `redact_text` handles token-shaped strings and selected query forms, but ordinary password assignments survive.

Independent reproduction records a persistence failure containing synthetic credentials and calls `build_diagnostics_bundle()`. Its returned failure message remains:

```text
connection failed password=PLANTED_PASSWORD https://alice:PLANTED_PASSWORD@example.test/
```

This is the actual bundle-construction path, not merely an isolated regex example. Strip URL credentials and sanitize credential assignments in free text, or omit raw exception messages from downloadable bundles in favor of structured safe descriptions. Add both forms to regression tests. The existing test only plants `hf_...` tokens and an `api_key` query parameter.

### P2 — Apply diffs parse launch arguments incorrectly

`backend/services/model_runtime_apply.py:653` treats every token starting with `-` as an option. For `['--n-gpu-layers', '-1']`, it produces `n_gpu_layers=true` and a fictitious setting `1=true`. Comparing negative values produces changes in fictitious numeric fields rather than the intended GPU-layer value.

The same helper returns `{'api_key': 'PLANTED_KEY'}` for `['--api-key', 'PLANTED_KEY']`; its credential filter only applies to environment keys. This conflicts with its own claim that secrets stay out and the intended safe comparison surface.

Compare structured effective configuration or use an argument parser aware of value types, aliases and `--name=value` syntax. Apply secret filtering to all representations. The context-size happy-path tests pass but do not cover these cases. Existing revision checks remain in the apply path.

### P2 — Navigation measurements are not a complete CI gate

`frontend/e2e/production-navigation.mjs:44` retrieves navigation timing but sums only resource entries for transferred bytes. Consequently the document is omitted. Its same-origin filter includes API responses, rather than restricting the metric to the document and assets as planned. Measurement ends when a control becomes visible, so document `duration` can still be zero before load completes.

No workflow invokes this script or sets `STUDIO_NAVIGATION_URL`. A new empty container also has no model control to satisfy the locator; the harness needs deterministic catalog setup and lifecycle management. A local GGUF-only fixture timed out looking for the control, while a safetensors fixture exposed it and completed all five samples. The script should define the tested library/view instead of relying on arbitrary server data.

Wire it into the production-serving validation, include navigation `transferSize`, define an explicit measurement completion point and asset filter, and seed a reproducible fixture. Preserve the separate static compressed asset-size gate. The new backend catalog-read/write-overlap test passes.

### P2 — Persistence failure diagnostics have a timing-sensitive test failure

The complete backend run fails `test_failed_store_write_is_remembered_without_a_document` because `persistence_status()['latest_failure']` is `None` immediately after the write raises. A focused rerun passes.

In `backend/store_io.py:65`, failure recording runs in a Future completion callback. A caller waiting on `future.result()` can wake before that callback records the event. Decide whether diagnostic recording is guaranteed before returning a failed write; either enforce that ordering or make the test wait for the explicitly asynchronous event. Do not treat the focused rerun as a clean full-suite result.

## Validation

| Check | Result |
| --- | --- |
| Full backend suite | **1 failed, 1,673 passed, 7 skipped**, 65.76 s |
| Focused rerun of failing diagnostics test | Passed; timing race remains |
| Frontend suite | **414 passed**, 37.47 s |
| Chromium fixture journeys | **14 passed**, 49.3 s |
| Production build | Passed, 1.12 s |
| Compressed asset-size budget | **229,865 bytes / 250,000** |
| Container smoke shell syntax | Passed |
| Container execution | Unavailable: Docker Desktop WSL integration is disabled |
| Whitespace check | Fails on four added lines in `frontend/src/styles/_base.css:359` |

Production navigation was run against a fresh local backend serving the production build, using a temporary safetensors catalog record, no active engine and no GPU. Five uncached Chromium runs produced document duration 122.3–190.7 ms, script-reported resource transfer 242,032–242,033 bytes, and time to visible control 258.0–323.2 ms. These pass the current 800 ms / 400,000 byte / 1,000 ms ceilings, subject to the measurement defects above. They are not container or remote-CI results.

Local Node remains 20.20.2; only Node 20 is installed here. Node 24 is configured in CI but was not executed in this review. No image was published and no live model was built or launched.

Temporary evidence: `/tmp/studio-six-{backend,frontend,browser,build,probes,navigation-safetensors}.log`, `/tmp/studio-six-diagnostic-rerun.log`, and `/tmp/studio-six-probes.py`. The local verification server was stopped afterward.

Recommended order: repair source-build runtime failures and diagnostic redaction, fix the failure-record race and Apply parser, then complete the production navigation gate and obtain a Node 24/container CI run.
