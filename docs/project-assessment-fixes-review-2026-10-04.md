**Verification of assessment fixes — 4 October 2026**

**Follow-up implementation: the three findings below are now fixed.** Default GET/HEAD requests receive the 20-second deadline after Axios merges defaults; positive explicit deadlines remain unchanged and mutations receive no new deadline. Model configuration now distinguishes launch-manifest plans using the capability flag, refreshes the pending plan and model observation after selective apply, and refreshes the plan after persisted template changes.

Added regression coverage using the actual Axios adapter configuration and real Pinia stores for legacy response normalization, successful restart, and stopped-model publication. Audio fixtures now wait for configuration loading to complete and clear session drafts; configuration component tests automatically unmount. This improves synchronization and cleanup without claiming a definitive root cause for the earlier intermittent failure.

Follow-up validation: **415 frontend tests passed across 47 files**, production build passed, and `git diff --check` passed. Chromium verified that successful selective apply removes the restart action and updates the state description, and that legacy pending changes retain the reload button. Evidence: `/tmp/studio-repair-{targeted,frontend,build,browser}.log`. This follow-up changed frontend code/tests only; backend validation below is from the preceding review. The broader assessment plan remains separate.

The remainder records the original review and reproductions before those fixes.

Reviewed the uncommitted changes on `2323179` following the [project assessment](project-assessment-2026-10-04.md). This is a review of the corrective patch, not a declaration that the entire improvement plan is complete. No application code was changed during this review.

The patch fixes the reproduced library polling leak and moves the identified metadata I/O off the event loop. It also introduces useful runtime observation quality, preserves the last known running state, and ages out header health. Three confirmed issues remain in the new behavior.

**1. P2 — The default read deadline does not take effect.**

In [the API client](../frontend/src/api/client.js), the request interceptor assigns its 20-second timeout only when `config.timeout == null`. Axios has already merged its defaults before invoking the interceptor, so an ordinary `axios.get('/api/status')` arrives with `timeout: 0`. The null check fails and the request retains an unlimited timeout.

An adapter probe using the installed Axios package and the actual interceptor source observed `timeout: 0` for a default GET and `timeout: 1234` for an explicitly configured GET. Consequently, a stalled library read can keep its in-flight flag set and prevent subsequent refreshes; repeated header polling can accumulate requests.

Make the interceptor account for Axios's merged zero default, preserving deliberately configured deadlines and any explicitly supported opt-out. Add a test that inspects the adapter's effective configuration, rather than testing an unmerged input object.

**2. P2 — Successful selective apply leaves the old pending plan on screen.**

In [ModelConfig](../frontend/src/views/ModelConfig.vue), `showApplyLlamaSwap` now depends on `swapConfigPending.models`. The selective branch of `applyLlamaSwapFromModelConfig()` fetches that plan before applying, then refreshes only the cheap stale flag after success. The pending-plan watcher responds to transitions to `stale: true`, so clearing the stale flag does not replace the old plan.

Chromium reproduction with controlled API responses: open a model whose action is `restart_now`, confirm the selective restart, return a successful apply response and `stale: false`, and make subsequent pending-plan responses empty. The page still says “Saved changes for this model are not in use yet” and retains the “Restart this model” button. No pending-plan request is made after the apply. This can lead users to attempt a second apply even though the first succeeded.

Refresh or invalidate the model-specific plan after apply, and reconcile the runtime observation used by the page. Cover successful running-model restart and stopped-model publication in an integration test. Also use the same invalidation policy for other configuration writes, including persisted templates.

**3. P2 — Legacy mode loses the model-page reload action.**

The new `Array.isArray(pending.models)` branch in [ModelConfig](../frontend/src/views/ModelConfig.vue) treats any models array as an authoritative per-model plan. However, [the engine store](../frontend/src/stores/engines.js) always normalizes an absent `data.models` field to `[]`, including legacy responses when `LAUNCH_MANIFESTS_ENABLED=0`.

Chromium reproduction: return a valid legacy pending response with `applicable: true`, `pending: true`, and a changes list, alongside `stale: true`. The model page hides “Reload proxy” and claims that the saved settings are published and running. This affects the model-page action; it does not establish that the separate header apply flow is unavailable.

Use the explicit `launch_manifests` capability flag to distinguish per-model plans from legacy deployment-wide state. Test through the real store normalization, since manually constructed component fixtures can omit the array and miss this regression.

**Confirmed improvements**

| Earlier finding | Current verification |
| --- | --- |
| Blocking file-size and quantization metadata I/O | The identified SDK work uses a dedicated four-worker executor; HEAD fallback uses async HTTP with a four-request concurrency limit. The new responsiveness, concurrency, and timeout regression tests pass. The executor limits active workers; this is not proof of a bounded submission queue or hard cancellation of running SDK calls. |
| Poll survives navigation during an active refresh | Original Chromium reproduction now stops at three model requests: initial library load, the delayed poll, and Audio's own initial fetch. The previous fourth request from the departed library is absent. |
| Redundant safetensors polling | Removed from the library's initial fetch and polling path. |
| Routine polling while hidden | New visibility/poll lifecycle test passes. |
| Proxy failure looks like a verified stopped model | Backend tests verify explicit `unreachable` and `stale` quality, preserved running state, and the last successful timestamp. Row/list presentation and the library outage banner consume the new fields. |
| Header remains healthy after refresh failure | New header test passes; the implementation also expires old observations. |
| No return to management login on expiry | Handler/component test passes. Actual cookie expiry and reauthentication across a backend restart were not exercised. |

**Validation**

- Backend: **1,613 passed, 7 skipped**, 67.95 seconds.
- Frontend first full run: **396 passed, 1 failed**. Failure: `ModelConfig.audioProfiles.test.js`, “preserves unknown audio.cpp keys on save”; the expected PUT had not occurred when asserted.
- That audio test file passed independently: **14 passed**. A second full frontend run passed: **397 passed across 42 files**, 30.45 seconds. Treat the first failure as unresolved intermittent test behavior, not a confirmed production save defect or a consistently clean test run.
- Production build passed in **1.07 seconds**.
- Chromium fixtures verified selective apply, legacy apply visibility, and the original navigation/polling reproduction. Sampled library/Search/Engines/Audio screens had no uncaught page errors.
- `git diff --check` passed for the application patch.

Checks used the existing Python 3.12.3 / Node 20.20.2 environment. Supported Node 24 clean installation, Docker startup, actual inference, and real proxy/network lifecycle behavior remain outside this verification. The browser used controlled API responses and did not restart real models.

Fix the three findings before marking the first reliability milestone complete. The broader onboarding, durable-history retention, catalog/storage efficiency, compression, client handoff, architecture, and release-harness work remains in the original plan.

Raw evidence is retained locally in `/tmp/studio-fixes-{backend,frontend,frontend-rerun,audio-rerun,build,browser,poll}.log`, `/tmp/studio-fixes-browser.cjs`, and `/tmp/studio-fixes-assessment-browser.json`. These temporary files are supplementary; the reproduction steps and conclusions are recorded above.
