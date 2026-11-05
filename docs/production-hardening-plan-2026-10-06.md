---
name: Production hardening and configuration recovery
overview: Close production validation, stabilize asynchronous tests, verify persistence recovery, simplify diagnostic exports, and then deliver configuration backup/restore and accessibility improvements.
status: proposed
---

# Production hardening and configuration recovery

Plan prepared against `68e093e` and the current working tree. This document authorizes no deployment or implementation by itself. Existing uncommitted application changes remain outside this planning task.

The preceding review closed its reproduced code defects, but did not execute Docker or Node 24 CI locally. An audio-upload test also failed intermittently before passing on rerun. These are validation gaps to resolve, not reasons to repeat the earlier feature work.

## Delivery order

| Slice | Priority | Deliverable | Estimated effort |
| --- | --- | --- | --- |
| 1. Production validation | P0 | Reproducible Node 24/container evidence and release gates | 1–2 days |
| 2. Deterministic asynchronous tests | P0 | Explained and corrected upload-test flakiness | 1–2 days |
| 3. Persistence failure and restart recovery | P0 | API/process-level failure coverage and necessary fixes | 3–5 days |
| 4. Structured diagnostic exports | P1 | Safe error codes and descriptions without raw exception exports | 1–2 days |
| 5. Configuration backup and restore | P1 | Versioned export, preview and recoverable import | 4–6 days |
| 6. Accessibility of core workflows | P1 | Keyboard, focus, announcements and zoom improvements | 2–3 days |

These are planning ranges for one engineer, including focused tests: approximately 12–20 engineering days. Use one reviewable PR per slice; split slice 5 into format/export and preview/import if needed. Start with slices 1–3. Slice 5 depends on the transaction/recovery guarantees from slice 3 and the export policy from slice 4. Accessibility can proceed once the affected workflow behavior is stable.

## 1. Close production validation

Build on `.github/workflows/ci.yml`, `scripts/container-smoke.sh`, and `scripts/measure-production-navigation.sh`. The image job in that workflow builds, tests, and publishes the container.

- Run clean dependency installation, frontend tests, browser journeys and the production build on Node 24. Preserve non-secret tests for fork PRs and explicit skips for secret-dependent checks.
- Exercise the exact loaded image that will be published: fresh writable data volume, no GPU, liveness/readiness, served frontend assets, graceful shutdown, and seeded navigation measurements. Do not rebuild between measurement and push.
- The navigation script now adjusts mount ownership; verify it with deliberately different host/container UIDs rather than assuming that the code change proves portability. Prefer narrowly scoped permissions for the disposable fixture.
- Retain image identity, runtime versions, probe output, navigation samples and failure logs as CI artifacts. Keep local shell syntax checks distinct from container execution evidence.
- Confirm the published image cannot bypass the required smoke and measurement steps, and attestation uses its registry manifest digest.

**Acceptance:** a supported-version CI run is green; failed readiness, failed shutdown and failed navigation prevent push; the fresh and seeded containers work under a UID mismatch; cleanup removes containers and fixture data after success and failure. Required credentials being unavailable is reported as an unresolved execution dependency, never as a pass. Verification must not trigger a release solely to obtain evidence.

## 2. Make asynchronous tests deterministic

Start with `frontend/src/components/audio/AudioModelConfig.test.js` and the component's reference-audio loading/upload behavior.

- Reproduce the intermittent upload failure with recorded test order and relevant loading state. Determine whether it is a component race, test readiness assumption, stale mock or leaked asynchronous work.
- Replace fixed chains of `flushPromises()` with waits for observable prerequisites: the assets tab is rendered, initial loading has settled, and the upload control is usable.
- Control delayed API responses explicitly. Verify an upload performed after readiness, a delayed initial load, and an unmount during pending work.
- Ensure each test releases response gates and removes mounted components, subscriptions and timers. Do not increase global timeouts or use automatic retries to mask failure.

**Acceptance:** the original failure has an explained trigger and a regression that fails before the correction. The affected file passes 20 repeated local runs with varied controlled response ordering, followed by a clean full frontend run on Node 24. Repetition supports confidence; it is not the sole evidence of correctness or a permanent slow CI gate.

## 3. Verify persistence under failure

Cover `backend/data_store.py`, `backend/store_io.py`, `backend/operations/supervisor.py`, and representative save/start/finish API paths.

- Inject disk-full and I/O errors at temporary-file write, fsync and atomic replacement boundaries. Include replacement-success/acknowledgment-failure cases so the API does not promise that every error means nothing was saved.
- Saturate the real bounded queue through API-driven operations. Verify rejected starts release reservations, accepted work remains tracked, and rejected terminal updates can be reconciled or retried without leaving permanent resource ownership.
- Run isolated child processes against temporary data directories. Kill the writer at explicit synchronization points, restart the application, and inspect durable configuration and operation status.
- Exercise concurrent independent edits alongside failures. Check actual files and fresh-process reads, not only in-memory state.
- Define the contract for ambiguous outcomes: refresh/reconcile before retry, show an appropriate message, and avoid duplicate side effects. Preserve the last readable document and useful recovery material.

**Acceptance:** acknowledged saves survive restart; interrupted writes leave a complete old or new document, never truncated state; failed/rejected operations do not become UI successes; active/terminal history agrees after reconciliation; unrelated committed edits survive. Liveness remains responsive during blocked persistence. Tests must not modify repository `data/` or require a GPU/network download.

Before implementing multi-document restore, write down the transaction boundary. Existing per-document atomic writes alone do not establish an atomic restore across settings, model configs and templates.

## 4. Simplify diagnostic exports

Update `backend/diagnostics.py`, persistence-event recording and the diagnostic endpoint/footer contract.

- Store bounded structured events with stable codes such as `STORE_QUEUE_FULL` and `STORE_WRITE_FAILED`, timestamp, operation category and a safe description. Retain the distinction between proxy-health observations and running-model observations.
- Exclude raw exception messages, command lines, request headers and arbitrary nested settings from downloadable diagnostics. Use an explicit field/schema allowlist. Treat log handling separately; do not claim this makes every application log safe to share.
- Keep recursive redaction as defense in depth for permitted string fields, not as the primary export boundary. Include the known bearer/basic, URL-userinfo and assignment cases in regressions.
- Preserve enough information to distinguish saturation, failed durability and stale observations. Add a bundle schema version and keep the ring bounded.

**Acceptance:** synthetic secrets placed in exceptions, nested settings, URLs and headers do not appear in the actual downloaded response. Known event codes map to useful UI descriptions. Unknown exception types produce a safe generic event rather than exposing arbitrary text. Access to the bundle follows the existing management authorization rules.

## 5. Add configuration backup and restore

Keep the first version a **configuration backup**, not a model-file or machine backup.

**Included:** portable application preferences, model configuration templates, and saved per-model settings keyed by stable model references. Record archive schema and application versions. Use a single size-limited JSON document to avoid archive extraction complexity.

**Excluded:** credentials, model weights, reference-audio files, build/install artifacts, machine-specific executable paths, runtime/PID state, operation history and published/running launch generations. Never serialize the data directory wholesale. Clearly state these limits before export and in the document metadata.

- Export a consistent snapshot under the storage coordination boundary. Apply the explicit schema/export policy from slice 4.
- Import begins with server-side validation and a read-only preview showing additions, replacements, skipped entries and unresolved model references. Require explicit mapping for missing models; importing must not download files or install engines.
- Default conflicts to keep-existing. Allow explicit replacement per item; avoid introducing recursive merge rules in version 1. Missing or excluded credential fields must preserve existing credentials.
- Bind the apply request to the exact validated input and current document revisions. If affected state changes after preview, reject the stale plan and request a fresh preview.
- Implement staged writes and durable recovery information for changes spanning multiple documents, with a documented commit/recovery protocol. Create a recoverable pre-import snapshot before applying.
- Restore saved settings only. Publication/restart remains a separate existing Apply action with its normal revision checks. The preview must explain this consequence.

**Acceptance:** supported export/import round-trip preserves included settings; malformed, oversized, unsupported-version and executable/path-injection inputs are rejected without mutation; omitted credentials remain unchanged; stale previews cannot overwrite concurrent edits. Crash tests at each restore phase recover to a consistent pre-import or completed state. Import never automatically launches, stops or reconfigures a running model.

## 6. Improve accessibility of core workflows

Review Models, Configure/Apply, Connect, download status, diagnostics and the new restore preview.

- Verify keyboard access, visible focus, meaningful accessible names and logical order for menus, controls and dialogs.
- Restore focus to the initiating control after dialogs close; provide a sensible fallback when that control disappears. Ensure modal focus containment and Escape behavior match the interaction.
- Announce actionable progress changes and failures through appropriate live regions. Avoid reading every polling update or progress percentage aloud.
- Test at a 390 px viewport and 200% browser zoom. Keep primary actions reachable and prevent clipping or overlapping status content. Adapt comparison/restore tables for narrow views.
- Add focused browser assertions for keyboard paths and accessible names. Include manual screen-reader checks for announcements and focus behavior; automated scans alone cannot establish that these flows work.

**Acceptance:** a keyboard-only user can configure/apply/connect and preview/confirm/cancel a restore; focus returns predictably; errors are announced once and remain discoverable; core workflows remain usable at the target width/zoom. Record browser, screen reader and manual results without claiming a formal accessibility certification.

## Review and completion rules

- For each PR, state the user-visible behavior, failure contract, focused evidence and remaining execution limits. Run broader suites after integration changes; avoid unrelated refactors while fixing these paths.
- Keep CPU fixture journeys separate from live GPU/build validation. Make external requirements explicit and do not label skipped tests as verified behavior.
- Reassess timing ceilings only from repeatable measurements and recorded rationale; do not loosen them merely to make CI pass.
- Update the verification report after each slice with its actual commit, commands/results and unresolved findings. A plan checkbox closes only when its acceptance evidence exists.
- No new engines, database migration, distributed scheduling, automatic credential backup or release publication is included in this plan.

**First implementation step:** establish the supported Node 24/container run and investigate the audio-upload failure. Use those results to refine slice 3 before beginning backup/restore.
