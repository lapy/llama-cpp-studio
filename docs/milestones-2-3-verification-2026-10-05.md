# Milestones 2 and 3 verification — 2026-10-05

Reviewed commit `b7c4590` and subsequently the working-tree fixes against the completion criteria in [the project assessment](project-assessment-2026-10-04.md). **The reported blockers are now resolved. Milestones 2 and 3 pass the checked fixture journeys and local regression budgets.** Application source and user data were not changed by this verification.

## Third re-check: reported blockers closed

The latest working-tree fixes pass re-verification:

- **Queue saturation recovery:** the original independent 32-slot saturation probe now leaves `_pending_writes` and resource ownership empty after rejection. Starting a replacement operation after draining succeeds. The new supervisor regression additionally rejects a terminal update, verifies restoration of the running record/resource, and successfully retries that terminal update after draining. Terminal-record eviction and queue-cap tests pass.
- **Browser journey:** the fixture now starts with no installed model, navigates through search/download, rejects the first download and save, and verifies successful retries. It proceeds through configure/apply/start/connect, rejects the first connection test and verifies recovery. Separate cases cover descriptor delay and failure/reload recovery. All three Playwright tests pass at the mobile viewport. This closes the specific missing installation and mutation/connection recovery coverage reported previously; it is representative fixture coverage, not every failure permutation or live engine installation.
- **Transfer budget:** the Node and Python asset walkers now include the route preload graph, library CSS and icon font. The production build totals **229,523 gzip bytes across 12 files**, below the **250,000-byte ceiling**. This is consistent with the prior browser-observed asset set plus HTML/favicon, and closes the identified omission. It remains a compressed build-asset budget, not a full HTTP-wire or API-payload measurement. The documentation correctly distinguishes regression ceilings from baseline speedup claims.
- **Earlier defects:** the independent cache-interleaving and HTTP encoding probes remain fixed. Background persistence retained operation metadata and produced an 11.1 ms maximum heartbeat gap during a queued 300 ms write. Installer provenance and its API regression pass in the backend suite.

| Latest validation | Result |
| --- | --- |
| Full backend suite | **1,668 passed, 7 skipped**, 114.39 s |
| Frontend suite | **415 passed**, 32.45 s |
| Chromium fixture journeys | **3 passed**, 11.9 s |
| Production build | Passed, 1.02 s |
| Cold Models asset budget | **229,523 / 250,000 bytes** |
| `git diff --check` | Passed |

No remaining blocker was reproduced within the reviewed scope. The changes are still uncommitted on top of `b7c4590`. Live GPU/inference behavior and remote CI were not validated; local Node is still 20.20.2 rather than the declared/CI Node 24 target. Those limits should not be confused with failed milestone checks. Logs are `/tmp/studio-m23-recheck3-{backend,frontend,browser,build,overflow,probes}.log`.

The following sections preserve review history. Their open findings and incomplete verdicts are superseded by this third re-check.

## Second re-check: browser harness, budgets and bounded persistence

The latest working tree adds a Playwright/CI harness, performance ceilings, a 32-write queue limit and terminal-record eviction. These address the previous omissions, but do not yet close both milestones.

### P1 — Queue rejection leaves an operation's resource reserved

`backend/operations/supervisor.py:319` changes `_records` and increments `_pending_writes` before `run_store(write)` at line 341 can reject the submission. `start_operation` also reserves its resource before persistence. When `StoreIoBusy` is raised, the write closure never runs, so its cleanup cannot undo either state change.

Independent reproduction using the real executor and a temporary store:

1. Occupy all 32 persistence slots with writes blocked on a threading event.
2. Call `start_operation('rejected', 'build', 'engine:demo')` on the event loop; it raises `StoreIoBusy`.
3. Release the writes and drain the queue.
4. The pool reports zero pending writes and disk has no operation, but the supervisor retains `{'rejected': 1}` in `_pending_writes` and `{'engine:demo': 'rejected'}` in `_resources`.
5. A new operation targeting that resource raises `ResourceBusyError` even though the queue is empty.

Roll back state and resource ownership when submission is rejected, and test retry after saturation through the supervisor rather than only the queue helper. Also define safe handling for rejected terminal writes, so the persisted operation cannot silently remain running. The new eviction rule preserves pending-write IDs, making a rejected submission an eviction leak too.

### Milestone 2 — Harness exists, but full journey coverage is incomplete

All **three Playwright tests pass** in Chromium at the configured mobile viewport (8.4 seconds). They cover delayed descriptors, descriptor failure/reload recovery, and successful configure/apply/start/connect.

However, `frontend/e2e/model-journey.spec.mjs:225` seeds `installed: true`; the test named “install, configure, apply, start, and connect” performs no installation. Save/apply/start/connect handlers always succeed. Add the actual fixture-controlled install transition and failed/delayed mutation or connection recovery assertions before claiming the entire milestone's acceptance journey. This is a coverage gap, not evidence those flows are broken.

### Milestone 3 — Transfer budget undercounts the navigation

`frontend/e2e/cold-models-budget.mjs:22` and the equivalent Python helper count HTML, entry assets and one level of library JavaScript imports. They omit the library stylesheet, CSS-loaded fonts and potentially deeper dependencies.

The production build passes and the script reports **193,661 gzip bytes across nine files**. A fresh Chromium navigation to `/models` requests assets whose gzip sizes total **228,987 bytes before adding HTML**. The omitted `ModelLibrary-*.css` and `primeicons-*.woff2` account for 35,666 bytes. These are recompressed sizes of browser-observed asset requests, not a measurement of HTTP wire bytes. The observed set remains below 250,000 bytes, but the current gate would miss growth in omitted assets. Use browser-observed production transfers or a complete asset dependency graph, and label a JS/CSS-only budget accordingly if that is the intended scope.

The performance tests otherwise pass, including warm catalog, history cleanup and retained-history ceilings. Their introductory claim that all ceilings are below old costs is inaccurate: the 1,000-model ceiling is 180 ms versus the previously recorded 162.96 ms baseline, with different HTTP-versus-handler measurement boundaries. Treat these as new regression ceilings, not proof every old cost would fail.

Validation: **1,667 backend tests passed, 7 skipped (66.96 seconds); 415 frontend tests passed (32.48 seconds); three browser tests passed; production build passed (1.13 seconds); diff whitespace check passed.** The original independent cache, compression and background-persistence probes still pass. Logs use `/tmp/studio-m23-recheck2-*`; additional probes are `/tmp/studio-m23-overflow.py` and `/tmp/studio-m23-transfer.log`. Local Node remains 20.20.2, below the declared/CI Node 24 target.

**Current verdict:** milestone 2 is closer but needs the missing journey cases; milestone 3 needs saturation recovery and an accurate transfer gate. The sections below are prior verification history, superseded where this section reports newer evidence.

## Re-check of working-tree fixes

Re-ran the complete backend suite: **1,659 passed, 7 skipped in 58.44 seconds**. `git diff --check` passed. The fixes are uncommitted changes on top of `b7c4590`.

- **Cache race closed:** the original independent interleaving probe now reads port 2345 and preserves it after the unrelated update. Reads compare revisions before and after loading, retry replacements, and avoid caching unconfirmed contents. A matching regression is included in `test_catalog_costs.py`.
- **Audio review defect closed for new installations:** the actual installer now stamps download provenance. The new installer-record/API regression confirms `config_reviewed=false` before saving and `true` with a user stamp afterward. Historical engine maps retain legacy treatment. Download services and persisted template application also handle provenance.
- **Compression defect closed:** the original HTTP probe now receives identity for `gzip;q=0, identity;q=1`. Both identity and gzip responses have `Vary: Accept-Encoding`. Added tests cover relative quality, wildcard exclusions, Brotli preference and refusal of all available encodings.
- The independent background-write probe remained responsive (11.2 ms maximum heartbeat gap during a queued 300 ms write), and request-gated writes preserved operation metadata.

**Milestone 2 remains incomplete only against its broader acceptance scope:** a committed browser harness for the full install/configure/apply/start/connect journey and its failure/recovery cases is still absent. **Milestone 3 still needs agreed benchmark/transfer budgets and bounded-resource validation:** executor submissions remain uncapped, and supervisor terminal-record eviction remains unresolved. No changes addressing those gaps were present.

Frontend files were unchanged, so frontend tests, build and browser journeys were not repeated for this backend-only re-check; their previous results below remain historical evidence. Logs: `/tmp/studio-m23-recheck-backend.log` and `/tmp/studio-m23-recheck-probes.log`.

The remainder records the original findings and measurements; the three defect descriptions below are historical and superseded by this re-check.

## Findings that prevent completion

### P1 — Cached documents can overwrite a concurrent writer

In `backend/data_store.py:343`, `_remember_document` obtains the file's stamp *after* `_load_document` has read its contents. Another store/process can replace the file between those operations. The reader then associates old contents with the replacement file's stamp. `_mutate` at line 489 trusts that cache under its interprocess lock and writes the stale contents back, losing the other writer's change.

Deterministic reproduction with two real `DataStore` instances sharing a temporary directory:

1. Wrap reader A's `_load_document`: call the original loader, then have writer B set `proxy_port=2345`, then return the original loaded contents.
2. Call A's `get_settings()`. A returns port 2000 while B and disk contain 2345.
3. Have A update the unrelated `public_inference_url` setting.
4. A fresh store now reads port **2000**: B's committed change was lost.

This is a controlled interleaving of actual storage operations, not a mocked cache result. The existing sequential cache-invalidation test does not exercise it. Bind cached contents to the revision actually read, validate/retry changes during reads, and preserve the mutation lock guarantees. Add this interleaving as a regression before closing milestone 3.

### P2 — Installing an audio model incorrectly counts as configuration review

`backend/routes/models.py:266` considers any config containing an `engines` dictionary reviewed. The audio installer automatically constructs that dictionary in `backend/services/audio_model_installer.py:1234` and returns it without a `config_reviewed_at` stamp.

Passing that installer-shaped normalized config to `_config_was_reviewed` returns **true** with no user save. A mobile Chromium fixture using the resulting catalog flag displays **“Next: Start the model”**, skipping review. The existing test covers a simpler legacy download record, not this real installer shape.

Use explicit review provenance for new downloads. If historical engine maps must remain compatible, distinguish migrated records from newly installed defaults. Cover an actual audio installation record and its subsequent config save. This violates milestone 2's explicit requirement that downloading does not imply review.

### P2 — Static compression ignores rejected encodings

`backend/static_assets.py:20` strips quality parameters and treats every encoding token as accepted. An HTTP request with `Accept-Encoding: gzip;q=0, identity;q=1` receives **Content-Encoding: gzip**. Identity responses also omit `Vary: Accept-Encoding` at line 107, despite serving alternate representations at the same URL.

Honor quality values, exclusions and wildcard negotiation; emit the appropriate Vary header on identity and compressed responses. Add HTTP regressions for these cases before closing milestone 3.

## Milestone 2 evidence

Implemented and verified:

- Real Chromium direct entry to `/models` at 390×844 with each of the nine registered engine IDs, including Unsloth. Runnable descriptors advanced to “Download a model”; an installed but non-runnable descriptor stayed at engine preparation. No horizontal overflow or page JavaScript errors in these fixtures.
- A delayed descriptor response withheld the checklist until it resolved. A failed response left the engine step incomplete; a page reload after recovery advanced correctly. Failure currently looks like missing readiness rather than an explicit descriptor-loading error.
- Compact onboarding and capability-aware connection construction have passing component/unit tests. Backend tests cover embedding requests, bounded chat requests and public URL settings.
- Browser verification of selective Apply: pending changes showed the model-specific restart action; applying refreshed state, removed that action and showed published/running settings. Legacy pending responses retained the proxy-reload action.

Still required: fix review provenance and commit browser journeys covering the complete install/configure/apply/start/connect sequence with success, delay, failure and recovery. No committed browser runner was found; the current CI runs pytest/Vitest. This review's temporary browser fixtures establish selected behavior, not the entire milestone's acceptance journey. Real installation, GPU launch and inference were not exercised.

## Milestone 3 evidence

The suite covers durable retention across restart, preservation of active work, a 200-record recent terminal cap, failed-write cache safety, compact catalog fields, off-loop reads/writes, transport reuse and ordinary gzip serving.

Independent probes confirm background persistence no longer stalls the event loop: a queued 300 ms write produced a maximum 11.3 ms gap on a 10 ms heartbeat. A start-and-finish pair inside one request, with deliberately delayed writes, retained its operation kind, resource and detail. The earlier helper-test failure and background persistence issues are resolved in this snapshot.

Matched local benchmarks compared original commit `2323179` with `b7c4590`, using the same interpreter, synthetic records and 15 calls per size. Catalog timing includes handler execution and JSON serialization; the proxy lookup is stubbed. Each model has its own group and a small config containing engine, context size and GPU-layer count.

| Measurement | Original | Current |
| --- | ---: | ---: |
| 10-model median | 1.06 ms | 0.68 ms |
| 100-model median | 7.08 ms | 3.10 ms |
| 1,000-model median | 162.96 ms | 32.84 ms |
| 1,000-model first call | 139.53 ms | 158.35 ms |
| 1,000-model maximum heartbeat gap, 5 ms target | 185.78 ms | 91.55 ms |
| 1,000-model JSON bytes | 1,160,010 | 1,304,010 |
| Repeated operation upsert after seeding 10,000 successes, median | 1,349.00 ms | 4.25 ms |
| First upsert after that seed | 1,318.61 ms | 1,074.24 ms |
| Durable records after that upsert | 10,001 | 1 |

Warm-read and retained-history gains are substantial. The initial history cleanup still costs roughly a second. Cold catalog latency did not improve in this run, and the small-config catalog payload grew about 12.4% because added metadata outweighs removed parameters. Detailed-config omission is tested, but a universal transfer reduction is not established. Timing is indicative local evidence, not an agreed performance budget.

Actual ASGI HTTP serving compressed a repetitive 15,000-byte JS fixture to 84 bytes with gzip and returned the expected encoding header. This verifies the serving path, not cold-browser transfer improvement for the full application. The negotiation defect above remains.

Still required: repair the cache race and encoding negotiation; add repeatable benchmark/budget gates for agreed catalog/history sizes and real cold-navigation transfer. Define persistence backpressure: the executor has one worker, but its submission queue and tracked pending futures have no explicit capacity limit. Also assess eviction of the supervisor's in-memory `_records`, which currently retains terminal entries until explicit forgetting or startup reconciliation. Durable retention alone does not bound those memory costs.

## Validation and limitations

- Backend: **1,655 passed, 7 skipped**, 61.11 seconds.
- Frontend: **415 passed across 47 files**, 32.71 seconds.
- Production build: passed, 1.02 seconds.
- Python 3.12; local Node 20.20.2 is below the declared Node 24 minimum. Supported-version CI should still be required.
- All storage probes used temporary directories. Browser APIs were fixtures; no user models were installed, launched or modified.

Temporary evidence: `/tmp/studio-m23-current-{backend,frontend,build,probes,bench}.log`, `/tmp/studio-m23-baseline-bench.log`, `/tmp/studio-m23-current-browser-apply.log`, `/tmp/studio-m23-journey.log`, and corresponding probe/browser scripts. The baseline benchmark completed its timing measurements before a separate current-only review-provenance check failed because the old commit lacks that helper. Temporary artifacts are not repository dependencies; reproduction details and relevant results are recorded here.

Close milestone 2 after the review-provenance fix and durable journey coverage. Close milestone 3 after persistence safety, negotiation and bounded-resource checks pass, with agreed performance budgets demonstrated.
