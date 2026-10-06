# Production hardening verification — 2026-10-06

| Slice | Status |
| --- | --- |
| 1. Production validation | **Partial:** Node 24 clean install, 421 frontend tests, build and asset budget pass. Container execution remains blocked. |
| 2. Deterministic asynchronous tests | **Complete for the identified failure:** dropped-event regression and async lifecycle protections verified. |
| 3. Persistence failure and restart recovery | **Complete.** Document save, restart reconciliation, action admission, confirmation handling, and restart ownership are implemented. The action gates are backend-verified by **1,728 passing tests**. They are not browser-verified. Container validation remains an open slice 1 dependency. |
| 4. Structured diagnostic exports | **Complete for the downloadable bundle and footer.** Exported events use stable codes and fixed descriptions. Raw exception text, command lines, request headers, and nested settings are not in the bundle. Raw logs remain outside that claim. |
| 5. Configuration backup and restore | **Complete within the recorded scope.** Journal durability, startup ordering, repeated recovery, and corrupt-journal behavior are in place. The latest backend evidence is **1,746 passed, 7 skipped**. Five Node 24 restore journeys passed. This does not close container validation. |
| 6. Accessibility | **Partial.** Keyboard completion, dialog focus return, error announcements, and 390 px / 200% zoom checks passed in the browser. A manual screen-reader pass was not run. |

Container smoke, UID handling, navigation, and shutdown remain **unexecuted**, not failed application checks. Browser journeys ran later with staged Chromium libraries. The workflow changes support future validation but do not replace container execution.

No image was published.

## Slice 1 — production validation

| Check | Result |
| --- | --- |
| Clean `npm ci` on Node 24.21.0 (npm 11.19.0) | Passed, 11 seconds |
| Frontend unit tests | **421 passed**, 26.07 seconds |
| Production build and compressed asset budget | Passed in 0.95 seconds; **230,166 / 250,000 bytes** |
| Browser journeys | **18 passed**, 52.2 seconds, on Node 24.21.0. The system still has no `libnspr4.so`; Chromium loaded the copy staged at `/tmp/studio-browser-libs/root/usr/lib/x86_64-linux-gnu` |
| Container smoke, UID mismatch, navigation, shutdown | Not run. Docker Desktop’s daemon is not running (`dockerDesktopLinuxEngine` pipe is absent). `docker` is not on the WSL PATH |
| Publish prevented by a failed probe | Not executed. The publish job still smokes and measures before push, and pull requests do not push |

Shell syntax checks and `scripts/container-fixture-test.sh` passed. Those checks only cover script parsing and the host/container uid choice. They are not container execution evidence.

The publish workflow now writes image identity, runtime versions, probe output, navigation samples, and failure logs under `artifacts/container-validation` and uploads that directory after the image job. A failed smoke or navigation step still stops the job before push. Fresh and seeded data directories are aligned to the image user only after a deliberate uid mismatch is shown to be unwritable, using modes `0750` and `0640`.

## Slice 2 — audio upload flake

The full-suite failure was `uploadReferenceAudio` called **0** times at `AudioModelConfig.test.js`. The change handler never ran.

Vue ignores a DOM event whose `_vts` is not newer than the listener’s attach time. Vue Test Utils sets `_vts` to `Date.now() + 1`. Mounting while fake timers are set to a future clock attaches the listener at that future time. Restoring real timers and calling `trigger()` then drops the event. The same shape showed up once in this session as `AudioWorkspace.test.js` “sends HeartMuLa style tags from the music form,” also with **0** calls, before the dispatch correction.

happy-dom overrides `dispatchEvent` on some prototypes and those overrides call the original parent method, so wrapping `EventTarget.prototype` alone does not see the event. `frontend/vitest.setup.js` documents that cause and wraps each override. Do not remove it when changing timers. A test that only passes with real timers, or only when run by itself, is this failure.

A separate component bug was also fixed: an older reference-audio load could replace a newer one, and a load or upload that finished after unmount could still toast.

`still uploads when the change listener was attached under a future clock` uses `trigger()` after that future-clock mount. It failed before the dispatch correction and passes after it. The file also waits until the assets tab is visible and loading text is gone, and it holds list and upload responses until the test releases them.

After that correction, `AudioModelConfig.test.js` passed **20/20** local runs (19 tests each). The clean Node 24 frontend suite above includes that file.

## Slice 3 — persistence failure and restart recovery

Each YAML document is replaced on its own. `settings.yaml`, `models.yaml`, `model_config_templates.yaml`, and `operations.yaml` do not share a commit. A crash between two of those writes can leave one document new and another old. That is not an atomic restore. Multi-document restore is slice 5. It can depend on the guarantees below, and it has to record its own recovery state. The same boundary is stated on `backend/data_store.py`.

`os.replace` is atomic replacement: a reader sees the complete previous document or the complete new one. Syncing the temporary file before the replace makes those new bytes crash-durable before the name changes. Syncing the directory afterward makes the directory entry crash-durable. A failed directory sync does not make the replacement unknown; this process already sees the new file, and a later crash can still lose an unsynced directory entry.

`committed: true` means this process observed the replace return. `committed: false` means it observed that the replace did not happen, so the previous state is unchanged. That previous state may be the absence of the document: a first write that fails before replace leaves no file. `committed: "unknown"` means the outcome could not be established. Callers refresh and do not repeat a side effect until that refresh succeeds.

A missing result file does not prove that a side effect did not occur. `OPERATION_COMPLETION_EVIDENCE` records the evidence available at restart. Every current kind is `unproven`: `build`, `download`, `install`, `install_source`, `remove`, `sync`, `sync_source`, `update`, `runtime_apply`, and `operation`. An active row of an unproven kind becomes `unknown` and is not replayed. A result file that is already present does not turn that active row into `succeeded`. A kind added later stays `unproven` until it is listed there with evidence that the side effect did not happen. Absence from the map is not that proof. `interrupted` stays available for a future kind that can prove the side effect did not occur, and for a row already stored with that status. Launch journals are a separate `phase: interrupted` record and are not this operation status.

`backend/tests/test_persistence_failure.py` uses temporary config directories only. It does not touch repository `data/` and does not use a GPU or a network download.

| Check | Result |
| --- | --- |
| Queue rejection through HTTP start and finish | A full persistence queue returns **503** `STORE_QUEUE_FULL` with `committed: false`. The rejected start holds no resource and writes no operation. After the queue drains, the start succeeds. A rejected finish returns the same 503, leaves the resource owned, and leaves the on-disk status `running`. A second start of that resource is **409**. After the queue drains, the finish is stored and the resource is released. |
| Queued operation write fails at fsync | The start response is **500** `STORE_WRITE_FAILED`, `committed: false`, with no resource held and no operation on disk. A finish that fails the same way is **500**, the resource stays owned, and the on-disk status stays `running`. |
| Disk-full at temp write, fsync, and replace | `PUT /api/settings/inference` returns **500** `STORE_WRITE_FAILED`, `committed: false`, and says the previous state is unchanged, including that a first save wrote no document. A new process still reads the old URL. The response does not include the filesystem path or the raw OS error. |
| Replacement succeeded, acknowledgement failed | The same route returns **500** with `committed: true` and says the document was replaced and acknowledgement failed. A new process reads the new URL. |
| Acknowledged save | A successful settings save is still present when a new process reads the file. |
| Process killed between write, fsync, and replace | The parent waits on a pipe the child writes at the checkpoint, then sends `SIGKILL`. Death before fsync or before replace leaves the previous complete settings document. Death after replace leaves the new complete document. The first settings write, killed before replace, leaves no `settings.yaml` rather than a truncated one. Earlier documents from that startup still parse. |
| Independent edit | A settings write that fails at fsync does not remove a model document written afterward. A new process sees the old settings URL and the new model. |
| Liveness during a blocked write | `GET /api/live` returned **200** `{"live": true}` while another thread was stopped at the settings fsync checkpoint. |
| Killed operation, restart | A child was killed after the `running` row was replaced, during execution, before the terminal replace, and after the terminal replace. The parent then ran reconciliation in a new supervisor. The first three became `unknown`: none of those kinds can prove the side effect did not happen, and a missing result file is not that proof. The replaced terminal row stayed `failed`. No case started the operation again. In-memory reservations were empty afterward. |
| Reconciliation write fails | The durable row stays `running`. `POST /api/operations/reconcile` returns **500** `RECONCILE_UNKNOWN` with `committed: "unknown"`. |
| Save and recovery UI | Connect and model configuration keep the edited values for a full queue and for `committed: false`, explain the pause or the missed save, and retry only from a Try again control. `committed: true` reloads the stored document and does not offer another save until that reload works. If the reload fails, the edits stay and Refresh remains. The activity tray shows the same uncertain recovery when reconciliation cannot be established, including when Refresh itself fails. |

The full backend suite passed after source sync and CUDA uninstall joined the same admission gate: **1728 passed, 7 skipped**, 48.94 seconds. The earlier **1,724** passing run supports the behavior covered before those two paths. The full frontend suite on Node 24.21.0 passed **433** tests in 26.98 seconds.

Document saves now share `classifyPersistenceError`. A full queue or `committed: false` keeps the draft and offers Try again. `committed: true` or `"unknown"` keeps the draft, hides Try again, and holds another save until Refresh succeeds. After a successful refresh the acknowledgement can stay visible and a later save is a new deliberate action.

Covered document saves:

- Connect public URL
- Model configuration save, and apply when the stored change was acknowledged but the follow-up failed
- Configuration template save, apply-and-save, and delete
- Hugging Face token save in the library and in search, and token clear in the library
- Routing document save
- Save-only build settings for llama.cpp, a llama.cpp version, audio.cpp, an audio.cpp version, LMDeploy, 1Cat vLLM, and SGLang

Slice 3’s implementation is complete, including action admission, confirmation handling, and restart ownership. Document save stays closed. The action gates are backend-verified by the **1,728** passing tests. Container validation remains an open slice 1 dependency. Multi-document restore stays in slice 5.

`failed` does not by itself mean the action is safe to retry. `exclusive_action` stores `failed` and releases the reservation only when the exception has `before_side_effect`. A preflight rejection, or a proxy-status error raised before any possible side effect, can do that. A timeout or status error after dispatch has no such flag, so the row stays `unknown` and a later attempt still needs the current confirmation. A row that is already terminal is left as it is.

`effect_started: false` is negative evidence only because the durable change to "may have started" is awaited before the request is sent or the process is launched. A crash after that replacement, including one that happens before the process actually starts, stays unknown and does not authorize replay. A durability failure whose outcome is unknown records that the effect may have started and does not send the request.

A fresh `stopped` observation does not prove an earlier start request cannot still take effect. A verified `running` model is not started again, and a verified `stopped` model is not stopped again. Another attempt needs the earlier one finished, fenced, or confirmed for that operation and its current state token. Consuming that token and reserving the replacement are one `operations.yaml` replacement. A replacement that does not happen (`committed: false`) leaves the previous token valid, so the user can confirm it again. A replacement that lands keeps the new row and invalidates the old token, so a second concurrent request cannot start the same work. If that new row later stays unresolved, recovery uses its new token.

Runtime Apply receipts are tied to the same operation and generation. The launcher uses `pending-launch.json` when its revision matches, so the receipt's launch id is the apply operation id. A start is complete only when that launch id is the live process and the revision is the desired one. A stop is negative only when the journal's prior launch id is still that live process and the pointer is unchanged.

| Action | Completion evidence | Lookup | Safe retry |
| --- | --- | --- | --- |
| Start | The proxy's verified state is `running`. `effect_started: false` means the request was not sent, and only after the true write failed before dispatch. | The operation row, then `GET /running` only when deciding a new attempt. | Safe when the request was never sent. A verified `running` model is not started again. A verified `stopped` model does not by itself allow another start. Otherwise withheld until a confirmation names this operation and its current state token. The earlier attempt is not replayed. |
| Stop | The proxy's verified state is `stopped`. | The same row and running list. | Safe when the request was never sent. Unnecessary when it is verified `stopped`. A model that is still verified `running` or `loading` does not prove the earlier stop request has finished, so that retry stays withheld until the same bound confirmation. |
| Runtime Apply | An unchanged published pointer and, when a stop was only requested, the same prior launch id still alive. A start is complete only when the live receipt's launch id is this operation and its revision is the desired generation. | The apply journal, the published pointer, and launch receipts. The proxy does not have to be up. | Safe only for that negative evidence. A completed result is not applied again. Anything else stays `unknown`. Another apply is rejected until `confirm_operation_id` and `confirm_state` match that journal's current token. A stale token does not match. |
| Build and install | No current kind can prove the build or install did not run after it was started. A saved settings document is not completion. `effect_started: false` is the only negative proof, and only when that false row was durable before the process launch. | The operation row for that resource. A partial directory or a missing result file is not negative proof. | The HTTP handlers reject a second attempt while a row for that resource is active or unknown. A durable not-started row can be retried. An unproven attempt needs `confirm_operation_id` and `confirm_state` for that row; the response explains that prior work may already have happened. Startup does not replay. A new kind stays unproven until it is listed with proof the side effect did not happen. |
| Activate and delete | No current kind can prove the version change did not happen after it was started. | Activate uses `engine:{engine}`, so it overlaps every installation of that engine. Delete uses `engine:{engine}:{install_dir}` and overlaps that checkout plus the engine-wide activate. A different installation path does not overlap the delete key. | A second activate or delete is rejected while that row is active or unknown. A rejection that happens before any side effect is stored as `failed` and can be retried. A failure after the change may have started stays `unknown`. |
| Global Apply and active profile | Same as runtime Apply. Both use `proxy:runtime`, which overlaps every `runtime-apply:{model}` in either direction. | The operation row for that key, plus the apply journal when a model apply is involved. | A model apply is rejected while a global apply or active-profile change is unresolved, and a global apply is rejected while a model apply is unresolved. A preflight rejection before unload or publish is `failed`. A validator timeout, or a proxy status error after dispatch, stays `unknown`. |
| Download and catalog install | No current kind can prove the files were not written after the worker was launched. | Hugging Face work uses `hf:{repo}` or `hf:{repo}:{filename}`. A repo-wide key overlaps every file key in that repo, and a file key overlaps the repo key. Two different files in the same repo do not overlap each other. A package without a repo uses `audio-package:{id}`. A local import uses `audio-import:{path}`. | Both the model-download routes and catalog install admit through that key before the worker starts. Refresh uses the repo key. Projector, MTP, and DFlash use the file key. |
| Source sync | No current kind can prove the checkout was not changed after the worker was launched. | `sync_source` and `update` use `engine:{engine}:{install_dir}`, the same key as a build or delete of that installation. That key overlaps engine-wide activate. A different installation path does not overlap that key. These kinds also depend on `cuda:toolkit`, so a sync is still rejected while any other CUDA-dependent start, or a CUDA install or uninstall, owns the toolkit. | A sync is rejected while a build, update, activate, or delete of that same checkout is active or unknown, and each of those is rejected while the sync is unresolved. Startup does not replay. |
| CUDA uninstall | No current kind can prove the toolkit removal did not run after it was started. | Install and uninstall both use `cuda:toolkit`, not a version subdirectory. `build`, `install`, `install_source`, `sync_source`, and `update` that record an engine depend on that key. `remove` and `activate` do not. | Uninstall is rejected while a CUDA install or any of that dependent work is unresolved, and the reverse. Ownership stays with the uninstall until its termination is verified. |
| Cancel | Cancellation requested is `cancelling`. Verified termination is a later terminal row, written when the worker observes the stop. | The same operation row and its resource key. | A second cancel returns success with `terminated: false` and does not finish the row. The resource stays owned until that verified finish. A restart that cannot establish worker termination leaves the row `cancelling`, reclaims the resource, and does not classify it `unknown` or replay it. |

A stored settings document is still not evidence that a following build, install, sync, or uninstall finished. Those HTTP paths reject a second attempt while the previous one is active or unknown.

Queries and previews need ordinary error and retry handling. They do not commit a document:

- Catalog search
- Command preview
- Update checks

Connect tests are classified by what they actually do. A test that only probes an endpoint is a query. A test that stores a URL or starts a runtime is a document save or a side-effecting action, and follows that contract instead.

The activity dock ignores pointer events except on the toggle and the panel. The recovery banner did not opt back in, so its Refresh control was visible while the footer received the click. The banner now accepts pointer events.

With that correction, `npm run test:browser` on Node 24.21.0 passed **18** journeys in 52.2 seconds, using the staged libraries above. The same Node and libraries later ran four of those journeys again in 8.0 seconds: a rejected save and Try again, a replaced document whose acknowledgement failed, a replaced document whose refresh failed, and `RECONCILE_UNKNOWN` after one reconcile request. Those journeys verify document-save and reconciliation recovery. They do not establish browser coverage of the action gates.

## Slice 4 — structured diagnostic exports

The downloadable bundle is schema version **1**. It copies an explicit field list. Recursive redaction still runs on those permitted strings, including bearer and basic assignments, URL userinfo, and `password=`-style assignments. That redaction is not the export boundary. A nested settings object, a command line, and a request header are omitted even when they contain no pattern the redactor would recognize.

Persistence events in the ring no longer store exception text. `STORE_QUEUE_FULL` is saturation. `STORE_WRITE_FAILED` is a durability failure and may include `phase` (`temp_write`, `fsync`, `replace`, `acknowledge`) and `committed` (`true`, `false`, or `"unknown"`). Any other exception becomes `PERSISTENCE_FAILED` with a fixed description and no exception type or message. The ring stays at 16 events. Proxy health and the running-model observation remain separate fields. This does not make application logs safe to share; the store thread still logs the original exception.

The footer maps those codes to fixed sentences. It does not render `message`, `description`, or `exception_type` from the status payload. A planted secret in those fields stays off the page.

`GET /api/diagnostics/bundle` stays on the management API. In remote mode an unauthenticated request is **401** and a bearer token is accepted. In local mode a non-loopback client is **403**.

After this change the full backend suite passed **1,733 tests, 7 skipped**, in 76.06 seconds. The earlier **1,728** remains the evidence for the action gates. On Node 24.21.0, with the staged Chromium libraries, the footer journey passed in 2.7 seconds: a queue-full count, the uncommitted-save sentence, a separate running-model observation, and no planted exception text.

Document save stays closed. Container smoke, UID handling, navigation, and shutdown remain unexecuted. Docker is still independently blocked.

## Slice 5 — configuration backup and restore

A backup is one JSON document, schema **1**, kind `llama-cpp-studio-config-backup`. It records the application version and the same inclusion and exclusion list the plan requires. Included fields are portable preferences (`public_inference_url`, `proxy_port`), per-model settings keyed by `provider:id`, `huggingface:id`, or `catalog:id`, configuration templates, and routing profiles and selectors. Credentials, weights, reference audio, build artifacts, executable paths, runtime state, operation history, and launch generations are not serialized. The encoded document is limited to 1 MiB.

The four configuration documents are still replaced one at a time. `config_restore.yaml` is the commit record. It holds the pre-import snapshot, the intended documents, the preview revisions, and the plan id. The prepared journal is durable before the first document replacement: the temporary file is synced, `os.replace` returns, and the directory entry is synced. Mode `0600` sets permissions only. A failed directory sync does not start that first replacement.

Startup holds ordinary configuration writes, runs recovery, and releases the hold before later startup writes, including an environment token, and before requests are served. Recovery's own writes do not wait on that hold. A second interruption during rollback is retried and still ends in the pre-import state. A journal that cannot be read, or a terminal journal that does not match the documents, stops recovery, leaves configuration unchanged, and is not deleted. The journal is removed only after the `completed` or `rolled_back` record has been replaced and its directory entry synced.

Apply rechecks the preview revisions under the store lock. A mismatch is `BACKUP_STALE` and writes nothing. Conflicts default to keep-existing. `replace` substitutes one item and does not merge nested values. A model reference with no local row stays unresolved until the request maps it to an existing catalog id or skips it. Import does not create a model, download files, install an engine, or start, stop, or publish a model.

`GET /api/config-backup` exports the document. `POST /api/config-backup/preview` is read-only. `POST /api/config-backup/apply` requires the preview's `plan_id`. `POST /api/config-backup/reconcile` finishes an interrupted restore and does not apply the backup again.

The Restore screen selects a backup, shows additions, replacements, skipped items, and unresolved models, and keeps the saved-versus-running notice visible. Unresolved models are mapped or skipped. Conflicts start as keep-existing. Restore stays disabled until the current preview is applicable. A stale plan asks for a new preview. An uncertain response hides Restore and offers Reconcile, without sending the apply again.

Process-kill checks at journal prepare, after the first document replace, and after the last document replace recovered to the pre-import state or the completed restore. The credential and the model file path stayed in place. A preview whose revisions changed before apply did not overwrite the newer edit. Malformed, oversized, unsupported, credential, and path-bearing backups were rejected without a write. The earlier **1,742** passing run supports the backend work before these transaction guarantees. The full suite then passed **1,746 tests, 7 skipped**, in 71.42 seconds.

On Node 24.21.0, with the staged Chromium libraries, the five restore journeys passed in 8.0 seconds: a backup that cannot be read, a stale preview, mapping an unresolved model, a completed restore, and an uncertain response reconciled without a second apply. Those journeys, with the **1,746 passed, 7 skipped** backend run, are the latest slice 5 evidence. They do not close the container-validation gap.

## Slice 6 — accessibility

Keyboard-only use can preview, confirm, and cancel a restore. Connect and Apply open from the keyboard, trap focus in the dialog, close with Escape, and return focus to the button that opened them. A public URL that is not http or https is announced once in the Connect dialog and stays on the page. Copy actions name the value they copy. The Hugging Face token field is labeled. Download diagnostics is a named footer link. Persistence failures use a status region; proxy and runtime ages stay readable and are not live regions, so polling does not repeat them. Activity announces a task only when it fails or finishes, and hides the updating percentage from the accessibility tree.

At the 390×844 viewport, the restore page does not scroll sideways. At 200% page zoom, Connect, Download diagnostics, the restore heading, and the backup file control remain visible after scrolling.

On Node 24.21.0, with the staged Chromium libraries, three accessibility journeys passed in 8.7 seconds. Focus return also has three unit tests. This is not an accessibility certification.

A manual screen-reader pass was not run. This environment has no Orca, espeak, or speech-dispatcher session, so announcements and focus were checked through browser roles, names, and focus rather than heard speech.

## Still unresolved

- Start Docker and run `scripts/container-smoke.sh` and `scripts/measure-production-navigation.sh` against the image that would be pushed. Record the artifact directory. Do not push to obtain that evidence.
- A supported-environment CI run is still required before slice 1 can close. The local browser run used staged libraries rather than libraries installed on the machine.
