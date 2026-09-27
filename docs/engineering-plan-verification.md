# Verification of the first engineering improvement plan

Latest review: 2026-09-27, uncommitted changes on **`ec0c041`**.

**Verdict: all three findings from the preceding review are resolved in the tested snapshot. No new blocking finding was identified in this follow-up delta. This does not mark the entire original engineering plan complete.**

Reviewed an independent copy at `/tmp/llama-final-recheck-4te19eil`. A SHA-256 comparison found no repository-file drift during verification. No application code was changed by the reviewer; only this report was updated.

## Closure of the three findings

| Finding | Change | Independent verification |
| --- | --- | --- |
| P1: Shutdown leaves native-build subprocesses alive | Native and audio command runners now handle `asyncio.CancelledError` by terminating their processes before propagating cancellation. | Started a real native command through `LlamaManager._run_command_streaming`, registered it with the supervisor, and drained it. The task was cancelled and the process was no longer alive. The new regression test also covers a process that ignores SIGTERM. |
| P2: Shutdown misses cancellation cleanup after the worker finishes | The supervisor tracks cleanup tasks independently; managers register their cancellation task, and drain waits for remaining cleanup after draining workers. | Allowed the worker to finish while holding cancellation cleanup behind a gate. Drain remained pending, the manager remained busy, and status stayed `cancelling` until cleanup was released. |
| P2: Studio accepts a proxy-invalid candidate before unloading | Apply now runs the installed proxy's `-validate` against a temporary candidate before unloading. Validation errors, missing binaries and validator execution failures stop the apply. | Used actual pinned llama-swap v260. A valid candidate passed. A candidate using `${PORT}` in its proxy URL without using it in its command was rejected before unload. The local binary archive matched the Dockerfile's pinned SHA-256. |

The ordinary installer-cancellation probe still passes: while a real SIGTERM-resistant process is stopping, the manager remains busy and progress reports `cancelling`. Only after termination/reaping does progress become `cancelled` and the manager become idle.

## Earlier review findings

The fixes established in preceding reviews remain present:

- Durable interrupted operation outcomes are restored into progress snapshots; unsupported automatic requeueing was removed.
- Shared data-root resolution covers the affected stores/managers, and tests force a temporary root.
- Docker installs `requirements.lock`, matching CI's dependency specification.
- Legacy audio comparison uses sidecar payloads rather than generation filenames for the tested equivalent-config case.
- Proxy sidecar revision allocation preserves the working generation across restart and failed publication.
- Proxy apply observes reload success/rejection instead of relying solely on health.
- Publishing runs the complete backend suite.

## Verification evidence

- Full backend suite: **1,547 passed, 15 skipped**, **no warnings reported**, in **99.41 seconds**.
- An isolated rerun of `test_download_gguf_bundle_task_and_projector_task_update_store` passed in 0.42 seconds after the full run temporarily slowed in that area. No failure was reproduced there; the full suite completed successfully.
- Independent real-process and controlled-ordering probes confirmed shutdown cleanup and cancellation ownership.
- Actual llama-swap v260 verified both candidate acceptance and rejection before disruption. No model was launched by the validator probe.
- Frontend files were unchanged and were not retested. The preceding frontend verification passed **373 tests** and the production build.
- No clean dependency installation, Docker image build/startup, GPU inference, live engine installation, or browser workflow was performed.

Evidence files:

- `/tmp/llama-final-backend.log`
- `/tmp/llama-final-isolated.log`
- `/tmp/final_plan_probes.py`
- `/tmp/final_validation_probe.py`
- `/tmp/llama-final-recheck-4te19eil/review-snapshot-hashes.json`

## Remaining original-plan scope

Closing these findings is not evidence that every original workstream is complete. The broader items previously identified remain outside this patch: frontend view decomposition, checked API/task-event contracts, browser workflow coverage, actual container startup and network-boundary verification, and complete lifecycle coverage across all operation types. The dependency-lock change also still needs clean-install/image verification.

The reviewed corrective patch is ready for the next integration checks. Keep those broader acceptance criteria explicitly tracked rather than treating this follow-up review as full production or original-plan sign-off.
