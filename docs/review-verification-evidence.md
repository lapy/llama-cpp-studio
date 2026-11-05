# Verification evidence

This is the supporting evidence for [delivery map](delivery-map.md) and
[change inventory](change-inventory.md). Counts and commands belong here; the
delivery map states outcomes.

## Environment

- Verification ran in the project's Linux dev container
  `llama-cpp-studio-dev` (repo bind-mounted at `/app`), Python 3.12.3.
- The suite ran as the non-root `appuser`. This matters: one Rust migration test
  asserts read-only-directory semantics that root bypasses, so a root run
  reports a false failure for
  `test_onecat_vllm_source_build.py::test_build_env_moves_readonly_rustup_home`.
- Frontend gate: Node 24.22.2, `node_modules` from the container.
- The Windows host was used for editing and git only. The suite is not
  supported there (POSIX `/proc` and permission semantics); the Windows run was
  stopped once it hit those platform-only failures.

## Exact revision tested

- Base: `7e0c7e2`.
- Frozen checkpoint: `1fd002e` (`checkpoint/engine-definitions-2026-11-04`).
- Review series tip: `HEAD` of `release/engine-definitions` (the commit that
  ships this document).
- Assembled-tree check performed: `git diff --stat review-tree` was empty after
  the series was built. The `review-tree` tag points at the frozen checkpoint
  plus the review-only documents and the two focused tests; see
  [change inventory](change-inventory.md) for the reconstruction.
- Per-commit import check: each of the eight commits was checked out in a
  worktree and imported `backend.main` successfully inside the dev container, so
  the dependency order does not produce an unimportable intermediate tree.

## Suites

### Backend

```
docker exec -u appuser -w /app llama-cpp-studio-dev \
  python -m pytest backend/tests -q -p no:cacheprovider --basetemp=/tmp/appuser-final
```

Result: **2169 passed, 8 skipped** in 245.15 s, exit 0 (final review
`release/engine-definitions` tip). This includes the review follow-up tests for
pending snapshots on failed installs (venv, native and audio source/sync),
installation-identity vs package-version separation, transform-asset digest
coverage, snapshot read failure, removal pruning, cancellation after publish and
per-attempt publication tracking.

The eight skips are pre-existing platform/environment skips, not tests added by
this series.

### Frontend

```
docker exec -u appuser -w /app llama-cpp-studio-dev npm run test:frontend:gate
```

Result: **CLEAN — exit 0, 75 test files, 655 tests passed, no unhandled
errors** in 58.51 s (final review tip). This includes the activation, structured
error and retained-reference confirmation tests for the generic panel.

### Focused tests added by this series

```
python -m pytest backend/tests/test_pre_refactor_installation_smoke.py \
                 backend/tests/test_nativecraft_journey.py -q
```

Result: **6 passed** (5 upgrade-smoke tests, 1 third-fork journey).

- The upgrade smoke writes a pre-refactor `engines.yaml` by hand (no
  `schema_version`, no snapshot ledger), then checks listing, generic/compat
  status equality, activation, launch from the old venv, removal, and retry of a
  snapshot-less install.
- The third-fork journey runs the real recipe executor, checkout, CMake
  configure/build dispatch (fake POSIX toolchain), binary location, publication
  row creation, and GGUF launch compilation against the fixture's declared
  `native-server` layout. It also asserts the published snapshot is
  version-associated and that a changed definition refuses the retry, plus the
  BLAS policy suppressing `GGML_BLAS` flags.

### Review follow-up tests

- **Install/retry/removal:** snapshot read failure raises instead of disabling
  the guard; explicit version removal prunes only that version's snapshots;
  native source publication records a version-associated snapshot; cancellation
  after publish does not fail a ready/activated row.
- **Frontend:** the generic panel activates an inactive version through the
  shared version route, renders structured 409 details as their message, and
  offers the retirement confirmation when removal is blocked by a launch
  reference; the build-options compatibility wrapper is asserted to return its
  payload.

## Static gates

All run in the dev container against the assembled tree:

| Gate | Command | Result |
| --- | --- | --- |
| Python lint | `npm run lint:python` | pass |
| Python format (curated list) | `npm run format:python:check` | 114 files formatted |
| Frontend lint | `npm run lint:frontend` | pass |
| Engine literal ratchet | `npm run lint:engine-literals` | no growth beyond committed inventory |
| Build defaults | `npm run build-defaults:check` | current |
| Legacy route inventory | `npm run legacy-routes:check` | current |
| Version | `npm run check:version` | `1.0.0` |
| Engine parity inventory | `npm run engine-parity:check` | current |
| OpenAPI | `npm run openapi:check` | current |
| Prettier | `npm run format:check` | pass |
| Stylelint | `npm run style:check` | pass |
| API types | `npm run typecheck:api` | pass |

## Gate defects found while assembling this revision

These were baseline drift in the uncommitted tree, not new product changes.
They are repaired in the owning commits so the final gates run clean:

1. `scripts/git-hooks/pre-commit` listed two tests that do not exist
   (`test_sglang_legacy_plan_shadow.py`, `test_onecat_legacy_plan_shadow.py`).
2. Four listed Python files were not `ruff format` clean; one carried a UTF-8
   BOM that made the formatter panic instead of reporting the diff.
3. `docs/legacy-route-inventory.md` was stale after the new upgrade smoke test
   was added to the LMDeploy coverage row.
4. `docs/engine-parity-inventory.json` was stale for the same reason.
5. `GenericEnginePanel.vue` used `currentColor` where the stylelint config
   requires `currentcolor` (CSS keyword is case-insensitive; no visual change).

## Defect found by the strengthened third-fork journey

`backend/engines/definitions/nativecraft.yaml` declared the `artifact.locate`
output as `native_path`, while the shared handler produces `server_path`. The
fixture's source recipe therefore failed with "step did not produce declared
outputs ['native_path']" and could never have built anything. The declaration
now produces/requires `server_path`, and the fixture gained the GGUF runtime
profile the launch half of the journey needs.

This is a test-asset correction, not a shipped-engine behaviour change. It is
recorded because the previous proof only instantiated the shared class and
asserted attributes, which is exactly the weak proof the review asked to
replace.

## Hardware evidence

Preserved from the campaign pass and **not rerun on this revision** (the
workspace has no GPU):

- A real llama.cpp source build on a remote Tesla V100-SXM2-32GB with CUDA 12.8
  through the shared `CmakeSourceInstaller`, producing
  `build/bin/llama-server` (build 10630, commit 72190d715).
- The declared GGUF profile started that server; `/health` passed and
  `/v1/chat/completions` returned a real completion (about 226 tok/s generated).

Details: hardware verification evidence maintained privately outside the tracked documentation.

Coverage limit: **native llama.cpp only**. Unsloth release assets, LMDeploy,
vLLM, 1Cat-vLLM, SGLang, SGLang V100 and audio have no representative-hardware
evidence. Their tests use fixtures and fake toolchains.

## Explicit non-goals of this verification

- No CUDA compilation or real inference for venv/audio families in this
  workspace (no GPU).
- No Playwright browser E2E run; the frontend gate is the Vitest suite.
- No multi-day soak or restart-under-load test.
- No test of a second audio engine on hardware; the capability is proven by
  declaration/identity tests, not a second shipped engine.
