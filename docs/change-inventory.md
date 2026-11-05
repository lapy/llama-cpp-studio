# Change inventory and commit boundaries

Source of truth: the frozen checkpoint `1fd002e`
(`checkpoint/engine-definitions-2026-11-04`) on top of `main` at `7e0c7e2`.
The review branch is reconstructed from that tree; the final commit must have a
tree identical to the checkpoint plus the review documents and two focused
tests described below.

Totals: **279 changed paths** in the checkpoint (125 tracked edits/deletions,
154 added, including 59 backend test modules plus a harness and 3 fixture
files), split into eight dependent commits. Six review-added files land in their
owning commit: two tests and four documents. One fixture (`nativecraft.yaml`) is
deliberately touched twice — it first lands with the definition foundation and
is corrected by the launch commit — so the commit entries below sum to 286 for
285 unique paths.

## Commit series

| # | Commit | Scope | Paths | Depends on |
| --- | --- | --- | --- | --- |
| 1 | `chore(dev): add reproducible development container` | Dev container, compose file, dockerignore, dev init, pre-commit format list repair | 5 | — |
| 2 | `feat(engines): make validated YAML definitions authoritative` | Schema, loader, bootstrap, startup gate, nine definitions + four fixtures, descriptor contracts, declaration tests | 53 | 1 |
| 3 | `refactor(engines): share install hosts, primitives and retry snapshots` | `python/` hosts, native build runner, snapshot ledger, audio identity, llama/audio installers, shared audio defaults, host tests | 51 | 2 |
| 4 | `refactor(engines): serve lifecycle from definitions and thin compatibility delegates` | Specialised recipes/facades, `LIFECYCLE` + generic router, compat prefixes, startup wiring, retry/removal tests, pre-refactor upgrade smoke | 57 | 3 |
| 5 | `refactor(launch): compile declared profiles and the GGUF family` | Profile compiler, GGUF mechanism, launch revisions/digests, scanner, native route, launch parity/characterisation tests, nativecraft journey and fixture correction | 30 | 3 |
| 6 | `feat(api): expose descriptors and keyed engine state` | Descriptor endpoint, generated OpenAPI, frontend registry/store/test fixtures, API inventory gates | 23 | 4, 5 |
| 7 | `refactor(ui): render engines from descriptors and shared forms` | Engine panels, dialogs, forms, views, e2e selectors, generated build defaults, package scripts | 44 | 6 |
| 8 | `docs: replace campaign history with delivery map and migration notes` | Delivery map, this inventory, verification evidence, migration notes, campaign-doc banners, obsolete-test/tooling deletions, dead-branch audit | 23 | 1–7 |

The order is a dependency order, not just the requested grouping:

- 2 must precede 3 because the hosts import the definition API.
- 3 must precede 4 because the specialised recipes and routes call the shared
  hosts and the snapshot helpers.
- 4 and 5 are siblings: both depend on 3, and neither imports the other. In the
  chosen order, 4 registers the lifecycle records that 5's launcher can read.
- 6 depends on the API/descriptors from 4 and the launch surface from 5.
- 7 depends on the store/registry contract from 6.
- 8 is documentation and deletion only; it contains no correctness-critical
  production change.

## Deletions and their replacements

| Deleted | Replacement in the same commit | Notes |
| --- | --- | --- |
| `backend/engines/llama_cpp/source_build.py` (1,702 lines) | `backend/engines/python/cmake_source.py` (commit 3) | Git detects a 70% rename; the file is the shared `CmakeSourceInstaller` |
| `test_audio_align_profiles.py`, `test_audio_analysis_profiles.py`, `test_audio_gen_profiles.py`, `test_audio_sep_profiles.py`, `test_audio_vc_profiles.py` | Unified `test_audio_task_profiles.py` + `audio_profile_fixtures.py` (commit 3) | Per-family copies replaced by one parameterised contract |
| `test_audio_cpp_ui_removed.py` | None (commit 8) | Guard for a UI surface removed long before this branch; obsolete |
| `test_models_routes_architecture_contract.py` | None (commit 8) | Guard against a duplicated models router that the route split removed; no remaining subject |
| `test_proxy_parity_harness.py`, `scripts/proxy_parity.py` | None (commit 8) | Legacy parity harness superseded by `scripts/parity-inventory.py` (commit 6) |

`test_app_smoke.py` loses only a permanently skipped SSE test; it stays with the
cleanup commit.

## Test placement

Tests stay with the behaviour they verify:

- Definition/schema/descriptor declarations → commit 2.
- Shared host mechanics, snapshots, retry guard, audio identity → commit 3.
- Specialised recipes, lifecycle routes, compatibility delegates, removal and
  the pre-refactor upgrade smoke → commit 4.
- Profile/GGUF compilation, discovery, launch parity and the nativecraft
  journey → commit 5.
- Descriptor/keyed-store/server-contract tests → commit 6.
- Component/view/e2e tests → commit 7.
- Only obsolete guards and the architecture audit → commit 8.

## Boundary calls worth reviewing

- **`backend/main.py` is in commit 4, not 2.** It wires startup definition
  verification and data directories (foundation), but also registers lifecycle
  records and mounts the generic/audio routers. Splitting a single file would
  create an intermediate tree that cannot import; the file lands with the
  lifecycle slice. The foundation commit is still importable because the
  registry bootstraps definitions at import on its own.
- **`backend/engines/adapters.py` is in commit 3, not 2.** It looks like
  registry wiring, but it resolves definition-backed installers and native
  hosts; those modules arrive in commit 3.
- **`backend/routes/engine_versions.py` and the compat route modules are in
  commit 4, with `engine_lifecycle.py`.** Putting the generic router later would
  leave the compatibility delegates importing a module that does not exist yet.
- **`artifacts`/generated API JSON** (`frontend/src/api/openapi.json`,
  `frontend/src/engines/build-defaults.generated.json`) land with the frontend
  commits that regenerate and check them, not with the backend that produces
  them.
- **The nativecraft fixture is touched twice.** It first lands in commit 2 as
  the fixture existed in the checkpoint. Commit 5 applies the recipe-output and
  runtime-profile corrections proven necessary by the new journey test, so the
  failure and the fix are reviewable together. This is why commit 5 shows 30
  path entries for 29 unique files.

## Review follow-ups applied

Findings from the pre-publication review were fixed in their owning commits:

- **Install/retry/removal:** every install attempt records a pending snapshot
  against its pre-publication row name — venv, native release/source and audio
  source/sync — so an explicitly requested version that fails still refuses a
  retry after the definition changed; publication also records the authoritative
  version snapshot. Installation identity is kept out of the resolved package
  version, so a timestamp-pip row name is never passed to pip. The definition
  digest now covers referenced transform-asset content, so an asset-only edit
  changes installation identity. An unreadable snapshot ledger fails closed;
  explicit version removal prunes that version's snapshots; cancellation after
  publish no longer marks a ready/activated row failed, and the publication
  marker is cleared per attempt.
- **Frontend:** YAML-only engines can activate an installed version; structured
  409 details render as their message rather than `[object Object]`; removing an
  engine referenced by a launch generation offers the retirement confirmation;
  reloading mid-operation resumes status polling; the generic card refresh
  includes server-backed engines; `fetchAudioCppBuildOptions` returns the
  catalog payload and its test asserts it.
- **Docs:** one feature PR (not eight stacked PRs) is the stated strategy;
  rollback preserves `install_snapshots.yaml` and starts from a configuration
  backup instead of deleting provenance.

## Packaging decisions

- **One feature PR containing the eight ordered commits**, reviewed
  commit-by-commit and released together. The commits are review units, not
  independently mergeable products or separate releases. Eight stacked PRs are
  not used; switch only if reviewers explicitly need independently mergeable
  changes.
- One feature branch, one release.
- The checkpoint branch is local-only and is not merged.
- The pre-commit hook fix (two stale format paths) is in commit 1 because the
  gate must be runnable by every later commit; it changes no product code.
- Campaign documents are kept but marked historical; the delivery map is the
  entry point.
