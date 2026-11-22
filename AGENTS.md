# AGENTS.md

Guidance for coding agents working on **Studio** (llama-cpp-studio). Read this
before changing code. It describes the repository's actual boundaries, gates,
and development loop; deeper detail lives in the linked documents.

## What Studio is

Studio is a self-hosted inference workspace with four cooperating parts:

1. A **FastAPI management service** (`backend/main.py`, `backend/routes/`) for
   models, engines, configuration, operations, and the Vue application.
2. A **native inference proxy** (`backend/proxy/`) that owns routing, engine
   process generations, readiness, streaming, access policy, and usage
   accounting.
3. A **Vue 3 frontend** (`frontend/`) built with Vite, Pinia, and PrimeVue.
4. **Declarative engine definitions** (`backend/engines/definitions/*.yaml`)
   plus shared install/build/launch primitives and domain hosts.

Supporting state lives under the Studio data directory (`STUDIO_DATA_DIR`,
`./data` by default, `/app/data` in the container). Configuration and
operational state are user data, not source files.

## Read these first

| Topic                                                 | Document                                                                                                                           |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Setup, ports, engine matrix                           | [`README.md`](README.md)                                                                                                           |
| Module responsibilities, persistence, concurrency, QA | [`docs/current-engineering-status.md`](docs/current-engineering-status.md)                                                         |
| Engine-definition architecture and migration rules    | [`docs/config-driven-engine-refactoring-plan.md`](docs/config-driven-engine-refactoring-plan.md)                                   |
| Engine host/presentation boundaries                   | [`docs/engine-host-and-presentation-boundaries.md`](docs/engine-host-and-presentation-boundaries.md)                               |
| Frontend interaction and validation rules             | [`docs/frontend-standards.md`](docs/frontend-standards.md)                                                                         |
| Isolated remote/standalone agent environments         | [`docs/agent-dev-stack.md`](docs/agent-dev-stack.md), [`docs/agent-dev-standalone.md`](docs/agent-dev-standalone.md)               |
| API and runtime behavior                              | [`docs/api-reference.md`](docs/api-reference.md), [`docs/features.md`](docs/features.md)                                           |
| Proxy operations                                      | [`docs/proxy-operations.md`](docs/proxy-operations.md)                                                                             |
| audio.cpp contract/compatibility                      | [`docs/audio-cpp-discovery.md`](docs/audio-cpp-discovery.md), [`docs/audio-cpp-compatibility.md`](docs/audio-cpp-compatibility.md) |

If this file and a linked document disagree, the linked document's specific
contract wins for its area; update this file when a rule changes.

## Non-negotiable rules

1. **Work in a branch/worktree.** Never commit directly to `main`. Use one
   worktree per concurrent agent; do not edit another agent's worktree.
2. **Engine behavior is definition-first.** Add or change a YAML definition,
   recipe, transform, or runtime profile before adding engine-specific Python
   or frontend branches. New engine-ID dispatch outside `backend/engines/` is a
   ratchet failure. A genuinely new mechanism gets one reusable primitive/host,
   not a new engine class or route.
3. **Do not grow the compatibility surface accidentally.** Preserve existing
   API response shapes, error semantics, launch-spec fields, persisted state,
   and route behavior. Prefer additive changes; update consumers and tests
   together. New legacy routes are forbidden; `legacy-routes:check` enforces
   the inventory.
4. **Never hand-edit generated artifacts.** OpenAPI, build defaults, parity
   inventory, module budgets, and UI review screenshots have generators. Use
   them (see [Generated artifacts](#generated-artifacts)).
5. **Do not grow budgeted modules.** Extract a cohesive unit, then lower the
   budget with `scripts/module-size-ratchet.py --update`. Budgets have no
   slack; stale budgets also fail.
6. **One event loop.** The backend is a single API process with one asyncio
   loop. Never run blocking scans, subprocess probes, catalog assembly, or
   full-document rewrites on it. Use the existing blocking pool and async
   primitives (see [Backend rules](#backend-rules)).
7. **Tests never touch live state or the public network.** `backend/tests/conftest.py`
   isolates `STUDIO_DATA_DIR` and blocks non-local sockets unless a test is
   explicitly marked `network`. Use fixtures, not real downloads or live
   engines.
8. **Never commit secrets, model files, runtime data, logs, tokens, or
   `docs/private/` content.** `.env`, `data/`, logs, databases, model weights,
   and local evidence directories are ignored for a reason.
9. **Do not bypass repository gates.** Branch protection, required checks, and
   admin bypass are owner decisions. If a gate fails, fix the cause or report
   it; do not disable protection or force a merge.

## Repository map

| Path                                                            | Responsibility                                                 |
| --------------------------------------------------------------- | -------------------------------------------------------------- |
| `backend/main.py`                                               | Application lifecycle, route registration, static frontend     |
| `backend/routes/`                                               | Management API routers                                         |
| `backend/engines/`                                              | Engine registry, definitions, installers, scanners, activation |
| `backend/engines/definitions/*.yaml`                            | Authoritative engine definitions (schema version 2)            |
| `backend/engines/definitions/assets/`                           | Versioned declarative transforms for fork installers           |
| `backend/engines/definitions/reference/`                        | Schema falsification fixtures; **not loaded at runtime**       |
| `backend/engines/python/`                                       | Shared generic hosts/primitives (CMake, venv, release assets)  |
| `backend/engines/audio_cpp/`                                    | audio.cpp domain host and integration contract                 |
| `backend/proxy/`                                                | Launch compiler, native process manager, routing, accounting   |
| `backend/operations/`                                           | Durable operation admission, progress, recovery                |
| `backend/data_store.py`, `backend/store_io.py`                  | Configuration persistence and durability                       |
| `backend/model_catalog/`, `backend/services/`                   | Model acquisition, runtime application, benchmarks             |
| `backend/access_policy.py`, `backend/inference_auth.py`         | Management and inference authorization                         |
| `frontend/src/views/`, `components/`, `composables/`, `stores/` | Vue workspaces and state                                       |
| `frontend/src/api/`                                             | Axios client, typed contracts, generated OpenAPI               |
| `frontend/e2e/`                                                 | Playwright browser, accessibility, and layout journeys         |
| `devcontrol/`, `scripts/agent-dev/`                             | Development-only control plane; not in production images       |
| `scripts/`                                                      | Generation, ratchets, smoke tests, container validation        |
| `docs/`                                                         | Behavior, architecture, and evidence documents                 |

## Development environment

### Preferred: Linux dev container

The dev container is built from the runtime stage of the `Dockerfile` with the
repository bind-mounted at `/app`:

```bash
docker compose -f docker-compose.dev.yml up -d --build
docker compose -f docker-compose.dev.yml exec studio-dev bash
```

The container installs `requirements-dev.lock` into its venv and `npm ci` into a
named `node_modules` volume. Ports `5173`, `8080`, `8081`, and `2000` bind to
host loopback. `.devcontainer/` attaches VS Code to the same service.

Two test-specific caveats:

- Run the backend suite as the unprivileged image user:
  ```bash
  docker compose -f docker-compose.dev.yml exec --user appuser studio-dev \
    python -m pytest backend/tests -q
  ```
  A root shell bypasses the permission bits exercised by the rustup-home
  relocation tests.
- Run `npm run build` once before `backend/tests/test_performance_budgets.py`;
  it inspects the production bundle.

For a remote GPU host, use the isolated stack in
[`docs/agent-dev-stack.md`](docs/agent-dev-stack.md). For a single container
without Docker-socket access, use
[`docs/agent-dev-standalone.md`](docs/agent-dev-standalone.md). Neither shares
the live Studio checkout, data root, ports, or token.

### Local (Linux/macOS)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.lock
npm ci
npm run check:runtime
npm run dev:all
```

`npm run check:runtime` enforces **Node.js 24.x** and **Python 3.12**. The npm
backend scripts invoke `python`, so activate the venv first. On Windows, run
Linux tooling in the dev container rather than relying on host Python/Node;
`npm run kill-ports` is the only PowerShell-host-specific helper.

### Isolation and ports

- `STUDIO_DATA_DIR` selects an explicit data root. Tests set it to a temp
  directory; never point it at a live deployment.
- Local development defaults: Vite `5173`, management API `8081`, native proxy
  `2000`.
- Container/production defaults: UI and management API `8080`, native proxy
  `2000`.

### Toolchain pins are intentional

Verify the source of truth in `package.json`, `.github/workflows/ci.yml`, and the
`Dockerfile` before changing any of these. As of the current tree:

- **Node 24** (`>=24 <25`, `packageManager` npm 11.21.0), **Python 3.12**.
- **Playwright `1.64.1`** and the CI browser image
  `mcr.microsoft.com/playwright:v1.64.1-noble` must move together. Playwright
  1.65.1 crashed Firefox in this image; do not bump one without the other and a
  browser CI run.
- **`@vue/test-utils` 2.5.1** is pinned; 2.6 breaks the audio preset edit test.
- **Prettier 3.9.10** is pinned; 3.10 reformats budgeted views and creates
  unrelated line-count churn.
- The shipped image is Ubuntu 26.04 with glibc 2.43 and Python 3.12. CUDA
  12.8/12.9 builds use GCC 12/14; **GCC 15 is rejected by nvcc**. CUDA 13 drops
  Volta/SM70, so V100 paths stay on CUDA 12.
- A glibc/CUDA 12 header shim and host-compiler selection live in
  `backend/cuda_host_compiler.py`, `backend/cuda_glibc_compat.py`, and
  `backend/engines/cuda/glibc_c23_math_compat.h`. Do not remove them or select
  bare `g++` for CUDA builds.

If a dependency or compiler upgrade is necessary, change the pin, update the
lockfile/CI image, and run the affected build plus browser/container evidence.

## Daily commands

Run from the repository root with the venv activated and `node_modules`
installed.

| Goal                           | Command                                                            |
| ------------------------------ | ------------------------------------------------------------------ |
| Runtime check                  | `npm run check:runtime`                                            |
| Backend + frontend dev servers | `npm run dev:all`                                                  |
| Backend suite                  | `python -m pytest backend/tests -q`                                |
| One backend test file          | `python -m pytest backend/tests/test_engine_contracts.py -q`       |
| Last failed backend tests      | `python -m pytest backend/tests -q --lf`                           |
| Frontend unit suite            | `npm run test:frontend`                                            |
| One frontend test file         | `npm run test:frontend -- frontend/src/views/ModelLibrary.test.js` |
| Browser journeys               | `npm run test:browser`                                             |
| Accessibility journeys         | `npm run test:a11y`                                                |
| Python/static aggregate        | `npm run lint`                                                     |
| Frontend lint                  | `npm run lint:frontend`                                            |
| TypeScript API boundary        | `npm run typecheck:api`                                            |
| CSS lint                       | `npm run style:check`                                              |
| Prettier check                 | `npm run format:check`                                             |
| OpenAPI freshness              | `npm run openapi:check`                                            |
| Production build               | `npm run build`                                                    |
| Build-defaults freshness       | `npm run build-defaults:check`                                     |
| Engine parity inventory        | `npm run engine-parity:check`                                      |
| Legacy route inventory         | `npm run legacy-routes:check`                                      |
| Module-size ratchet            | `python scripts/module-size-ratchet.py`                            |
| Version consistency            | `npm run check:version`                                            |

Use narrow tests while iterating, but run the full relevant suite before a PR.
For cross-cutting backend changes, run the full backend suite. For frontend
behavior, run the frontend unit suite and the relevant Playwright spec. For UI
or interaction changes, also run the browser suite and accessibility checks.

## Backend rules

### Request path and concurrency

- Routes are `async def`; do not call synchronous HTTP, `subprocess.run`,
  filesystem scans, or expensive model/catalog work on the event loop.
- Use `backend.concurrency`'s blocking pool (`run_blocking`/existing helpers)
  for blocking work, and existing async subprocess/streaming helpers for
  builds and logs.
- Long-running or mutating work must use the operation admission/progress
  machinery in `backend/operations/`; do not invent a background thread or a
  fire-and-forget task for it.
- Cancellation must propagate to child processes/process groups; use the
  existing cancellation primitives.

### Persistence and state

- Read/write configuration only through `backend/data_store.py` /
  `backend/store_io.py` and the repository's store APIs. Do not write YAML
  documents directly or repair live state by editing files.
- Preserve the distinction between desired configuration, published launch
  revision, and verified running revision. Do not collapse them.
- Schema changes need migration/backward-compatibility behavior for existing
  installations and version rows. Never make an existing row unreadable.
- Absolute paths in persisted state are user data; validate them, do not
  normalize them into the repository or tests.

### API and contracts

- Follow existing router, Pydantic, error, and response conventions.
- Management authorization and inference authorization are separate. Do not
  expose management operations through inference credentials or vice versa.
- When a Pydantic/OpenAPI-visible contract changes, export the schema:
  ```bash
  npm run openapi:export
  npm run openapi:check
  ```
  Commit `frontend/src/api/openapi.json` together with the code. The pre-commit
  hook may update it automatically only when backend files are fully staged.
- Do not add a bespoke route per engine when a generic/definition-derived
  route exists. Extend the shared router or declare a typed extension.

### Engine code boundaries

- `backend/engines/registry.py` is a **derived view** of validated YAML
  definitions. Do not add a parallel literal registry.
- Engine-specific behavior belongs in:
  - the YAML definition;
  - `backend/engines/definitions/assets/<engine>/` for versioned transforms;
  - a domain package (`backend/engines/audio_cpp/`) or a declared generic host
    for genuine algorithms.
- Do not add engine-ID allowlists, `if engine == ...` branches, or new
  hardcoded route prefixes outside `backend/engines/`. The engine-literal
  ratchet freezes the remaining compatibility sites; growth requires an
  explicit, reviewed inventory regeneration.
- The reference definitions under `backend/engines/definitions/reference/` are
  shadow fixtures, not runtime definitions.

## Engine changes

For a new engine or an installation/runtime behavior change:

1. **Choose the closest existing definition** and inherit the relevant
   templates (`_templates.yaml`, `extends`) instead of copying unrelated code.
2. **Declare** identity, install kind, release/source recipes, artifact
   handling, runtime profiles, parameter discovery, launch policy, settings
   defaults, data directories, and presentation `ui_kind` as applicable.
3. **Reuse primitives** from `backend/engines/definitions/primitives.py` and the
   shared hosts. Add one reusable primitive only when the mechanism does not
   exist; never a Python import hook or embedded shell script in YAML.
4. **Express fork rewrite steps as versioned transform assets** under
   `definitions/assets/<engine>/`, with applicability checks and descriptive
   errors. Preserve dependency checkouts and retry/sync semantics.
5. **Respect launch profiles.** Runtime JIT and engine servers often invoke
   `nvcc`, `cc`, or `git` directly; put required compiler/environment defaults
   in the profile's `env.defaults`, not only in the install script. Keep
   per-engine environment scoped; do not set global process variables that
   leak into other engines.
6. **Test** at the layers you changed:
   - `backend/tests/test_engine_definitions_*` and schema/loader tests;
   - recipe plan/argv/env and fake-process tests;
   - profile-launch parity tests (`test_*_profile_launch_parity.py`);
   - lifecycle/activation/update tests;
   - `npm run test:backend:engine` for the definition/launch suite.
7. **Regenerate** `docs/engine-parity-inventory.json`/`.md` with
   `python scripts/parity-inventory.py`, then run the literal and module
   ratchets. A new literal site is a reviewed diff, not an accident.
8. **Document** user-visible behavior in `README.md`, `docs/features.md`, and
   the relevant engine/compatibility docs.
9. **Validate on real hardware** for the build/inference path you touched.
   Mocked toolchains and CPU fixtures do not establish CUDA/ROCm/GPU behavior;
   state exactly what hardware was used and what remains unverified.

## Frontend rules

- Vue 3 SFCs with the existing `<script setup>`/component conventions; use
  PrimeVue components and the shared design tokens/styles.
- Network calls go through the shared Axios client and API modules/stores. Do
  not scatter raw `fetch`/Axios calls through views or components.
- Keep API-facing TypeScript in `frontend/src/api/configuration.ts` and the
  generated contract in sync. Do not hand-edit `openapi.json`.
- Preserve accessibility and interaction contracts in
  [`docs/frontend-standards.md`](docs/frontend-standards.md): accessible names,
  keyboard/tab behavior, focus return, dialogs, live-region announcements,
  reduced motion, responsive reflow, and both themes.
- URL state may contain view metadata (tab, filter, pagination, selected
  identifier). Never put secrets, request bodies, proxy settings, or
  credentials in URLs.
- Do not create a second state path: use the existing Pinia stores/composables
  and their persistence/error semantics.
- Tests are colocated `*.test.js` files and use Vitest + happy-dom. Mock at the
  API/client boundary with fixtures; do not depend on real timers, network, or
  a live backend.
- Playwright journeys use fixture data. For UI changes, run the relevant spec
  and `npm run test:a11y`; a scan alone does not prove keyboard/screen-reader
  behavior.

## Generated artifacts

| Artifact                                             | Generator / check                                                  | Rule                                                                                              |
| ---------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `frontend/src/api/openapi.json`                      | `npm run openapi:export` / `openapi:check`                         | Regenerate with backend contract changes                                                          |
| `frontend/src/engines/build-defaults.generated.json` | `python scripts/export-build-defaults.py` / `build-defaults:check` | Regenerate when llama build defaults change                                                       |
| `docs/engine-parity-inventory.{json,md}`             | `python scripts/parity-inventory.py` / `engine-parity:check`       | Regenerate after behavior migration or literal growth                                             |
| `scripts/module-size-budgets.json`                   | `python scripts/module-size-ratchet.py --update`                   | Only after extracting code; never raise a budget to fit growth                                    |
| `docs/ui-review/*.png`                               | `frontend/e2e/ui-refactor.spec.mjs`                                | Browser-generated; do not hand-edit. Commit only intentional UI refreshes, never incidental churn |
| `VERSION`, package/backend metadata                  | `npm run check:version`                                            | Bump only for a release the owner requested                                                       |

`scripts/legacy-route-inventory.py --check` freezes the compatibility-route
inventory. Do not add a new legacy route to make a check pass.

## Git and pull requests

- Branch from current `origin/main`; name branches by intent (for example
  `fix/audio-cpp-prebuilt-safety`, `feat/engine-runtime-profile`).
- Use conventional commit prefixes matching the history (`fix(scope):`,
  `feat(scope):`, `build:`, `ci:`, `docs:`).
- Keep the diff focused. Do not mix formatting-only changes, dependency
  upgrades, and behavior changes unless they are the same reviewable unit.
- Run the relevant checks before pushing. A PR is not ready while a required
  check is red, even if the failure is unrelated; report it explicitly.
- PR body: what changed, why, exact commands/results, risks, and hardware
  validation (or what was not run). Link the issue when one exists.
- The repository enforces linear history and squash merges. Rebase on
  `origin/main` instead of adding merge commits. Do not force-push over another
  agent's branch or after review without coordination.
- Required checks on `main`: `dependency-audit`, `shell-syntax`, `backend`,
  `frontend`, `browser (chromium)`, `browser (firefox)`. The `image` job runs
  on non-PR pushes/tags, not on PRs, so a Dockerfile change needs local
  build/smoke evidence in addition to PR CI.
- Dependency PRs are not automatically safe. Current intentional holds:
  `@vue/test-utils` 2.5.1, Playwright/`playwright-core` 1.64.1, Prettier 3.9.10.
  Update the package and its matching CI image/pins together and run the
  affected suites.

### Before opening a PR

```bash
npm run check:runtime
python -m pytest backend/tests -q
npm run test:frontend
npm run lint
npm run typecheck:api
npm run style:check
npm run format:check
npm run openapi:check
npm run build-defaults:check
npm run engine-parity:check
npm run legacy-routes:check
npm run check:version
```

Add `npm run build`, `npm run test:browser`, and `npm run test:a11y` for
frontend/UI changes. For backend changes, include the targeted tests that prove
the new behavior. For engine work, include the definition/launch suite and
state the hardware/engine version used.

## Definition of done

- [ ] Behavior is implemented in the right layer; no new engine-ID dispatch.
- [ ] Existing API/launch/persistence contracts are preserved or explicitly
      migrated, with consumers and tests updated.
- [ ] Targeted tests and the full relevant suite pass.
- [ ] Generated artifacts are regenerated; no unrelated drift.
- [ ] Ratchets (`engine-literals`, `module-size`, `legacy-routes`,
      `build-defaults`, `openapi`, `version`) pass.
- [ ] User-facing docs are updated for behavior changes.
- [ ] No secrets, models, runtime data, logs, caches, or local evidence are
      staged.
- [ ] PR body records commands/results, hardware scope, and known limits.
- [ ] CI is green; image changes have separate local image/smoke evidence.

## What validation does and does not prove

- Unit tests, browser tests, and container smoke tests establish software
  behavior. They do **not** establish GPU throughput, driver compatibility, or
  every engine/model/device combination.
- CPU fixtures do not exercise CUDA/TensorRT/ROCm kernels. Mocked toolchains do
  not prove a real build.
- Accessibility scans are a starting point; manual keyboard/screen-reader
  journeys are still required for interaction claims.
- A successful production build does not prove that a model will load or infer
  correctly on a given engine.

When a claim depends on hardware or an external engine, say what was actually
run and what remains unverified. Do not convert a documented limitation into a
green test.

## Common failure modes

1. **Blocking the event loop** with a synchronous probe or full-document
   rewrite. Use the blocking pool and existing async subprocess helpers.
2. **Editing configuration documents directly** instead of using the store;
   this bypasses serialization, durability, and recovery behavior.
3. **Adding an engine-ID branch/route** instead of definition/kind dispatch;
   the literal ratchet catches it, but the design is already wrong.
4. **Forgetting a generated artifact**, producing a stale OpenAPI/build
   defaults/parity inventory or a module-budget mismatch.
5. **Running host tooling with the wrong runtime.** Use Node 24 and Python 3.12;
   in a dev container, use its venv/`node_modules`, not host ones.
6. **Bumping Playwright without its CI browser image** or test-utils/Prettier
   without re-running the suites that exposed the incompatibility.
7. **Using GCC 15 for CUDA 12.8/12.9**, or reusing a stale build workspace from
   a previous image. The workspace key includes host/toolchain identity; use
   the CUDA host-compiler selector and glibc shim already in the tree.
8. **Committing UI review PNGs** produced incidentally by browser tests.
9. **Touching live data or making real network/model downloads in tests.**
10. **Merging red or bypassing protection** to "finish" a task. Report the
    blocking gate instead.
