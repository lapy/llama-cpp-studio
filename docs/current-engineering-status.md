# Current engineering status

Updated 2026-10-07. This is the canonical engineering status; dated verification documents are historical evidence.

## Product and recovery guarantees

- Configuration YAML mutations are serialized across threads and processes, reject corrupt input, and use atomic replacement. Persistence responses distinguish unchanged, replaced, and unknown outcomes.
- Durable operation rows and resource admission fences prevent automatic replay when a side effect may already have started. Confirmation tokens are revision-bound and consumed atomically.
- Configuration backup schema 1 exports portable preferences, model settings, templates, and routing without credentials, weights, executable paths, runtime state, or operation history. Restore uses a durable multi-document journal and revision-bound preview.
- The Restore UI supports mapping, conflict choices, stale-plan recovery, uncertain-response reconciliation, keyboard navigation, focus return, and responsive zoom behavior.
- Every configuration mutation preserves a private pre-change revision under `data/config/history/`. The UI shows a redacted diff and selectively restores one preference, model configuration, template, routing profile, or selector. Restore checks the current document revision while holding the store lock and never starts or publishes a model.
- A running model can execute a bounded local streaming benchmark. Results record time to first token, throughput, token count, total duration, the saved configuration revision and fingerprint, and observed total GPU memory. Results remain local in `benchmarks.yaml`.

## Engineering controls

- `VERSION` is the application version source for FastAPI metadata, configuration exports, and `package.json`; `scripts/check-version.py` rejects drift. The container copies the same file.
- FastAPI response models cover the backup, history, and benchmark contracts. A deterministic checked-in OpenAPI document is regenerated with `npm run openapi:export`, and CI rejects schema drift.
- The new boundary client in `frontend/src/api/configuration.ts` supplies TypeScript contracts for restore, history, and benchmark flows. Further endpoints can migrate to the same pattern without a frontend rewrite.
- Large views now delegate backup/history and benchmarking to cohesive feature components. Existing engine and model screens remain a modular-monolith migration target rather than being rewritten at once.
- ESLint, Ruff, Stylelint, version consistency, scoped Prettier formatting, unit suites, the production build, and the compressed-asset budget run in CI. Property-based state-machine coverage exercises action admission and confirmation invariants.
- npm and Python dependency audits run on every CI change. Dependabot tracks npm, pip, and GitHub Actions. The publish workflow scans the exact smoke-tested image with Trivy, writes a CycloneDX SBOM, and only then pushes and attests it.

## Validation and limits

The latest verification performed with this change is recorded in its review or commit message. Run the following locally to reproduce the software-only checks:

```bash
source .venv/bin/activate
npm run lint
npm run style:check
npm run format:check
npm run openapi:check
pip-audit -r requirements.lock
npm audit --omit=dev --audit-level=high
python -m pytest backend/tests -q
npm run test:frontend
npm run build
node frontend/e2e/cold-models-budget.mjs
```

Container smoke, host/container UID handling, navigation measurements, shutdown, Trivy, and image SBOM generation require a running Docker daemon. Real GPU and model compatibility require suitable hardware and downloaded runtimes. Browser role/name/focus checks do not replace a manual screen-reader pass.

## Next maintenance targets

- Extend typed request and response models and the TypeScript API boundary as endpoints change.
- Continue extracting cohesive workflows from the largest engine, search, and model-configuration files when those areas are edited.
- Add comparable benchmark baselines only after hardware, engine build, model, and runtime revision metadata can be matched; current results deliberately report measurements without claiming cross-machine comparability.
- Expand formatting coverage as legacy files are touched; current lint and CSS correctness gates cover the full frontend while the formatting gate covers newly migrated modules.
