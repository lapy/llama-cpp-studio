# Studio documentation

Studio documentation describes the behavior available in the repository's current
implementation. Start with the [project README](../README.md) for installation and a
feature overview.

## Using Studio

- [Setup and overview](../README.md): screenshots, deployment, engine matrix, first-model
  workflow, ports, storage, environment variables, and troubleshooting.
- [Complete feature guide](features.md): every workspace, model acquisition, configuration
  and change review, templates, identity/aliases, runtime actions, audio, proxy policies,
  System tools, backup/history, and benchmarks.
- [API reference](api-reference.md): management endpoints and schemas, native inference
  paths, authorization, request examples, and outcome handling.

## Operating inference

- [Proxy operations](proxy-operations.md): live routing, request history, client keys,
  quotas, streaming, queues/timeouts, process supervision, accounting storage, and
  diagnostic tooling.
- [Engine observability flags](engine-observability-flags.md): which logging and metrics
  flags Studio applies automatically, which log destination/suppression flags it reserves,
  and the per-engine rationale.
- [Runtime architecture](launch-manifests-plan.md): configuration state, launch compilation,
  immutable revisions, per-model and sequential batch application, rollback, and
  reconciliation.
- [Runtime behavior cases](launch-manifests-use-cases.md): concrete save/apply, alias,
  process, failure, and recovery behavior.
- [Audio contract and discovery](audio-cpp-discovery.md): structured metadata,
  task/endpoint selection, typed options, assets/voices, model inspection, and audio
  transport boundaries.

## Engineering

- [Remote development stack for agents](agent-dev-stack.md): isolated checkout/state/ports,
  optional GPU selection, SSH and JSON/MCP control, guarded patches/tests/logs/reload,
  checkpoints, rollback, and operation history.
- [Self-contained development image](agent-dev-standalone.md): one app container with
  checkout/state/data in named volumes, SSH/MCP control, and no Docker socket or host OS
  changes.
- [Engineering reference](current-engineering-status.md): module responsibilities,
  persistence/authorization boundaries, operation recovery, quality gates, and validation
  scope.
- [Engine behavior parity inventory](engine-parity-inventory.md): generated routes,
  per-engine operation endpoints, response-shape divergences, settings defaults, and
  hardcoded engine-ID literal sites. Regenerate with
  `python scripts/parity-inventory.py`; `npm run lint:engine-literals` fails when literals
  outside `backend/engines/` grow beyond this committed baseline.
- [Feature implementation map](studio-native-refactoring-plan.md): feature-to-source and
  validation coverage map.
- [Frontend standards](frontend-standards.md): frontend accessibility, styling, API
  boundary, component, and validation conventions.
- [Third-party notices](../THIRD_PARTY_NOTICES.md): attribution and licensing information.

## Release history and verification

- [Delivery map](delivery-map.md): gains, architecture, compatibility, and remaining work
  for the config-driven engine definitions release.
- [Change inventory](change-inventory.md): the dependent commit series and exact files.
- [Verification evidence](review-verification-evidence.md): commands, totals, and hardware
  coverage limits.
- [Migration notes](engine-definitions-migration-notes.md): upgrade and rollback behavior
  for existing installations.

The running service supplies interactive management API documentation at `/docs` and its
machine-readable schema at `/openapi.json`. The
[checked-in schema](../frontend/src/api/openapi.json) provides the repository's generated
API contract.
