# Studio documentation

The documentation describes the features and behavior available in the repository's current implementation.

## Using Studio

- [Setup and overview](../README.md): deployment, engine matrix, first-model workflow, ports, storage, environment variables, and troubleshooting.
- [Complete feature guide](features.md): every workspace, model acquisition, configuration controls, identity/aliases, runtime actions, audio, proxy policies, System tools, templates, backup/history, and benchmarks.
- [API reference](api-reference.md): management endpoints and schemas, native inference paths, authorization, request examples, and outcome handling.

## Operating inference

- [Proxy operations](proxy-operations.md): live routing, request history, client keys, quotas, streaming, queues/timeouts, process supervision, accounting storage, and diagnostic tooling.
- [Runtime architecture](launch-manifests-plan.md): configuration state, launch compilation, immutable revisions, per-model and sequential batch application, rollback, and reconciliation.
- [Runtime behavior cases](launch-manifests-use-cases.md): concrete save/apply, alias, process, failure, and recovery behavior.
- [Audio contract and discovery](audio-cpp-discovery.md): structured metadata, task/endpoint selection, typed options, assets/voices, model inspection, and audio transport boundaries.

## Engineering

- [Engineering reference](current-engineering-status.md): module responsibilities, persistence/authorization boundaries, operation recovery, quality gates, and validation scope.
- [Feature implementation map](studio-native-refactoring-plan.md): feature-to-source and validation coverage map.
- [Frontend standards](frontend-standards.md): frontend accessibility, styling, API boundary, component, and validation conventions.
- [Third-party notices](../THIRD_PARTY_NOTICES.md): attribution and licensing information.

The running service supplies interactive management API documentation at `/docs` and its machine-readable schema at `/openapi.json`. The [checked-in schema](../frontend/src/api/openapi.json) provides the repository's generated API contract.
