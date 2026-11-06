# Engine host and presentation boundaries

Status: accepted (2026-11-06)

## Context

The engine refactor moved identity, recipes, lifecycle, launch profiles,
catalog surfaces and most UI behavior into definitions. Two boundaries kept
resisting a literal "zero engine-specific code" reading:

1. **Native hosts.** Audio bundle/model/CLI orchestration and native CMake
   toolchain mechanics are domain algorithms, not recipes.
2. **Frontend presentation.** Icons, copy, specialty panels and visual
   grouping are presentation, not engine capability data.

## Decision 1 — declared domain hosts

A host may remain engine-specific when it implements a domain algorithm.

Rules:

- The definition's `native_host` selects the host. No engine-ID map or route
  branch may select it.
- A host is addressed by a stable host name (`cmake_source`, `release_assets`,
  `audio_cpp`), not by an engine ID.
- A second engine declaring the same host must need YAML only.
- Host code lives under `backend/engines/python/` (generic mechanisms) or the
  domain package (audio).
- `AudioCppInstaller` is the declared audio domain host. It is not a
  compatibility shim and is not scheduled for deletion.
- `CmakeBuildHost` and `GenericReleaseInstaller` are the shared generic hosts.

Completion test: adding a generic venv, CMake or release engine needs no
Python host edit; adding an audio-family engine needs no installer edit.

## Decision 2 — server-authoritative descriptors + presentation overlay

`GET /api/engines` is authoritative for:

- engine existence and label
- install/runtime/scanner kinds
- lifecycle verbs
- artifact/task capabilities
- launch policy and `ui_kind`

The frontend presentation overlay is optional and keyed by `ui_kind`
(`audio`, `sglang`, `native`, `venv`) or a local descriptor. It owns icons,
card subtitles, dialog copy and specialty panels.

Rules:

- An engine with no local presentation entry must render through the generic
  card/panel using the server descriptor.
- Local descriptors may win for shipped engines, but cannot be required for an
  engine to function.
- Do not move icon/copy/visual design into the backend; add `ui_kind` when a
  new presentation family is genuinely needed.

Completion test: a server-described engine with no local frontend entry gets a
labeled card and generic presentation.

## Consequences

- "Fully YAML-defined" means no engine-ID dispatch or allowlists; it does not
  mean no domain hosts or no presentation code.
- Frozen frontend literal caps remain a separate ratchet; they may only shrink
  as presentation becomes descriptor/kind driven.
- Hardware verification remains independent of this boundary.
