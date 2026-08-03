# Temporary audio.cpp model-spec adapter

Studio’s temporary adapter bridges **upstream audio.cpp model-spec migration** so the UI and sidecars still expose peer-model path options before every family is fully typed.

**Implementation:** `backend/audio_cpp_model_contracts.py`  
**Consumed by:** version scan (`backend/engine_param_scanner.py`), task profiles / sidecar session fields (`backend/audio_task_profiles.py`), and any path that reads `capabilities.family_dependencies`.

This adapter is intentional and shrink-only. Do not grow Studio-only peer graphs beyond keys that real loaders already accept.

---

## Why it exists

Upstream audio.cpp is moving family metadata into typed contracts:

| Tree | Role |
| --- | --- |
| `model_specs/<family>.json` with `schema_version: 1` | Long-term source of truth (install layout **and** typed options / capabilities / `dependencies`) |
| `model_specs_v1/<family>.json` | Migration preview: richer typed-shaped metadata for families not yet on `schema_version: 1` in `model_specs/` |
| Loader CLI options (`--list-loaders`, `--help`, runtime `*.*_model_path`) | What the binary actually accepts today |

Until migration finishes:

1. Many active families still lack a typed `schema_version` in `model_specs/`.
2. Some loaders already take peer paths (Whisper for VeVo2, aligner for OuteTTS) that are **not** listed in upstream `dependencies` yet.
3. Preview specs often name options `foo_path` while live loaders prefer `foo_model_path`.

Studio therefore adapts at contract-load time so operators can configure peer paths without hardcoding ASR/TTS family lists in the UI layer.

---

## What “temporary” means

- **Temporary** = Studio behavior that should disappear as upstream lands typed specs with complete `dependencies` and aligned option keys.
- **Not temporary** = reading typed `model_specs/` with `schema_version: 1`, normalizing declared `dependencies`, and building sidecar fields from those contracts.

Adapter identity string:

```text
pre_v1_model_specs_v1
```

(`TEMPORARY_PRE_V1_ADAPTER` in code.)

A human-readable note is persisted on scan as `capabilities.temporary_pre_v1_adapter` and listed in `contract_warnings` when any family still needs the adapter.

---

## Two adapter layers

The temporary adapter has two independent fill-ins. A family can use one, both, or neither.

### 1. Pre-v1 `model_specs_v1` overlay

For a family **without** `schema_version` in `model_specs/<family>.json`, Studio loads `model_specs_v1/<family>.json` when that preview looks typed-shaped (has `dependencies`, `options`, and/or `capabilities` + `category`, or an explicit schema version).

Layout fields (`sources`, `packages`, `package_defaults`, `layouts`) are merged from `model_specs/` when present so install/discovery metadata is not dropped.

Normalized contract flags:

| Field | Typical value for overlay |
| --- | --- |
| `source` | `model_specs_v1` |
| `typed` | `false` (unless the preview itself carries `schema_version`) |
| `temporary` | `true` |
| `adapter` | `pre_v1_model_specs_v1` |

Examples that commonly use this path today: `qwen3_asr`, `miotts`, `vibevoice_asr` (and other families still preview-only on a given checkout).

### 2. Temporary peer dependency seeds

Some runtime peers exist in loaders but are missing from upstream `dependencies`. Studio fills **only those gaps** from `TEMPORARY_PEER_DEPENDENCY_SEEDS`.

| Family | Seeded option (public key after alias) | Peer | Scope | Kind |
| --- | --- | --- | --- | --- |
| `vevo2` | `vevo2.whisper_model_path` | `whisper` | load | `external` |
| `outetts` | `outetts.aligner_model_path` (from `aligner_path`) | `qwen3_forced_aligner` | session | `model` |

Rules:

1. Prefer upstream `dependencies` when present.
2. Skip a seed if the same `option_key`, local `option`, peer `family`, or `*_path` / `*_model_path` alias is already declared.
3. Mark each seeded row with `temporary_seed: true`.
4. Mark the contract with `temporary_peer_seeds: true` when any seed was applied.
5. Typed contracts (`schema_version` present) stay `typed: true` / `temporary: false`; peer seeds alone do **not** force the pre-v1 adapter flag onto typed families.
6. If a family has seeds but **no** usable spec file at all, Studio synthesizes a minimal stub (`source: temporary_peer_seed`) so peer fields still appear.

**Not seeded (by design):**

- `ace_step.dit_model_path` — package **variant** inside the model root, not a peer family.
- Invented Studio-only companion graphs that loaders do not accept.

Drop each seed row as soon as upstream declares the equivalent dependency.

---

## Contract load preference order

`load_family_contract(source_root, family)`:

1. **Typed** `model_specs/<family>.json` with `schema_version` → stable path.
2. Else **preview** `model_specs_v1/<family>.json` (merged with layout from `model_specs/` when available) → temporary overlay.
3. Else rich but unversioned content already in `model_specs/` (rare).
4. Else peer-seed stub if the family is listed in `TEMPORARY_PEER_DEPENDENCY_SEEDS`.
5. Always run `apply_temporary_peer_dependency_seeds` on the result.

`load_family_contracts` discovers families from both spec directories and always includes seed-table families so peer-only coverage is not skipped when no JSON exists yet.

---

## Option key aliasing (`*_path` → `*_model_path`)

Preview / schema drafts often use local names like `forced_aligner_path`, `vad_path`, `codec_path`. Live loaders historically accept `*.*_model_path`.

`public_option_key()`:

- Builds `family.local` and, when local ends with `_path` but not `_model_path`, also `family.<stem>_model_path`.
- If a `known_keys` set from a binary scan is provided, prefers a key that the binary advertised.
- Otherwise prefers the `*_model_path` candidate (temporary default).

Normalized dependency rows always carry the resolved public `option_key` used by Studio params and sidecars.

Remove the alias preference once upstream loaders and typed specs share one public naming scheme.

---

## How Studio uses adapted contracts

On audio.cpp version scan:

1. Load contracts for known loader families from the active source checkout.
2. Persist `capabilities.family_dependencies` (normalized peer rows).
3. Persist `family_contract_sources`, `family_contract_temporary`, and when needed:
   - `temporary_pre_v1_adapter_families`
   - `temporary_pre_v1_adapter` (note string)
4. Include an adapter warning in `contract_warnings` while any temporary family remains.
5. Fold contract content into the version `contract_fingerprint` so peer/spec drift can trigger operator review / defaults migration.

For model config UI and sidecar generation:

- Session/load peer path fields come from contract `dependencies` via `dependency_sidecar_fields` / `sidecar_session_fields_for` — not from hardcoded per-family ASR lists.
- Optional labels/placeholders live in `DEPENDENCY_FIELD_ENRICHMENT` (including temporary seed keys).

A family is counted as still using the temporary adapter when any of:

- `temporary == true`
- `temporary_peer_seeds == true`
- `adapter == pre_v1_model_specs_v1`

(`temporary_adapter_families()`.)

---

## Declared vs seeded dependencies (reference)

**Already covered by upstream preview/typed `dependencies` (no Studio seed needed when present):**

| Family | Peers (typical) |
| --- | --- |
| `qwen3_asr` | `qwen3_forced_aligner`, bundled `silero_vad` |
| `miotts` | `miocodec`, optional `qwen3_asr` (best-of-N) |
| `vibevoice_asr` | bundled `silero_vad` |

**Studio seeds until upstream declares them:**

| Family | Public option key | Notes |
| --- | --- | --- |
| `vevo2` | `vevo2.whisper_model_path` | External Whisper dir used by VC / S2S / SVC feature extraction |
| `outetts` | `outetts.aligner_model_path` | Optional forced aligner for cloning when the package lacks an embedded aligner; runtime also accepts `outetts.aligner_path` / `forced_aligner_model_path` |

---

## End state (when to delete the adapter)

Remove the temporary adapter when **all** of the following are true for every Studio-supported family:

1. Typed `model_specs/<family>.json` with `schema_version: 1` is authoritative for options, capabilities, and tasks.
2. Every real peer path the loaders require is declared in that family’s `dependencies`.
3. Public option keys match runtime (`*_path` vs `*_model_path` aliasing no longer needed).
4. `model_specs_v1/` is unused for runtime (migration reference only, or gone).
5. `TEMPORARY_PEER_DEPENDENCY_SEEDS` is empty and `temporary_adapter_families()` returns `[]` on a current checkout scan.

Until then: shrink rows and overlay usage family-by-family; do not add speculative peers.

---

## Operator checklist

When activating or updating an audio.cpp build:

1. Inspect scan capabilities for `temporary_pre_v1_adapter_families` and `contract_warnings`.
2. For families listed there, expect peer path fields in model config even if upstream typed specs still show empty `dependencies`.
3. Prefer configuring peers that match installed packages under `data/models/audio-cpp/`.
4. After upstream migrates a family (typed + declared deps), re-scan; that family should leave the temporary list and any `temporary_seed` rows for it should be removable from Studio.

---

## Related: package downloads

Studio package installs now prefer upstream ``tools/model_manager_v2.py`` (packages
from ``model_specs/*.json``). The legacy ``model_manager.py`` /
``model_manager_deprecated.py`` path remains only for composite/converter leftovers.
See ``backend/audio_cpp_model_managers.py`` and the README audio.cpp section.

| `TEMPORARY_PEER_DEPENDENCY_SEEDS` | Explicit gap-fill peer table |
| `load_family_contract` / `load_family_contracts` | Preference order + stub + seeds |
| `apply_temporary_peer_dependency_seeds` | Gap merge with skip-if-declared |
| `public_option_key` / `_model_path_alias` | Preview→runtime key mapping |
| `temporary_adapter_families` | Families still on the temporary path |
| `DEPENDENCY_FIELD_ENRICHMENT` | UI copy for dependency path fields |
| `backend/tests/test_audio_cpp_model_contracts.py` | Overlay, seed, skip-if-declared, stub coverage |
