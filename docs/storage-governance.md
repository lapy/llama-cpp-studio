# Storage governance

Status: implemented in the single-operator hardening release (WS2).

Studio is a single-operator, single-host workspace. This document describes the
Storage section in **System → Storage**, the management API behind it, and the
safety rules that decide what Studio may reclaim. Every change here is additive:
the new surface reads the existing data roots and never rewrites configuration.

## What the surface answers

One scan answers three questions:

1. What is using disk inside the data root (and the configured Hugging Face
   cache and ccache, which may live elsewhere)?
2. Which of that storage does Studio own and no other surface manages?
3. What can be removed safely right now, and why are the remaining units
   refused?

Storage is split into seven **disjoint** categories. "Disjoint" means one file's
bytes are reported in exactly one category; a nested cache is subtracted from
the install that contains it.

| Category          | Location                                                                     | Ownership                                             | Cleanup                                                                   | Manage link        |
| ----------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------- | ------------------ |
| `models`          | `$STUDIO_DATA_DIR/models/<namespace>/<unit>`                                 | Model payload directories (for example audio bundles) | Orphan payload directories with no catalog row                            | `/models`          |
| `hf_cache`        | `HUGGINGFACE_HUB_CACHE` / `HF_HUB_CACHE` / `HF_HOME/hub`                     | Hugging Face downloads                                | Cached repositories that no catalog model uses                            | `/models`          |
| `engine_installs` | Engine version roots (`data/llama-cpp`, `data/audio-cpp/builds`, venv roots) | Engines screen                                        | **Report only** — remove versions from Engines                            | `/engines`         |
| `cuda_toolkits`   | `$STUDIO_DATA_DIR/cuda`                                                      | System hardware                                       | **Report only** — manage toolkits from System hardware                    | `/system#hardware` |
| `jit_caches`      | Per-install `caches/`, `build-workspaces/`, `ccache`, `temp/cuda_installers` | Regenerable build artifacts                           | Engine caches, stale compile workspaces, ccache, CUDA installer downloads | none               |
| `history_backups` | `$STUDIO_DATA_DIR/config/history`                                            | Local pre-change configuration revisions              | Prune revisions older than the keep count                                 | none               |
| `accounting_db`   | `$STUDIO_DATA_DIR/config/proxy-accounting.sqlite3`                           | Proxy request accounting                              | **Report only** — reset/compact from Proxy                                | `/proxy`           |

Shared assets under `models/audio-cpp/reference-audio` and
`models/audio-cpp/community-voices` are owned by the audio features and are
always reported as `SHARED_DATA`; cleanup never removes them. Catalog model
payloads are reported as `CATALOG_MODEL` and are removed through the Models
screen, not here. This is deliberate: there is no second destructive path for
catalog models, engine installs, CUDA toolkits or the accounting database.

## API

All routes are management routes under `/api` and require the same management
authorization as the rest of the control plane. The API **never accepts a
filesystem path**. Units are named by opaque ids returned from the inventory,
and every id is resolved against a fresh scan on the server.

### `GET /api/storage/inventory?refresh=false`

Returns the cached scan (10 second TTL) or, with `refresh=true`, a new scan.
The scan runs on the shared blocking pool and is bounded by
`MAX_SCAN_ENTRIES` (100000 filesystem entries) and `MAX_UNITS_PER_CATEGORY`
(200 units per category).

```json
{
  "schema_version": 1,
  "generated_at": "2026-11-22T20:00:00+00:00",
  "cached": false,
  "filesystem": {
    "data_root": "/data",
    "total_bytes": 1000000000,
    "used_bytes": 400000000,
    "free_bytes": 600000000
  },
  "scan": {
    "quality": "complete",
    "cached": false,
    "entries_scanned": 12345,
    "max_scan_entries": 100000,
    "max_units_per_category": 200,
    "truncated": false,
    "category_count": 7,
    "notes": []
  },
  "categories": [
    {
      "id": "models",
      "label": "Model files",
      "description": "Payload directories in Studio's model store. Directories with a catalog row are managed from Models.",
      "bytes": 4096,
      "unmanaged_bytes": 0,
      "unit_count": 1,
      "cleanable": true,
      "cleanable_bytes": 4096,
      "blocked": null,
      "manage_path": "/models",
      "outside_data_root": false,
      "available": true,
      "truncated": false,
      "notes": [],
      "units": [
        {
          "id": "5f0d1c2a9e8b7d6c3a11",
          "label": "audio-cpp/orphan",
          "kind": "model_payload",
          "bytes": 4096,
          "cleanable": true,
          "blocked": null,
          "manage_path": null,
          "outside_data_root": false,
          "detail": {}
        }
      ]
    }
  ]
}
```

`scan.quality` is `partial` when the scan hit an entry cap or could not read the
catalog; sizes may then be lower than reality. `blocked` objects always carry
`{code, message}`. A category that is report-only also marks every unit with
`REPORT_ONLY` and a `manage_path`; a unit blocked for another reason carries
that reason instead.

### `POST /api/storage/cleanup/preview`

```json
{ "category": "models", "unit_ids": ["5f0d1c2a9e8b7d6c3a11"], "keep_history": 50 }
```

- `unit_ids` is optional; omitting it selects every unit in the category. At
  most 200 ids are accepted.
- `keep_history` is only used by `history_backups` (1–100, default 50).
- The server re-reads the cached inventory, applies every refusal guard and
  returns a plan:

```json
{
  "plan_id": "<sha256>",
  "category": "models",
  "category_label": "Model files",
  "keep_history": null,
  "units": [{ "id": "5f0d1c2a9e8b7d6c3a11", "label": "audio-cpp/orphan", "bytes": 4096 }],
  "refused": [
    {
      "id": "…",
      "label": "audio-cpp/keep",
      "bytes": 4096,
      "code": "CATALOG_MODEL",
      "message": "A catalog model uses this directory. Remove the model from Models."
    }
  ],
  "reclaimable_bytes": 4096,
  "confirmation_phrase": "DELETE 1 MODEL DIRECTORIES",
  "scope_cap_units": 200,
  "expires_at": "2026-11-22T20:05:00+00:00",
  "notice": "Review the selected units. The plan is refused if the inventory changes before it is applied."
}
```

`plan_id` is a digest of the category, keep count and each selected unit's id,
byte count and modification identity. Plans live in memory for five minutes
and are single-use; a restart drops them. The confirmation phrase uses `DELETE`
for the three directory categories and `PRUNE` for configuration history.

### `POST /api/storage/cleanup/apply`

```json
{ "plan_id": "<sha256>", "confirm": true }
```

Apply requires `confirm: true` and a matching plan. It re-scans the tree, checks
that every selected unit is still cleanable and unchanged, re-checks the scope
cap, deletes on the blocking pool and returns:

```json
{
  "plan_id": "<sha256>",
  "category": "models",
  "category_label": "Model files",
  "outcome": "completed",
  "freed_bytes": 4096,
  "removed": [{ "id": "…", "label": "audio-cpp/orphan", "bytes": 4096 }],
  "failed": [],
  "notice": "Everything selected was removed."
}
```

`outcome` is `completed`, `partial` or `refused`. Per-unit failures stay in
`failed` with `{code, message}`. The inventory cache is cleared after an apply.

Error responses use `{"detail": {"code": "...", "message": "..."}}`:

| Status | Code                      | Meaning                                                      |
| ------ | ------------------------- | ------------------------------------------------------------ |
| 400    | `CONFIRMATION_REQUIRED`   | Apply was called with `confirm` not `true`.                  |
| 400    | `CATEGORY_NOT_CLEANABLE`  | The category is report-only; use its manage link.            |
| 409    | `PLAN_STALE`              | Plan missing, expired, already applied, or the tree changed. |
| 409    | `CLEANUP_SCOPE_TOO_LARGE` | More than 200 units were selected.                           |
| 409    | `CLEANUP_IN_FLIGHT`       | Another cleanup is running.                                  |
| 409    | `NOTHING_TO_CLEAN`        | The plan has no selected units.                              |
| 409    | `ACTION_IN_FLIGHT`        | The shared operation supervisor already holds a cleanup.     |
| 503    | `STORAGE_SCAN_FAILED`     | The revalidation scan failed; nothing was deleted.           |

## Safety guards

- **Realpath containment.** Every unit is resolved with `os.path.realpath` and
  must stay below its category root; the category root itself can never be a
  target. Symlinks are skipped during the scan and refused at delete time.
- **No path input.** `unit_ids` are matched against the server's own scan; a
  value that is not an id (including a path) becomes `UNKNOWN_UNIT` and no
  filesystem call receives it.
- **Running models and launch references.** A unit that a live model process
  uses is refused with `RUNNING_MODEL`; a published or retained launch
  generation refuses with `RETAINED_LAUNCH_REFERENCE`. The check is repeated
  immediately before the delete.
- **Active operations.** An in-progress HF download refuses its repository
  (`DOWNLOAD_IN_PROGRESS`), an engine build/install/update refuses the affected
  caches (`BUILD_IN_PROGRESS`), a CUDA install refuses the installer cache
  (`CUDA_INSTALL_IN_PROGRESS`), and an audio model install refuses model-store
  cleanup.
- **Pending restore.** A configuration restore journal blocks history pruning
  with `CONFIG_RECOVERY_IN_PROGRESS` until it is reconciled.
- **Scope.** A plan may contain at most 200 units; the cap is checked again at
  apply time.
- **Shared data.** Reference audio and community voices are never cleanable.
- **Exclusive action.** Apply runs inside `exclusive_action("storage_cleanup",
"storage:cleanup")`, so it gets the same admission fence, durable operation
  row and busy conflict behavior as other mutating actions.

## Frontend

System gains a fifth section, **Storage** (`/system#storage`), backed by the
Pinia `storage` store. It shows the data root, measured bytes, free space and
scan freshness, then one card per category with a share bar, unit count,
reclaimable bytes, blocked reasons and manage links. A per-category disclosure
lists every unit. "Review cleanup" prepares the plan and opens a modal
confirmation dialog that lists the selected units and refusals, shows the
confirmation phrase, and keeps the delete button disabled until the operator
checks the permanent-deletion box. Results report freed bytes and list removed
and failed units. Status uses `role="status"`; failures use `role="alert"` and
never expose scan paths.

## Operational notes

- A first scan on a large HF cache is I/O-heavy; it runs on the blocking pool
  and is cached for 10 seconds. The UI's refresh button forces a new walk.
- The Hugging Face cache may be outside `STUDIO_DATA_DIR` (for example Docker
  sets `HUGGINGFACE_HUB_CACHE=/app/data/hf-cache/hub`). The inventory flags
  that with `outside_data_root` and still validates containment inside the
  cache root.
- `STUDIO_CCACHE_DIR` overrides the compiler cache location. The CUDA installer
  cache is `$STUDIO_DATA_DIR/temp/cuda_installers` and only contains downloaded
  installers; installed toolkits are never touched.
- Pruning history is file-only: it removes old revision snapshots under
  `config/history`, not the current configuration and not configuration
  backups.
- Cleanup does not run GC for the accounting database, does not remove engine
  versions, and does not remove unrelated `temp/`, `logs/` or `runtime/` data.

## Validation

```bash
python -m pytest backend/tests/test_storage_inventory.py \
  backend/tests/test_storage_cleanup.py backend/tests/test_storage_routes.py -q
npm run test:frontend -- frontend/src/stores/storage.test.js \
  frontend/src/components/system/StoragePanel.test.js \
  frontend/src/views/SystemView.test.js
```

The backend tests build temporary data roots and HF caches; they never read the
developer's real storage and never download. Container/browser evidence is
recorded by the release lead together with the OpenAPI regeneration, because
this branch intentionally does not edit the generated contract.
