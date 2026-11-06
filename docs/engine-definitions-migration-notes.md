# Engine definitions migration notes

This release changes where engine behaviour is defined, not where installations
live. There is no configuration migration step. These notes are for operators
upgrading an existing installation and for reviewers checking rollback.

## Upgrade behaviour

- **Engine IDs and version rows are unchanged.** `engines.yaml` keeps its
  historical shape: `versions`, `active_version`, `binary_path`, `venv_path`,
  `install_dir`, `build_config`, `source_ref`, `repository_source`.
- **Install directories are unchanged.** Native source builds still publish
  under the engine's install root; venv engines keep `venv_path`; audio keeps
  its prepared bundles.
- **No snapshot ledger is backfilled.** `install_snapshots.yaml` is created only
  when a new install is scheduled. An install that predates it has no recorded
  digest, and retries proceed without the new drift refusal. This is deliberate:
  refusing an upgrade because the old install cannot prove which YAML produced
  it would strand every existing installation.
- **The first write to an old engine does not rewrite its rows.** The store's
  read path leaves the document alone; the only additions are the new ledger and
  the existing settings keys.
- **Compatibility URLs stay live for this release.** Existing scripts and
  clients keep their per-engine prefixes; the generic `/api/engines/...` surface
  is additive.

The focused check for this path is
`backend/tests/test_pre_refactor_installation_smoke.py`. It writes a
pre-refactor document by hand (no `schema_version`, no snapshots), then verifies
listing, generic/compat status equality, activation, launch from the old venv,
removal of an inactive version, and retry of a snapshot-less install.

## Changes an operator will notice

| Change | What to expect |
| --- | --- |
| Startup definition gate | A malformed or drifted definition now refuses startup with a definition error instead of running with a partially understood engine. Fix the YAML or restore the definition file. |
| Retry after a definition edit | If a version has a recorded snapshot and its definition digest changed, retry is refused with "reinstall instead of retrying". Reinstall that version to adopt the new definition. |
| Generic native install bodies | The target is taken from the URL path. Sending a conflicting `engine`/`repository_source` body to the generic route returns 400; legacy routes keep their old behaviour. |
| Unimplemented generic verbs | A verb an engine does not implement returns 405 instead of a fabricated success. |
| Audio extension routing | The router is mounted once per engine declaring `runtime_kind: audio_cpp`; request identity comes from the path. Existing `/api/audio-cpp` paths are unchanged. |
| Installer classes | The per-engine venv modules and maps are removed, and Unsloth's installer moved to the generic release host. Importing the old venv/Unsloth classes is no longer supported. Native hosts are selected by the definition's `native_host`. |
| First apply after upgrade | Every definition-backed engine, including audio.cpp, gets a new launch revision because manifests now record the definition digest. The upgrade does not restart a running model; **Restart this model** applies it, **Use on next start** publishes it for the next actual start. |

## Rollback

Back up the configuration directory (`engines.yaml`, `settings.yaml`, model
configuration and `install_snapshots.yaml`) before upgrading. This release does
not rewrite version rows, but the backup is the only recovery point for
operator edits made after the upgrade.

- **Preserve `install_snapshots.yaml` when rolling back.** It is the retry
  provenance for versions installed by this release: definition digest plus the
  resolved inputs that produced the layout. The previous release ignores the
  file and `engines.yaml` stays readable without it. Do not delete it as a
  routine rollback step — deleting it discards that provenance, and a later
  retry can no longer detect that the recorded layout came from a different
  definition. If a retry is refused after a definition change, reinstall the
  version instead of deleting its snapshot.
- No version row was converted in place, so checking out the previous release
  leaves `engines.yaml` readable as-is.
- Launch revisions created by this release record a `profile_digest`. Reverting
  to the previous release may ignore that field; it does not change the
  executable or argv, and no running process is modified by the revert.
- The legacy route prefixes remain, so a client rollback does not require an API
  change.

## Known gaps

- **Requirement 9 is transitional:** the SGLang runtime selector copy and
  audio-package branches still carry engine-specific UI, and frozen literal
  caps remain. Everything on the generic onboarding path is
  descriptor-driven.
- **Hardware coverage is native-llama-only.** A real llama.cpp CUDA source build
  and GGUF launch/inference were verified on a V100. Unsloth release assets, the
  venv engines (vLLM, 1Cat, SGLang, SGLang V100, LMDeploy) and audio need their
  own representative-hardware pass.
- **Retry of a failed latest PyPI install can re-resolve latest.** If neither
  the row nor the snapshot recorded a concrete package version, retry cannot
  reproduce the attempted version. Release-asset installs (Unsloth, 1Cat-vLLM)
  now record their resolved tag in the pending snapshot. Reinstall an explicit
  PyPI version when reproducibility matters.
- **Removal and rollback differ for the ledger.** Removing a version prunes its
  snapshot rows (so a later same-name install cannot select a stale digest).
  Rolling back the release does not prune anything.
- **Fixture-only engines** (`yamlcraft`, `launchcraft`, `launchforge`,
  `nativecraft`) are not shipped engines and are not registered in production.
  The `nativecraft` fixture is the third-fork proof; its journey is
  `backend/tests/test_nativecraft_journey.py`.

## Reviewer starting points

1. `docs/delivery-map.md` — outcome, architecture, compatibility, risks.
2. `docs/change-inventory.md` — exact commit boundaries and dependency order.
3. `docs/review-verification-evidence.md` — commands, totals and limits.
4. The risky paths, in order: pre-refactor upgrade smoke, retry/snapshot guard,
   launch digest behaviour, compatibility URL equality, and the frontend store
   contract.
