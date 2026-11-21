# audio.cpp latest-release canary

Studio's audio integration is validated against pinned v0.9.1 fixtures in the test suite. Upstream ships quickly (`v0.9.1` on Oct 7 → `v0.9.5` on Nov 14, 2026) and has already broken the integration at the ABI level, so the fixture suite alone cannot tell us that the newest engine release still matches the contract assumptions Studio makes.

`scripts/audio-cpp-canary.py` closes that gap. It resolves the newest `0xShug0/audio.cpp` GitHub release, downloads the Linux x64 CPU prebuilt, verifies the published `sha256` digest when one is available, extracts the archive, runs `audiocpp_cli --list-loaders --json`, composes the contract with `compose_version_contract`, and fails loudly when an assumption no longer holds.

The canary is a scheduled/dispatch signal, not a merge gate. `.github/workflows/audio-canary.yml` runs it on Mondays at 06:17 UTC and on manual dispatch with `permissions: contents: read`. It deliberately has no `push` trigger: upstream breakage is not caused by a Studio diff, and a known-red canary must not turn every merge into a red workflow. The JSON report is uploaded as the `audio-cpp-canary-report` artifact on every run, pass or fail.

> **Runs on the shipped ABI.** The workflow executes inside `ubuntu:26.04` (glibc 2.43), matching the Studio image, so a prebuilt ABI failure means the base image or the selected artifact regressed. Do not make the canary green by skipping the ABI check.

## What it checks

Every check emits a pass/warn/fail line and a `checks[]` entry in the JSON report. The run fails when any check fails; warnings do not fail it.

| Check                 | Failure kind                                          | Contract assumption                                                                                       |
| --------------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `release`             | `download_integrity`                                  | The release resolves (GitHub API, or the public release page when the API quota is exhausted).            |
| `asset_selection`     | `download_integrity`                                  | The release publishes a Linux x64 CPU prebuilt (`select_prebuilt`), or `--asset` names one.                |
| `download`            | `download_integrity`                                  | The asset downloads over HTTP(S) to a temporary directory.                                                 |
| `asset_digest`        | `download_integrity`                                  | The downloaded bytes match the release's published `sha256:` digest (a missing digest is a warning).       |
| `archive_extraction`  | `download_integrity`                                  | The archive extracts safely and contains `audiocpp_server` and `audiocpp_cli`.                             |
| `model_manager`       | `probe_error` / `contract_drift`                      | `tools/model_manager_v2.py list --json` returns package rows. A missing script is a warning.               |
| `cli_probe`           | `abi_unsupported` / `probe_error`                     | The CLI can execute on this host and `--list-loaders --json` exits 0.                                     |
| `loader_json`         | `probe_error` / `contract_drift`                      | The payload is JSON, `schema_version` is 1, and at least one family parses.                               |
| `contract_composition`| `contract_drift`                                      | `compose_version_contract` returns the Studio contract schema with operations.                            |
| `required_operations` | `contract_drift`                                      | `speech`, `transcriptions`, `transcriptions_details`, `alignments`, and `tasks_run` exist.                |
| `advertised_endpoints`| `contract_drift`                                      | Every advertised `api_endpoints` value maps to a registry operation or the generic `/v1/tasks/run` route. |
| `drift_warnings`      | `contract_drift`                                      | Composer drift warnings are reported; unreviewed warnings fail. `--allow-drift` downgrades only this check. |

Families, tasks, endpoints, and operations counts are printed in the log and stored in `report.counts`. The canary never downloads models, starts the server, or sends inference requests.

## Failure taxonomy

Failed checks carry one of four `failure` values. They are also collected in `report.failure_kinds` and named in the final `result:` line.

| Kind                 | Meaning                                                                                       | First action                                                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `contract_drift`     | The engine now contradicts a Studio assumption: schema, families, endpoints, operations, catalog. | Read the exact endpoint/operation/warning names in the report; update the registry/composer and its tests, then refresh fixtures. |
| `abi_unsupported`    | The binary cannot execute here (glibc/loader/ELF mismatch), so no contract observation is possible. | Compare `details.required_glibc` with `details.host_glibc`; move the canary host/base image forward or pick a compatible asset.   |
| `download_integrity` | Release lookup, asset selection, download, digest, size, or archive extraction failed.         | Re-run once; if it persists, open the release page and check the asset names/digest, or pass `--tag`/`--asset`.                  |
| `probe_error`        | The CLI or model manager ran but produced no usable probe output.                             | Read `details.stderr_tail`; a flag/interface change usually needs Studio probe code plus a new fixture.                         |

## Run locally

The CLI binary is Linux x64, so run the canary on Linux. It downloads roughly 70 MB per run from GitHub.

```bash
pip install -r requirements-dev.lock
python scripts/audio-cpp-canary.py --report /tmp/audio-cpp-canary.json
```

Useful options:

```bash
# Pin a release instead of the latest one.
python scripts/audio-cpp-canary.py --tag v0.9.3

# Use an explicitly named asset (renamed/new archive layout).
python scripts/audio-cpp-canary.py --asset audio-v0.9.3-bin-ubuntu-x64-cpu-portable.tar.gz

# Keep the downloaded/extracted temporary directory for debugging.
python scripts/audio-cpp-canary.py --tag v0.9.5 --keep

# Report drift warnings without failing that check (unresolved endpoints still fail).
python scripts/audio-cpp-canary.py --allow-drift
```

Release resolution uses the GitHub API first. When the unauthenticated 60-requests/hour quota is exhausted (normal on shared CI runner IPs), the script falls back to the public release page and reads the same per-asset `sha256:` digests from it, so digest verification still runs. `--http-timeout` and `--probe-timeout` shorten waits; no token or secret is required.

Against `v0.9.3` on a glibc 2.39 host the run passes:

```text
[PASS] release: release v0.9.3 resolved via github-api (34 assets)
[PASS] asset_selection: audio-v0.9.3-bin-ubuntu-x64-cpu-portable.tar.gz (cpu-portable prebuilt)
[PASS] asset_digest: sha256 02156f7a...c1af3929 matches the published digest
[PASS] model_manager: model_manager_v2.py list --json returned 412 package row(s)
[PASS] loader_json: schema_version=1 families=133 tasks=16 endpoints=3
[PASS] advertised_endpoints: all 3 advertised api_endpoints value(s) resolve
result: v0.9.3: PASS (families=133 tasks=16 endpoints=3 operations=9)
```

`operations` is the current registry size, so it follows Studio's contract table as it grows; it is not a property of the engine release.

Against `v0.9.5` on the same host it fails as intended:

```text
[FAIL] cli_probe: audio.cpp CLI cannot execute on this host: requires GLIBC_2.43, host glibc is 2.39 (abi_unsupported)
result: v0.9.5: FAIL (abi_unsupported) (families=0 tasks=0 endpoints=0 operations=0)
```

## Reading a failure

The report schema is `audio-cpp-canary/v1`:

```json
{
  "schema": "audio-cpp-canary/v1",
  "passed": false,
  "failure_kinds": ["abi_unsupported"],
  "release": {"tag": "v0.9.5", "source": "github-api", "asset_count": 34},
  "asset": {"name": "audio-v0.9.5-bin-...tar.gz", "sha256": "4e413b1d..."},
  "host": {"platform": "Linux-...", "python": "3.12.3", "glibc": "2.39"},
  "counts": {"families": 0, "tasks": 0, "endpoints": 0, "operations": 0},
  "endpoints": {"advertised": [], "unresolved": []},
  "required_operations": {"required": ["speech", "..."], "missing": []},
  "drift": {"warnings": [], "unexpected": [], "allow_drift": false},
  "model_manager": {"script": "tools/model_manager_v2.py", "packages": 462},
  "checks": [
    {
      "id": "cli_probe",
      "status": "failed",
      "failure": "abi_unsupported",
      "message": "audio.cpp CLI cannot execute on this host: requires GLIBC_2.43, host glibc is 2.39 ...",
      "details": {"required_glibc": "2.43", "host_glibc": "2.39", "returncode": 1}
    }
  ],
  "summary": "v0.9.5: FAIL (abi_unsupported) (families=0 tasks=0 endpoints=0 operations=0)"
}
```

Endpoint drift names the exact advertised value and family, for example:

```text
[FAIL] advertised_endpoints: 1 advertised api_endpoints value(s) do not resolve to a Studio
operation or the generic /audioapi/v1/tasks/run route: /v1/future/route (future_family)
```

Exit status is `0` when every check passed (warnings allowed) and `1` otherwise. In the workflow, the failing step makes the scheduled/dispatch run red and the `if: always()` upload still preserves the report artifact.

## Refresh fixtures

`backend/tests/fixtures/audio_cpp_v093_list_loaders.json` is a trimmed real capture from `v0.9.3` (8 of 133 families, all three advertised endpoints, and the newer `wake`/`cls`/`midi`/`turn` task vocabulary). The real payload uses a family-keyed `loaders` object, not a list; keep that shape when refreshing it.

Capture a new payload on a host whose glibc can run the release (the dev box at `10.0.1.9:2223` runs glibc 2.39, which runs `v0.9.3`'s CPU prebuilt):

```bash
ssh -p 2223 10.0.1.9
mkdir -p /tmp/audio-cpp-fixture && cd /tmp/audio-cpp-fixture
tag=v0.9.3
curl -fL -o "$tag.tar.gz" \
  "https://github.com/0xShug0/audio.cpp/releases/download/$tag/audio-$tag-bin-ubuntu-x64-cpu-portable.tar.gz"
tar -xzf "$tag.tar.gz"
./audiocpp_cli --list-loaders --json > loaders.json
```

Trim the JSON to representative families (cover every advertised endpoint and any newly added task), replace the fixture, update the expected counts in `backend/tests/test_audio_cpp_canary.py`, then run:

```bash
python -m pytest backend/tests/test_audio_cpp_canary.py -q
python -m ruff check scripts/audio-cpp-canary.py backend/tests/test_audio_cpp_canary.py
```

Never commit the release archive or binaries; only the trimmed JSON fixture belongs in the repository. When a new release breaks the contract, the report's exact names and the live failure are the input to the registry/composer update, and the refreshed fixture is the regression that keeps the fix pinned.
