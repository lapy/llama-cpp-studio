# audio.cpp engine compatibility

Studio validates its audio.cpp integration against a specific upstream release and
reports a compatibility ceiling. The ceiling is advisory: a newer engine version still
installs, activates, and serves requests, but Studio reports it as **not validated**
instead of implying support.

The backend owns the decision in
[`backend/engines/audio_cpp/compatibility.py`](../backend/engines/audio_cpp/compatibility.py):
`VALIDATED_ENGINE_VERSION = "0.9.5"` is the newest release whose Linux prebuilt and
`--list-loaders --json` / model `--inspect` surface were actually exercised against
Studio. Change it only after exercising the next release on a real install (see
[Updating the ceiling](#updating-the-ceiling)).

## Version matrix

| audio.cpp version                        | Status      | Meaning                                                                                                                                                    |
| ---------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `0.9.5`                                   | supported   | Validated ceiling. Source-build and prebuilt loader/inspect, real inference, and the new task surface were exercised; the prebuilt needs glibc 2.43 (the 26.04 image provides it). |
| `0.9.1` – `0.9.4`                         | supported   | At or below the ceiling. 0.9.1 is the minimum build Studio can activate because it introduced the structured `--list-loaders --json` surface.               |
| below `0.9.1`                             | unsupported | Cannot activate in Studio (no structured loader JSON). Reported unsupported rather than "older than the ceiling".                                           |
| `0.9.6` and newer                         | unsupported | Not validated. Still installs and runs unblocked, but its surface is outside the exercised set until the ceiling is deliberately bumped.                    |
| branch, commit, or local build label      | unknown     | Studio cannot map the label to a release, so it makes no claim in either direction.                                                                        |
| no active version                         | unknown     | Nothing is installed/active to classify.                                                                                                                    |

"Unsupported" does **not** mean blocked: Studio never refuses an install, activation, or
request because of this classification. It means the surface is outside what was
validated, so failures are not treated as Studio regressions until the release is
exercised. The engine UI may show the note; the API always carries it.

## Linux prebuilt note

Linux prebuilts for v0.9.4 and v0.9.5 link **glibc 2.43**. The Studio image is Ubuntu
26.04 (glibc 2.43), so they run as shipped; on an older host (for example `ubuntu:24.04`,
glibc 2.39) the install preflight rejects the archive and points at a source build. This
is an artifact-toolchain limit, not an engine or Studio shim regression.

## Upstream movement after the ceiling

Upstream is at `v0.9.5` (2026-11-14) and releases frequently. The changes that matter to
Studio's integration surface since the previously validated line:

- **0.9.2** — normalized route aliases and option spellings.
- **0.9.3** — changed `instruct`/`instructions` handling.
- **0.9.4** — malformed request options now return `400` instead of being tolerated.
- **0.9.5** — inline `voice_ref`, `models-root`, and explicit model-variant routing;
  validated end-to-end, including a real CPU inference run.

Upstream issue [0xShug0/audio.cpp#852](https://github.com/0xShug0/audio.cpp/issues/852)
(machine-readable integration contract) remains open without a maintainer commitment to a
date. Studio therefore keeps its shim and its own ceiling.

## API fields

These fields are additive on the existing responses; no endpoint or existing key changed.

| Field                       | Type          | Meaning                                                                                                     |
| --------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------- |
| `engine_version`            | string / null | Installed version label (`v0.9.4`, `v0.9.5`, `source-main-…`, …).                                            |
| `validated_through`         | string        | Newest validated release, always `0.9.5`.                                                                    |
| `engine_supported`          | bool / null   | `true` within the validated range, `false` outside it, `null` when the version is unknown.                    |
| `engine_compat_note`        | string        | Short human-readable explanation of the classification.                                                      |
| `upstream_contract_available` | bool        | Status only. Whether the opt-in audio.cpp#852 probe found a contract document for the active install.       |

Surfaced on:

- `GET /api/audio-cpp/status` — top-level fields, including `upstream_contract_available`.
- `GET /api/audio-cpp/contract` — `engine_version` plus the three compatibility fields.
- `GET /api/llama-versions` — audio.cpp rows carry the four compatibility fields so the
  Engines view can annotate a version before/while it is active.

An unknown version is reported as `engine_supported: null` and never as `false`: "we did
not look" and "we looked and it failed" are different answers.

The probe behind `upstream_contract_available` is opt-in: it runs only when
`AUDIO_CPP_UPSTREAM_CONTRACT=1`, and with the flag off nothing is read or executed. With
the flag on, a server binary that does not implement `--describe-server`, or a checkout
with no known contract file, simply reports `false` — no error, no retry, and no change to
engine install, activation, or inference.

## Installing a newer engine from source

When a release prebuilt does not run on the host (for example the glibc 2.43 artifacts on
`ubuntu:24.04`), build the release from source instead:

1. Open **Engines**, select audio.cpp, and set the tracking ref to the release tag
   (`v0.9.4`, `v0.9.5`) or to the branch/commit under test in Build settings. Older
   releases used the legacy `release-X.Y[.Z]` tag form; Studio still recognizes it, but
   current upstream tags are `vX.Y.Z`.
2. Build from source (the same release tag registers under its tag name whether it came
   from a prebuilt or a source build).
3. Activate the new version and run the parameter scan; the status payload then reports
   `engine_supported: false` with the ceiling note while the version runs normally.

The API equivalents are `PUT /api/audio-cpp/build-settings` followed by
`POST /api/audio-cpp/build-source` with `source_ref` and `source_ref_type: "release"`.
Source builds still need the host toolchain/CMake setup described in
[audio.cpp contract, discovery, and transport](audio-cpp-discovery.md#build-configuration-contract);
Studio's supported runtime backends remain CPU and CUDA.

## What "unknown" means

A bare branch, commit SHA, or local build label has no release version Studio can compare
against the ceiling. Studio records `status: "unknown"`, `engine_supported: null`, and a
note; it does not guess that a branch tip is newer or older than 0.9.5, and it does not
downgrade it to unsupported. If you want a branch under test to count as validated,
exercise it and record the release it corresponds to.

## Updating the ceiling

1. Install and activate the candidate release on a host with the target libc/toolkit.
2. Exercise version scan (`--list-loaders --json`), model inspect/help, one request per
   task family you intend to claim, and the audio workspace forms.
3. Update `VALIDATED_ENGINE_VERSION` in `backend/engines/audio_cpp/compatibility.py`.
4. Update this matrix, the pinned constant test in
   `backend/tests/test_audio_cpp_compatibility.py`, and any stale wording in
   [the discovery guide](audio-cpp-discovery.md) / [the shim plan](audio-cpp-integration-shim-plan-2026-10-15.md).
5. Keep the status/contract fields and the non-blocking behavior unchanged.

## Related

- [audio.cpp contract, discovery, and transport](audio-cpp-discovery.md)
- [audio.cpp integration contract shim plan](audio-cpp-integration-shim-plan-2026-10-15.md)
- [0xShug0/audio.cpp#852](https://github.com/0xShug0/audio.cpp/issues/852)
