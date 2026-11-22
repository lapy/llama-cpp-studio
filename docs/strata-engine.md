# Strata engine status

Status: **experimental registration only** (2026-11-22). The engine is declared
in `backend/engines/definitions/strata.yaml`, but Studio does not install or
launch it yet. This page records what is declared, why the usual definition-only
path does not apply, and what a supervised integration needs.

## What Strata is

[Strata](https://github.com/Niko1221/Strata) is a local inference engine for
consumer NVIDIA, AMD, Intel Arc and Apple Silicon GPUs. Unlike the Python-venv
engines, it ships as a managed distribution:

- The **Terminal** (`strata`) installs a per-backend engine package from a
  signed update feed.
- `strata launch` acquires the model, prepares a per-model run config
  (`strata-<model>.json`), starts a background server and returns.
- `strata serve --profile <run-config> --port <port>` is the **foreground**
  server that exposes the OpenAI- and Anthropic-compatible API.

Studio's native proxy owns the engine process: it spawns a foreground process,
waits for readiness, supervises it, and treats process exit as the end of the
generation. `strata launch` is a background supervisor and does not fit that
contract, and the run config it produces is model-specific preparation output,
not a launch-time argument.

## What Studio declares today

`backend/engines/definitions/strata.yaml` declares the engine identity,
`prebuilt` distribution kind, a `strata` runtime kind, the
`prepared_bundle` artifact/package contract, text-generation capabilities, the
supported consumer-GPU families, the upstream repository and the managed
`strata` data directory. There is deliberately **no** `lifecycle`, `recipes` or
`runtime_profiles` block: advertising install or launch actions before a host
implements them would make the UI lie. Studio therefore sees a prepared Strata
profile as the artifact (`prepared_bundle`), not a raw GGUF file.

- **Where it is visible:** `GET /api/engines` includes the descriptor, the
  capability-filter inputs and the hardware policy. Test fixtures can load the
  definition through the generic loader and startup gates.
- **Where it is not visible:** the Engines workspace only renders engines that
  expose a lifecycle record, and the model engine picker only offers engines
  compatible with the model's artifact format. Strata has neither today, so it
  does not appear as an installable or selectable engine.
- **Model compatibility:** excluded from GGUF inference because the declared
  artifact is a prepared Strata profile, not the raw shards.

## What a supervised integration needs

1. A release-archive native host: resolve `strata-control-<version>-linux-x86_64.tar.gz`
   from the GitHub release (or the signed feed), verify the published digest,
   extract it into an install directory, and locate `bin/strata`.
2. A model-preparation step: run the bundled `setup.py` in control mode against
   the Studio model path to produce the run config, with progress and
   cancellation, and record it as a version artifact.
3. A foreground runtime profile: `strata serve --engine strata --profile <run-config>
   --port <port>`, with the readiness `/health` contract, managed cache variables
   (`XDG_CACHE_HOME`, `CUDA_CACHE_PATH`) and the Studio model id mapped to
   Strata's served model name.
4. Generic native lifecycle wiring (install/status/check-updates/cancel/remove)
   over `/api/engines/{engine_id}` so no new legacy route prefix is needed.

Until those land, install and run Strata with its own Terminal. Studio will not
claim to manage what it cannot supervise.
