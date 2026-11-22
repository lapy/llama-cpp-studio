# Strata engine: planned integration

Status: **planned, not registered** (2026-11-22). Studio ships no `strata`
engine definition and offers no Strata install or launch action. This page
keeps the integration analysis and the contract a future native host must
satisfy, so the work is not lost.

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

## Why it is not registered

A registration without an install/launch path would appear in `GET /api/engines`
and in the model engine picker while every action it advertises would fail. A
full Strata model-preparation stack is out of scope for the single-operator
hardening release, so the definition was retired rather than shipped inert
(the generic native lifecycle it would use is implemented — see
[engine onboarding](engine-onboarding.md)).

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
4. A definition with `native_host` and a `lifecycle` block (plus its `release`
   recipe), which serves install/status/check-updates/cancel/build-settings/
   remove on `/api/engines/{engine_id}`. `install-source` would require a
   declared `source` recipe; nothing here needs a legacy route prefix.

Until those land, install and run Strata with its own Terminal. Studio will not
claim to manage what it cannot supervise.
