# llama.cpp Studio

`llama.cpp Studio` is a local control plane for downloading, configuring, and serving LLMs from a single machine.

The project combines:

- a Vue 3 frontend
- a FastAPI backend
- YAML-backed state under `data/`
- a unified `llama-swap` OpenAI-compatible endpoint on port `2000`

Today, the app manages five runtime families:

- `llama.cpp` for GGUF models
- `ik_llama.cpp` for GGUF models
- `LMDeploy` for safetensors models
- `1Cat-vLLM` for vLLM-backed models
- `audio.cpp` for prepared audio model bundles (TTS, ASR, VAD, and related tasks)

This README has been rebuilt to match the current repository layout and runtime behavior.

## What the app does

- Search Hugging Face and the audio.cpp package catalog for compatible models
- Download GGUF quantizations, optional `mmproj` projector files, safetensors bundles, and prepared audio.cpp packages
- Store model and engine state in YAML instead of SQLite
- Build `llama.cpp`, `ik_llama.cpp`, and `audio.cpp` from source and manage multiple installed versions
- Install LMDeploy from PyPI or from source into a dedicated virtual environment
- Install CUDA Toolkit versions into the persistent app data directory
- Configure models per engine using a parameter catalog parsed from the active runtime binary
- Serve models through one OpenAI-compatible endpoint exposed by `llama-swap`
- Stream progress and notifications over Server-Sent Events

## Ports and endpoints

| Purpose | Docker / container | Local dev |
| --- | --- | --- |
| Web UI + FastAPI API | `http://localhost:8080` | frontend: `http://localhost:5173`, backend API: `http://localhost:8081` |
| OpenAI-compatible model endpoint | `http://localhost:2000` | `http://localhost:2000` |
| OpenAPI docs | `http://localhost:8080/docs` | `http://localhost:8081/docs` |
| Raw schema | `http://localhost:8080/openapi.json` | `http://localhost:8081/openapi.json` |

In local dev, Vite proxies `/api` from `5173` to `127.0.0.1:8081`.

## How the system is wired

```text
Browser UI (Vue 3)
  -> FastAPI backend
    -> YAML config in data/config/
    -> Hugging Face downloads in data/models/ and data/hf-cache/
    -> engine installs in data/llama-cpp/, data/lmdeploy/, and data/audio-cpp/
    -> CUDA installs in data/cuda/
    -> llama-swap config in data/llama-swap-config.yaml
  -> llama-swap on :2000
    -> llama.cpp / ik_llama.cpp / LMDeploy / 1Cat-vLLM / audio.cpp runtimes
```

The backend starts `llama-swap` automatically when there is at least one active runtime binary available from any registered engine (including audio-only installs).

## First-run workflow

1. Start the app.
2. Open `Engines`.
3. Build and activate a `llama.cpp` or `ik_llama.cpp` version for GGUF models.
4. If you want safetensors support, install and activate LMDeploy.
5. If you want audio tasks (TTS, ASR, VAD, and related), build and activate `audio.cpp` from source.
6. If you need gated Hugging Face access, set `HUGGINGFACE_API_KEY` or enter a token in the UI.
7. Open `Search`, find a model, and download or install it.
8. Open `Models`, configure the model, and choose its engine.
9. If the UI says the `llama-swap` config is stale, apply the pending config.
10. Start the model from the library.
11. Call it through `http://localhost:2000/v1/...`.

Important:

- Saving model config updates the YAML store immediately.
- Applying pending `llama-swap` config rewrites `data/llama-swap-config.yaml` and unloads models before regenerating proxy state.
- GGUF models require an active `llama.cpp` or `ik_llama.cpp` build.
- safetensors models require an active LMDeploy install.
- audio.cpp models require a prepared bundle installed or imported locally, plus an active `audio.cpp` build.

## Docker quick start

### Prerequisites

- Docker
- Docker Compose
- For NVIDIA GPU use: NVIDIA drivers on the host plus the NVIDIA Container Toolkit

### CPU mode

```bash
git clone <repo-url>
cd llama-cpp-studio
docker compose -f docker-compose.cpu.yml up --build
```

This mode:

- exposes `8080` for the UI/API
- exposes `2000` for `llama-swap`
- mounts `./data` to `/app/data`
- mounts `./backend` to `/app/backend`
- enables backend reload with `RELOAD=true`

This is the best Docker option for backend-focused development. The frontend is still the built bundle from the image, not a live Vite dev server.

### NVIDIA / CUDA mode

```bash
docker compose -f docker-compose.cuda.yml up --build -d
```

This mode:

- exposes the same ports: `8080` and `2000`
- mounts `./data` to `/app/data`
- reserves NVIDIA GPUs for the container
- disables backend reload

### Manual image build

```bash
docker build -t llama-cpp-studio .

docker run -d \
  --name llama-cpp-studio \
  -p 8080:8080 \
  -p 2000:2000 \
  -v "$(pwd)/data:/app/data" \
  llama-cpp-studio
```

For NVIDIA GPUs, add `--gpus all`.

### After startup

Open:

- UI: `http://localhost:8080`
- OpenAPI docs: `http://localhost:8080/docs`
- model endpoint: `http://localhost:2000/v1/models`

## Local development

### Prerequisites

- Node.js 20+
- Python 3
- a virtual environment tool such as `venv`
- a `python` executable on `PATH` if you want to use the provided `npm` scripts as-is

If you want to build runtimes on the host instead of inside Docker, you will also need native build tooling such as:

- `cmake`
- `build-essential`
- `git`
- `pkg-config`
- `libopenblas-dev` for OpenBLAS-backed CPU builds

### Install dependencies

The repository scripts use `python`, so if your system only provides `python3` you should either add a `python` alias or translate the commands below accordingly.

```bash
npm install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### Run frontend and backend together

```bash
npm run dev:all
```

That starts:

- frontend Vite dev server on `5173`
- backend on `8081`

You can also run them separately:

```bash
npm run dev:frontend
npm run dev:backend
```

### Production-style local run

Build the frontend, then let FastAPI serve the built assets:

```bash
npm run build
python backend/main.py
```

When running outside Docker, the app stores persistent state under `./data`.

## Testing

```bash
npm run test
```

Or run suites individually:

```bash
npm run test:frontend
python -m pytest backend/tests -q
```

## What lives in `data/`

The app is built around a persistent writable data directory. In Docker that is `/app/data`. Outside Docker it is `./data`.

Typical layout:

```text
data/
  config/
    models.yaml
    engines.yaml
    settings.yaml
    engine_params_catalog.yaml
    audio-cpp/
      servers/
  models/
    gguf/
    safetensors/
    audio-cpp/
  hf-cache/
  llama-cpp/
  lmdeploy/
  audio-cpp/
    builds/
    tools/
  cuda/
  logs/
  temp/
  llama-swap-config.yaml
```

What these are used for:

- `config/models.yaml`: downloaded models and per-model configuration
- `config/engines.yaml`: installed engine versions, active versions, and build settings
- `config/settings.yaml`: app settings such as Hugging Face token and proxy port
- `config/engine_params_catalog.yaml`: parsed CLI parameter catalog used by the model config UI
- `config/audio-cpp/servers/`: generated one-model `audiocpp_server` JSON sidecars for `llama-swap`
- `models/`: downloaded GGUF, safetensors, and prepared audio.cpp bundles
- `models/audio-cpp/`: versioned prepared audio model packages
- `hf-cache/`: Hugging Face cache
- `llama-cpp/`: source checkouts and build artifacts for `llama.cpp` and `ik_llama.cpp`
- `lmdeploy/`: LMDeploy virtual environments and source installs
- `audio-cpp/builds/`: source checkouts and build artifacts for `audio.cpp`
- `audio-cpp/tools/`: isolated Python virtual environment for the upstream model manager
- `cuda/`: CUDA Toolkit installs managed by the app
- `logs/`: installer and background task logs
- `llama-swap-config.yaml`: generated proxy configuration

## Runtime and engine behavior

### GGUF

GGUF models are managed as quantized entries grouped by Hugging Face repo. They run through either:

- `llama.cpp`
- `ik_llama.cpp`

Current engine management behavior:

- multiple versions can be installed and retained
- versions can be activated or deleted
- updates build the latest source ref with the saved build settings
- parameter support is discovered by scanning the active binary's `--help` output

### safetensors

safetensors repos are managed as logical model bundles and run through LMDeploy.

Current LMDeploy flows:

- install latest or specific version from PyPI
- install from a source repository and branch
- keep multiple installs in the engine registry
- activate or remove installs from the UI

### CUDA

CUDA is managed as a persistent install under `data/cuda`.

The Docker image is prepared to use:

- `CUDA_HOME`
- `CUDA_PATH`
- `LD_LIBRARY_PATH`
- NCCL-related include and library paths

The app can install multiple CUDA versions and keeps a `current` symlink for the active one.

### audio.cpp

`audio.cpp` is a native engine for prepared audio model bundles. It is gated by `AUDIO_CPP_ENABLED` (enabled by default). Maturity is per surface, not a single blanket label:

| Surface | Maturity |
| --- | --- |
| Speech (`/v1/audio/speech`) and ASR (`/v1/audio/transcriptions`) via `llama-swap` | Primary Studio path; treat as stable once smoke-tested against your pin |
| Generic tasks via `/upstream/{model}/v1/tasks/run` | Supported but second-class until `llama-swap` routes `/v1/tasks/run` |
| Catalog discovery from upstream JSON (`--list-loaders --json`, package `family` / `standalone`, `--inspect --json`) | Stable on modern audio.cpp tips that advertise those contracts |
| Heuristic discovery fallback (fuzzy package→family matching) | Experimental; logged via `discovery_source`; controlled by `AUDIO_CPP_HEURISTIC_DISCOVERY` (default on) |

Supported tasks include TTS, ASR, VAD, diarization, separation, generation, voice conversion, speech-to-speech, and alignment. Build backends include CPU, CUDA, and Vulkan on Linux; Metal is exposed only when the host supports it.

Current audio.cpp flows:

- build/update from a user-chosen tracking ref (branch/tag) persisted in engine settings; Update syncs in place when possible
- on activate/sync, rescan `--help` / `--list-loaders` (JSON preferred) and rediscover packages from the active checkout (no Studio package YAML mirror); persist a `capability_delta` for operator review
- after contract drift, review affected models and optionally batch-migrate request defaults (`POST /api/audio-cpp/migrate-defaults`)
- discover verified-compatible packages through the normalized model catalog (`/api/model-catalog/search`), preferring authoritative package JSON fields over fuzzy matching
- install direct HF snapshots, composite/converter packages via the upstream model manager, or import a local prepared directory
- configure family, task, mode, backend, device, load options, and session options from scanned model profiles (generic request UI for unknown families); instructions policy prefers upstream inspect JSON
- run one `audiocpp_server` process per Studio model behind `llama-swap` using a generated JSON sidecar

Model manager behavior:

- Python/Torch are required only by the upstream `tools/model_manager.py`
- Studio creates an isolated helper venv under `data/audio-cpp/tools/` on first use
- installs are validated with `audiocpp_cli --inspect` before promotion into `data/models/audio-cpp/`

Pinned upstream versions:

| Component | Pin |
| --- | --- |
| audio.cpp repository | `https://github.com/0xShug0/audio.cpp.git` |
| Tracking ref | User-configurable (bootstraps from GitHub latest release / default branch) |
| llama-swap | v240 |

Known limitation (deferred — no Studio reverse-proxy shim): `llama-swap` v240 routes `/v1/audio/speech`, `/v1/audio/transcriptions`, and `/v1/audio/voices`, but not `/v1/tasks/run`. Generic non-OpenAI audio tasks use the direct upstream fallback at `/upstream/{model}/v1/tasks/run` until `llama-swap` adds that route. When it lands, Studio will bump the `LLAMA_SWAP_VERSION` pin and point examples at `/v1/tasks/run` through the proxy.

## Model configuration behavior

Each model stores configuration per engine. In practice that means:

- switching a model from `llama.cpp` to `ik_llama.cpp` or `LMDeploy` does not destroy the other engine sections
- the UI shows parameters based on the scanned catalog for the active engine
- unsupported flags can be hidden in the config view
- raw custom CLI args can be appended
- the saved `llama-swap` command can be previewed from the UI and API

Useful model-related routes:

- `GET /api/models`
- `POST /api/models/search`
- `POST /api/models/download`
- `GET /api/model-catalog/search`
- `POST /api/model-catalog/install`
- `POST /api/model-catalog/import`
- `GET /api/models/{id}/config`
- `PUT /api/models/{id}/config`
- `GET /api/models/{id}/saved-llama-swap-cmd`
- `POST /api/models/{id}/preview-llama-swap-cmd`
- `POST /api/models/{id}/start`
- `POST /api/models/{id}/stop`

## Serving models

The user-facing inference endpoint is the `llama-swap` proxy on port `2000`.

Useful requests:

```bash
curl http://localhost:2000/v1/models
```

```bash
curl http://localhost:2000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "replace-with-a-model-id-from-v1-models",
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ]
  }'
```

The `model` value should come from `GET /v1/models`. It may be a sanitized Hugging Face repo plus quantization or a custom alias if you set one in model config.

### Audio models (TTS / ASR / voices)

Standard OpenAI-compatible audio routes are proxied through port `2000`:

```bash
curl http://localhost:2000/v1/audio/voices \
  -H "Authorization: Bearer local"
```

```bash
curl http://localhost:2000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "replace-with-an-audio-model-id",
    "input": "Hello from audio.cpp",
    "voice": "default"
  }' \
  --output speech.wav
```

```bash
curl http://localhost:2000/v1/audio/transcriptions \
  -H "Authorization: Bearer local" \
  -F model="replace-with-an-asr-model-id" \
  -F file="@sample.wav"
```

For generic audio tasks exposed by `/v1/tasks/run`, use the direct upstream path until `llama-swap` adds unified routing:

```bash
curl http://localhost:2000/upstream/replace-with-model-id/v1/tasks/run \
  -H "Content-Type: application/json" \
  -d '{"task":"vad","input":{"audio_path":"/path/in/container.wav"}}'
```

## App API surface

The FastAPI app exposes a small number of main route groups:

- `/api/models`: model library, downloads, config, start/stop, metadata
- `/api/model-catalog`: normalized search, install, import, and task status for verified packages
- `/api/engines`: engine capability descriptors used by the UI
- `/api/llama-versions`: engine versions, build settings, source builds, CUDA actions
- `/api/audio-cpp`: audio.cpp build, activation, status, and update checks
- `/api/lmdeploy`: LMDeploy install/remove/status/update checks
- `/api/status`: system status and proxy health
- `/api/gpu-info`: GPU and CPU capability information
- `/api/events`: Server-Sent Events for progress and notifications
- `/api/llama-swap`: stale/apply/pending proxy configuration endpoints

OpenAPI docs are available at `/docs`.

## Environment variables

Most users only need a few environment variables:

| Variable | Purpose |
| --- | --- |
| `HUGGINGFACE_API_KEY` | Access gated Hugging Face models and authenticated downloads |
| `HF_HUB_ENABLE_HF_TRANSFER=1` | Enable faster Hugging Face transfer support when available |
| `HF_HOME` | Base Hugging Face cache directory |
| `HUGGINGFACE_HUB_CACHE` | Hugging Face hub cache directory |
| `CUDA_VISIBLE_DEVICES` | Limit visible GPUs to comma-separated indices/UUIDs (e.g. `0,1`). Do not set `all` — leave unset for all devices |
| `RELOAD` | Enable or disable backend auto-reload |
| `BACKEND_CORS_ORIGINS` | Comma-separated allowed origins |
| `BACKEND_CORS_ALLOW_CREDENTIALS` | Toggle credentialed CORS requests |
| `CPU_ONLY_MODE` | Force GPU detection into CPU-only mode |
| `AUDIO_CPP_ENABLED` | Enable or disable the audio.cpp integration (`1` default, set `0`/`false` to disable) |
| `AUDIO_CPP_HEURISTIC_DISCOVERY` | Allow fuzzy package→family discovery when upstream JSON omits fields (`1` default; set `0` to require package JSON contracts) |

Advanced / less common:

| Variable | Purpose |
| --- | --- |
| `LMDEPLOY_BIN` | Override the LMDeploy executable path used by the backend |
| `CMAKE` or `CMAKE_EXECUTABLE` | Override the CMake executable used for source builds |

## Troubleshooting

### `data/` is not writable

The app needs write access to the mounted data directory. If the container logs complain about `/app/data` permissions, fix ownership on the host volume before continuing.

### No models can start

Check these in order:

1. an engine version is installed and active
2. the model was downloaded successfully
3. pending `llama-swap` changes were applied
4. `http://localhost:2000/health` is reachable

### Config changes are not reflected at inference time

Saving model config updates the database state, but the generated proxy config may still be stale. Use the UI's apply flow or call:

```bash
curl -X POST http://localhost:8080/api/llama-swap/apply-config
```

### Parameter list is empty or outdated

Use the `Engines` page action to rescan CLI parameters for the active engine. The backend builds the parameter registry from the runtime binary's `--help` output.
