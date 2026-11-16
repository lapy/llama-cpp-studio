# Per-engine CUDA toolkit binding

_Last reviewed: 2026-11-16._

Source builds of the llama.cpp family can be pinned to one installed CUDA toolkit
instead of whichever toolkit is active globally. The binding is stored in the
engine's build settings as `cuda_toolkit` (an installed toolkit version such as
`12.9`; empty means “Active Studio toolkit”).

## Resolution order

1. If `cuda_toolkit` is set, `_check_cuda_toolkit_available(preferred)` asks the
   CUDA installer for `get_cuda_env(version)` and validates that the resulting
   `CUDA_HOME` has `bin/nvcc` and a complete toolkit. An invalid or missing
   binding fails the build with a clear error instead of silently falling back.
2. Otherwise the active Studio-managed toolkit is used, then the system
   `CUDA_HOME`/`CUDA_PATH` environment as before.

The chosen root fills `CUDA_HOME`, `PATH`, and `LD_LIBRARY_PATH` for CMake; the
existing build-provenance record shows the resulting toolkit in the build task
metadata.

## UI

The llama.cpp build-settings dialog shows a **CUDA toolkit** select populated
from `cudaStatus.installed_versions`, with “Active Studio toolkit” as the
default. It is shown only for the llama-family build dialog; other engines keep
their existing installer flow.

## Scope

The binding is deliberate and per engine: changing the globally active toolkit
does not silently change a build whose engine has an explicit binding. Because
the compiled binary links against that toolkit, rebuild the engine to change the
binding.
