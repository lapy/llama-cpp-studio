# Runtime auto-fit

_Last reviewed: 2026-11-15._

Modern llama.cpp sizes a model for the machine at load time. The server enables
`--fit` by default: it measures the actual model, context, and compute buffers
and then chooses the context size and per-device GPU layer counts (including
partial MoE tensor offload) that fit the available memory. Studio should not
invent those numbers; it should point users back at the engine when they have
overridden the inputs the fitter owns.

## Canonical flags and defaults (llama.cpp server, 2026)

| Flag                             | Meaning                                        | Default |
| -------------------------------- | ---------------------------------------------- | ------- |
| `-fit`, `--fit [on\|off]`        | Fit unset arguments to device memory           | `on`    |
| `-fitt`, `--fit-target MiB0,...` | Target margin per device                       | `1024`  |
| `--fit-ctx N`                    | Minimum context auto-fit may choose            | `4096`  |
| `-c`, `--ctx-size N`             | Prompt context size; `0` = loaded from model   | `0`     |
| `-ngl`, `--gpu-layers N`         | Layers in VRAM; exact number, `auto`, or `all` | `auto`  |
| `-fa`, `--flash-attn`            | Flash attention: `on`, `off`, or `auto`        | `auto`  |

## Rationale

An explicit `--ctx-size` prevents the fitter from choosing the context, and a
numeric `--n-gpu-layers` prevents it from choosing the layer count; `--fit off`
disables both. A fixed 4K or 8K context chosen without measurement can waste
VRAM (too small) or fail to start (too large); a blanket KV-cache estimate is
less accurate than the fitter's own measurement, especially for MoE models and
multi-GPU setups. Flash attention is independent of `--fit` and keeps its own
`auto` default, so Studio does not touch it here. Studio does not recommend
numeric context sizes, layer counts, thread counts, batch sizes, or memory
estimates.

Instead, the model configuration page identifies exactly the settings that
override auto-fit:

- `fit = off`
- `ctx_size != 0`
- a numeric `n_gpu_layers` (not `auto`/`all`)

and offers one action that restores the engine defaults:

- `fit = on`
- `ctx_size = 0`
- clear the `n_gpu_layers` override (the numeric control cannot hold `auto`)

The action only edits the draft. Saving and applying remain the action bar's job.

## Detection

Auto-fit support is capability-based, never engine-id-based: the scanned
parameter list must contain the `fit` parameter (key `fit`, primary flag
`--fit`, or `--fit` in its flags). Engines whose scans do not expose `fit` get no
auto-fit card or action.

## GPU admission is not a fitter

The proxy's GPU admission preflight (`vram_admission`, see
[proxy operations](proxy-operations.md#gpu-admission)) answers a narrower
question than auto-fit: it decides only whether a start is obviously impossible
on the selected GPUs before a process is spawned. It does not choose a context
size, does not choose layer counts, does not pass any argument, and does not add
a model-configuration field.

The two mechanisms are intentionally different kinds of evidence:

- Auto-fit measures the actual model, context and compute buffers inside the
  engine and then chooses what fits. That measurement is authoritative.
- Admission compares a coarse estimate - an operator override, a previous load's
  ready-time delta, or the file-size lower bound - against free memory. It can be
  wrong in both directions, which is why `enforce` refuses only a known deficit
  from an override or an observation, why a weights-only deficit only warns, and
  why an unknown estimate or failed probe admits.

Consequences for operators:

- Auto-fit remains the authority for memory sizing. Restoring the auto-fit
  defaults is still the right fix for an overridden context or layer count; a
  generous `vram_estimates` entry is not a substitute for `fit = on`.
- A warning is not a prediction of OOM; it says the estimate exceeds the
  available headroom after the safety margin. The estimate and its source are
  shown with every decision so the operator can judge the evidence.
- Admission estimates reset when the Studio process restarts, because a load
  observation is only valid for that process lifetime. Set an override when a
  model's residency must be known before the first load.

## Sources

- [llama.cpp `tools/server/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
  — `--fit`, `--fit-ctx`, `--fit-target`, `--ctx-size`, `--gpu-layers`, and
  `--flash-attn` defaults.
- [`common/fit.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/common/fit.cpp)
  — the measurement/fitting implementation that assigns context and per-device
  layer counts.
