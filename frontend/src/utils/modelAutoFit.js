/**
 * Auto-fit detection and restoration for engines that expose the `--fit`
 * parameter in their scanned launch API.
 *
 * Current llama.cpp enables `--fit` by default: it measures the real model,
 * context, and compute buffers, then chooses context and per-device GPU layer
 * counts (including partial MoE tensor offload) to fit the available memory.
 * An explicit `--ctx-size` prevents the fitter from choosing the context, and a
 * numeric `--n-gpu-layers` prevents it from choosing the layer count; `--fit
 * off` disables both. Flash attention is NOT part of the fitter: its own
 * default is `auto`, independent of `--fit`.
 *
 * Canonical llama.cpp server flags/defaults (2026-11-15):
 *   -fit,   --fit [on|off]      default 'on'
 *   --fit-ctx N                 default 4096 (minimum context auto-fit may choose)
 *   -fitt,  --fit-target MiB    default 1024 (target margin per device)
 *   -c,     --ctx-size N        default 0 (0 = loaded from model)
 *   -ngl,   --gpu-layers N      default 'auto' (exact number, 'auto', or 'all')
 *   -fa,    --flash-attn        default 'auto'
 *
 * Sources (cite when changing these rules):
 *   https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
 *   https://github.com/ggml-org/llama.cpp/blob/master/common/fit.cpp
 *
 * Detection is capability-based, never engine-id-based: an engine supports
 * auto-fit exactly when its scanned parameter list contains `fit` (key `fit`,
 * primary flag `--fit`, or `--fit` in its flags).
 */

/** Flag carried by the `fit` parameter in the scanned CLI catalog. */
export const AUTO_FIT_FLAG = '--fit'

/**
 * The only parameters this module restores. Everything else (flash attention,
 * threads, batch, parallel, sampling values) is deliberately left alone:
 * auto-fit does not own those decisions.
 */
export const AUTO_FIT_OVERRIDE_KEYS = Object.freeze(['fit', 'ctx_size', 'n_gpu_layers'])

function positiveNumber(value) {
  const number = Number(value)
  return Number.isFinite(number) && number > 0 ? number : null
}

function optionValues(param) {
  const options = Array.isArray(param?.options) ? param.options : []
  return options
    .map((option) => (option && typeof option === 'object' ? option.value : option))
    .filter((value) => value != null)
}

/**
 * Find the engine's `fit` parameter. Returns `null` when the scanned parameter
 * list does not advertise auto-fit.
 *
 * @param {Array<object>|null|undefined} parameters
 * @returns {object|null}
 */
export function findAutoFitParam(parameters) {
  const list = Array.isArray(parameters) ? parameters : []
  return (
    list.find(
      (param) =>
        param &&
        (param.key === 'fit' ||
          param.primary_flag === AUTO_FIT_FLAG ||
          (Array.isArray(param.flags) && param.flags.includes(AUTO_FIT_FLAG))),
    ) || null
  )
}

/** Defaults quoted from the engine's own parameter metadata (never invented). */
export function autoFitDefaults(parameters) {
  const list = Array.isArray(parameters) ? parameters : []
  const byKey = new Map(list.filter((param) => param?.key).map((param) => [param.key, param]))
  return {
    fit: byKey.get('fit')?.default ?? null,
    fitCtx: positiveNumber(byKey.get('fit_ctx')?.default),
    fitTarget: positiveNumber(byKey.get('fit_target')?.default),
  }
}

/** True when the value is an explicit numeric layer/context override. */
function isNumericOverride(value) {
  if (value == null || value === '') return false
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase()
    if (!normalized || normalized === 'auto' || normalized === 'all') return false
  }
  return Number.isFinite(Number(value))
}

function isContextOverride(value) {
  return isNumericOverride(value) && Number(value) !== 0
}

function isFitDisabled(value) {
  if (value == null || value === '') return false
  if (value === false) return true
  return ['off', 'false', '0'].includes(String(value).trim().toLowerCase())
}

/** The control only accepts text when it is a select/string, not an int input. */
function acceptsAutoString(param) {
  if (!param) return false
  if (optionValues(param).includes('auto')) return true
  if (param.type === 'string') return true
  if (param.type === 'select' && !optionValues(param).length) return true
  return false
}

function contextRow(param, current) {
  return {
    key: 'ctx_size',
    label: param.label || 'Context size',
    current,
    autoLabel: 'Auto (0)',
    restore: { mode: 'set', value: 0 },
    reason:
      'An explicit --ctx-size prevents auto-fit from choosing the context. Setting 0 makes the engine load the model context and fit it to memory instead.',
  }
}

function layerRow(param, current) {
  const restore = acceptsAutoString(param) ? { mode: 'set', value: 'auto' } : { mode: 'clear' }
  return {
    key: 'n_gpu_layers',
    label: param.label || 'GPU layers',
    current,
    autoLabel: 'Auto',
    restore,
    reason:
      'A numeric layer count prevents auto-fit from choosing layers. Clearing it lets the engine choose per-device layers, including partial MoE offload.',
  }
}

function fitRow(param, current) {
  const offersOn = optionValues(param).includes('on')
  const restore =
    offersOn || param.type === 'string' || (param.type === 'select' && !optionValues(param).length)
      ? { mode: 'set', value: 'on' }
      : { mode: 'clear' }
  return {
    key: 'fit',
    label: param.label || 'Auto-fit',
    current: current === false ? 'off' : current,
    autoLabel: 'On',
    restore,
    reason:
      'Auto-fit is explicitly off, so the engine keeps fixed context and layer settings instead of sizing them to memory. Turning it on lets the engine measure the model at load time.',
  }
}

/**
 * Settings currently overridden that defeat auto-fit, in restore order.
 * Only keys present in the scanned parameter list are considered, and a
 * non-empty result implies the engine advertises `fit`.
 *
 * @param {{ parameters?: Array<object>, config?: object }} input
 * @returns {Array<{key:string,label:string,current:*,autoLabel:string,restore:{mode:'set'|'clear',value?:*},reason:string}>}
 */
export function detectAutoFitOverrides({ parameters, config } = {}) {
  const list = Array.isArray(parameters) ? parameters : []
  if (!findAutoFitParam(list)) return []
  const byKey = new Map(list.filter((param) => param?.key).map((param) => [param.key, param]))
  const values = config && typeof config === 'object' ? config : {}
  const rows = []

  const fit = byKey.get('fit')
  if (fit && isFitDisabled(values.fit)) rows.push(fitRow(fit, values.fit))

  const ctx = byKey.get('ctx_size')
  if (ctx && isContextOverride(values.ctx_size)) rows.push(contextRow(ctx, values.ctx_size))

  const layers = byKey.get('n_gpu_layers')
  if (layers && isNumericOverride(values.n_gpu_layers)) {
    rows.push(layerRow(layers, values.n_gpu_layers))
  }

  return rows
}

/**
 * Complete auto-fit state for the current engine + draft.
 *
 * @param {{ parameters?: Array<object>, config?: object }} input
 */
export function buildAutoFitState({ parameters, config } = {}) {
  const fitParam = findAutoFitParam(parameters)
  if (!fitParam) {
    return {
      supported: false,
      fitParam: null,
      overrides: [],
      active: false,
      defaults: { fit: null, fitCtx: null, fitTarget: null },
    }
  }
  const overrides = detectAutoFitOverrides({ parameters, config })
  return {
    supported: true,
    fitParam,
    overrides,
    active: overrides.length === 0,
    defaults: autoFitDefaults(parameters),
  }
}
