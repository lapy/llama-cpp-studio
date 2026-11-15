/**
 * Ordered, engine-agnostic groups for the Runtime tab's model parameters.
 *
 * The launch API exposes only flat parameter metadata (key, label, default,
 * flags, supported, ...), so grouping is pattern-based. Rules are evaluated in
 * `PARAM_GROUPS` order and the first match wins; every unmatched key falls
 * through to "Other". The patterns are deliberately narrow: a key belongs to a
 * group only when its name starts with a documented token (or contains a
 * clearly-owned token such as `gpu`/`threads`). Ambiguous names stay in "Other"
 * instead of being guessed — for example `logits_all` is an output control, not
 * a logging setting, and `session_len` is an engine-specific context length.
 *
 * Documented key families (lowercase key matching):
 *  - Context & tokens: `ctx*`, `n_ctx*`, `batch*`, `n_batch*`, `ubatch*`,
 *    `n_ubatch*`, `parallel*`, `n_parallel*`, `cont_batching*`, `cache_reuse*`,
 *    `n_keep*`, `n_discard*`, `session_len`
 *  - Memory & GPU: `gpu` as a whole token (`gpu`, `gpu_mode`, `main_gpu`,
 *    `n_gpu_layers`, `gpu_split`, `use_gpu`), `split*`, `tensor*`, `*offload*`,
 *    `mlock*`, `mmap*`, `flash_attn*`, `cache_type*`
 *  - Sampling & decoding: `temperature*`, `temp`/`temp_*`, `top_*`, `min_p*`,
 *    `repeat*`, `presence_penalty*`, `frequency_penalty*`, `seed*`, `grammar*`,
 *    `mirostat*`, `dry`/`dry_*`, `min_keep*`, `typical_p*`, `tfs_z*`, `xtc_*`,
 *    `dynatemp*`, `logit_bias*`, `n_probs*`, `n_predict*`, `ignore_eos*`,
 *    `sampler*`
 *  - Server & API: `host*`, `port`/`port_*`, `api_key*`, `*timeout*`, `slot*`,
 *    `props*`, `metrics*`, `chat_template*`, `jinja*`, `embedding*`, `rerank*`
 *  - Multimodal: `mmproj*`, `image*`, `vision*`, `audio*`, `clip*`
 *  - Performance: `*threads*`, `numa*`, `prio*`, `poll*`, `cpu` as a whole
 *    token (`cpu`, `cpu_moe`, `use_cpu`)
 *  - Logging & debug: `verbose*`, `verbosity*`, `log`/`log_*`, `logfile*`,
 *    `debug*`
 *  - Other: anything else.
 *
 * When an engine adds a genuinely new key family, add a narrow pattern here and
 * a representative test in modelParamGroups.test.js; do not broaden a pattern
 * speculatively.
 */

/** @typedef {{ id: string, title: string, description: string, icon: string, defaultOpen: boolean }} ParamGroup */

/** @type {ReadonlyArray<Readonly<ParamGroup>>} */
export const PARAM_GROUPS = Object.freeze(
  [
    {
      id: 'context',
      title: 'Context & tokens',
      description: 'Window size, batching, and how context is reused between requests.',
      icon: 'pi pi-window-maximize',
      defaultOpen: true,
    },
    {
      id: 'memory',
      title: 'Memory & GPU',
      description: 'GPU offload, split modes, cache types, and host memory mapping.',
      icon: 'pi pi-microchip',
      defaultOpen: true,
    },
    {
      id: 'sampling',
      title: 'Sampling & decoding',
      description: 'How tokens are chosen once the model is loaded.',
      icon: 'pi pi-sliders-h',
      defaultOpen: false,
    },
    {
      id: 'server',
      title: 'Server & API',
      description: 'Network surface, limits, and OpenAI-compatible API features.',
      icon: 'pi pi-server',
      defaultOpen: false,
    },
    {
      id: 'multimodal',
      title: 'Multimodal',
      description: 'Projectors and media inputs for image, vision, and audio models.',
      icon: 'pi pi-images',
      defaultOpen: false,
    },
    {
      id: 'performance',
      title: 'Performance',
      description: 'CPU threading, affinity, and runtime scheduling controls.',
      icon: 'pi pi-gauge',
      defaultOpen: false,
    },
    {
      id: 'logging',
      title: 'Logging & debug',
      description: 'Engine diagnostics, log files, and verbosity.',
      icon: 'pi pi-list',
      defaultOpen: false,
    },
    {
      id: 'other',
      title: 'Other',
      description: 'Parameters that do not match a known category yet.',
      icon: 'pi pi-ellipsis-h',
      defaultOpen: false,
    },
  ].map((group) => Object.freeze(group)),
)

/** @type {ReadonlyMap<string, Readonly<ParamGroup>>} */
const GROUP_BY_ID = new Map(PARAM_GROUPS.map((group) => [group.id, group]))

/**
 * First match wins. Keep this list in `PARAM_GROUPS` order and keep each
 * pattern anchored/narrow; see the module comment for the documented families.
 *
 * @type {ReadonlyArray<readonly [string, ReadonlyArray<RegExp>]>}
 */
const GROUP_RULES = Object.freeze([
  [
    'context',
    [
      /^ctx($|_)/,
      /^n_ctx($|_)/,
      /^batch($|_)/,
      /^n_batch($|_)/,
      /^ubatch($|_)/,
      /^n_ubatch($|_)/,
      /^parallel($|_)/,
      /^n_parallel($|_)/,
      /^cont_batching($|_)/,
      /^cache_reuse($|_)/,
      /^n_keep($|_)/,
      /^n_discard($|_)/,
      /^session_len$/,
    ],
  ],
  [
    'memory',
    [
      /(^|_)gpu($|_)/,
      /^split($|_)/,
      /^tensor($|_)/,
      /offload/,
      /^mlock($|_)/,
      /^mmap($|_)/,
      /^flash_attn($|_)/,
      /^cache_type($|_)/,
    ],
  ],
  [
    'sampling',
    [
      /^temperature($|_)/,
      /^temp($|_)/,
      /^top($|_)/,
      /^min_p($|_)/,
      /^repeat($|_)/,
      /^presence_penalty($|_)/,
      /^frequency_penalty($|_)/,
      /^seed($|_)/,
      /^grammar($|_)/,
      /^mirostat($|_)/,
      /^dry($|_)/,
      /^min_keep($|_)/,
      /^typical_p($|_)/,
      /^tfs_z($|_)/,
      /^xtc($|_)/,
      /^dynatemp($|_)/,
      /^logit_bias($|_)/,
      /^n_probs($|_)/,
      /^n_predict($|_)/,
      /^ignore_eos($|_)/,
      /^samplers?($|_)/,
    ],
  ],
  [
    'server',
    [
      /^host($|_)/,
      /^port($|_)/,
      /^api_key($|_)/,
      /timeout/,
      /^slots?($|_)/,
      /^props($|_)/,
      /^metrics($|_)/,
      /^chat_template($|_)/,
      /(^|_)jinja($|_)/,
      /^embedding($|_)/,
      /^rerank($|_)/,
    ],
  ],
  ['multimodal', [/^mmproj($|_)/, /^image($|_)/, /^vision($|_)/, /^audio($|_)/, /^clip($|_)/]],
  [
    'performance',
    [/(^|_)threads?($|_)/, /^numa($|_)/, /^prio($|_)/, /^poll($|_)/, /(^|_)cpu($|_)/],
  ],
  ['logging', [/^verbose($|_)/, /^verbosity($|_)/, /^log($|_)/, /^logfile/, /^debug($|_)/]],
])

/**
 * Group id for a parameter metadata object (or a raw key string).
 *
 * @param {{ key?: string } | string | null | undefined} param
 * @returns {string}
 */
export function groupIdForParam(param) {
  const key = typeof param === 'string' ? param : param?.key
  const normalized = String(key || '')
    .trim()
    .toLowerCase()
  if (!normalized) return 'other'
  for (const [groupId, patterns] of GROUP_RULES) {
    if (patterns.some((pattern) => pattern.test(normalized))) return groupId
  }
  return 'other'
}

/**
 * Ordered group metadata for a parameter. Unknown keys return the frozen
 * "Other" entry.
 *
 * @param {{ key?: string } | string | null | undefined} param
 * @returns {Readonly<ParamGroup>}
 */
export function groupForParam(param) {
  return GROUP_BY_ID.get(groupIdForParam(param)) || GROUP_BY_ID.get('other')
}

/**
 * Group params in `PARAM_GROUPS` order, preserving the caller's order inside
 * each group. Empty groups are omitted so callers can render only real
 * sections. `count` is the number of params in the group and `modifiedCount`
 * counts params for which `isModified` returns true (defaults to a boolean
 * `param.modified` flag).
 *
 * @template {{ key: string, modified?: boolean }} T
 * @param {ReadonlyArray<T>} params
 * @param {{ isModified?: (param: T) => boolean }} [options]
 * @returns {Array<Readonly<ParamGroup> & { params: T[], count: number, modifiedCount: number }>}
 */
export function groupParameters(params, { isModified = (param) => param?.modified === true } = {}) {
  /** @type {Map<string, T[]>} */
  const buckets = new Map(PARAM_GROUPS.map((group) => [group.id, []]))
  for (const param of params || []) {
    if (!param || typeof param !== 'object' || !param.key) continue
    buckets.get(groupIdForParam(param)).push(param)
  }
  const groups = []
  for (const group of PARAM_GROUPS) {
    const items = buckets.get(group.id)
    if (!items.length) continue
    groups.push({
      ...group,
      params: items,
      count: items.length,
      modifiedCount: items.filter((param) => isModified(param)).length,
    })
  }
  return groups
}

/**
 * Multi-word AND search over a parameter's label, key, description, and flags.
 * Mirrors the Runtime toolbar: an empty query matches nothing, and
 * `hideUnsupported` removes params the scanned engine build does not support.
 *
 * @param {{ label?: string, key?: string, description?: string, flags?: string[], supported?: boolean }} param
 * @param {string} query
 * @param {{ hideUnsupported?: boolean }} [options]
 * @returns {boolean}
 */
export function paramMatchesSearch(param, query, { hideUnsupported = false } = {}) {
  if (!param?.key) return false
  if (hideUnsupported && param.supported === false) return false
  const raw = String(query ?? '')
    .trim()
    .toLowerCase()
  if (!raw) return false
  const haystack = [
    param.label || '',
    param.key || '',
    param.description || '',
    ...(Array.isArray(param.flags) ? param.flags : []),
  ]
    .join(' ')
    .toLowerCase()
  return raw
    .split(/\s+/)
    .filter(Boolean)
    .every((token) => haystack.includes(token))
}
