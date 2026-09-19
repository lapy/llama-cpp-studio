/**
 * Parse engine CLI command text into catalog parameter values.
 * Inverse of llama-swap structured token emission.
 */

/** Flags Studio always injects or owns — never import, even if the catalog lists them. */
export const STUDIO_RESERVED_FLAGS = new Set([
  '--alias',
  '--help',
  '--hf-file',
  '--hf-repo',
  '--host',
  '--hostname',
  '--listen',
  '--mmproj',
  '--model',
  '--model-path',
  '--port',
  '--server-port',
  '--usage',
  '--version',
  '-h',
  '-m',
])

/** Config keys that map to Studio-owned routing / bind / weights. */
export const STUDIO_RESERVED_KEYS = new Set([
  'alias',
  'hf_file',
  'hf_repo',
  'host',
  'hostname',
  'listen',
  'mmproj',
  'model',
  'model_path',
  'port',
  'server_port',
])

export const STUDIO_ENV_PREFIX = 'LLAMA_STUDIO_'
export const SWAP_ENV_KEY_RE = /^[A-Za-z_][A-Za-z0-9_]*$/
/** Env names that collide with Studio-owned listen / identity, plus the reserved prefix. */
export const STUDIO_RESERVED_ENV_KEYS = new Set([
  'HOST',
  'HOSTNAME',
  'PORT',
  'SERVER_PORT',
])
const COMMAND_WORDS = new Set([
  'sglang',
  'serve',
  'vllm',
  'lmdeploy',
  'api_server',
  'launch_server',
  'export',
])

const PYTHON_MODULE_RE = /^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$/
const MACRO_RE = /^\$\{.+\}$/
const ENV_VAR_RE = /^\$[A-Za-z_][A-Za-z0-9_]*$/
const NUMERIC_TOKEN_RE = /^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/

/**
 * POSIX-ish tokenizer (quotes, backslash, line continuations).
 * @param {string} text
 * @returns {{ tokens: string[], parseError: string | null }}
 */
export function tokenizeCli(text) {
  const src = String(text ?? '')
  const tokens = []
  let buf = ''
  let i = 0
  let quote = null
  let parseError = null

  const flush = () => {
    if (buf.length) {
      tokens.push(buf)
      buf = ''
    }
  }

  while (i < src.length) {
    const ch = src[i]

    if (quote === "'") {
      if (ch === "'") quote = null
      else buf += ch
      i += 1
      continue
    }

    if (quote === '"') {
      if (ch === '\\' && i + 1 < src.length) {
        buf += src[i + 1]
        i += 2
        continue
      }
      if (ch === '"') quote = null
      else buf += ch
      i += 1
      continue
    }

    if (ch === '\\') {
      if (i + 1 >= src.length) {
        buf += '\\'
        i += 1
        continue
      }
      const next = src[i + 1]
      if (next === '\n') {
        i += 2
        continue
      }
      if (next === '\r') {
        i += src[i + 2] === '\n' ? 3 : 2
        continue
      }
      buf += next
      i += 2
      continue
    }

    if (ch === "'" || ch === '"') {
      quote = ch
      i += 1
      continue
    }

    if (/\s/.test(ch)) {
      flush()
      i += 1
      continue
    }

    buf += ch
    i += 1
  }

  if (quote) {
    parseError = 'Unclosed quote in command text'
    flush()
  } else {
    flush()
  }

  return { tokens, parseError }
}

export function isOptionToken(token) {
  if (!token || token === '-') return false
  if (token === '--') return true
  if (!token.startsWith('-')) return false
  if (NUMERIC_TOKEN_RE.test(token)) return false
  return true
}

export function splitFlagEquals(token) {
  if (!token || token === '--' || !token.startsWith('-')) {
    return { flag: token, value: undefined, inline: false }
  }
  const eq = token.indexOf('=')
  if (eq <= 1) return { flag: token, value: undefined, inline: false }
  return {
    flag: token.slice(0, eq),
    value: token.slice(eq + 1),
    inline: true,
  }
}

function tokenBasename(token) {
  const cleaned = String(token || '').replace(/^['"]|['"]$/g, '')
  const parts = cleaned.split(/[\\/]/)
  return parts[parts.length - 1] || cleaned
}

function isPythonBinary(token) {
  const base = tokenBasename(token).toLowerCase()
  return base === 'python' || /^python\d/.test(base)
}

export function isIgnorablePositional(token) {
  if (!token) return true
  if (['&&', '||', ';', '|', '&'].includes(token)) return true
  if (MACRO_RE.test(token) || ENV_VAR_RE.test(token)) return true
  if (isPythonBinary(token)) return true
  const base = tokenBasename(token)
  if (COMMAND_WORDS.has(base.toLowerCase())) return true
  if (/^(llama-server|llama-cli|llama-batched-bench|llama-swap)(\.exe)?$/i.test(base)) return true
  if (base.includes('${')) return true
  return false
}

export function splitEnvAssignment(token) {
  if (!token || token.startsWith('-')) return null
  const eq = token.indexOf('=')
  if (eq <= 0) return null
  const key = token.slice(0, eq)
  if (!SWAP_ENV_KEY_RE.test(key)) return null
  return { key, value: token.slice(eq + 1) }
}

export function canonicalEnvKey(key) {
  const name = String(key || '')
  if (name.toUpperCase() === 'CUDA_VISIBLE_DEVICES') return 'CUDA_VISIBLE_DEVICES'
  if (name.toUpperCase() === 'LD_LIBRARY_PATH') return 'LD_LIBRARY_PATH'
  return name
}

export function studioEnvSkipReason(key) {
  const name = String(key || '')
  if (!name) return null
  if (name.startsWith(STUDIO_ENV_PREFIX)) {
    return 'LLAMA_STUDIO_* keys are reserved and ignored by Studio'
  }
  const upper = name.toUpperCase()
  if (STUDIO_RESERVED_ENV_KEYS.has(upper)) {
    const mapped = upper === 'SERVER_PORT' ? 'server_port' : upper.toLowerCase()
    return studioReasonForKey(mapped)
  }
  return null
}

export function flagTakesValue(param) {
  if (!param) return true
  return param.value_kind !== 'flag'
}

export function coerceBool(raw) {
  if (typeof raw === 'boolean') return raw
  const text = String(raw ?? '').trim().toLowerCase()
  if (['1', 'true', 'yes', 'on'].includes(text)) return true
  if (['0', 'false', 'no', 'off'].includes(text)) return false
  return null
}

function optionValues(param) {
  const out = []
  for (const item of param?.options || []) {
    if (item && typeof item === 'object') {
      if (item.value != null && item.value !== '') out.push(item.value)
    } else if (item != null && item !== '') {
      out.push(item)
    }
  }
  return out
}

export function delimitedEnumSeparator(param) {
  return param?.value_kind === 'semicolon_enum' ? ';' : ','
}

export function isDelimitedEnumParam(param) {
  return ['csv_enum', 'semicolon_enum'].includes(param?.value_kind) || param?.type === 'multiselect'
}

export function coerceCliValue(raw, param, { polarity } = {}) {
  const kind = param?.value_kind || 'scalar'
  if (kind === 'flag') {
    if (polarity === 'negative') return false
    if (raw == null || raw === '') return true
    const parsed = coerceBool(raw)
    return parsed == null ? true : parsed
  }
  if (raw == null) return null
  if (kind === 'json_object' || param?.type === 'json') {
    if (typeof raw === 'object') return raw
    const trimmed = String(raw).trim()
    if (!trimmed) return null
    try {
      return JSON.parse(trimmed)
    } catch {
      return String(raw)
    }
  }
  if (kind === 'repeatable') {
    return raw === '' ? [] : [raw]
  }
  if (isDelimitedEnumParam(param)) {
    return String(raw)
      .split(delimitedEnumSeparator(param))
      .map((part) => part.trim())
      .filter(Boolean)
  }
  if (param?.type === 'int' || param?.scalar_type === 'int') {
    const text = String(raw).trim()
    if (!text || !NUMERIC_TOKEN_RE.test(text)) return raw
    const n = Number.parseInt(text, 10)
    return Number.isNaN(n) ? raw : n
  }
  if (param?.type === 'float' || param?.scalar_type === 'float') {
    const text = String(raw).trim()
    if (!text || !NUMERIC_TOKEN_RE.test(text)) return raw
    const n = Number.parseFloat(text)
    return Number.isNaN(n) ? raw : n
  }
  if (param?.type === 'bool') {
    const parsed = coerceBool(raw)
    return parsed == null ? raw : parsed
  }
  return raw
}

export function mergeParsedValues(existing, incoming, param) {
  if (existing === undefined) return incoming
  if (param?.value_kind === 'repeatable') {
    const left = Array.isArray(existing) ? existing : existing == null ? [] : [existing]
    const right = Array.isArray(incoming) ? incoming : incoming == null ? [] : [incoming]
    return [...left, ...right]
  }
  if (isDelimitedEnumParam(param)) {
    const left = Array.isArray(existing) ? existing : []
    const right = Array.isArray(incoming) ? incoming : []
    const seen = new Set(left.map(String))
    const out = [...left]
    for (const item of right) {
      const key = String(item)
      if (seen.has(key)) continue
      seen.add(key)
      out.push(item)
    }
    return out
  }
  return incoming
}

/**
 * @param {Array<Record<string, any>>} catalogParams
 * @returns {Map<string, { param: Record<string, any>, polarity: 'positive' | 'negative' }>}
 */
export function buildFlagIndex(catalogParams) {
  const index = new Map()
  for (const param of catalogParams || []) {
    if (!param || !param.key) continue
    const flags = new Set()
    for (const flag of [param.primary_flag, param.negative_flag, ...(param.flags || [])]) {
      if (flag) flags.add(String(flag))
    }
    for (const flag of flags) {
      const polarity = param.negative_flag && flag === param.negative_flag ? 'negative' : 'positive'
      if (!index.has(flag)) index.set(flag, { param, polarity })
    }
  }
  return index
}

export function quoteCliToken(token) {
  const text = String(token ?? '')
  if (text === '') return "''"
  if (!/[\s'"\\]/.test(text)) return text
  return `'${text.replace(/'/g, `'\\''`)}'`
}

export function joinCliTokens(tokens) {
  return (tokens || []).map(quoteCliToken).join(' ').trim()
}

export function valuesEqual(a, b) {
  if (Object.is(a, b)) return true
  if (a == null && b == null) return true
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false
    return a.every((item, idx) => valuesEqual(item, b[idx]))
  }
  if (a && b && typeof a === 'object' && typeof b === 'object' && !Array.isArray(a) && !Array.isArray(b)) {
    try {
      return JSON.stringify(a) === JSON.stringify(b)
    } catch {
      return false
    }
  }
  if (a != null && b != null && String(a) === String(b)) return true
  return false
}

export function formatCliValue(value, { emptyLabel = 'Not set' } = {}) {
  if (value === undefined) return emptyLabel
  if (value === null) return 'Default'
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (Array.isArray(value)) return value.length ? value.join(', ') : '(empty)'
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value)
    } catch {
      return String(value)
    }
  }
  return String(value)
}

export function studioSkipReason(flag, param) {
  const key = param?.key
  if (key && STUDIO_RESERVED_KEYS.has(key)) return studioReasonForKey(key)
  if (STUDIO_RESERVED_FLAGS.has(flag)) {
    const mapped = flag.replace(/^--/, '').replace(/-/g, '_')
    if (STUDIO_RESERVED_KEYS.has(mapped)) return studioReasonForKey(mapped)
    if (flag === '-m') return studioReasonForKey('model')
    if (flag === '-h') return 'Help / version flags are not imported'
    return 'Managed by Studio'
  }
  if (param?.reserved) return 'Reserved by this engine catalog — Studio manages this option'
  return null
}

function studioReasonForKey(key) {
  if (['host', 'hostname', 'listen', 'port', 'server_port'].includes(key)) {
    return 'Studio / llama-swap owns the listen address and port'
  }
  if (['model', 'model_path', 'hf_repo', 'hf_file', 'mmproj'].includes(key)) {
    return 'Studio supplies the model and companion paths'
  }
  if (key === 'alias') return 'API alias is configured on this page, not from CLI --alias'
  return 'Managed by Studio'
}

export function unsupportedSkipReason(param) {
  if (param?.supported === false) {
    return 'Not available in this engine build (deprecated or unsupported)'
  }
  return null
}

function consumeOptionalValue(tokens, index, inline, inlineValue) {
  if (inline) {
    return { value: inlineValue, consumed: 0 }
  }
  if (index + 1 < tokens.length && !isOptionToken(tokens[index + 1])) {
    return { value: tokens[index + 1], consumed: 1 }
  }
  return { value: undefined, consumed: 0 }
}

function recordParsedValue(bucket, param, flag, value, tokens, extra = {}) {
  const prev = bucket.get(param.key)
  if (
    prev
    && param.value_kind !== 'repeatable'
    && !isDelimitedEnumParam(param)
    && !valuesEqual(prev.value, value)
  ) {
    extra.onReplace?.(flag, param)
  }
  bucket.set(param.key, {
    key: param.key,
    value: mergeParsedValues(prev?.value, value, param),
    sourceFlag: flag,
    tokens: [...(prev?.tokens || []), ...tokens],
    param,
    ...extra.fields,
  })
}

/**
 * @param {string} text
 * @param {Array<Record<string, any>>} catalogParams
 * @returns {{
 *   params: Array<{ key: string, value: any, sourceFlag: string, tokens: string[], param: Record<string, any> }>,
 *   reserved: Array<{ flag: string, key?: string, value?: string, tokens: string[], reason: string }>,
 *   env: Array<{ key: string, value: string, tokens: string[] }>,
 *   unsupported: Array<{ key: string, value: any, sourceFlag: string, tokens: string[], param: Record<string, any>, reason: string }>,
 *   unknown: Array<{ tokens: string[], reason: string }>,
 *   ignored: Array<{ tokens: string[], reason: string }>,
 *   warnings: string[],
 *   parseError: string | null,
 *   leftoverTokens: string[],
 * }}
 */
export function parseCliCommand(text, catalogParams = []) {
  const { tokens, parseError } = tokenizeCli(text)
  const flagIndex = buildFlagIndex(catalogParams)
  const paramsByKey = new Map()
  const envByKey = new Map()
  const unsupportedByKey = new Map()
  const reserved = []
  const unknown = []
  const ignored = []
  const warnings = []
  const onReplace = (flag, param) => {
    warnings.push(`Later ${flag} replaces earlier value for ${param.key}`)
  }

  const recordEnv = (rawKey, rawValue, consumedTokens) => {
    const key = canonicalEnvKey(rawKey)
    const value = rawValue == null ? '' : String(rawValue)
    const skip = studioEnvSkipReason(key)
    if (skip) {
      reserved.push({
        flag: `${key}=${value}`,
        key,
        value,
        tokens: consumedTokens,
        reason: skip,
      })
      return
    }
    if (!value.trim()) {
      warnings.push(`Empty value for environment variable ${key}`)
      return
    }
    const prev = envByKey.get(key)
    if (prev && !valuesEqual(prev.value, value)) {
      warnings.push(`Later ${key} replaces earlier environment value`)
    }
    envByKey.set(key, {
      key,
      value,
      tokens: [...(prev?.tokens || []), ...consumedTokens],
    })
  }

  let i = 0
  while (i < tokens.length) {
    const token = tokens[i]

    if (token === '--') {
      ignored.push({ tokens: tokens.slice(i), reason: 'end of options' })
      break
    }

    if (token === '-m' && tokens[i + 1] && PYTHON_MODULE_RE.test(tokens[i + 1])) {
      ignored.push({ tokens: [token, tokens[i + 1]], reason: 'python module' })
      i += 2
      continue
    }

    if (token === 'export' && tokens[i + 1]) {
      const exported = splitEnvAssignment(tokens[i + 1])
      if (exported) {
        recordEnv(exported.key, exported.value, [token, tokens[i + 1]])
        i += 2
        continue
      }
    }

    const envAssign = splitEnvAssignment(token)
    if (envAssign) {
      recordEnv(envAssign.key, envAssign.value, [token])
      i += 1
      continue
    }

    if (!isOptionToken(token)) {
      if (isIgnorablePositional(token)) {
        ignored.push({ tokens: [token], reason: 'command prefix' })
      } else {
        unknown.push({ tokens: [token], reason: 'unrecognized token' })
      }
      i += 1
      continue
    }

    const { flag, value: inlineValue, inline } = splitFlagEquals(token)
    const hit = flagIndex.get(flag)
    const studioReason = studioSkipReason(flag, hit?.param)

    if (studioReason) {
      const taken = consumeOptionalValue(tokens, i, inline, inlineValue)
      const consumed = [token, ...tokens.slice(i + 1, i + 1 + taken.consumed)]
      reserved.push({
        flag,
        key: hit?.param?.key,
        value: taken.value,
        tokens: consumed,
        reason: studioReason,
      })
      i += 1 + taken.consumed
      continue
    }

    if (!hit) {
      const taken = consumeOptionalValue(tokens, i, inline, inlineValue)
      const consumed = [token, ...tokens.slice(i + 1, i + 1 + taken.consumed)]
      unknown.push({
        tokens: consumed,
        reason: 'Not in this engine catalog (unknown or removed)',
      })
      i += 1 + taken.consumed
      continue
    }

    const { param, polarity } = hit
    const takesValue = flagTakesValue(param)
    let raw
    const consumed = [token]
    if (takesValue) {
      if (inline) {
        raw = inlineValue
      } else if (i + 1 < tokens.length && !isOptionToken(tokens[i + 1])) {
        raw = tokens[i + 1]
        consumed.push(raw)
        i += 1
      } else {
        warnings.push(`Missing value for ${flag}`)
        i += 1
        continue
      }
    } else {
      raw = inline ? inlineValue : undefined
    }

    const incoming = coerceCliValue(raw, param, { polarity })
    const allowed = optionValues(param)
    if (allowed.length && incoming != null && !Array.isArray(incoming) && !allowed.map(String).includes(String(incoming))) {
      warnings.push(`${flag} value ${raw} is not in the catalog options for ${param.key}`)
    }

    const deprecatedReason = unsupportedSkipReason(param)
    if (deprecatedReason) {
      recordParsedValue(unsupportedByKey, param, flag, incoming, consumed, {
        onReplace,
        fields: { reason: deprecatedReason },
      })
      i += 1
      continue
    }

    recordParsedValue(paramsByKey, param, flag, incoming, consumed, { onReplace })
    i += 1
  }

  return {
    params: [...paramsByKey.values()],
    env: [...envByKey.values()],
    reserved,
    unsupported: [...unsupportedByKey.values()],
    unknown,
    ignored,
    warnings,
    parseError,
    leftoverTokens: unknown.flatMap((row) => row.tokens),
  }
}

export function buildImportPreview(parsed, currentValues = {}) {
  return (parsed?.params || []).map((row) => {
    const present = Object.prototype.hasOwnProperty.call(currentValues, row.key)
    const currentValue = present ? currentValues[row.key] : undefined
    return {
      ...row,
      currentValue,
      change: changeKind(currentValue, row.value, { present }),
      label: row.param?.label || row.key,
    }
  })
}

export function currentEnvLookup(currentEnv, key) {
  const env = currentEnv && typeof currentEnv === 'object' && !Array.isArray(currentEnv) ? currentEnv : {}
  const exact = Object.prototype.hasOwnProperty.call(env, key) ? key : null
  if (exact) return { present: true, key: exact, value: env[exact] }
  const upper = String(key || '').toUpperCase()
  const found = Object.keys(env).find((name) => name.toUpperCase() === upper)
  if (found) return { present: true, key: found, value: env[found] }
  return { present: false, key, value: undefined }
}

export function buildEnvImportPreview(parsed, currentEnv = {}) {
  return (parsed?.env || []).map((row) => {
    const current = currentEnvLookup(currentEnv, row.key)
    return {
      ...row,
      currentValue: current.present ? current.value : undefined,
      change: changeKind(current.present ? current.value : undefined, row.value, { present: current.present }),
    }
  })
}

export function changeKind(currentValue, incomingValue, { present = false } = {}) {
  if (!present) return 'new'
  if (valuesEqual(currentValue, incomingValue)) return 'unchanged'
  return 'update'
}
