/**
 * Build a human-readable diff between the working model config and the saved
 * snapshot. Values are grouped so the review dialog can explain what Save will
 * change without exposing raw JSON.
 */

const TOP_LEVEL_LABELS = {
  engine: 'Engine',
  model_alias: 'API alias',
  swap_aliases: 'Alias swaps',
  set_params_by_id: 'Request parameter aliases',
  custom_args: 'Custom arguments',
  runtime_env: 'Environment variables',
  runtime_env_unset: 'Unset environment variables',
  gpu_mode: 'GPU selection',
  gpu_devices: 'GPU devices',
  load_options: 'Load options',
  session_options: 'Session options',
}

const ROUTING_KEYS = new Set(['model_alias', 'swap_aliases', 'set_params_by_id'])
const RUNTIME_KEYS = new Set([
  'custom_args',
  'runtime_env',
  'runtime_env_unset',
  'gpu_mode',
  'gpu_devices',
  'load_options',
  'session_options',
])
const INTERNAL_KEYS = new Set(['engines', 'last_reviewed_fingerprint'])

function hasOwn(value, key) {
  return Object.prototype.hasOwnProperty.call(value || {}, key)
}

function jsonEqual(left, right) {
  if (left === right) return true
  try {
    return JSON.stringify(left) === JSON.stringify(right)
  } catch {
    return false
  }
}

function inlineValue(value) {
  if (value === undefined) return 'Default'
  if (value === null) return 'Not set'
  if (typeof value === 'boolean') return value ? 'On' : 'Off'
  if (typeof value === 'number') return String(value)
  if (typeof value === 'string') return value === '' ? 'Empty' : value
  if (Array.isArray(value)) {
    if (!value.length) return 'None'
    if (value.every((item) => ['string', 'number', 'boolean'].includes(typeof item))) {
      return value.map(String).join(', ')
    }
    return JSON.stringify(value)
  }
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

function summarizeAliases(value) {
  if (!Array.isArray(value) || !value.length) return 'None'
  return value
    .map((alias) => {
      const suffix = String(alias?.sub_id || '').trim() || '(base)'
      const params = alias?.params && typeof alias.params === 'object' ? alias.params : {}
      const entries = Object.entries(params)
      if (!entries.length) return suffix
      const summary = entries
        .slice(0, 4)
        .map(([key, parameter]) => `${key}=${inlineValue(parameter)}`)
        .join(', ')
      const more = entries.length > 4 ? ` +${entries.length - 4} more` : ''
      return `${suffix}: ${summary}${more}`
    })
    .join(' · ')
}

function summarizeValue(key, value) {
  if (key === 'set_params_by_id') return summarizeAliases(value)
  if (key === 'runtime_env' && value && typeof value === 'object' && !Array.isArray(value)) {
    const entries = Object.entries(value)
    if (!entries.length) return 'None'
    return entries
      .slice(0, 8)
      .map(([name, setting]) => `${name}=${inlineValue(setting)}`)
      .join(', ')
  }
  return inlineValue(value)
}

function changeKind(before, after) {
  if (before === undefined && after !== undefined) return 'added'
  if (before !== undefined && after === undefined) return 'removed'
  return 'changed'
}

function buildRow(key, label, before, after) {
  if (jsonEqual(before, after)) return null
  return {
    key,
    label,
    kind: changeKind(before, after),
    before: summarizeValue(key, before),
    after: summarizeValue(key, after),
  }
}

function sortedUnion(left, right) {
  return [...new Set([...Object.keys(left || {}), ...Object.keys(right || {})])].sort()
}

/**
 * @param {object} current Working config.
 * @param {object} saved Last saved config.
 * @param {{paramLabels?: Record<string,string>, engineLabel?: string}} [options]
 * @returns {Array<{id:string,label:string,rows:Array}>}
 */
export function buildConfigChanges(current = {}, saved = {}, options = {}) {
  const currentEngine = String(current?.engine || '')
  const savedEngine = String(saved?.engine || '')
  const paramLabels = options.paramLabels || {}
  const currentSection = (current?.engines && current.engines[currentEngine]) || {}
  const savedSection = (saved?.engines && saved.engines[savedEngine]) || {}

  const routingRows = []
  const runtimeRows = []
  const paramRows = []
  const sectionKeys = sortedUnion(currentSection, savedSection)

  const topKeys = sortedUnion(current, saved).filter((key) => !INTERNAL_KEYS.has(key))
  for (const key of topKeys) {
    // The working form mirrors routing/runtime fields from the active engine
    // section. Compare the section copy only, or every change appears twice.
    if (sectionKeys.includes(key)) continue
    const before = hasOwn(saved, key) ? saved[key] : undefined
    const after = hasOwn(current, key) ? current[key] : undefined
    const label = TOP_LEVEL_LABELS[key] || options.paramLabels?.[key] || key
    const row = buildRow(key, label, before, after)
    if (!row) continue
    if (key === 'engine') runtimeRows.push(row)
    else if (ROUTING_KEYS.has(key)) routingRows.push(row)
    else if (RUNTIME_KEYS.has(key)) runtimeRows.push(row)
    else paramRows.push(row)
  }

  for (const key of sectionKeys) {
    const before = hasOwn(savedSection, key) ? savedSection[key] : undefined
    const after = hasOwn(currentSection, key) ? currentSection[key] : undefined
    const label = paramLabels[key] || TOP_LEVEL_LABELS[key] || key
    const row = buildRow(key, label, before, after)
    if (!row) continue
    if (ROUTING_KEYS.has(key)) routingRows.push(row)
    else if (RUNTIME_KEYS.has(key)) runtimeRows.push(row)
    else paramRows.push(row)
  }

  const groups = []
  if (routingRows.length) {
    groups.push({ id: 'routing', label: 'Routing & aliases', rows: routingRows })
  }
  if (runtimeRows.length) {
    groups.push({ id: 'runtime', label: 'Runtime & environment', rows: runtimeRows })
  }
  if (paramRows.length) {
    const label = options.engineLabel
      ? `Engine settings · ${options.engineLabel}`
      : 'Engine settings'
    groups.push({ id: 'params', label, rows: paramRows })
  }
  return groups
}

export function countConfigChanges(groups = []) {
  return groups.reduce((total, group) => total + (group.rows?.length || 0), 0)
}
