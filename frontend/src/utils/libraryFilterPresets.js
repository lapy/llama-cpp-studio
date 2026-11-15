/**
 * Saved model-library filter presets.
 *
 * Presets are local view metadata: a name plus the URL-addressable filter
 * fields. Everything is stored under one versioned key so a future schema can
 * ignore incompatible data instead of guessing at it.
 */

export const LIBRARY_PRESETS_KEY = 'llama-studio.library.filter-presets.v1'
export const LIBRARY_PRESETS_VERSION = 1
export const MAX_LIBRARY_PRESETS = 20
export const MAX_PRESET_NAME_LENGTH = 60
const MAX_FILTER_TEXT_LENGTH = 200

const FILTER_STATUSES = ['all', 'running', 'stopped', 'attention']
const FILTER_SORTS = ['name', 'size', 'recent']
const FILTER_DEFAULTS = Object.freeze({
  q: '',
  status: 'all',
  engine: 'all',
  task: 'all',
  sort: 'name',
})

const STATUS_TEXT = {
  running: 'Running',
  stopped: 'Stopped',
  attention: 'Needs attention',
}
const SORT_TEXT = {
  name: 'Name',
  size: 'Size',
  recent: 'Recent use',
}

let fallbackIdCounter = 0

function cleanText(value, maxLength) {
  return typeof value === 'string' ? value.trim().slice(0, maxLength) : ''
}

function createPresetId() {
  const uuid = globalThis.crypto?.randomUUID?.()
  if (uuid) return `preset-${uuid}`
  fallbackIdCounter += 1
  return `preset-${Date.now().toString(36)}-${fallbackIdCounter}`
}

/** Keep only the filters a URL can restore, with defaults for invalid values. */
export function normalizePresetFilters(filters = {}) {
  const input = filters && typeof filters === 'object' ? filters : {}
  return {
    q: cleanText(input.q, MAX_FILTER_TEXT_LENGTH),
    status: FILTER_STATUSES.includes(input.status) ? input.status : FILTER_DEFAULTS.status,
    engine: cleanText(input.engine, MAX_FILTER_TEXT_LENGTH) || FILTER_DEFAULTS.engine,
    task: cleanText(input.task, MAX_FILTER_TEXT_LENGTH) || FILTER_DEFAULTS.task,
    sort: FILTER_SORTS.includes(input.sort) ? input.sort : FILTER_DEFAULTS.sort,
  }
}

function resolveStorage(storage) {
  if (storage && typeof storage.getItem === 'function') return storage
  try {
    return typeof localStorage === 'undefined' ? null : localStorage
  } catch {
    return null
  }
}

function persistPresets(presets, storage) {
  const target = resolveStorage(storage)
  if (!target) throw new Error('Saved presets are unavailable in this browser.')
  try {
    target.setItem(
      LIBRARY_PRESETS_KEY,
      JSON.stringify({ version: LIBRARY_PRESETS_VERSION, presets }),
    )
  } catch {
    throw new Error('Saved presets could not be written. Browser storage may be full.')
  }
}

/** Parse a raw localStorage value; malformed or foreign data yields no presets. */
export function parsePresets(raw) {
  if (!raw) return []
  let payload
  try {
    payload = JSON.parse(raw)
  } catch {
    return []
  }
  const entries = Array.isArray(payload)
    ? payload
    : payload?.version === LIBRARY_PRESETS_VERSION && Array.isArray(payload.presets)
      ? payload.presets
      : null
  if (!entries) return []
  const presets = []
  const names = new Set()
  for (const entry of entries) {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) continue
    const name = cleanText(entry.name, MAX_PRESET_NAME_LENGTH)
    if (!name || names.has(name.toLowerCase())) continue
    names.add(name.toLowerCase())
    presets.push({
      id: cleanText(entry.id, 100) || createPresetId(),
      name,
      filters: normalizePresetFilters(entry.filters),
      createdAt: typeof entry.createdAt === 'string' ? entry.createdAt : '',
    })
    if (presets.length >= MAX_LIBRARY_PRESETS) break
  }
  return presets
}

export function loadPresets(storage) {
  const target = resolveStorage(storage)
  if (!target) return []
  try {
    return parsePresets(target.getItem(LIBRARY_PRESETS_KEY))
  } catch {
    return []
  }
}

export function savePreset(name, filters, options = {}) {
  const presetName = cleanText(name, MAX_PRESET_NAME_LENGTH)
  if (!presetName) throw new Error('Enter a name for the preset.')
  const presets = loadPresets(options.storage)
  if (presets.some((preset) => preset.name.toLowerCase() === presetName.toLowerCase())) {
    throw new Error(`A preset named “${presetName}” already exists. Delete it before saving again.`)
  }
  if (presets.length >= MAX_LIBRARY_PRESETS) {
    throw new Error(
      `You can save up to ${MAX_LIBRARY_PRESETS} presets. Delete one before adding another.`,
    )
  }
  const preset = {
    id: cleanText(options.id, 100) || createPresetId(),
    name: presetName,
    filters: normalizePresetFilters(filters),
    createdAt: typeof options.now === 'string' ? options.now : new Date().toISOString(),
  }
  const next = [...presets, preset]
  persistPresets(next, options.storage)
  return { preset, presets: next }
}

export function deletePreset(id, options = {}) {
  const presets = loadPresets(options.storage)
  const next = presets.filter((preset) => preset.id !== id)
  if (next.length !== presets.length) persistPresets(next, options.storage)
  return next
}

/** Short human summary used next to a preset name. */
export function describePresetFilters(filters) {
  const normalized = normalizePresetFilters(filters)
  const parts = []
  if (normalized.q) parts.push(`Name: ${normalized.q}`)
  if (normalized.status !== FILTER_DEFAULTS.status) {
    parts.push(`Status: ${STATUS_TEXT[normalized.status]}`)
  }
  if (normalized.engine !== FILTER_DEFAULTS.engine) parts.push(`Engine: ${normalized.engine}`)
  if (normalized.task !== FILTER_DEFAULTS.task) parts.push(`Task: ${normalized.task}`)
  if (normalized.sort !== FILTER_DEFAULTS.sort) parts.push(`Sort: ${SORT_TEXT[normalized.sort]}`)
  return parts.length ? parts.join(' · ') : 'No filters'
}
