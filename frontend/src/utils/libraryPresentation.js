/**
 * Pure presentation helpers shared by the model-library cards and list views.
 *
 * Keeping these out of ModelLibrary.vue lets the view own interactions while
 * card/list identity, formatting and flattening stay independently testable.
 */
import { modelRuntimeState } from './modelPresentation'
import { defaultEngineForFormat } from '@/engines/registry'
import { engineLabel } from '@/composables/engineVersionDelete'

// Decimal (1000) so MB/GB match Hugging Face
export function formatLibraryBytes(bytes) {
  if (!bytes) return ''
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let index = 0
  let value = bytes
  while (value >= 1000 && index < units.length - 1) {
    value /= 1000
    index += 1
  }
  return `${value.toFixed(1)} ${units[index]}`
}

export function formatLibraryDate(iso) {
  if (!iso) return ''
  try {
    return new Intl.RelativeTimeFormat('en', { numeric: 'auto' }).format(
      Math.round((new Date(iso) - Date.now()) / 86400000),
      'day',
    )
  } catch {
    return iso.slice(0, 10)
  }
}

export function groupEngine(group) {
  const quant = primaryQuant(group)
  return (
    (quant?.config && quant.config.engine) || (quant ? defaultEngineForFormat(quant.format) : '')
  )
}

export function prettyEngine(engineId) {
  if (!engineId) return ''
  const label = engineLabel(engineId)
  return label === 'another engine' ? String(engineId) : label
}

export function groupTitle(group) {
  return group?.base_model_name || group?.huggingface_id || 'Model'
}

export function primaryQuant(group) {
  if (!group || !Array.isArray(group.quantizations) || !group.quantizations.length) return null
  return group.quantizations[0]
}

export function isStandaloneGroup(group) {
  if (!group || !Array.isArray(group.quantizations) || !group.quantizations.length) return false
  if (group.quantizations.length === 1) return true
  return group.quantizations.every(
    (quant) =>
      quant.format === 'safetensors' ||
      ['prepared_bundle', 'builtin'].includes(quant.artifact?.package_kind),
  )
}

/** Sum of file_size across all quantizations in a GGUF group (bytes). */
export function groupTotalFileSize(group) {
  if (!group?.quantizations?.length) return 0
  return group.quantizations.reduce((sum, quant) => sum + (Number(quant.file_size) || 0), 0)
}

export function quantStatus(quant) {
  return String(quant?.status || quant?.run_state || '').toLowerCase()
}

export function lineIsRunning(quant) {
  if (!quant || quant.runtime_quality === 'unreachable') return false
  const status = quantStatus(quant)
  return Boolean(quant.is_active) || status === 'ready' || status === 'loading'
}

export function groupIsRunning(group) {
  return (group?.quantizations || []).some((quant) => lineIsRunning(quant))
}

/** Worst runtime observation in a group; a verified group needs every quant verified. */
export function groupRuntimeQuality(group) {
  const quants = group?.quantizations || []
  if (quants.some((quant) => quant?.runtime_quality === 'unreachable')) return 'unreachable'
  if (quants.some((quant) => quant?.runtime_quality === 'stale')) return 'stale'
  return 'verified'
}

export function groupNeedsAttention(group) {
  if (groupRuntimeQuality(group) !== 'verified') return true
  return (group?.quantizations || []).some((quant) => {
    const status = quantStatus(quant)
    return status === 'failed' || status === 'error' || quant?.last_error
  })
}

/** Most recent use/download timestamp in a group, in epoch milliseconds. */
export function groupRecentStamp(group) {
  const stamps = (group?.quantizations || []).map(
    (quant) => Date.parse(quant?.last_used_at || quant?.downloaded_at || '') || 0,
  )
  return Math.max(0, ...stamps)
}

function formatKind(format) {
  if (format === 'gguf') return 'GGUF'
  if (format === 'safetensors') return 'Safetensors'
  return ''
}

function sameIdentity(left, right) {
  const normalize = (value) =>
    String(value || '')
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '')
  const a = normalize(left)
  const b = normalize(right)
  return Boolean(a) && a === b
}

function listMeta(group, quant, includeVariant) {
  const title = groupTitle(group)
  const parts = []
  const repo = group?.huggingface_id
  if (repo && !sameIdentity(title, repo)) parts.push(repo)
  const engine = prettyEngine(quant?.config?.engine || groupEngine(group))
  if (engine) parts.push(engine)
  const variant = quant?.quantization
  const showVariant = includeVariant && variant && variant !== title
  const kind = formatKind(quant?.format)
  if (kind && !showVariant) parts.push(kind)
  if (showVariant) parts.push(variant)
  const taskLabels = { asr: 'ASR', tts: 'Speech', embeddings: 'Embeddings' }
  const tasks = (quant?.tasks || group?.tasks || [])
    .filter((task) => task && task !== 'text-generation')
    .slice(0, 2)
    .map((task) => taskLabels[task] || task)
  if (tasks.length) parts.push(tasks.join(', '))
  return parts.join(' · ')
}

function notableStatus(quant, stateFor) {
  const state = stateFor(quant)
  return state.key === 'stopped' ? null : state
}

/** Flatten visible model groups into the rows shown by the list layout. */
export function buildLibraryLines(groups, { stateFor = modelRuntimeState } = {}) {
  const lines = []
  for (const group of groups) {
    const quants = group.quantizations || []
    const title = groupTitle(group)
    if (quants.length > 1) {
      lines.push({
        key: `group:${group.huggingface_id}`,
        kind: 'group',
        title,
        meta: listMeta(group, primaryQuant(group), false),
        size: groupTotalFileSize(group) ? formatLibraryBytes(groupTotalFileSize(group)) : '',
        status: '',
        statusKey: '',
        running: groupIsRunning(group),
        groupId: group.huggingface_id,
        quant: null,
      })
      for (const quant of quants) {
        const status = notableStatus(quant, stateFor)
        lines.push({
          key: quant.id,
          kind: 'variant',
          title: quant.quantization || quant.name || 'Variant',
          meta: '',
          size: quant.file_size ? formatLibraryBytes(quant.file_size) : '',
          status: status?.label || '',
          statusKey: status?.key || '',
          statusDetail: status?.detail || '',
          running: lineIsRunning(quant),
          quant,
          groupId: group.huggingface_id,
        })
      }
      continue
    }
    const quant = quants[0] || null
    const status = quant ? notableStatus(quant, stateFor) : null
    lines.push({
      key: quant?.id || group.huggingface_id,
      kind: 'single',
      title,
      meta: listMeta(group, quant, true),
      size: quant?.file_size ? formatLibraryBytes(quant.file_size) : '',
      status: status?.label || '',
      statusKey: status?.key || '',
      statusDetail: status?.detail || '',
      running: lineIsRunning(quant),
      quant,
      groupId: group.huggingface_id,
    })
  }
  return lines
}
