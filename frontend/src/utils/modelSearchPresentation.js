/**
 * Pure presentation helpers for model-search results.
 *
 * The search view, its result card and its variant rows all render the same
 * catalog payload; keeping these derivations here means the components stay
 * markup-only and the view keeps ownership of store state and actions.
 */

import { isAudioEngineId } from '@/engines/registry'

const RECOMMENDED_QUANT_IDS = Object.freeze([
  'Q4_K_M',
  'Q5_K_M',
  'Q4_K_S',
  'Q6_K',
  'Q8_0',
  'Q3_K_M',
])

/** Stable color per pipeline_tag (PrimeVue Tag severities). Unknown tags get a deterministic hash color. */
const PIPELINE_TAG_SEVERITY = Object.freeze({
  'text-generation': 'success',
  'text2text-generation': 'success',
  'fill-mask': 'info',
  'question-answering': 'info',
  summarization: 'warning',
  translation: 'warning',
  /* Multimodal / vision — avoid `contrast` (often near-white on light tags in dark UI) */
  'image-text-to-text': 'info',
  'image-to-text': 'info',
  'visual-question-answering': 'warning',
  'automatic-speech-recognition': 'info',
  'feature-extraction': 'secondary',
  'text-classification': 'info',
  'token-classification': 'info',
  'sentence-similarity': 'info',
  'reinforcement-learning': 'secondary',
  robotics: 'secondary',
  'depth-estimation': 'warning',
  'tabular-classification': 'info',
  'tabular-regression': 'info',
  'table-question-answering': 'info',
  'object-detection': 'warning',
  'image-classification': 'warning',
  'audio-classification': 'info',
  'audio-to-audio': 'info',
  'text-to-speech': 'info',
  'text-to-audio': 'info',
  'voice-activity-detection': 'secondary',
  other: 'secondary',
})

const PIPELINE_SEVERITY_POOL = ['success', 'info', 'warning', 'secondary']

export function pipelineTagSeverity(pipelineTag) {
  if (pipelineTag == null || pipelineTag === '') return 'secondary'
  const k = String(pipelineTag).toLowerCase().trim()
  if (PIPELINE_TAG_SEVERITY[k]) return PIPELINE_TAG_SEVERITY[k]
  let h = 0
  for (let i = 0; i < k.length; i++) h = (h * 31 + k.charCodeAt(i)) >>> 0
  return PIPELINE_SEVERITY_POOL[h % PIPELINE_SEVERITY_POOL.length]
}

export function catalogVariantKey(variant) {
  return String(variant?.id || variant?.label || '').toUpperCase()
}

export function isRecommendedCatalogVariant(variant) {
  if (variant?.default) return true
  const key = catalogVariantKey(variant)
  return RECOMMENDED_QUANT_IDS.some(
    (id) =>
      key === id || key.endsWith(`-${id}`) || key.includes(`_${id}`) || key.includes(`-${id}`),
  )
}

export function recommendedCatalogVariantScore(variant) {
  const key = catalogVariantKey(variant)
  const idx = RECOMMENDED_QUANT_IDS.findIndex((id) => key === id || key.endsWith(`-${id}`))
  return idx === -1 ? 1000 : idx
}

export function catalogSourceUrl(result) {
  if (result?.provider === 'huggingface' && result?.source?.id) {
    return `https://huggingface.co/${result.source.id}`
  }
  if (isAudioEngineId(result?.provider)) {
    return 'https://github.com/0xShug0/audio.cpp'
  }
  return null
}

export function catalogFormatLabel(result) {
  const format = result?.artifact_format
  if (format === 'gguf') return 'GGUF'
  if (format === 'safetensors') return 'Safetensors'
  if (format === 'mixed') return 'Prepared audio'
  if (format === 'builtin') return 'Built-in'
  return ''
}

export function catalogCardSubtitle(result) {
  if (result?.provider === 'huggingface') {
    return result.source?.id || String(result.provider_item_id || '').split(':')[0] || ''
  }
  return result?.provider_item_id || result?.id || ''
}

// Use decimal (1000) so MB/GB match Hugging Face's display
export function formatBytes(bytes) {
  if (!bytes) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0
  let val = bytes
  while (val >= 1000 && i < units.length - 1) {
    val /= 1000
    i++
  }
  return `${val.toFixed(1)} ${units[i]}`
}

export function formatNumber(n) {
  if (n == null) return ''
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return String(n)
}

/** Card model-index context length → short label for the result row */
export function formatContextLength(ctx) {
  if (ctx == null || ctx === '') return ''
  const n = Number(ctx)
  if (!Number.isFinite(n)) return String(ctx)
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M ctx`
  if (n >= 10_000) return `${Math.round(n / 1000)}k ctx`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k ctx`
  return `${n} ctx`
}

export function formatLanguageHint(langs) {
  if (langs == null) return ''
  const arr = Array.isArray(langs) ? langs : [langs]
  const flat = arr.map((l) => String(l).trim()).filter(Boolean)
  if (!flat.length) return ''
  return flat.slice(0, 4).join(', ')
}

export function formatSearchUpdatedAt(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const diff = Date.now() - d.getTime()
  const day = 86400000
  if (diff < 0) return 'updated'
  if (diff < day) return 'updated today'
  if (diff < 7 * day) return `updated ${Math.floor(diff / day)}d ago`
  if (diff < 60 * day) return `updated ${Math.floor(diff / (7 * day))}w ago`
  return `updated ${d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}`
}

export function catalogResultSizeSummary(result) {
  const sizes = (result?.install_variants || [])
    .map((variant) => Number(variant?.size_bytes) || 0)
    .filter((size) => size > 0)
  if (!sizes.length) return ''
  if (sizes.length === 1) return formatBytes(sizes[0])
  return `from ${formatBytes(Math.min(...sizes))}`
}

export function catalogCardMeta(result) {
  const items = []
  const author = result?.metadata?.raw?.author || result?.author
  const downloads = result?.downloads ?? result?.metadata?.downloads
  const likes = result?.likes ?? result?.metadata?.likes
  if (author) items.push({ key: 'author', icon: 'pi pi-user', label: author })
  if (downloads != null && downloads !== '') {
    items.push({ key: 'downloads', icon: 'pi pi-download', label: formatNumber(downloads) })
  }
  if (likes != null && likes !== '') {
    items.push({ key: 'likes', icon: 'pi pi-heart', label: formatNumber(likes) })
  }
  const raw = result?.metadata?.raw || {}
  const parameters = result?.metadata?.parameters || raw.parameters
  const contextLabel = formatContextLength(result?.metadata?.context_length ?? raw.context_length)
  const languageLabel = formatLanguageHint(
    (result?.languages || []).length ? result.languages : raw.language,
  )
  const license = result?.metadata?.license || raw.license
  const updated = formatSearchUpdatedAt(
    result?.metadata?.last_modified || raw.updated_at || raw.last_modified,
  )
  if (parameters) {
    items.push({ key: 'parameters', icon: 'pi pi-sliders-h', label: parameters })
  }
  if (contextLabel) {
    items.push({ key: 'context', icon: 'pi pi-window-maximize', label: contextLabel })
  }
  if (languageLabel) {
    items.push({ key: 'language', icon: 'pi pi-globe', label: languageLabel })
  }
  if (license) {
    items.push({ key: 'license', icon: 'pi pi-file', label: license })
  }
  if (updated) {
    items.push({ key: 'updated', icon: 'pi pi-calendar', label: updated })
  }
  if (isAudioEngineId(result?.provider)) {
    const format = result?.metadata?.format || result?.install_variants?.[0]?.format
    const precision = result?.metadata?.precision || result?.install_variants?.[0]?.precision
    if (format || precision) {
      items.push({
        key: 'format',
        icon: 'pi pi-box',
        label: [format, precision].filter(Boolean).join(' · '),
      })
    }
    const backend =
      result?.metadata?.manager_backend || result?.install_variants?.[0]?.manager_backend
    if (backend === 'v2') {
      items.push({ key: 'manager', icon: 'pi pi-cog', label: 'spec v2' })
    }
  }
  const sizeSummary = catalogResultSizeSummary(result)
  if (sizeSummary) {
    items.push({ key: 'size', icon: 'pi pi-box', label: sizeSummary })
  }
  return items
}

export function catalogPrimaryBadges(result) {
  const badges = []
  const tasks = result?.tasks || []
  if (result?.family) {
    badges.push({ key: 'family', label: result.family, severity: 'secondary' })
  }
  for (const task of tasks.slice(0, 2)) {
    badges.push({
      key: `task-${task}`,
      label: task,
      severity: pipelineTagSeverity(task),
    })
  }
  const modalities = [
    ...(result?.input_modalities || []).map((m) => `${m} in`),
    ...(result?.output_modalities || []).map((m) => `${m} out`),
  ]
  if (modalities.length) {
    badges.push({
      key: 'modalities',
      label: modalities.slice(0, 2).join(' · '),
      severity: 'secondary',
      tooltip: modalities.join(' · '),
    })
  }
  return badges
}

export function catalogInstallMethodLabel(result, variant) {
  if (!isAudioEngineId(result?.provider)) return ''
  return (
    variant?.method_label ||
    {
      direct: 'Direct HF',
      composite: 'Assemble',
      converter: 'Convert',
      bundled: 'Bundled asset',
    }[variant?.method] ||
    ''
  )
}

export function catalogInstallMethodSeverity(variant) {
  if (variant?.method === 'direct') return 'secondary'
  if (variant?.method === 'bundled') return 'success'
  if (variant?.method === 'composite') return 'info'
  if (variant?.method === 'converter') return 'warn'
  return 'secondary'
}

export function catalogManagerBackendLabel(variant) {
  const backend = String(variant?.manager_backend || '')
    .trim()
    .toLowerCase()
  if (backend === 'v2') return 'Spec v2'
  if (backend === 'legacy') return 'Legacy'
  return ''
}

export function catalogInstallMethodHint(result, variant) {
  if (!isAudioEngineId(result?.provider)) return ''
  if (variant?.method_hint) return variant.method_hint
  if (variant?.uses_model_manager || ['composite', 'converter'].includes(variant?.method)) {
    return 'Installed via the legacy audio.cpp model manager (assemble/convert), not a plain HF snapshot.'
  }
  if (variant?.manager_backend === 'v2') {
    return 'Installed via model_manager_v2 from model_specs package metadata.'
  }
  if (result?.gated || variant?.gated) {
    return 'Gated Hugging Face repo — a Hugging Face token with access is required.'
  }
  return ''
}

export function catalogHasProjector(result) {
  return result?.artifact_format === 'gguf' && (result.metadata?.raw?.mmproj_files || []).length > 0
}

export function catalogHasMtp(result) {
  return result?.artifact_format === 'gguf' && (result.metadata?.raw?.mtp_files || []).length > 0
}

export function catalogHasDflash(result) {
  return result?.artifact_format === 'gguf' && (result.metadata?.raw?.dflash_files || []).length > 0
}

export function catalogHasDspark(result) {
  return result?.artifact_format === 'gguf' && (result.metadata?.raw?.dspark_files || []).length > 0
}

function parseDsparkLabel(filename) {
  const upper = (filename || '').toUpperCase()
  for (const label of ['Q8_0', 'BF16', 'F16', 'F32', 'Q6_K', 'Q5_K', 'Q4_K_M']) {
    if (upper.includes(label)) return label
  }
  return 'Default'
}

function parseProjectorPrecision(filename) {
  const upper = (filename || '').toUpperCase()
  if (upper.includes('BF16')) return 'BF16'
  if (upper.includes('F16')) return 'F16'
  if (upper.includes('F32')) return 'F32'
  return null
}

function parseMtpLabel(filename) {
  const upper = (filename || '').toUpperCase()
  const match = upper.match(
    /\b(Q[0-9]+(?:_[A-Z0-9]+)?|IQ[0-9]_[A-Z0-9]+|UD-[A-Z0-9_]+|BF16|F16|F32)\b/,
  )
  if (match) return match[1]
  const base =
    String(filename || '')
      .split('/')
      .pop() || ''
  if (/^mtp[-_]/i.test(base) && !/\bQ\d/i.test(base)) return 'Default'
  return 'Default'
}

function parseDflashLabel(filename) {
  const upper = (filename || '').toUpperCase()
  const match = upper.match(
    /\b(Q[0-9]+(?:_[A-Z0-9]+)?|IQ[0-9]_[A-Z0-9]+|UD-[A-Z0-9_]+|BF16|F16|F32)\b/,
  )
  if (match) return match[1]
  return 'Default'
}

export function getProjectorOptions(mmprojFiles = []) {
  const byPrecision = new Map()
  mmprojFiles.forEach((file) => {
    const precision = parseProjectorPrecision(file.filename)
    if (!precision || byPrecision.has(precision)) return
    byPrecision.set(precision, {
      label: precision,
      value: file.filename,
      size: file.size || 0,
    })
  })

  return [
    { label: 'None', value: '', size: 0 },
    ...Array.from(byPrecision.values()).sort((a, b) => a.label.localeCompare(b.label)),
  ]
}

export function getMtpOptions(mtpFiles = []) {
  const byLabel = new Map()
  mtpFiles.forEach((file) => {
    const label = file.label || parseMtpLabel(file.filename)
    if (byLabel.has(label)) return
    byLabel.set(label, {
      label,
      value: file.filename,
      size: file.size || 0,
    })
  })
  const preferred = ['Q8_0', 'Q4_0', 'Default', 'BF16', 'F16']
  return [
    { label: 'None', value: '', size: 0 },
    ...Array.from(byLabel.values()).sort((a, b) => {
      const ia = preferred.indexOf(a.label)
      const ib = preferred.indexOf(b.label)
      if (ia !== -1 || ib !== -1) return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib)
      return a.label.localeCompare(b.label)
    }),
  ]
}

export function getDflashOptions(dflashFiles = []) {
  const byLabel = new Map()
  dflashFiles.forEach((file) => {
    const label = file.label || parseDflashLabel(file.filename)
    if (byLabel.has(label)) return
    byLabel.set(label, {
      label,
      value: file.filename,
      size: file.size || 0,
    })
  })
  const preferred = ['BF16', 'F16', 'Default', 'Q8_0', 'Q4_0']
  return [
    { label: 'None', value: '', size: 0 },
    ...Array.from(byLabel.values()).sort((a, b) => {
      const ia = preferred.indexOf(a.label)
      const ib = preferred.indexOf(b.label)
      if (ia !== -1 || ib !== -1) return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib)
      return a.label.localeCompare(b.label)
    }),
  ]
}

export function getDsparkOptions(dsparkFiles = []) {
  const byLabel = new Map()
  dsparkFiles.forEach((file) => {
    const label = file.label || parseDsparkLabel(file.filename)
    if (byLabel.has(label)) return
    byLabel.set(label, {
      label,
      value: file.filename,
      size: file.size || 0,
    })
  })
  const preferred = ['Q8_0', 'BF16', 'F16', 'Default']
  return [
    { label: 'None', value: '', size: 0 },
    ...Array.from(byLabel.values()).sort((a, b) => {
      const ia = preferred.indexOf(a.label)
      const ib = preferred.indexOf(b.label)
      if (ia !== -1 || ib !== -1) return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib)
      return a.label.localeCompare(b.label)
    }),
  ]
}

export function catalogProjectorOptions(result) {
  return getProjectorOptions(result?.metadata?.raw?.mmproj_files || [])
}

export function catalogMtpOptions(result) {
  return getMtpOptions(result?.metadata?.raw?.mtp_files || [])
}

export function catalogDflashOptions(result) {
  return getDflashOptions(result?.metadata?.raw?.dflash_files || [])
}

export function catalogDsparkOptions(result) {
  return getDsparkOptions(result?.metadata?.raw?.dspark_files || [])
}

export function catalogVariantActionLabel(result) {
  return result?.provider === 'huggingface' ? 'Download' : 'Install'
}

export function catalogVariantsSummaryTitle(result) {
  const count = (result?.install_variants || []).length
  if (result?.artifact_format === 'gguf') {
    return `${count} quantization${count === 1 ? '' : 's'}`
  }
  return `${count} install option${count === 1 ? '' : 's'}`
}

export function catalogVariantsSummaryMeta(result) {
  const sizeSummary = catalogResultSizeSummary(result)
  const labels = (result?.install_variants || [])
    .map((variant) => variant.label || variant.id)
    .filter(Boolean)
  if (!labels.length) return sizeSummary || 'Open to choose a variant'
  const preview = labels.slice(0, 4).join(' · ')
  const remaining = labels.length - 4
  const labelSummary = remaining > 0 ? `${preview} · +${remaining} more` : preview
  return sizeSummary ? `${sizeSummary} · ${labelSummary}` : labelSummary
}

export function catalogVariantsBrowseLabel(result) {
  return result?.artifact_format === 'gguf' ? 'Browse quants' : 'Browse options'
}
