/**
 * Pure helpers for the proxy's GPU-admission surfaces.
 *
 * The view keeps the draft state; these helpers turn estimate rows into the
 * settings map the API accepts, summarise a running generation's memory, and
 * turn the last admission decision into one operator-facing notice. All are
 * pure so they can be unit tested without a mounted view or a live backend.
 */

export function vramEstimatePairs(rows) {
  const estimates = {}
  for (const row of rows || []) {
    const model = String(row?.model || '').trim()
    const mib = Number(row?.mib)
    if (model && Number.isFinite(mib) && mib > 0) estimates[model] = Math.round(mib)
  }
  return estimates
}

export function vramSummary(vram, formatNumber = (value) => Number(value).toLocaleString()) {
  if (!vram) return 'Not observed'
  const bits = []
  if (vram.observed_mib != null) bits.push(`${formatNumber(vram.observed_mib)} MiB observed`)
  if (vram.estimate_mib != null) {
    bits.push(`${formatNumber(vram.estimate_mib)} MiB ${vram.estimate_source || 'estimate'}`)
  }
  if (vram.quality === 'contended') bits.push('contended sample')
  return bits.join(' · ') || 'Not observed'
}

export function vramNotice(snapshot, formatNumber = (value) => Number(value).toLocaleString()) {
  const decision = snapshot?.last_admission
  if (!decision || !['warned', 'rejected'].includes(decision.decision)) return null
  const estimate =
    decision.estimate_mib != null
      ? `${formatNumber(decision.estimate_mib)} MiB ${decision.estimate_source || ''}`.trim()
      : 'an unknown estimate'
  const available =
    decision.available_mib != null ? formatNumber(decision.available_mib) : 'unknown'
  const text = `${decision.model || 'A model'} needs ${estimate}, more than the ${available} MiB available for the selected GPU(s). ${decision.reason || ''}`
  return {
    severity: decision.decision === 'rejected' ? 'error' : 'warn',
    text: text.trim(),
  }
}
