export const STRATEGY_OPTIONS = [
  {
    value: 'warm',
    label: 'warm',
    hint: 'Prefer a target that is already loaded; otherwise start the first available.',
  },
  {
    value: 'pin',
    label: 'pin',
    hint: 'Always route to the first target (ignore warm cache).',
  },
  {
    value: 'spillover',
    label: 'spillover',
    hint: 'Fill each target up to the spillover limit, then overflow to the next.',
  },
]

export function strategyHint(strategy) {
  const found = STRATEGY_OPTIONS.find((opt) => opt.value === strategy)
  return found?.hint || ''
}

export function parseTargets(text) {
  return String(text || '')
    .split(/[,;\n]+/)
    .map((part) => part.trim())
    .filter(Boolean)
}

export function normalizeTargetList(raw) {
  if (Array.isArray(raw)) {
    const out = []
    const seen = new Set()
    for (const item of raw) {
      const target = String(item || '').trim()
      if (!target || seen.has(target)) continue
      seen.add(target)
      out.push(target)
    }
    return out
  }
  return parseTargets(raw)
}

export function metadataFromRows(rows) {
  const out = {}
  for (const row of rows || []) {
    const key = String(row?.key || '').trim()
    if (!key) continue
    out[key] = row?.value != null ? String(row.value) : ''
  }
  return out
}

export function metadataToRows(metadata) {
  if (!metadata || typeof metadata !== 'object' || Array.isArray(metadata)) {
    return [{ key: '', value: '' }]
  }
  const entries = Object.entries(metadata)
  if (!entries.length) return [{ key: '', value: '' }]
  return entries.map(([key, value]) => ({
    key,
    value: value == null ? '' : String(value),
  }))
}

/**
 * Collect catalog ids usable as selector/profile pin targets.
 * Includes stable llama-swap ids, routing aliases, and setParamsByID sub-ids.
 */
export function collectCatalogTargetIds(quantizations = []) {
  const ids = new Set()
  for (const model of quantizations) {
    if (!model || typeof model !== 'object') continue
    for (const key of ['llama_swap_id', 'proxy_name', 'routing_name']) {
      const value = String(model[key] || '').trim()
      if (value) ids.add(value)
    }
    const cfg = model.config && typeof model.config === 'object' ? model.config : {}
    const engines = cfg.engines && typeof cfg.engines === 'object' ? cfg.engines : {}
    const sections = [
      cfg,
      ...Object.values(engines).filter((section) => section && typeof section === 'object'),
    ]
    for (const section of sections) {
      const alias = String(section.model_alias || '').trim()
      if (alias) ids.add(alias)
      const variants = Array.isArray(section.set_params_by_id) ? section.set_params_by_id : []
      for (const variant of variants) {
        const sub = String(variant?.sub_id || '').trim()
        if (!sub) continue
        const base =
          String(section.model_alias || '').trim() ||
          String(model.llama_swap_id || model.proxy_name || '').trim()
        if (base) ids.add(`${base}:${sub}`)
        ids.add(sub)
      }
    }
  }
  return [...ids].sort((a, b) => a.localeCompare(b))
}

export function validateRoutingForm(selectorRows, profileRows) {
  const errors = []
  const selectorIds = new Set()
  const profileIds = new Set()

  for (const [idx, row] of selectorRows.entries()) {
    const id = String(row.id || '').trim()
    const n = idx + 1
    if (!id) {
      errors.push(`Selector ${n}: id is required`)
      continue
    }
    if (selectorIds.has(id)) {
      errors.push(`Selector id «${id}» is duplicated`)
    }
    selectorIds.add(id)
    const targets = normalizeTargetList(row.targets ?? row.targetsText)
    if (!targets.length) {
      errors.push(`Selector «${id}»: at least one target is required`)
    }
    if (!['warm', 'pin', 'spillover'].includes(row.strategy)) {
      errors.push(`Selector «${id}»: strategy must be warm, pin, or spillover`)
    }
  }

  for (const [idx, row] of profileRows.entries()) {
    const id = String(row.id || '').trim()
    const n = idx + 1
    if (!id) {
      errors.push(`Profile ${n}: id is required`)
      continue
    }
    if (profileIds.has(id)) {
      errors.push(`Profile id «${id}» is duplicated`)
    }
    profileIds.add(id)
    const pins = (row.pins || []).filter((pin) => String(pin.key || '').trim())
    if (!pins.length) {
      errors.push(`Profile «${id}»: at least one pin is required`)
    }
  }

  for (const id of selectorIds) {
    if (profileIds.has(id)) {
      errors.push(`Id «${id}» is used by both a profile and a selector`)
    }
  }

  return errors
}

export function buildRoutingPayload(selectorRows, profileRows) {
  const selectors = {}
  for (const row of selectorRows) {
    const id = String(row.id || '').trim()
    if (!id) continue
    const block = {
      strategy: row.strategy || 'warm',
      targets: normalizeTargetList(row.targets ?? row.targetsText),
    }
    if (row.name?.trim()) block.name = row.name.trim()
    if (row.description?.trim()) block.description = row.description.trim()
    if (row.unlisted) block.unlisted = true
    const metadata = metadataFromRows(row.metadataRows)
    if (Object.keys(metadata).length) block.metadata = metadata
    if (block.strategy === 'spillover') {
      block.settings = {
        spillover: Number(row.spillover) > 0 ? Number(row.spillover) : 1,
      }
    }
    selectors[id] = block
  }

  const profiles = {}
  for (const row of profileRows) {
    const id = String(row.id || '').trim()
    if (!id) continue
    const pins = {}
    for (const pin of row.pins || []) {
      const key = String(pin.key || '').trim()
      if (!key) continue
      pins[key] = String(pin.target ?? '').trim()
    }
    profiles[id] = {
      description: String(row.description || '').trim(),
      pins,
    }
  }
  return { profiles, selectors }
}
