export function parseTargets(text) {
  return String(text || '')
    .split(/[,;\n]+/)
    .map((part) => part.trim())
    .filter(Boolean)
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
    if (!parseTargets(row.targetsText).length) {
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
      targets: parseTargets(row.targetsText),
    }
    if (row.name?.trim()) block.name = row.name.trim()
    if (row.description?.trim()) block.description = row.description.trim()
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
