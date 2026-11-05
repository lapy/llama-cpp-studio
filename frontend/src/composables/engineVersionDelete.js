import { ENGINE_LABELS, serverEngineLabel } from '../engines/registry.js'

export function engineLabel(engineId) {
  // YAML campaign S5: live server descriptors win the name (identity is
  // one system); the local table + raw id are the offline fallback. For
  // the nine, the two sources carry byte-identical labels by backend
  // registry-equivalence pin, so rendered copy is stable either way.
  return serverEngineLabel(engineId) || ENGINE_LABELS[engineId] || engineId || 'another engine'
}

export function activeVersionDeletePlan(version) {
  const selected = Array.isArray(version?.selected_models)
    ? version.selected_models.filter(Boolean)
    : []
  const dormant = Array.isArray(version?.dormant_models)
    ? version.dormant_models.filter((row) => row?.name)
    : []
  if (version?.is_active && selected.length) {
    return {
      allowed: false,
      message: `This engine is selected by ${selected.join(', ')}. Choose another engine for those models before deleting the active version.`,
    }
  }
  const dormantLines = dormant.map((row) => `${row.name} (using ${engineLabel(row.engine)})`)
  if (dormantLines.length) {
    const lead = version?.is_active
      ? `Delete the active version "${version.version}"?`
      : `Delete version "${version.version}"?`
    const detail =
      dormantLines.length === 1
        ? `${dormantLines[0]} still has saved settings for this engine, but that model is using another engine.`
        : `These models still have saved settings for this engine, but each is using another engine: ${dormantLines.join('; ')}.`
    return {
      allowed: true,
      message: `${lead} ${detail} Delete it anyway?`,
    }
  }
  if (version?.is_active) {
    return {
      allowed: true,
      message: `Delete the active version "${version.version}"? No model is using this engine.`,
    }
  }
  return {
    allowed: true,
    message: `Delete version "${version?.version || version?.id}"?`,
  }
}
