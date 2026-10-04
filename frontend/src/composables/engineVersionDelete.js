const ENGINE_LABELS = {
  llama_cpp: 'llama.cpp',
  ik_llama: 'ik_llama.cpp',
  unsloth_llama: 'Unsloth llama.cpp',
  lmdeploy: 'LMDeploy',
  '1cat_vllm': '1Cat-vLLM',
  sglang: 'SGLang',
  sglang_v100: 'SGLang V100',
  vllm: 'vLLM',
  audio_cpp: 'audio.cpp',
}

export function engineLabel(engineId) {
  return ENGINE_LABELS[engineId] || engineId || 'another engine'
}

export function activeVersionDeletePlan(version) {
  const selected = Array.isArray(version?.selected_models) ? version.selected_models.filter(Boolean) : []
  const dormant = Array.isArray(version?.dormant_models) ? version.dormant_models.filter((row) => row?.name) : []
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
    const detail = dormantLines.length === 1
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
