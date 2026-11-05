/** One vocabulary for observed model lifecycle state across every workspace. */
export function modelRuntimeState(model) {
  const quality = String(model?.runtime_quality || 'verified').toLowerCase()
  const state = String(model?.status || model?.run_state || '').toLowerCase()
  if (quality === 'unreachable' || quality === 'unknown')
    return { key: 'unknown', label: 'Status unavailable', severity: 'warn', actionable: false }
  if (quality === 'stale')
    return {
      key: 'stale',
      label: model?.is_active ? 'Last seen running' : 'Status stale',
      severity: 'warn',
      actionable: false,
    }
  if (state === 'failed' || state === 'error' || model?.last_error)
    return { key: 'error', label: 'Failed', severity: 'danger', actionable: false }
  if (state === 'loading' || state === 'starting')
    return { key: 'loading', label: 'Loading', severity: 'warn', actionable: false }
  if (state === 'ready')
    return { key: 'ready', label: 'Ready', severity: 'success', actionable: true }
  if (model?.is_active)
    return { key: 'running', label: 'Running', severity: 'success', actionable: true }
  return { key: 'stopped', label: 'Stopped', severity: 'secondary', actionable: false }
}

export function groupRuntimeState(group) {
  const states = (group?.quantizations || []).map(modelRuntimeState)
  const priority = ['unknown', 'stale', 'error', 'loading', 'ready', 'running', 'stopped']
  return (
    states.sort((a, b) => priority.indexOf(a.key) - priority.indexOf(b.key))[0] ||
    modelRuntimeState(null)
  )
}

export function taskLabel(value) {
  return (
    {
      'text-generation': 'Text generation',
      embeddings: 'Embeddings',
      tts: 'Speech',
      asr: 'Transcription',
      vc: 'Voice conversion',
      gen: 'Audio generation',
    }[value] || value
  )
}

export function formatLabel(value) {
  return ({ gguf: 'GGUF', safetensors: 'Safetensors', audio_cpp: 'Audio package' })[value] || value
}
