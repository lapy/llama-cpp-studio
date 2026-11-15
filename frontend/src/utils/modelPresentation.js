/** One vocabulary for observed model lifecycle state across every workspace. */
export function modelRuntimeState(model, options = {}) {
  const quality = String(model?.runtime_quality || 'verified').toLowerCase()
  const state = String(model?.status || model?.run_state || '').toLowerCase()
  if (options.isStarting)
    return {
      key: 'starting',
      label: 'Starting',
      severity: 'info',
      actionable: false,
      detail: 'Waiting for the model to report Running.',
    }
  if (options.isStopping)
    return {
      key: 'stopping',
      label: 'Stopping',
      severity: 'warn',
      actionable: false,
      detail: 'Waiting for the model to stop.',
    }
  if (quality === 'unreachable' || quality === 'unknown')
    return {
      key: 'unknown',
      label: 'Status unavailable',
      severity: 'warn',
      actionable: false,
      detail: 'The inference proxy could not be reached, so this status may be out of date.',
    }
  if (quality === 'stale')
    return {
      key: 'stale',
      label: 'Status unavailable',
      severity: 'warn',
      actionable: false,
      detail: model?.is_active
        ? 'Last seen running. The inference proxy is unreachable.'
        : 'The last known status is stale. The inference proxy is unreachable.',
    }
  if (state === 'failed' || state === 'error' || model?.last_error)
    return {
      key: 'error',
      label: 'Failed',
      severity: 'danger',
      actionable: false,
      detail: model?.last_error || 'The last start attempt failed.',
    }
  if (state === 'loading' || state === 'starting')
    return {
      key: 'starting',
      label: 'Starting',
      severity: 'info',
      actionable: false,
      detail: 'The model is loading. Stop cancels this start.',
    }
  if (state === 'ready' || model?.is_active)
    return {
      key: 'running',
      label: 'Running',
      severity: 'success',
      actionable: true,
      detail:
        state === 'ready'
          ? model?.is_active
            ? 'Ready and accepting requests.'
            : 'Ready to accept requests.'
          : 'The model process is running.',
    }
  return {
    key: 'stopped',
    label: 'Stopped',
    severity: 'secondary',
    actionable: false,
    detail: 'The model is not running.',
  }
}

export function groupRuntimeState(group) {
  const states = (group?.quantizations || []).map((quant) => modelRuntimeState(quant))
  const priority = ['unknown', 'stale', 'error', 'starting', 'running', 'stopped']
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
  return { gguf: 'GGUF', safetensors: 'Safetensors', audio_cpp: 'Audio package' }[value] || value
}
