import { isAudioPackage } from './audioModel'

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

/**
 * A downloaded model needs its first configuration review until a user save
 * stamps it. Only a stopped model offers setup: running, starting, failed and
 * unknown models keep their own state-driven primary. Audio packages use the
 * audio workspace instead of the text-model setup path.
 */
export function modelNeedsSetup(model, runtimeState = null) {
  if (!model || model.config_reviewed !== false) return false
  if (isAudioPackage(model)) return false
  // A never-saved model stays a setup candidate even when the proxy cannot
  // confirm its state: is_active===false plus a stopped/unknown/stale state
  // means it is not running, and editing config is safe. Only an active model
  // keeps its state-driven primary.
  if (model.is_active) return false
  const state = runtimeState || modelRuntimeState(model)
  return ['stopped', 'unknown', 'stale'].includes(state.key)
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

/**
 * Normalize a client-facing API name the way the native catalog publishes it:
 * lower case, path/whitespace runs to `-`, only `[a-z0-9._-]`, collapsed and
 * trimmed. Returns `fallback` when the name normalizes to nothing so callers
 * can show the stable runtime id instead.
 */
export function normalizeApiName(value, fallback = '') {
  const normalized = String(value ?? '')
    .trim()
    .toLowerCase()
    .replace(/[/\\\s]+/g, '-')
    .replace(/[^a-z0-9._-]/g, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^[._-]+|[._-]+$/g, '')
  return normalized || fallback
}
