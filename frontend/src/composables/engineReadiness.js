/**
 * Shared engine-card and checklist decisions from /api/engines descriptors.
 * Installed-but-not-runnable builds stay out of the ready set.
 */

const AUDIO_TASKS = new Set(['vad', 'asr', 'diar', 'sep', 'gen', 'tts', 'clon', 'vc', 's2s', 'align'])

export function engineCardOrder(descriptor) {
  if (descriptor?.runnable) return 0
  if (Number(descriptor?.installed_versions) > 0) return 1
  return 2
}

export function engineCardCta(descriptor, { installed = false, active = false } = {}) {
  const hasActive = Boolean(descriptor?.active_version) || active
  const hasInstall = Number(descriptor?.installed_versions) > 0 || installed
  if (hasActive) return 'Manage'
  if (hasInstall) return 'Activate'
  return 'Install'
}

export function engineMatchesFilters(descriptor, { task = 'all', hardware = 'all' } = {}) {
  if (!descriptor || (task === 'all' && hardware === 'all')) return true
  const tasks = new Set(descriptor.tasks || [])
  const formats = new Set(descriptor.artifact_formats || [])
  // R77: category membership follows the descriptor's declared tasks/formats
  // and its cpu_supported field — no engine-id special cases.
  const isAudioEngine = [...tasks].some((item) => AUDIO_TASKS.has(item))
  if (task === 'audio') {
    if (!isAudioEngine) return false
  } else if (task === 'embeddings') {
    if (!descriptor.supports_embeddings && !tasks.has('embeddings')) return false
  } else if (task === 'text') {
    if (tasks.size && !tasks.has('text-generation') && !tasks.has('text2text-generation')) return false
  }
  if (hardware === 'cpu') {
    if (!formats.has('gguf') && !descriptor.cpu_supported) return false
  }
  // R78: GPU-architecture membership is a declared capability, not an id list.
  if (hardware === 'sm70') {
    if (!(descriptor.gpu_archs || []).includes('sm70')) return false
  }
  return true
}

export function runnableEngineIds(descriptors) {
  return (descriptors || [])
    .filter((descriptor) => descriptor && descriptor.enabled !== false && descriptor.runnable)
    .map((descriptor) => descriptor.id)
}
