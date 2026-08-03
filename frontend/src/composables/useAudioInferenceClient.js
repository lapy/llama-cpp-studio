/**
 * OpenAI-compatible audio inference via Studio /v1/audio (Approach A).
 * ASR multipart always goes through Studio so non-WAV uploads are converted.
 * Generic tasks use Studio /v1/audio/tasks/run (proxied to llama-swap upstream).
 */

/** Dedicated VC families that use /v1/tasks/run (not OpenAI speech). */
const DEDICATED_VC_FAMILIES = new Set(['vevo2', 'seed_vc', 'miocodec'])

/** TTS-family VC/clone that stays on /v1/audio/speech (mirrors backend fixtures). */
const SPEECH_VC_FAMILIES = new Set([
  'chatterbox',
  'moss_tts_nano',
  'moss_tts_local',
  'index_tts2',
])

export function audioInferenceModelId(model, config = null) {
  const cfg = config || model?.config || {}
  return (
    cfg.model_alias
    || model?.llama_swap_id
    || model?.proxy_name
    || model?.id
    || ''
  )
}

export function studioAudioBaseUrl() {
  if (typeof window !== 'undefined' && window.location?.origin) {
    return `${window.location.origin}/v1/audio`
  }
  return '/v1/audio'
}

function familyKey(config = {}) {
  return String(config.family || '').toLowerCase().replace(/-/g, '_')
}

/**
 * Resolve preferred API path for a model config.
 * Prefer inspect/install-recorded preferred_api_endpoint over family hardcodes.
 */
export function audioApiEndpoint(config = {}, model = null) {
  const preferred = (
    config.preferred_api_endpoint
    || config.api_endpoint
    || model?.preferred_api_endpoint
    || model?.manifest?.inspection?.preferred_api_endpoint
  )
  if (preferred) return String(preferred)

  const task = String(config.task || '').toLowerCase()
  const family = familyKey(config)

  if (['asr', 'stt', 'transcription'].includes(task)) {
    return '/v1/audio/transcriptions'
  }
  if (['tts', 'speech', 'clon', 'clone', 'vdes', 'design', 'voice_design'].includes(task)) {
    return '/v1/audio/speech'
  }
  if (['vc', 'voice_conversion', 'svc', 's2s'].includes(task)) {
    if (DEDICATED_VC_FAMILIES.has(family)) return '/v1/tasks/run'
    if (SPEECH_VC_FAMILIES.has(family)) return '/v1/audio/speech'
    if (config.speech_defaults && typeof config.speech_defaults === 'object') {
      return '/v1/audio/speech'
    }
    return '/v1/tasks/run'
  }
  return '/v1/tasks/run'
}

export function usesSpeechForConversion(config = {}, model = null) {
  return audioApiEndpoint(config, model) === '/v1/audio/speech'
    && ['vc', 'voice_conversion', 'svc', 's2s', 'clon', 'clone'].includes(
      String(config.task || '').toLowerCase(),
    )
}

async function readErrorDetail(response) {
  const text = await response.text().catch(() => '')
  if (!text) return `HTTP ${response.status}`
  try {
    const json = JSON.parse(text)
    if (typeof json.detail === 'string') return json.detail
    if (json.error?.message) return json.error.message
    if (json.message) return json.message
    return text
  } catch {
    return text
  }
}

/**
 * POST /v1/audio/speech — returns { blob, contentType }
 */
export async function synthesizeSpeech({
  modelId,
  input,
  voice,
  extras = {},
  signal,
} = {}) {
  const body = {
    model: modelId,
    input,
    ...extras,
  }
  if (voice != null && voice !== '') body.voice = voice

  const response = await fetch(`${studioAudioBaseUrl()}/speech`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: 'Bearer local',
    },
    body: JSON.stringify(body),
    signal,
  })
  if (!response.ok) {
    throw new Error(await readErrorDetail(response))
  }
  const contentType = response.headers.get('content-type') || 'audio/wav'
  const blob = await response.blob()
  return { blob, contentType }
}

/**
 * POST /v1/audio/transcriptions (multipart) — returns parsed JSON or { text }
 */
export async function transcribeAudio({
  modelId,
  file,
  filename,
  language,
  prompt,
  extras = {},
  signal,
} = {}) {
  const form = new FormData()
  form.append('model', modelId)
  form.append('file', file, filename || file.name || 'audio.wav')
  if (language) form.append('language', language)
  if (prompt) form.append('prompt', prompt)
  for (const [key, value] of Object.entries(extras)) {
    if (value == null || value === '') continue
    form.append(key, typeof value === 'string' ? value : String(value))
  }

  const response = await fetch(`${studioAudioBaseUrl()}/transcriptions`, {
    method: 'POST',
    headers: {
      Authorization: 'Bearer local',
    },
    body: form,
    signal,
  })
  if (!response.ok) {
    throw new Error(await readErrorDetail(response))
  }
  const contentType = response.headers.get('content-type') || ''
  if (contentType.includes('application/json')) {
    return response.json()
  }
  const text = await response.text()
  return { text }
}

/**
 * POST Studio /v1/audio/tasks/run → llama-swap /upstream/{model}/v1/tasks/run
 */
export async function runAudioTask({
  modelId,
  task,
  input = {},
  signal,
} = {}) {
  const response = await fetch(`${studioAudioBaseUrl()}/tasks/run`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: 'Bearer local',
    },
    body: JSON.stringify({ model: modelId, task, input }),
    signal,
  })
  if (!response.ok) {
    throw new Error(await readErrorDetail(response))
  }
  const contentType = response.headers.get('content-type') || ''
  if (contentType.includes('application/json')) {
    return response.json()
  }
  return { blob: await response.blob(), contentType }
}

export function taskKindFromConfig(config = {}) {
  const task = String(config.task || '').toLowerCase()
  if (['asr', 'stt', 'transcription'].includes(task)) return 'transcribe'
  if (['vc', 'voice_conversion'].includes(task)) return 'convert'
  if (['sep', 'separation'].includes(task)) return 'separate'
  if (['vad', 'diar', 'diarization', 'align', 'alignment'].includes(task)) return 'analyze'
  if (['gen', 'music', 'sfx'].includes(task)) return 'music'
  if (['design', 'vdes', 'voice_design'].includes(task)) return 'design'
  // clon/clone stay on Speech (TTS / clone) — OpenAI speech path
  if (['tts', 'speech', 'clon', 'clone'].includes(task) || !task) return 'speech'
  return 'speech'
}

/** Map workspace tab id for deep links from a model config. */
export function audioTabFromConfig(config = {}) {
  return taskKindFromConfig(config)
}
