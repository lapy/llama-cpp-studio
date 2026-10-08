import { sessionFetch } from '@/api/sessionFetch.js'

/**
 * Audio inference through the Studio origin. Studio handles transport and
 * format compatibility; audio.cpp owns request semantics and model behavior.
 */

export const AUDIO_CPP_TASKS_PATH = '/v1/tasks/run'
export const LLAMA_SWAP_AUDIO_TASKS_PATH = '/audioapi/v1/tasks/run'
const SPEECH_RATE_KEYS = ['speed', 'speaking_rate']

export function isGenericTaskEndpoint(endpoint) {
  const value = String(endpoint || '').trim()
  return value === AUDIO_CPP_TASKS_PATH || value === LLAMA_SWAP_AUDIO_TASKS_PATH
}

/** Request-option keys the model scan advertised as speed controls. */
export function acceptedSpeechRateKeys(registry) {
  const keys = []
  const sections = Array.isArray(registry?.sections) ? registry.sections : []
  for (const section of sections) {
    const params = Array.isArray(section?.params) ? section.params : []
    for (const param of params) {
      if (param?.scope !== 'request_option') continue
      const key = String(param?.key || '')
      if (SPEECH_RATE_KEYS.includes(key) && !keys.includes(key)) keys.push(key)
    }
  }
  return keys
}

/**
 * Top-level speech fields for rates the scan accepts.
 * audio.cpp reads ``speed`` / ``speaking_rate`` on the speech JSON body and
 * rejects them when the model has no speed control.
 */
export function speechRateRequestFields(defaults, acceptedKeys) {
  const accepted = new Set(acceptedKeys || [])
  const options = defaults?.options && typeof defaults.options === 'object' && !Array.isArray(defaults.options)
    ? defaults.options
    : {}
  const fields = {}
  for (const key of SPEECH_RATE_KEYS) {
    if (!accepted.has(key)) continue
    const raw = options[key] != null && options[key] !== '' ? options[key] : defaults?.[key]
    if (raw == null || raw === '' || typeof raw === 'boolean') continue
    const value = typeof raw === 'number' ? raw : Number(raw)
    if (!Number.isFinite(value) || value <= 0) continue
    fields[key] = value
  }
  return fields
}

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

/** Same host as the Studio UI, with the configured llama-swap proxy port. */
export function llamaSwapBaseUrl(proxyPort) {
  const port = Number(proxyPort)
  const resolved = Number.isFinite(port) && port > 0 ? port : 2000
  if (typeof window !== 'undefined' && window.location?.hostname) {
    const protocol = window.location.protocol || 'http:'
    return `${protocol}//${window.location.hostname}:${resolved}`
  }
  return `http://localhost:${resolved}`
}

/**
 * Normalize Studio/workspace bodies to audio.cpp ``request`` object fields.
 */
export function tasksRunRequestObject(input = {}) {
  let requestObj = input
  if (requestObj && typeof requestObj === 'object' && !Array.isArray(requestObj)) {
    if (requestObj.request && typeof requestObj.request === 'object') {
      requestObj = requestObj.request
    } else if (requestObj.input && typeof requestObj.input === 'object') {
      requestObj = requestObj.input
    }
  }
  if (!requestObj || typeof requestObj !== 'object' || Array.isArray(requestObj)) {
    return {}
  }
  const normalized = {}
  for (const [key, value] of Object.entries(requestObj)) {
    if (value == null || value === '') continue
    if (['model', 'task', 'input', 'request', 'busy_timeout_ms'].includes(key)) continue
    normalized[key] = value
  }
  if (!('audio' in normalized)) {
    if (normalized.audio_path) {
      normalized.audio = normalized.audio_path
      delete normalized.audio_path
    } else if (normalized.source_audio) {
      normalized.audio = normalized.source_audio
    }
  }
  if (!('voice_ref' in normalized) && normalized.target_voice) {
    normalized.voice_ref = normalized.target_voice
  }
  return normalized
}

/**
 * Resolve preferred API path for a model config.
 * Prefer inspect/install-recorded preferred_api_endpoint over family hardcodes.
 *
 * Note: voice conversion (vc/svc/s2s) always uses llama-swap ``/audioapi/v1/tasks/run``
 * — the OpenAI speech endpoint has no source-audio field (audio.cpp webui/CLI agree).
 */
export function audioApiEndpoint(config = {}, model = null) {
  const task = String(config.task || '').toLowerCase()
  if (['vc', 'voice_conversion', 'svc', 's2s'].includes(task)) {
    return LLAMA_SWAP_AUDIO_TASKS_PATH
  }

  const preferred = (
    config.preferred_api_endpoint
    || config.api_endpoint
    || model?.preferred_api_endpoint
    || model?.manifest?.inspection?.preferred_api_endpoint
  )
  if (preferred) {
    const value = String(preferred)
    return isGenericTaskEndpoint(value) ? LLAMA_SWAP_AUDIO_TASKS_PATH : value
  }

  if (['align', 'alignment'].includes(task)) {
    return '/v1/audio/alignments'
  }
  if (['asr', 'stt', 'transcription'].includes(task)) {
    return '/v1/audio/transcriptions'
  }
  if (['tts', 'speech', 'clon', 'clone', 'vdes', 'design', 'voice_design'].includes(task)) {
    return '/v1/audio/speech'
  }
  return LLAMA_SWAP_AUDIO_TASKS_PATH
}

/** @deprecated Conversion always uses tasks/run; kept for call-site compatibility. */
export function usesSpeechForConversion(_config = {}, _model = null) {
  return false
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

  const response = await sessionFetch(`${studioAudioBaseUrl()}/speech`, {
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
  details = false,
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

  const endpoint = details ? 'transcriptions/details' : 'transcriptions'
  const response = await sessionFetch(`${studioAudioBaseUrl()}/${endpoint}`, {
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

function batchItem(item, index) {
  if (typeof item === 'string') return { name: `File ${index + 1}`, text: item }
  if (!item || typeof item !== 'object') {
    return { name: `File ${index + 1}`, text: String(item ?? '') }
  }
  const name = item.filename || item.file || item.name || item.id || `File ${index + 1}`
  const text = item.text ?? item.transcript ?? item.transcription
  return {
    name: String(name),
    text: typeof text === 'string' ? text : JSON.stringify(item, null, 2),
  }
}

export function parseBatchTranscriptionPayload(contentType, bodyText) {
  const type = String(contentType || '').toLowerCase()
  const text = String(bodyText || '')
  const trimmed = text.trim()
  if (type.includes('application/json') || trimmed.startsWith('{') || trimmed.startsWith('[')) {
    try {
      const payload = JSON.parse(trimmed)
      if (Array.isArray(payload)) return { results: payload.map(batchItem) }
      if (payload && typeof payload === 'object') {
        const list = payload.results || payload.transcriptions || payload.files
        if (Array.isArray(list)) return { ...payload, results: list.map(batchItem) }
        if (typeof payload.text === 'string') {
          return { text: payload.text, results: [batchItem(payload, 0)] }
        }
      }
    } catch {
      /* Fall through to the event-stream and plain-text readers. */
    }
  }
  if (type.includes('text/event-stream') || text.includes('\ndata:') || text.startsWith('data:')) {
    const events = []
    for (const line of text.split(/\r?\n/)) {
      if (!line.startsWith('data:')) continue
      const data = line.slice(5).trim()
      if (!data || data === '[DONE]') continue
      try {
        events.push(JSON.parse(data))
      } catch {
        events.push({ text: data })
      }
    }
    if (events.length) return { results: events.map(batchItem) }
  }
  return { text, results: text ? [{ name: 'Batch', text }] : [] }
}

/**
 * POST /v1/batches/transcriptions — one multipart request, up to 32 files.
 * Returns { results: [{ name, text }], text? }.
 */
export async function transcribeAudioBatch({
  modelId,
  files,
  language,
  prompt,
  signal,
} = {}) {
  const list = Array.from(files || []).filter(Boolean)
  if (!list.length) throw new Error('Choose at least one audio file.')
  if (list.length > 32) throw new Error('Batch transcription accepts at most 32 files.')
  const form = new FormData()
  form.append('model', modelId)
  for (const file of list) {
    form.append('file', file, file.name || 'audio.wav')
  }
  if (language) form.append('language', language)
  if (prompt) form.append('prompt', prompt)

  const origin = typeof window !== 'undefined' && window.location?.origin ? window.location.origin : ''
  const response = await sessionFetch(`${origin}/v1/batches/transcriptions`, {
    method: 'POST',
    headers: { Authorization: 'Bearer local' },
    body: form,
    signal,
  })
  if (!response.ok) throw new Error(await readErrorDetail(response))
  const contentType = response.headers.get('content-type') || ''
  return parseBatchTranscriptionPayload(contentType, await response.text())
}

/** Voice ids are resolved by the running engine, including its voice library. */
export async function fetchAudioVoices({ modelId, signal } = {}) {
  const response = await sessionFetch(
    `${studioAudioBaseUrl()}/voices?model=${encodeURIComponent(modelId)}`,
    { headers: { Authorization: 'Bearer local' }, signal },
  )
  if (!response.ok) throw new Error(await readErrorDetail(response))
  const result = await response.json()
  return Array.isArray(result.voices) ? result.voices.filter((id) => typeof id === 'string') : []
}

/** Forced alignment accepts a client upload and the known transcript. */
export async function alignAudio({ modelId, file, text, language, signal } = {}) {
  const form = new FormData()
  form.append('model', modelId)
  form.append('file', file, file.name || 'audio.wav')
  form.append('text', text)
  if (language) form.append('language', language)
  const response = await sessionFetch(`${studioAudioBaseUrl()}/alignments`, {
    method: 'POST',
    headers: { Authorization: 'Bearer local' },
    body: form,
    signal,
  })
  if (!response.ok) throw new Error(await readErrorDetail(response))
  return response.json()
}

/**
 * POST Studio /v1/tasks/run with audio.cpp ``{model, request}``.
 */
export async function runAudioTask({
  modelId,
  input = {},
  busyTimeoutMs,
  signal,
} = {}) {
  const request = tasksRunRequestObject(input)
  const body = { model: modelId, request }
  if (busyTimeoutMs != null) body.busy_timeout_ms = busyTimeoutMs

  const response = await sessionFetch(
    `${studioAudioBaseUrl().replace(/\/audio$/, '')}${AUDIO_CPP_TASKS_PATH.replace(/^\/v1/, '')}`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: 'Bearer local',
      },
      body: JSON.stringify(body),
      signal,
    },
  )
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
  if (['svc', 's2s'].includes(task)) return 'convert'
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

function base64ToUint8Array(b64) {
  const binary = atob(String(b64 || '').replace(/\s/g, ''))
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i)
  return bytes
}

/**
 * Decode audio.cpp ``/v1/tasks/run`` JSON into playable WAV blobs.
 * Primary track is ``audio`` (base64 WAV); separation also returns
 * ``named_audio_outputs[{id, audio}]``.
 */
export function extractAudioClipsFromTaskResult(result) {
  if (!result || typeof result !== 'object' || Array.isArray(result)) {
    return { clips: [], meta: null }
  }
  if (result.blob instanceof Blob) {
    return {
      clips: [{ id: 'audio', label: 'Audio', blob: result.blob, filename: 'audio.wav' }],
      meta: null,
    }
  }

  const clips = []
  if (typeof result.audio === 'string' && result.audio) {
    const bytes = base64ToUint8Array(result.audio)
    clips.push({
      id: 'audio',
      label: 'Audio',
      blob: new Blob([bytes], { type: 'audio/wav' }),
      filename: 'audio.wav',
      sample_rate: result.sample_rate,
      channels: result.channels,
    })
  }
  for (const [index, named] of (result.named_audio_outputs || []).entries()) {
    if (typeof named?.audio !== 'string' || !named.audio) continue
    const id = String(named.id || `stem-${index + 1}`)
    const bytes = base64ToUint8Array(named.audio)
    clips.push({
      id,
      label: id,
      blob: new Blob([bytes], { type: 'audio/wav' }),
      filename: `${id.replace(/[^a-zA-Z0-9._-]+/g, '_') || 'stem'}.wav`,
      sample_rate: named.sample_rate,
      channels: named.channels,
    })
  }

  const meta = { ...result }
  delete meta.audio
  delete meta.named_audio_outputs
  return { clips, meta: Object.keys(meta).length ? meta : null }
}
